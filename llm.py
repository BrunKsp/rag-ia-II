"""
RF03: Orquestração LLM com function calling.

Fluxo:
  1. System prompt define domínio e restrições
  2. LLM recebe as tools MCP no formato OpenAI
  3. Automatic function calling: LLM decide quais tools usar
  4. Backend executa as chamadas e devolve resultados ao LLM
  5. Resposta final inclui fontes documentais
  6. Guardrail de saída valida fundamentação
"""

import json
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv
from guardrails import validate_output, GuardrailError

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "meta/llama-3.3-70b-instruct")
MAX_TOOL_ROUNDS = 3

SYSTEM_PROMPT = """Você é um assistente especialista em zoneamento agroclimático e tábua de risco de safra do Brasil para a safra 2026-2027.

DOMÍNIO: Dados oficiais da Tábua de Risco Safra 2026-2027, publicada pelo Ministério da Agricultura via portarias oficiais.

RESPONSABILIDADES:
- Responder perguntas sobre riscos agrícolas usando SEMPRE as ferramentas disponíveis para buscar dados
- Citar as fontes (município, cultura, portaria) em todas as respostas
- Informar explicitamente quando não houver dados suficientes: "Não encontrei informações sobre..."
- Manter o foco exclusivo no domínio agroclimático

DADOS TÉCNICOS:
- Décadas (dec1-dec36): períodos de 10 dias do ano calendário
- Valores de risco: 0 a 20 (percentual de risco de sinistro por décade)
- Manejos: Irrigado, Sequeiro, entre outros
- Ciclos: identificados por código numérico

INSTRUÇÃO: Antes de responder, sempre use search_knowledge para buscar dados relevantes.
Responda em português brasileiro, de forma clara e objetiva."""

# Definição das tools no formato OpenAI (espelha as tools do servidor MCP)
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Consulta o banco vetorial por similaridade semântica e retorna os trechos mais relevantes sobre risco de safra.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Consulta em linguagem natural sobre risco agrícola",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Número de trechos a recuperar (padrão 5, máx 20)",
                    },
                    "topic": {
                        "type": "string",
                        "description": "Filtrar por cultura específica, ex: 'Soja', 'Milho'",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_document",
            "description": "Recupera o conteúdo completo de um documento por ID no formato 'Cultura_UF'.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": "ID do documento, ex: 'Soja_GO', 'Milho_MG', 'Cevada_Grãos_SP'",
                    }
                },
                "required": ["doc_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_topics",
            "description": "Lista todas as culturas e estados disponíveis no corpus da tábua de risco.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


async def run_inference(query: str, mcp_caller, history: list[dict] | None = None) -> dict:
    """
    Executa o pipeline RAG completo com function calling.

    Returns:
        dict com 'answer', 'sources', 'tool_calls'.
    """
    client = AsyncOpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY,
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": query})

    retrieved_docs: list[str] = []
    sources: list[dict] = []
    tool_calls_log: list[dict] = []
    query_embedding: list[float] | None = None

    # Gera embedding da query para armazenar junto ao log de Q&A
    try:
        from rag_core import _query_embedding
        query_embedding = _query_embedding(query)
    except Exception:
        pass

    for _ in range(MAX_TOOL_ROUNDS):
        response = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
            max_tokens=1024,
        )

        msg = response.choices[0].message

        if not msg.tool_calls:
            answer = msg.content or ""
            try:
                answer = validate_output(answer, retrieved_docs)
            except GuardrailError:
                answer = (
                    "Não foi possível gerar uma resposta fundamentada nos dados disponíveis. "
                    "Por favor, reformule sua pergunta especificando a cultura e/ou município."
                )
            return {
                "answer": answer,
                "sources": sources,
                "tool_calls": tool_calls_log,
                "query_embedding": query_embedding,
            }

        # Adiciona mensagem do assistente com tool_calls
        messages.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ],
        })

        # Executa cada tool call
        for tc in msg.tool_calls:
            tool_name = tc.function.name
            tool_args = json.loads(tc.function.arguments)
            tool_calls_log.append({"tool": tool_name, "args": tool_args})

            result = await mcp_caller(tool_name, tool_args)
            result_str = json.dumps(result, ensure_ascii=False)

            # Coleta fontes de search_knowledge
            if tool_name == "search_knowledge" and "results" in result:
                for item in result["results"]:
                    retrieved_docs.append(item["text"])
                    meta = item.get("metadata", {})
                    src = {
                        "doc_id": meta.get("doc_id", ""),
                        "topic": meta.get("topic", ""),
                        "uf": meta.get("uf", ""),
                        "portaria": meta.get("portaria", ""),
                        "score": item.get("score", 0),
                    }
                    if src not in sources:
                        sources.append(src)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result_str,
            })

    return {
        "answer": "Não foi possível completar a consulta. Tente uma pergunta mais específica.",
        "sources": sources,
        "tool_calls": tool_calls_log,
        "query_embedding": query_embedding,
    }
