"""
Motor de consulta RAG.
Busca contexto no ChromaDB e gera resposta com LLM da NVIDIA.
"""

import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
LLM_MODEL = "meta/llama-3.3-70b-instruct"
COLLECTION_NAME = "tabua_risco_safra"
TOP_K = 5

SYSTEM_PROMPT = """Você é um especialista em zoneamento agroclimático e tábua de risco de safra do Brasil.
Responda perguntas sobre riscos agrícolas com base nos dados fornecidos.
Os dados são da Tábua de Risco Safra 2026-2027, publicada pelo governo federal.
As décadas representam períodos de 10 dias. Os valores de risco vão de 0 a 20 (porcentagem de risco de sinistro).
Responda sempre em português, de forma clara e objetiva. Se não souber, diga que não encontrou nos dados."""


def get_query_embedding(client: OpenAI, text: str) -> list[float]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
        extra_body={"input_type": "query", "truncate": "END"},
    )
    return response.data[0].embedding


def build_context(results: dict) -> str:
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    parts = []
    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        parts.append(f"[{i}] {doc}")
    return "\n\n".join(parts)


def query_rag(question: str, top_k: int = TOP_K) -> str:
    client = OpenAI(
        base_url="https://integrate.api.nvidia.com/v1",
        api_key=NVIDIA_API_KEY,
    )

    chroma = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma.get_collection(COLLECTION_NAME)

    embedding = get_query_embedding(client, question)

    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    context = build_context(results)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Contexto recuperado da base de dados:\n\n{context}\n\n"
                f"Pergunta: {question}"
            ),
        },
    ]

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=1024,
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Quais culturas têm maior risco em Minas Gerais?"
    print(f"\nPergunta: {q}\n")
    answer = query_rag(q)
    print(f"Resposta:\n{answer}")
