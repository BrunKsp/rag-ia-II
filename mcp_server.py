"""
RF02: Servidor MCP com FastMCP expondo 3 tools de consulta RAG.

Tools:
  - search_knowledge: busca semântica no banco vetorial
  - get_document: recupera documento específico por ID
  - list_topics: lista culturas e estados disponíveis

Execução:
  python mcp_server.py                  # HTTP streamable na porta 8001
  MCP_PORT=8002 python mcp_server.py    # porta customizada
"""

import os
from fastmcp import FastMCP
from pydantic import Field
from dotenv import load_dotenv
import rag_core

load_dotenv()

mcp = FastMCP(
    name="Tábua de Risco Safra 2026-2027",
    instructions=(
        "Servidor de conhecimento sobre a tábua de risco agroclimático safra 2026-2027. "
        "Fornece acesso a dados de risco por cultura, município, estado e décade."
    ),
)


@mcp.tool()
def search_knowledge(
    query: str = Field(description="Consulta em linguagem natural sobre risco agrícola"),
    top_k: int = Field(default=5, ge=1, le=20, description="Número de trechos a recuperar"),
    topic: str | None = Field(default=None, description="Filtrar por cultura específica (opcional)"),
) -> dict:
    """
    Consulta o banco vetorial por similaridade semântica e retorna os trechos
    mais relevantes sobre risco de safra, com metadados e score de relevância.
    """
    return rag_core.search_knowledge(query=query, top_k=top_k, topic=topic)


@mcp.tool()
def get_document(
    doc_id: str = Field(
        description="ID do documento no formato 'Cultura_UF', ex: 'Soja_GO', 'Cevada_Grãos_MG'"
    ),
) -> dict:
    """
    Recupera o conteúdo completo e os metadados de um documento específico
    identificado pelo par cultura-estado (doc_id).
    """
    return rag_core.get_document(doc_id=doc_id)


@mcp.tool()
def list_topics() -> dict:
    """
    Lista todos os tópicos (culturas) e estados disponíveis no corpus da
    tábua de risco safra 2026-2027, com contagem de chunks por tópico.
    """
    return rag_core.list_topics()


if __name__ == "__main__":
    port = int(os.getenv("MCP_PORT", "8001"))
    print(f"Servidor MCP iniciando na porta {port}...")
    print(f"Endpoint: http://0.0.0.0:{port}/mcp")
    mcp.run(transport="streamable-http", host="0.0.0.0", port=port)
