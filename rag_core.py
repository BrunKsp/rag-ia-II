"""
Funções centrais do RAG: busca semântica, recuperação de documentos e listagem de tópicos.
Usadas tanto pelo servidor MCP quanto pelo backend.
"""

import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
EMBEDDING_MODEL = "nvidia/nv-embedqa-mistral-7b-v2"
COLLECTION_NAME = "tabua_risco_safra"
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(_BASE_DIR, "chroma_db")

_openai_client: OpenAI | None = None
_chroma_collection = None


def _get_client() -> OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NVIDIA_API_KEY,
        )
    return _openai_client


def _get_collection():
    global _chroma_collection
    if _chroma_collection is None:
        chroma = chromadb.PersistentClient(path=CHROMA_PATH)
        _chroma_collection = chroma.get_collection(COLLECTION_NAME)
    return _chroma_collection


def _query_embedding(text: str) -> list[float]:
    response = _get_client().embeddings.create(
        model=EMBEDDING_MODEL,
        input=[text],
        extra_body={"input_type": "query", "truncate": "END"},
    )
    return response.data[0].embedding


def search_knowledge(query: str, top_k: int = 5, topic: str | None = None) -> dict:
    """
    Consulta o banco vetorial por similaridade semântica.

    Args:
        query: Pergunta em linguagem natural sobre risco agrícola.
        top_k: Número de trechos a recuperar (1-20).
        topic: Filtrar por cultura específica (opcional).

    Returns:
        dict com 'query', 'results' (list de chunks com texto, metadata e score) e 'total'.
    """
    collection = _get_collection()
    embedding = _query_embedding(query)

    where = {"topic": topic} if topic else None

    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(max(1, top_k), 20),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": doc,
            "metadata": meta,
            "score": round(1 - float(dist), 4),
        })

    return {"query": query, "results": chunks, "total": len(chunks)}


def get_document(doc_id: str) -> dict:
    """
    Recupera todos os chunks de um documento pelo seu doc_id.

    Args:
        doc_id: ID no formato 'Cultura_UF', ex: 'Soja_GO' ou 'Cevada_Grãos_MG'.

    Returns:
        dict com 'doc_id', 'chunks' e 'total_chunks'. Retorna 'error' se não encontrado.
    """
    collection = _get_collection()

    results = collection.get(
        where={"doc_id": doc_id},
        include=["documents", "metadatas"],
    )

    if not results["documents"]:
        return {"error": f"Documento '{doc_id}' não encontrado.", "doc_id": doc_id}

    return {
        "doc_id": doc_id,
        "chunks": [
            {"text": doc, "metadata": meta}
            for doc, meta in zip(results["documents"], results["metadatas"])
        ],
        "total_chunks": len(results["documents"]),
    }


def list_topics() -> dict:
    """
    Lista todos os tópicos (culturas) e estados disponíveis no corpus.

    Returns:
        dict com 'topics' (nome, contagem de chunks, estados) e 'states'.
    """
    collection = _get_collection()
    results = collection.get(include=["metadatas"])

    topics: dict[str, dict] = {}
    all_states: set[str] = set()

    for meta in results["metadatas"]:
        topic = meta.get("topic", "")
        uf = meta.get("uf", "")

        if topic:
            if topic not in topics:
                topics[topic] = {"chunk_count": 0, "states": set()}
            topics[topic]["chunk_count"] += 1
            if uf:
                topics[topic]["states"].add(uf)

        if uf:
            all_states.add(uf)

    return {
        "topics": [
            {
                "name": name,
                "chunk_count": info["chunk_count"],
                "states": sorted(info["states"]),
            }
            for name, info in sorted(topics.items())
        ],
        "total_topics": len(topics),
        "states": sorted(all_states),
    }
