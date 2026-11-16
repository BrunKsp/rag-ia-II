"""
Armazenamento opcional no MongoDB para logs de Q&A e embeddings.
Se MONGO_URI não estiver configurado, todas as operações são no-op silenciosas.
"""

import os
from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "")
DB_NAME = os.getenv("MONGO_DB", "rag_safra")

_client = None
_db = None


def _get_db():
    global _client, _db
    if not MONGO_URI:
        return None
    if _db is None:
        from motor.motor_asyncio import AsyncIOMotorClient
        _client = AsyncIOMotorClient(MONGO_URI)
        _db = _client[DB_NAME]
    return _db


async def _ensure_indexes():
    db = _get_db()
    if db is None:
        return
    try:
        await db["qa_logs"].create_index("created_at")
        await db["qa_logs"].create_index("blocked")
    except Exception:
        pass


async def log_qa(
    query: str,
    query_embedding: list[float] | None,
    answer: str,
    sources: list[dict],
    tool_calls: list[dict],
    blocked: bool = False,
    block_reason: str = "",
) -> None:
    db = _get_db()
    if db is None:
        return
    try:
        doc = {
            "query": query,
            "answer": answer,
            "sources": sources,
            "tool_calls": tool_calls,
            "blocked": blocked,
            "block_reason": block_reason,
            "created_at": datetime.now(timezone.utc),
        }
        if query_embedding:
            doc["query_embedding"] = query_embedding
        await db["qa_logs"].insert_one(doc)
    except Exception:
        pass


async def save_embeddings(docs: list[dict]) -> int:
    db = _get_db()
    if db is None:
        return 0
    try:
        result = await db["embeddings"].insert_many(docs, ordered=False)
        return len(result.inserted_ids)
    except Exception:
        return 0


async def get_qa_logs(limit: int = 50, skip: int = 0, blocked_only: bool = False) -> list[dict]:
    db = _get_db()
    if db is None:
        return []
    try:
        query = {"blocked": True} if blocked_only else {}
        cursor = db["qa_logs"].find(query, {"query_embedding": 0}).sort("created_at", -1).skip(skip).limit(limit)
        docs = await cursor.to_list(length=limit)
        for doc in docs:
            doc["_id"] = str(doc["_id"])
        return docs
    except Exception:
        return []


async def stats() -> dict:
    db = _get_db()
    if db is None:
        return {"connected": False}
    try:
        total = await db["qa_logs"].count_documents({})
        blocked = await db["qa_logs"].count_documents({"blocked": True})
        return {"connected": True, "total_queries": total, "blocked_queries": blocked}
    except Exception:
        return {"connected": False}
