"""
RF05: Backend FastAPI.

Integra guardrails, MCP server e LLM em um único endpoint de chat.
O MCP server pode rodar separadamente (python mcp_server.py) ou
as tools são chamadas diretamente via rag_core como fallback.
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from guardrails import validate_input, GuardrailError
from llm import run_inference
import mongo_store

load_dotenv()

MCP_URL = os.getenv("MCP_URL", "http://localhost:8001/mcp")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Backend iniciado. MCP URL: {MCP_URL}")
    asyncio.create_task(mongo_store._ensure_indexes())
    yield


app = FastAPI(
    title="RAG Tábua de Risco Safra 2026-2027",
    description="API RAG sobre dados de risco agroclimático safra 2026-2027",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def call_mcp_tool(tool_name: str, args: dict) -> dict:
    """Chama tool via MCP server HTTP ou fallback para rag_core direto."""
    try:
        from fastmcp import Client
        async with Client(MCP_URL) as client:
            result = await client.call_tool(tool_name, args)
            if result and len(result) > 0:
                content = result[0]
                text = getattr(content, "text", None)
                if text:
                    return json.loads(text)
                return {"result": str(content)}
    except Exception as e:
        print(f"MCP call failed ({e}), usando fallback direto")

    # Fallback: importa rag_core diretamente
    import rag_core
    fn = getattr(rag_core, tool_name, None)
    if fn is None:
        raise HTTPException(500, f"Tool '{tool_name}' não encontrada")
    return fn(**args)


class ChatRequest(BaseModel):
    query: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    tool_calls: list[dict]
    blocked: bool = False
    block_reason: str = ""


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # RF04: Guardrail de entrada
    try:
        clean_query = validate_input(request.query)
    except GuardrailError as e:
        asyncio.create_task(
            mongo_store.log_qa(
                query=request.query,
                query_embedding=None,
                answer=str(e),
                sources=[],
                tool_calls=[],
                blocked=True,
                block_reason=e.reason,
            )
        )
        return ChatResponse(
            answer=str(e),
            sources=[],
            tool_calls=[],
            blocked=True,
            block_reason=e.reason,
        )

    try:
        result = await run_inference(
            query=clean_query,
            mcp_caller=call_mcp_tool,
            history=request.history,
        )
        asyncio.create_task(
            mongo_store.log_qa(
                query=clean_query,
                query_embedding=result.get("query_embedding"),
                answer=result["answer"],
                sources=result["sources"],
                tool_calls=result["tool_calls"],
            )
        )
        return ChatResponse(**{k: v for k, v in result.items() if k != "query_embedding"})
    except Exception as e:
        raise HTTPException(500, f"Erro na inferência: {str(e)}")


@app.get("/health")
async def health():
    mongo_stats = await mongo_store.stats()
    return {"status": "ok", "mcp_url": MCP_URL, "mongo": mongo_stats}


@app.get("/eval/logs")
async def eval_logs(
    limit: int = Query(50, ge=1, le=200),
    skip: int = Query(0, ge=0),
    blocked_only: bool = Query(False),
):
    """Retorna logs de Q&A para avaliação do modelo."""
    return await mongo_store.get_qa_logs(limit=limit, skip=skip, blocked_only=blocked_only)


@app.get("/topics")
async def topics():
    return await call_mcp_tool("list_topics", {})


@app.get("/document/{doc_id}")
async def document(doc_id: str):
    return await call_mcp_tool("get_document", {"doc_id": doc_id})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
