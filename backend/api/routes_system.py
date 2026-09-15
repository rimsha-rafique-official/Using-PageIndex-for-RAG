"""
System routes: health check, models, settings inspection.
"""
from __future__ import annotations
from fastapi import APIRouter

from backend.config import settings
from backend.api.deps import get_container
from backend.api.schemas import HealthResponse


router = APIRouter(prefix="/system", tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health():
    container = get_container()
    ollama_ok = await container.loader.check_health()
    tools = [t.name for t in container.pageindex.tools]
    return HealthResponse(
        ollama=ollama_ok,
        model=settings.ollama.model,
        pageindex_connected=bool(tools),
        pageindex_tools=tools,
    )


@router.get("/models")
async def list_models():
    container = get_container()
    models = await container.loader.list_available()
    return [
        {
            "name": m.name,
            "size_bytes": m.size_bytes,
            "digest": m.digest,
            "family": m.family,
            "parameter_size": m.parameter_size,
            "quantization": m.quantization,
        }
        for m in models
    ]


@router.get("/config")
async def config():
    return {
        "ollama": {
            "base_url": settings.ollama.base_url,
            "model": settings.ollama.model,
            "keep_alive": settings.ollama.keep_alive,
        },
        "generation": {
            "temperature": settings.generation.temperature,
            "top_p": settings.generation.top_p,
            "top_k": settings.generation.top_k,
            "num_ctx": settings.generation.num_ctx,
            "num_predict": settings.generation.num_predict,
        },
        "pageindex": {
            "transport": settings.pageindex.transport,
            "tools": {
                "list": settings.pageindex.tool_list_documents,
                "submit": settings.pageindex.tool_submit_document,
                "tree": settings.pageindex.tool_get_tree,
                "retrieve": settings.pageindex.tool_retrieve,
            },
        },
        "rag": {
            "max_nodes_in_context": settings.rag.max_nodes_in_context,
            "max_chars_per_node": settings.rag.max_chars_per_node,
        },
    }
