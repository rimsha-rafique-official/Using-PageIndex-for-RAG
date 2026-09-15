"""
FastAPI app. Run with:
    python -m backend.api.app
or
    uvicorn backend.api.app:app --host 127.0.0.1 --port 8000
"""
from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.api.deps import init_container, shutdown_container
from backend.api import routes_documents, routes_chat, routes_system
from backend.utils import get_logger


log = get_logger("api.app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_container()
    try:
        yield
    finally:
        await shutdown_container()


app = FastAPI(
    title="Local RAG — Ollama + PageIndex MCP",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes_system.router)
app.include_router(routes_documents.router)
app.include_router(routes_chat.router)


# Serve frontend if present
_frontend_dir = settings.project_root / "frontend"
if _frontend_dir.exists():
    app.mount("/ui", StaticFiles(directory=str(_frontend_dir), html=True), name="ui")


@app.get("/")
async def root():
    return {
        "name": "Local RAG",
        "ui": "/ui/",
        "docs": "/docs",
        "health": "/system/health",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api.app:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=False,
        log_level="info",
    )
