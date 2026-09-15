"""
Dependency container.

We instantiate the pipeline once at app startup and reuse it across
requests. Using a small container instead of FastAPI Depends keeps
things explicit and easy to follow.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from backend.config import settings
from backend.mcp import PageIndexService
from backend.models.inference import OllamaInference
from backend.models.loaders import OllamaLoader
from backend.rag import RAGPipeline
from backend.utils import get_logger


log = get_logger("api.deps")


@dataclass
class AppContainer:
    loader: OllamaLoader
    inference: OllamaInference
    pageindex: PageIndexService
    pipeline: RAGPipeline


_container: Optional[AppContainer] = None


async def init_container() -> AppContainer:
    global _container

    log.info("Initializing components...")

    loader = OllamaLoader()
    if not await loader.check_health():
        raise RuntimeError(
            f"Cannot reach Ollama at {settings.ollama.base_url}. "
            f"Start it with: `ollama serve`"
        )

    await loader.ensure_loaded(settings.ollama.model)

    inference = OllamaInference()
    pageindex = PageIndexService()
    await pageindex.start()

    pipeline = RAGPipeline(pageindex=pageindex, inference=inference)

    _container = AppContainer(
        loader=loader,
        inference=inference,
        pageindex=pageindex,
        pipeline=pipeline,
    )
    log.info("Ready.")
    return _container


async def shutdown_container() -> None:
    global _container
    if _container is None:
        return
    await _container.inference.close()
    await _container.loader.close()
    await _container.pageindex.stop()
    _container = None


def get_container() -> AppContainer:
    if _container is None:
        raise RuntimeError("Container not initialized")
    return _container
