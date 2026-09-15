"""
Ollama loader. Talks to the local Ollama daemon's REST API.

Why Ollama? It's how qwen3-vl:2b is already installed on the user's
machine. We don't ship our own runtime — Ollama owns the model files,
GGUF parsing, GPU offload, and so on.
"""
from __future__ import annotations
import httpx
from typing import Optional

from backend.config import settings
from backend.utils import get_logger
from .base import BaseLoader, ModelInfo


log = get_logger("loader.ollama")


class OllamaLoader(BaseLoader):
    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0):
        self.base_url = (base_url or settings.ollama.base_url).rstrip("/")
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _client_(self) -> httpx.AsyncClient:
        if self._client is None:
            # trust_env=False -> ignore Windows system proxy / env vars.
            # Ollama is always on localhost and never needs a proxy.
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=self.timeout,
                trust_env=False,
            )
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ---- BaseLoader ----

    async def check_health(self) -> bool:
        try:
            c = await self._client_()
            r = await c.get("/api/tags")
            return r.status_code == 200
        except Exception as e:
            log.warning("Ollama health check failed: %s", e)
            return False

    async def list_available(self) -> list[ModelInfo]:
        c = await self._client_()
        r = await c.get("/api/tags")
        r.raise_for_status()
        models = r.json().get("models", [])
        out: list[ModelInfo] = []
        for m in models:
            details = m.get("details", {}) or {}
            out.append(
                ModelInfo(
                    name=m.get("name", ""),
                    backend="ollama",
                    size_bytes=m.get("size"),
                    digest=m.get("digest"),
                    family=details.get("family"),
                    parameter_size=details.get("parameter_size"),
                    quantization=details.get("quantization_level"),
                )
            )
        return out

    async def ensure_loaded(self, model_name: str) -> ModelInfo:
        """
        Verify the model is installed. We do NOT auto-pull — pulling a
        model is a multi-GB operation and should be an explicit user step.
        """
        available = await self.list_available()
        for m in available:
            if m.name == model_name or m.name.startswith(model_name + ":"):
                # Warm up by sending an empty generate. This loads weights
                # into RAM/VRAM so the first real request is fast.
                c = await self._client_()
                try:
                    await c.post(
                        "/api/generate",
                        json={
                            "model": m.name,
                            "prompt": "",
                            "keep_alive": settings.ollama.keep_alive,
                        },
                        timeout=60.0,
                    )
                except Exception as e:
                    log.info("Warm-up skipped (%s)", e)
                log.info("Model ready: %s", m.name)
                return m

        raise RuntimeError(
            f"Model '{model_name}' not found in Ollama. "
            f"Run: `ollama pull {model_name}`"
        )