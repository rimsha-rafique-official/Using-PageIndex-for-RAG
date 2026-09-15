"""
Ollama inference engine.

Uses the /api/chat endpoint, which handles Qwen's chat template
automatically. Streams via NDJSON lines.
"""
from __future__ import annotations
import json
from typing import AsyncIterator, Optional

import httpx

from backend.config import settings
from backend.utils import get_logger
from .base import BaseInference, ChatMessage, GenerationParams, StreamChunk


log = get_logger("inference.ollama")


def _params_to_options(p: GenerationParams) -> dict:
    """Translate our generic params into Ollama's `options` block."""
    opts = {
        "temperature": p.temperature,
        "top_p": p.top_p,
        "top_k": p.top_k,
        "repeat_penalty": p.repeat_penalty,
        "num_ctx": p.num_ctx,
        "num_predict": p.num_predict,
    }
    if p.stop:
        opts["stop"] = p.stop
    if p.seed is not None:
        opts["seed"] = p.seed
    return opts


class OllamaInference(BaseInference):
    def __init__(
        self,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.base_url = (base_url or settings.ollama.base_url).rstrip("/")
        self.default_model = default_model or settings.ollama.model
        self.timeout = timeout or settings.ollama.request_timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _c(self) -> httpx.AsyncClient:
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

    # ---- BaseInference ----

    async def generate(
        self,
        messages: list[ChatMessage],
        params: Optional[GenerationParams] = None,
        model: Optional[str] = None,
    ) -> str:
        params = params or GenerationParams()
        body = {
            "model": model or self.default_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "options": _params_to_options(params),
            "keep_alive": settings.ollama.keep_alive,
        }
        c = await self._c()
        r = await c.post("/api/chat", json=body)
        r.raise_for_status()
        data = r.json()
        return data.get("message", {}).get("content", "").strip()

    async def stream(
        self,
        messages: list[ChatMessage],
        params: Optional[GenerationParams] = None,
        model: Optional[str] = None,
    ) -> AsyncIterator[StreamChunk]:
        params = params or GenerationParams()
        body = {
            "model": model or self.default_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
            "options": _params_to_options(params),
            "keep_alive": settings.ollama.keep_alive,
        }
        c = await self._c()
        async with c.stream("POST", "/api/chat", json=body) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("error"):
                    raise RuntimeError(f"Ollama error: {obj['error']}")
                msg = obj.get("message", {}) or {}
                text_piece = msg.get("content", "") or ""
                done = bool(obj.get("done"))
                if done:
                    yield StreamChunk(
                        text=text_piece,
                        done=True,
                        total_duration_ms=(obj.get("total_duration") or 0) // 1_000_000
                        if obj.get("total_duration")
                        else None,
                        prompt_tokens=obj.get("prompt_eval_count"),
                        completion_tokens=obj.get("eval_count"),
                    )
                else:
                    if text_piece:
                        yield StreamChunk(text=text_piece, done=False)