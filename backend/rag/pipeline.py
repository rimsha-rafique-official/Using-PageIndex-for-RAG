"""
RAG pipeline.

This is the orchestrator. It does NOT know about HTTP or transports —
it only knows: take a question, use PageIndexService to retrieve, hand
those sections to the local model, stream tokens out.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import AsyncIterator, Optional

from backend.config import settings
from backend.mcp import PageIndexService
from backend.models.inference import (
    BaseInference,
    ChatMessage,
    GenerationParams,
    StreamChunk,
)
from backend.models.prompts import build_rag_messages
from backend.utils import get_logger


log = get_logger("rag.pipeline")


@dataclass
class RAGAnswer:
    text: str
    sources: list[dict]
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_duration_ms: Optional[int] = None


class RAGPipeline:
    def __init__(
        self,
        pageindex: PageIndexService,
        inference: BaseInference,
    ):
        self.pageindex = pageindex
        self.inference = inference

    # ---- retrieval ----

    async def retrieve(self, doc_id: str, question: str) -> list[dict]:
        result = await self.pageindex.retrieve(
            doc_id, question, max_nodes=settings.rag.max_nodes_in_context
        )
        sections = result["sections"][: settings.rag.max_nodes_in_context]
        log.info("Retrieved %d sections for query: %s", len(sections), question[:60])
        return sections

    # ---- generation ----

    async def answer(
        self,
        doc_id: Optional[str],
        question: str,
        history: list[dict] | None = None,
        params: Optional[GenerationParams] = None,
    ) -> RAGAnswer:
        sections = await self.retrieve(doc_id, question) if doc_id else []
        messages = build_rag_messages(question, sections, history)
        text = await self.inference.generate(messages, params=params)
        return RAGAnswer(text=text, sources=sections)

    async def stream_answer(
        self,
        doc_id: Optional[str],
        question: str,
        history: list[dict] | None = None,
        params: Optional[GenerationParams] = None,
    ) -> AsyncIterator[dict]:
        """
        Yields events:
          {"event": "retrieval", "sections": [...]}
          {"event": "token", "text": "..."}
          {"event": "done", "stats": {...}}
        """
        sections = await self.retrieve(doc_id, question) if doc_id else []
        yield {"event": "retrieval", "sections": sections}

        messages = build_rag_messages(question, sections, history)
        async for chunk in self.inference.stream(messages, params=params):
            if chunk.text:
                yield {"event": "token", "text": chunk.text}
            if chunk.done:
                yield {
                    "event": "done",
                    "stats": {
                        "prompt_tokens": chunk.prompt_tokens,
                        "completion_tokens": chunk.completion_tokens,
                        "total_duration_ms": chunk.total_duration_ms,
                    },
                }
