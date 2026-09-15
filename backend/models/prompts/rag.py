"""
Prompt builders.

A builder takes structured data (retrieved sections, history,
user question) and returns a list[ChatMessage] ready for inference.
Keeping these as pure functions makes them trivial to test.
"""
from __future__ import annotations
from typing import Iterable

from backend.models.inference import ChatMessage
from backend.config import settings

from .system import RAG_SYSTEM, NO_DOC_SYSTEM


def build_rag_messages(
    question: str,
    retrieved_sections: list[dict],
    history: list[dict] | None = None,
) -> list[ChatMessage]:
    """
    retrieved_sections: list of {title, pages, text, node_id}
    history: list of {role, content}  (excluding the current question)
    """
    if not retrieved_sections:
        return [
            ChatMessage(role="system", content=NO_DOC_SYSTEM),
            ChatMessage(role="user", content=question),
        ]

    context_block = _format_context(retrieved_sections)
    user_content = (
        f"<context>\n{context_block}\n</context>\n\n"
        f"Question: {question}"
    )

    msgs: list[ChatMessage] = [ChatMessage(role="system", content=RAG_SYSTEM)]
    if history:
        for h in history[-6:]:  # cap history so the prompt stays bounded
            role = h.get("role")
            if role in ("user", "assistant"):
                msgs.append(ChatMessage(role=role, content=h.get("content", "")))
    msgs.append(ChatMessage(role="user", content=user_content))
    return msgs


def _format_context(sections: Iterable[dict]) -> str:
    parts: list[str] = []
    max_chars = settings.rag.max_chars_per_node
    for i, s in enumerate(sections, start=1):
        title = s.get("title") or f"Section {i}"
        pages = s.get("pages") or s.get("page_range") or ""
        text = (s.get("text") or "")[:max_chars]
        header = f"[{i}] {title}" + (f" (pages {pages})" if pages else "")
        parts.append(f"{header}\n{text}")
    return "\n\n---\n\n".join(parts)
