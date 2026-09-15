"""
Qwen-specific helpers.

We use Ollama's /api/chat which applies the model's bundled chat
template automatically, so we don't need to format ChatML manually.
This file exists for the case where you bypass /api/chat (e.g. the
/api/generate raw endpoint) and need the template yourself.

Qwen3 ChatML format:

    <|im_start|>system
    {system}<|im_end|>
    <|im_start|>user
    {user}<|im_end|>
    <|im_start|>assistant
"""
from __future__ import annotations
from backend.models.inference import ChatMessage


def render_chatml(messages: list[ChatMessage], add_generation_prompt: bool = True) -> str:
    parts: list[str] = []
    for m in messages:
        parts.append(f"<|im_start|>{m.role}\n{m.content}<|im_end|>")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "\n".join(parts)
