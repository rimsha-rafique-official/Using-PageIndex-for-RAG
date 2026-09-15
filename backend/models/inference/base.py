"""
Inference abstraction.

An inference engine takes a chat history + generation params and
yields tokens. Whether it's Ollama, llama.cpp, or vLLM is hidden.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal, Optional


Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str


@dataclass
class GenerationParams:
    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    num_ctx: int = 8192
    num_predict: int = 1024
    stop: list[str] = field(default_factory=list)
    seed: Optional[int] = None


@dataclass
class StreamChunk:
    """One streamed unit from the model."""
    text: str
    done: bool = False
    # Stats are populated on the final chunk only
    total_duration_ms: Optional[int] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class BaseInference(ABC):
    @abstractmethod
    async def generate(
        self,
        messages: list[ChatMessage],
        params: Optional[GenerationParams] = None,
        model: Optional[str] = None,
    ) -> str: ...

    @abstractmethod
    def stream(
        self,
        messages: list[ChatMessage],
        params: Optional[GenerationParams] = None,
        model: Optional[str] = None,
    ) -> AsyncIterator[StreamChunk]: ...
