"""
Loader abstraction.

A loader's job is to verify a model exists, return its metadata,
and warm it up if needed. It does NOT do inference — that's the
inference layer's job. Keeping these separate means we can swap
Ollama for llama.cpp / vLLM / mlx later by adding a new loader.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelInfo:
    name: str
    backend: str               # "ollama" | "llama-cpp" | ...
    size_bytes: Optional[int] = None
    digest: Optional[str] = None
    family: Optional[str] = None
    parameter_size: Optional[str] = None
    quantization: Optional[str] = None


class BaseLoader(ABC):
    """Contract every model loader must implement."""

    @abstractmethod
    async def check_health(self) -> bool: ...

    @abstractmethod
    async def list_available(self) -> list[ModelInfo]: ...

    @abstractmethod
    async def ensure_loaded(self, model_name: str) -> ModelInfo: ...
