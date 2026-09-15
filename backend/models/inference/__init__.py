from .base import BaseInference, ChatMessage, GenerationParams, StreamChunk, Role
from .ollama_inference import OllamaInference

__all__ = [
    "BaseInference",
    "ChatMessage",
    "GenerationParams",
    "StreamChunk",
    "Role",
    "OllamaInference",
]
