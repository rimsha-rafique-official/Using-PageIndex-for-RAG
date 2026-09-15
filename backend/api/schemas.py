"""
Request/response schemas. Pydantic v2 syntax.
"""
from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel, Field


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class GenerationOverrides(BaseModel):
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1, le=200)
    num_ctx: Optional[int] = Field(default=None, ge=512, le=131072)
    num_predict: Optional[int] = Field(default=None, ge=1, le=8192)
    repeat_penalty: Optional[float] = Field(default=None, ge=0.5, le=2.0)
    seed: Optional[int] = None


class ChatRequest(BaseModel):
    question: str
    doc_id: Optional[str] = None
    history: list[HistoryMessage] = Field(default_factory=list)
    params: Optional[GenerationOverrides] = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[dict]
    stats: dict = Field(default_factory=dict)


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    raw: dict


class DocumentInfo(BaseModel):
    doc_id: str
    name: Optional[str] = None
    pages: Optional[int] = None
    extra: dict = Field(default_factory=dict)


class HealthResponse(BaseModel):
    ollama: bool
    model: str
    pageindex_connected: bool
    pageindex_tools: list[str]
