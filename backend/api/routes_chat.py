"""
Chat routes:
  POST /chat           -> non-streaming JSON
  POST /chat/stream    -> Server-Sent Events
"""
from __future__ import annotations
import json
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from backend.api.deps import get_container
from backend.api.schemas import ChatRequest, ChatResponse
from backend.models.inference import GenerationParams


router = APIRouter(prefix="/chat", tags=["chat"])


def _params_from_request(req: ChatRequest) -> GenerationParams:
    p = GenerationParams()
    if req.params:
        if req.params.temperature is not None: p.temperature = req.params.temperature
        if req.params.top_p is not None:       p.top_p = req.params.top_p
        if req.params.top_k is not None:       p.top_k = req.params.top_k
        if req.params.num_ctx is not None:     p.num_ctx = req.params.num_ctx
        if req.params.num_predict is not None: p.num_predict = req.params.num_predict
        if req.params.repeat_penalty is not None: p.repeat_penalty = req.params.repeat_penalty
        if req.params.seed is not None:        p.seed = req.params.seed
    return p


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest):
    container = get_container()
    params = _params_from_request(req)
    try:
        result = await container.pipeline.answer(
            doc_id=req.doc_id,
            question=req.question,
            history=[h.model_dump() for h in req.history],
            params=params,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return ChatResponse(answer=result.text, sources=result.sources, stats={})


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    container = get_container()
    params = _params_from_request(req)

    async def event_gen() -> AsyncIterator[bytes]:
        try:
            async for ev in container.pipeline.stream_answer(
                doc_id=req.doc_id,
                question=req.question,
                history=[h.model_dump() for h in req.history],
                params=params,
            ):
                yield f"data: {json.dumps(ev)}\n\n".encode("utf-8")
        except Exception as e:
            err = {"event": "error", "message": str(e)}
            yield f"data: {json.dumps(err)}\n\n".encode("utf-8")

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )
