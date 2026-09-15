"""
Document routes: upload, list, get tree.
"""
from __future__ import annotations
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from backend.config import settings
from backend.api.deps import get_container
from backend.api.schemas import UploadResponse, DocumentInfo


router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    settings.pdf_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.pdf_dir / file.filename
    contents = await file.read()
    dest.write_bytes(contents)

    container = get_container()
    try:
        result = await container.pageindex.submit_pdf(dest)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"PageIndex submit failed: {e}")

    doc_id = (
        result.get("doc_name")
        or result.get("name")
        or result.get("filename")
        or result.get("doc_id")
        or result.get("id")
        or ""
    )
    if not doc_id:
        raise HTTPException(
            status_code=502,
            detail=f"PageIndex did not return any document identifier. Got: {result}",
        )

    return UploadResponse(doc_id=doc_id, filename=file.filename, raw=result)


@router.get("/", response_model=list[DocumentInfo])
async def list_documents():
    container = get_container()
    try:
        docs = await container.pageindex.list_documents()
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    out: list[DocumentInfo] = []
    for d in docs:
        # PageIndex uses `name` (incl. .pdf) as the addressable identifier
        # for get_document/get_page_content. We surface that as doc_id
        # so the rest of the app can keep using doc_id naming.
        name = d.get("name") or d.get("filename") or d.get("title") or ""
        out.append(
            DocumentInfo(
                doc_id=name,
                name=name,
                pages=d.get("pageNum") or d.get("pages") or d.get("page_count"),
                extra={
                    k: v for k, v in d.items()
                    if k not in {"doc_id", "id", "name", "pages", "pageNum"}
                },
            )
        )
    return out


@router.get("/{doc_id}/tree")
async def get_tree(doc_id: str):
    container = get_container()
    try:
        tree = await container.pageindex.get_tree(doc_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    return tree
