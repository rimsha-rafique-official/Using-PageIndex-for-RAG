"""
PageIndex service.

A domain layer on top of the raw MCP client. Hides PageIndex's tool
names and shapes from the rest of the app.

Current pageindex-mcp tool names (as of June 2026):
  process_document, browse_documents, search_documents,
  get_folder_structure, get_document, get_document_structure,
  get_page_content, get_document_image, remove_document
"""
from __future__ import annotations
import asyncio
import base64
import json
import socket
import threading
from functools import partial
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any

import httpx

from backend.config import settings
from backend.utils import get_logger
from .base import MCPTool
from .pageindex_client import PageIndexMCPClient


log = get_logger("pageindex.service")


class PageIndexService:
    def __init__(self, client: PageIndexMCPClient | None = None):
        self.client = client or PageIndexMCPClient()
        self._tools_by_name: dict[str, MCPTool] = {}

    async def start(self) -> None:
        await self.client.connect()
        tools = await self.client.list_tools()
        self._tools_by_name = {t.name: t for t in tools}

    async def stop(self) -> None:
        await self.client.close()

    @property
    def tools(self) -> list[MCPTool]:
        return list(self._tools_by_name.values())

    # ---- helpers ----

    def _pick_tool(self, preferred: str, *fallbacks: str) -> str:
        candidates = [preferred, *fallbacks]
        for c in candidates:
            if c in self._tools_by_name:
                return c
        for c in candidates:
            for name in self._tools_by_name:
                if c.lower() in name.lower():
                    return name
        raise RuntimeError(
            f"None of {candidates} found in PageIndex tools: "
            f"{list(self._tools_by_name)}"
        )

    @staticmethod
    def _ensure_pdf_suffix(doc_name: str) -> str:
        """PageIndex requires the full filename including .pdf."""
        return doc_name if doc_name.lower().endswith(".pdf") else f"{doc_name}.pdf"

    # ---- public API ----

    async def list_documents(self) -> list[dict]:
        """
        Returns docs — tries every known tool name across MCP versions.
        PageIndex has renamed this tool several times:
          recent_documents -> browse_documents (June 2026)
        """
        candidates = [
            "browse_documents",
            "recent_documents",
            "list_documents",
            "search_documents",
            "get_documents",
            "documents",
            "list_docs",
            "my_documents",
        ]
        tool_name = None
        for c in candidates:
            if c in self._tools_by_name:
                tool_name = c
                break
        # Last resort: anything with "document" but not process/get/remove
        if not tool_name:
            for name in self._tools_by_name:
                if (
                    "document" in name.lower()
                    and "process" not in name.lower()
                    and "get" not in name.lower()
                    and "remove" not in name.lower()
                ):
                    tool_name = name
                    break
        if not tool_name:
            log.warning(
                "No list-documents tool found. Available: %s",
                list(self._tools_by_name),
            )
            return []

        log.info("Using tool '%s' for list_documents", tool_name)
        r = await self.client.call_tool(tool_name, {})
        if r["is_error"]:
            raise RuntimeError(f"list_documents failed: {r['raw_text']}")
        data = r["parsed"]
        if isinstance(data, dict):
            for key in ("docs", "documents", "results", "items", "data"):
                if key in data and isinstance(data[key], list):
                    return data[key]
        if isinstance(data, list):
            return data
        return []

    async def submit_pdf(self, pdf_path: str | Path) -> dict:
        """
        Upload via MCP's process_document tool. The tool only accepts a
        URL, so we briefly serve the local file via an HTTP server,
        give PageIndex the URL, then tear the server down.
        """
        path = Path(pdf_path).resolve()
        if not path.exists():
            raise FileNotFoundError(path)

        try:
            name = self._pick_tool(
                settings.pageindex.tool_submit_document,
                "process_document", "upload_doc", "submit_pdf", "upload",
            )
            tool = self._tools_by_name[name]
            schema_props = (tool.input_schema or {}).get("properties", {})

            if "file_base64" in schema_props or "content_base64" in schema_props:
                key = "file_base64" if "file_base64" in schema_props else "content_base64"
                payload = {
                    key: base64.b64encode(path.read_bytes()).decode("ascii"),
                    "filename": path.name,
                }
                r = await self.client.call_tool(name, payload)
                if r["is_error"]:
                    raise RuntimeError(f"submit via MCP failed: {r['raw_text']}")
                return r["parsed"] if isinstance(r["parsed"], dict) else {"raw": r["raw_text"]}

            if "url" in schema_props:
                return await self._submit_via_local_url(name, path, schema_props)
        except RuntimeError as e:
            log.info("MCP submit not usable (%s); trying REST fallback", e)

        if not settings.pageindex.api_key:
            raise RuntimeError(
                "Upload failed. No usable MCP upload path and no "
                "PAGEINDEX_API_KEY set for REST fallback."
            )
        log.info("Uploading via PageIndex REST API: %s", path.name)
        async with httpx.AsyncClient(timeout=300.0, trust_env=False) as http:
            with open(path, "rb") as fp:
                files = {"file": (path.name, fp, "application/pdf")}
                resp = await http.post(
                    "https://api.pageindex.ai/api/v2/documents",
                    headers={"Authorization": f"Bearer {settings.pageindex.api_key}"},
                    files=files,
                )
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"PageIndex REST upload failed [{resp.status_code}]: {resp.text}"
                )
            return resp.json()

    async def _submit_via_local_url(
        self, tool_name: str, path: Path, schema_props: dict
    ) -> dict:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()

        serve_dir = str(path.parent)
        Handler = partial(SimpleHTTPRequestHandler, directory=serve_dir)
        httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        url = f"http://127.0.0.1:{port}/{path.name}"
        log.info("Serving %s at %s for PageIndex to fetch", path.name, url)

        try:
            payload: dict = {"url": url}
            if "filename" in schema_props:
                payload["filename"] = path.name
            r = await self.client.call_tool(tool_name, payload)
            if r["is_error"]:
                raise RuntimeError(f"{tool_name} failed: {r['raw_text']}")
            await asyncio.sleep(1.0)
            return r["parsed"] if isinstance(r["parsed"], dict) else {"raw": r["raw_text"]}
        finally:
            httpd.shutdown()
            httpd.server_close()
            log.info("Stopped local file server on port %s", port)

    async def get_tree(self, doc_name: str) -> dict:
        """Return the document's structure tree."""
        name = self._ensure_pdf_suffix(doc_name)
        r = await self.client.call_tool(
            "get_document_structure",
            {"doc_name": name, "wait_for_completion": True},
        )
        if r["is_error"]:
            raise RuntimeError(f"get_tree failed: {r['raw_text']}")
        return r["parsed"] if isinstance(r["parsed"], dict) else {"raw": r["raw_text"]}

    async def get_page_content(self, doc_name: str, pages: str) -> str:
        """
        Fetch text content for the given page spec.
        `pages` examples: "1-5", "1,3,5", "2-4,7".
        """
        name = self._ensure_pdf_suffix(doc_name)
        r = await self.client.call_tool(
            "get_page_content",
            {"doc_name": name, "pages": pages, "wait_for_completion": True},
        )
        if r["is_error"]:
            raise RuntimeError(f"get_page_content failed: {r['raw_text']}")
        payload = r["parsed"]
        if isinstance(payload, dict):
            for key in ("content", "text", "pages", "page_content"):
                v = payload.get(key)
                if isinstance(v, str):
                    return v
                if isinstance(v, list):
                    parts = []
                    for item in v:
                        if isinstance(item, str):
                            parts.append(item)
                        elif isinstance(item, dict):
                            parts.append(item.get("content") or item.get("text") or "")
                    if parts:
                        return "\n\n".join(parts)
            return json.dumps(payload, indent=2)[:50000]
        if isinstance(payload, str):
            return payload
        return r["raw_text"] or ""

    async def retrieve(
        self, doc_name: str, query: str, *, max_nodes: int | None = None
    ) -> dict:
        """
        Two-step retrieval:
          1. Pull the document structure (titles + summaries + page ranges)
          2. Heuristically pick the most relevant sections by keyword overlap
          3. Fetch those page ranges via get_page_content
          4. Return as normalized 'sections' list
        """
        max_nodes = max_nodes or settings.rag.max_nodes_in_context

        tree = await self.get_tree(doc_name)
        nodes = self._flatten_structure(tree.get("structure", []))
        if not nodes:
            log.warning("Document %s has no structure nodes", doc_name)
            return {"sections": [], "raw": tree}

        scored = self._rank_nodes(query, nodes)
        picked = scored[:max_nodes] or nodes[:max_nodes]

        sections: list[dict] = []
        for node in picked:
            start = node.get("start_index")
            end = node.get("end_index", start)
            if start is None:
                continue
            page_spec = f"{start}-{end}" if end and end != start else str(start)
            try:
                text = await self.get_page_content(doc_name, page_spec)
            except Exception as e:
                log.warning("get_page_content failed for %s: %s", page_spec, e)
                text = node.get("summary", "")
            sections.append({
                "node_id": node.get("node_id", ""),
                "title": node.get("title", ""),
                "pages": page_spec,
                "text": text,
                "summary": node.get("summary", ""),
            })

        log.info("Retrieved %d sections from %s", len(sections), doc_name)
        return {"sections": sections, "raw": tree}

    @staticmethod
    def _flatten_structure(structure: list, out: list | None = None) -> list[dict]:
        """Flatten the nested PageIndex tree into a flat list of leaf nodes."""
        if out is None:
            out = []
        for node in structure or []:
            if not isinstance(node, dict):
                continue
            children = node.get("nodes")
            if children and isinstance(children, list):
                PageIndexService._flatten_structure(children, out)
            else:
                out.append(node)
        if not out:
            for node in structure or []:
                if isinstance(node, dict):
                    out.append(node)
        return out

    @staticmethod
    def _rank_nodes(query: str, nodes: list[dict]) -> list[dict]:
        """Simple bag-of-words ranking. Good enough for top-K selection."""
        q_tokens = {
            t.lower()
            for t in query.replace("?", " ").replace(",", " ").split()
            if len(t) > 2
        }
        if not q_tokens:
            return list(nodes)

        scored: list[tuple[int, dict]] = []
        for n in nodes:
            text = f"{n.get('title','')} {n.get('summary','')}".lower()
            score = sum(1 for t in q_tokens if t in text)
            scored.append((score, n))
        scored.sort(key=lambda x: x[0], reverse=True)
        with_hits = [n for s, n in scored if s > 0]
        return with_hits if with_hits else [n for _, n in scored]