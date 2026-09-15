"""
PageIndex MCP client.

Uses the official `mcp` Python SDK (modelcontextprotocol/python-sdk).
Supports three transports per PageIndex's docs:

  1. http_api   -> https://api.pageindex.ai/mcp     (API key)
  2. http_chat  -> https://chat.pageindex.ai/mcp    (OAuth)
  3. local_npx  -> npx -y @pageindex/mcp            (stdio, local PDFs)

Install dependencies:
    pip install "mcp[cli]>=1.0"
"""
from __future__ import annotations
import json
from contextlib import AsyncExitStack
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from backend.config import settings
from backend.utils import get_logger
from .base import BaseMCPClient, MCPTool


log = get_logger("mcp.pageindex")


class PageIndexMCPClient(BaseMCPClient):
    def __init__(self, transport: Optional[str] = None):
        self.transport = transport or settings.pageindex.transport
        self._session: Optional[ClientSession] = None
        self._stack: Optional[AsyncExitStack] = None
        self._tools_cache: list[MCPTool] | None = None

    # ---- lifecycle ----

    async def connect(self) -> None:
        if self._session is not None:
            return
        self._stack = AsyncExitStack()

        if self.transport == "local_npx":
            params = StdioServerParameters(
                command=settings.pageindex.npx_command,
                args=list(settings.pageindex.npx_args),
                env=None,
            )
            read, write = await self._stack.enter_async_context(stdio_client(params))
            self._session = await self._stack.enter_async_context(
                ClientSession(read, write)
            )

        elif self.transport in ("http_api", "http_chat"):
            url = (
                settings.pageindex.api_url
                if self.transport == "http_api"
                else settings.pageindex.chat_url
            )
            headers: dict[str, str] = {}
            if self.transport == "http_api":
                if not settings.pageindex.api_key:
                    raise RuntimeError(
                        "PAGEINDEX_API_KEY env var is required for http_api transport. "
                        "Create one at https://dash.pageindex.ai/api-keys"
                    )
                headers["Authorization"] = f"Bearer {settings.pageindex.api_key}"

            # streamablehttp_client yields (read, write, get_session_id_callback)
            read, write, _ = await self._stack.enter_async_context(
                streamablehttp_client(url, headers=headers)
            )
            self._session = await self._stack.enter_async_context(
                ClientSession(read, write)
            )

        else:
            raise ValueError(f"Unknown transport: {self.transport}")

        await self._session.initialize()
        log.info("PageIndex MCP connected via %s", self.transport)

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
            self._session = None
            self._tools_cache = None

    # ---- protocol ----

    async def list_tools(self) -> list[MCPTool]:
        assert self._session is not None, "Call connect() first"
        if self._tools_cache is not None:
            return self._tools_cache
        result = await self._session.list_tools()
        tools = [
            MCPTool(
                name=t.name,
                description=t.description or "",
                input_schema=t.inputSchema or {},
            )
            for t in result.tools
        ]
        self._tools_cache = tools
        log.info("PageIndex exposes %d tools: %s", len(tools), [t.name for t in tools])
        return tools

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict:
        assert self._session is not None, "Call connect() first"
        result = await self._session.call_tool(name, arguments=arguments)
        # mcp returns a list of content blocks (text / image / resource).
        # We normalize to a single dict: try to parse JSON text content,
        # otherwise return the raw text.
        merged_text: list[str] = []
        raw_blocks: list[dict] = []
        for block in result.content:
            btype = getattr(block, "type", None)
            raw_blocks.append({"type": btype, "data": getattr(block, "text", None)})
            if btype == "text" and getattr(block, "text", None):
                merged_text.append(block.text)

        text = "\n".join(merged_text).strip()
        parsed: Any = None
        if text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = text

        return {
            "is_error": bool(getattr(result, "isError", False)),
            "parsed": parsed,
            "raw_text": text,
            "blocks": raw_blocks,
        }