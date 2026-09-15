"""
MCP client abstraction.

We expose a minimal surface — connect, list_tools, call_tool —
so the rest of the app doesn't know whether MCP is reached over
stdio, HTTP+SSE, or streamable HTTP.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: dict


class BaseMCPClient(ABC):
    @abstractmethod
    async def connect(self) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...

    @abstractmethod
    async def list_tools(self) -> list[MCPTool]: ...

    @abstractmethod
    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict: ...
