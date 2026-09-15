"""
Central configuration.
All paths, URLs, and tunables live here. Loaded from env when set.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Load .env if present (before any dataclass reads os.getenv)
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class OllamaConfig:
    base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model: str = os.getenv("OLLAMA_MODEL", "qwen3-vl:2b")
    request_timeout: float = float(os.getenv("OLLAMA_TIMEOUT", "300"))
    keep_alive: str = os.getenv("OLLAMA_KEEP_ALIVE", "5m")


@dataclass
class GenerationConfig:
    """Default sampling parameters. Override per-request in the API."""
    temperature: float = 0.3
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    num_ctx: int = 8192
    num_predict: int = 1024
    stop: list[str] = field(default_factory=list)


@dataclass
class PageIndexMCPConfig:
    """
    PageIndex MCP connection.

    Three transports are supported:
      - 'http_api'   : developer endpoint, needs API key
      - 'http_chat'  : PageIndex Chat endpoint (OAuth)
      - 'local_npx'  : run `npx -y pageindex-mcp` locally over stdio
    """
    transport: str = os.getenv("PAGEINDEX_TRANSPORT", "local_npx")

    # HTTP transports
    api_url: str = os.getenv(
        "PAGEINDEX_API_URL", "https://api.pageindex.ai/mcp"
    )
    chat_url: str = os.getenv(
        "PAGEINDEX_CHAT_URL", "https://chat.pageindex.ai/mcp"
    )
    api_key: Optional[str] = os.getenv("PAGEINDEX_API_KEY")

    # stdio transport — full path to npx.cmd (Windows needs .cmd extension)
    npx_command: str = os.getenv(
        "NPX_COMMAND",
        r"C:\Program Files\nodejs\npx.cmd",
    )
    npx_args: tuple = ("-y", "pageindex-mcp")

    # Tool names — updated to match pageindex-mcp as of June 2026.
    # The service layer also fuzzy-matches, so these are just the
    # preferred first-try names.
    tool_list_documents: str = "browse_documents"
    tool_submit_document: str = "process_document"
    tool_get_tree: str = "get_document_structure"
    tool_retrieve: str = "search_documents"


@dataclass
class RAGConfig:
    max_nodes_in_context: int = 8
    max_chars_per_node: int = 2500
    enable_streaming: bool = True
    cite_pages: bool = True


@dataclass
class APIConfig:
    host: str = os.getenv("API_HOST", "127.0.0.1")
    port: int = int(os.getenv("API_PORT", "8000"))
    cors_origins: list[str] = field(
        default_factory=lambda: ["http://localhost:5173", "http://localhost:8000", "*"]
    )


@dataclass
class Settings:
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    pageindex: PageIndexMCPConfig = field(default_factory=PageIndexMCPConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    api: APIConfig = field(default_factory=APIConfig)

    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    pdf_dir: Path = PROJECT_ROOT / "data" / "pdfs"


settings = Settings()