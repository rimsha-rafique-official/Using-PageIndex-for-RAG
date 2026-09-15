"""
Smoke test. Runs without the web server.

Verifies:
  1. Ollama is reachable and qwen3-vl:2b is installed
  2. PageIndex MCP connects and lists its tools
  3. End-to-end generate works

Usage:
    python scripts/smoke_test.py
"""
import asyncio
import sys
from pathlib import Path

# Allow running this script directly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env from project root BEFORE importing backend modules
from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from backend.config import settings  # noqa: E402
from backend.models.loaders import OllamaLoader  # noqa: E402
from backend.models.inference import OllamaInference, ChatMessage  # noqa: E402
from backend.mcp import PageIndexService  # noqa: E402


async def main():
    print("=" * 60)
    print("LOCAL RAG SMOKE TEST")
    print("=" * 60)

    # 1. Ollama
    print("\n[1/3] Checking Ollama...")
    loader = OllamaLoader()
    ok = await loader.check_health()
    print(f"     health: {'OK' if ok else 'FAIL'}")
    if not ok:
        print(f"     -> Cannot reach {settings.ollama.base_url}")
        print(f"     -> Run: ollama serve")
        return 1

    models = await loader.list_available()
    print(f"     installed models: {[m.name for m in models]}")
    try:
        info = await loader.ensure_loaded(settings.ollama.model)
        print(f"     loaded: {info.name} ({info.parameter_size}, {info.quantization})")
    except RuntimeError as e:
        print(f"     -> {e}")
        return 1

    # 2. Generate a tiny response
    print("\n[2/3] Generating with local model...")
    inf = OllamaInference()
    text = await inf.generate([
        ChatMessage(role="user", content="Say 'hello from qwen' in one short sentence.")
    ])
    print(f"     model says: {text!r}")
    await inf.close()
    await loader.close()

    # 3. PageIndex MCP
    print("\n[3/3] Connecting PageIndex MCP...")
    print(f"     transport: {settings.pageindex.transport}")
    if settings.pageindex.transport == "http_api" and not settings.pageindex.api_key:
        print("     -> WARN: PAGEINDEX_API_KEY not set. Skipping MCP test.")
        print("     -> Get a key at https://dash.pageindex.ai/api-keys")
        return 0

    svc = PageIndexService()
    try:
        await svc.start()
        print(f"     tools: {[t.name for t in svc.tools]}")
        docs = await svc.list_documents()
        print(f"     documents: {len(docs)}")
    except Exception as e:
        print(f"     -> MCP error: {e}")
        return 1
    finally:
        await svc.stop()

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))