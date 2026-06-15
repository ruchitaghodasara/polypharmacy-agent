# This project runs fully offline using Ollama.
# Install Ollama from https://ollama.com, then run:
#   ollama pull llama3.1:8b
#
# If get_llm() raises at startup, Ollama is not running.
# Fix with: ollama serve

"""
LLM configuration for the Polypharmacy Safety Agent — Ollama only.

Public API:
    get_llm()  → ChatOllama(model='llama3.1:8b', base_url='http://localhost:11434')

# NOTE: llama3.1:8b is smaller than cloud models — JSON parsing from this model
# may fail more often than GPT-4 or Gemini. The existing fallback templates in
# agents/report_generator.py and the NONE-on-failure path in interaction_auditor.py
# already handle this gracefully; do not change that logic.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

logger = logging.getLogger(__name__)

_OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
_OLLAMA_MODEL    = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")


def _check_ollama_reachable() -> None:
    """Raise RuntimeError with a friendly message if Ollama is not running."""
    import urllib.request
    import urllib.error
    try:
        urllib.request.urlopen(f"{_OLLAMA_BASE_URL}/api/tags", timeout=3)
    except (urllib.error.URLError, OSError):
        raise RuntimeError(
            f"Ollama is not running. Open a terminal and run: ollama serve\n"
            f"  (expected at {_OLLAMA_BASE_URL})"
        )


def get_llm():
    """Return a ChatOllama instance pointing at the local Ollama server."""
    _check_ollama_reachable()
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=_OLLAMA_MODEL,
        base_url=_OLLAMA_BASE_URL,
        temperature=0,
    )


# ── __main__ smoke test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print(f"\n=== llm_config smoke test  (model={_OLLAMA_MODEL}) ===\n")

    try:
        llm = get_llm()
    except RuntimeError as exc:
        print(f"[FAIL] {exc}")
        sys.exit(1)

    print(f"[INFO] Built: {type(llm).__name__}  model={_OLLAMA_MODEL}")

    try:
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content="Reply with just the word READY")])
        text = response.content if hasattr(response, "content") else str(response)
        print(f"[PASS] model={_OLLAMA_MODEL}  response='{text.strip()}'")
    except Exception as exc:
        print(f"[FAIL] Invocation error: {exc}")
        sys.exit(1)
