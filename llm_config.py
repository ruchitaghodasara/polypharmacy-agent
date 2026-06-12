"""
Centralised LLM provider configuration for the Polypharmacy Safety Agent.

Supported providers (set via LLM_PROVIDER env var):
  gemini   — Google Gemini 2.5 Flash via AI Studio (primary default)
  groq     — Llama-3.3-70b on Groq (fast, generous free tier)
  cerebras — Llama-3.1-70b on Cerebras (OpenAI-compat endpoint)
  ollama   — Local Llama 3.2 via Ollama (no API key required)

Auto-fallback: if the primary provider raises a RateLimitError the module
automatically retries once with LLM_FALLBACK (default: groq).

Public API:
    get_llm(provider=None)   → LangChain BaseChatModel
    get_eval_llm()           → always Cerebras (high-volume eval runs)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

logger = logging.getLogger(__name__)

# ── Provider factory ──────────────────────────────────────────────────────────

def _build_llm(provider: str):
    """
    Construct a LangChain chat model for the given provider name.
    Raises ValueError for unknown providers.
    """
    provider = provider.strip().lower()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=os.environ.get("GEMINI_API_KEY", ""),
        )

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=os.environ.get("GROQ_API_KEY", ""),
        )

    if provider == "cerebras":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            base_url="https://api.cerebras.ai/v1",
            api_key=os.environ.get("CEREBRAS_API_KEY", ""),
            model="llama3.1-70b",
            temperature=0,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model="llama3.2",
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        )

    raise ValueError(
        f"Unknown LLM provider '{provider}'. "
        "Choose from: gemini | groq | cerebras | ollama"
    )


# ── Public API ────────────────────────────────────────────────────────────────

def get_llm(provider: str | None = None):
    """
    Return a LangChain BaseChatModel for the requested provider.

    If *provider* is None, reads LLM_PROVIDER from the environment
    (default: 'gemini').

    On RateLimitError from the primary provider, automatically retries once
    with LLM_FALLBACK (default: 'groq') and logs a warning.
    """
    if provider is None:
        provider = os.environ.get("LLM_PROVIDER", "gemini")

    try:
        return _build_llm(provider)
    except Exception as exc:
        # Lazy import — only langchain_google_genai exposes this; other
        # providers surface rate limits as plain exceptions too.
        rate_limit_names = {"RateLimitError", "ResourceExhausted", "429"}
        if type(exc).__name__ in rate_limit_names or "429" in str(exc) or "rate" in str(exc).lower():
            fallback = os.environ.get("LLM_FALLBACK", "groq")
            if fallback != provider:
                logger.warning(
                    "Provider '%s' rate-limited (%s). Falling back to '%s'.",
                    provider,
                    type(exc).__name__,
                    fallback,
                )
                return _build_llm(fallback)
        raise


def get_eval_llm():
    """
    Return the LLM to use for high-volume evaluation runs.

    Always uses Cerebras (fast, generous free tier, no per-minute token cap
    comparable to cloud providers) regardless of LLM_PROVIDER.
    """
    return _build_llm("cerebras")


# ── __main__ smoke test ───────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    provider = os.environ.get("LLM_PROVIDER", "gemini")
    print(f"\n=== llm_config smoke test  (provider={provider}) ===\n")

    try:
        llm = get_llm()
    except Exception as exc:
        print(f"[FAIL] Could not build LLM for provider '{provider}': {exc}")
        sys.exit(1)

    print(f"[INFO] Built: {type(llm).__name__}")

    try:
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content="Reply with just the word READY")])
        text = response.content if hasattr(response, "content") else str(response)
        print(f"[PASS] provider={provider}  response='{text.strip()}'")
    except Exception as exc:
        print(f"[FAIL] Invocation error: {exc}")
        sys.exit(1)
