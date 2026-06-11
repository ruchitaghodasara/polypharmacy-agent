"""
Run this script to verify all external dependencies are reachable and
configured correctly before starting development.

Usage:
    python setup_check.py

Requires a .env file (copy from .env.example and fill in your keys).
"""

import os
import sys

# Load .env before any checks
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("ERROR: python-dotenv not installed. Run: pip install python-dotenv")
    sys.exit(1)

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
results: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str = "") -> None:
    results.append((name, passed, detail))
    status = PASS if passed else FAIL
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {name}{suffix}")


# ── 1. Anthropic API ──────────────────────────────────────────────────────────
print("\n--- Anthropic API ---")
try:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key or api_key.startswith("sk-ant-xxx"):
        record("ANTHROPIC_API_KEY set", False, "key looks like placeholder")
    else:
        client = anthropic.Anthropic(api_key=api_key)
        # Cheapest possible call: count tokens without sending a message
        response = client.messages.count_tokens(
            model=os.environ.get("CLAUDE_MODEL", "claude-haiku-4-5-20251001"),
            messages=[{"role": "user", "content": "ping"}],
        )
        record("Anthropic API reachable", True, f"token_count endpoint OK (input_tokens={response.input_tokens})")
except anthropic.AuthenticationError:
    record("Anthropic API reachable", False, "AuthenticationError — check ANTHROPIC_API_KEY")
except Exception as exc:
    record("Anthropic API reachable", False, str(exc))


# ── 2. Upstash Redis ──────────────────────────────────────────────────────────
print("\n--- Upstash Redis ---")
try:
    import redis as redis_lib

    host = os.environ.get("REDIS_HOST", "")
    port = int(os.environ.get("REDIS_PORT", 6380))
    password = os.environ.get("REDIS_PASSWORD", "")
    ssl = os.environ.get("REDIS_SSL", "true").lower() == "true"

    # Fallback: parse UPSTASH_REDIS_URL if individual vars not set
    url = os.environ.get("UPSTASH_REDIS_URL", "")
    if not host and url:
        r = redis_lib.from_url(url, decode_responses=True, socket_connect_timeout=5)
    elif host:
        r = redis_lib.Redis(
            host=host,
            port=port,
            password=password,
            ssl=ssl,
            decode_responses=True,
            socket_connect_timeout=5,
        )
    else:
        record("Redis credentials present", False, "Set REDIS_HOST or UPSTASH_REDIS_URL in .env")
        r = None

    if r is not None:
        pong = r.ping()
        record("Redis PING", pong, "PONG received" if pong else "no response")
        if pong:
            r.set("polypharmacy:setup_check", "ok", ex=60)
            val = r.get("polypharmacy:setup_check")
            record("Redis SET/GET round-trip", val == "ok", f"value={val!r}")
            r.delete("polypharmacy:setup_check")
except redis_lib.exceptions.AuthenticationError:
    record("Redis PING", False, "AuthenticationError — check REDIS_PASSWORD")
except redis_lib.exceptions.ConnectionError as exc:
    record("Redis PING", False, f"ConnectionError — {exc}")
except Exception as exc:
    record("Redis PING", False, str(exc))


# ── 3. ChromaDB ───────────────────────────────────────────────────────────────
print("\n--- ChromaDB ---")
try:
    import chromadb

    persist_dir = os.environ.get("CHROMA_PERSIST_DIR", "./data/chromadb")
    os.makedirs(persist_dir, exist_ok=True)

    chroma_host = os.environ.get("CHROMA_HOST", "")
    if chroma_host:
        chroma_port = int(os.environ.get("CHROMA_PORT", 8000))
        client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        mode = f"HTTP @ {chroma_host}:{chroma_port}"
    else:
        client = chromadb.PersistentClient(path=persist_dir)
        mode = f"local PersistentClient @ {persist_dir}"

    record("ChromaDB client created", True, mode)

    # Create a temporary collection and insert/query a document
    col = client.get_or_create_collection("setup_check_tmp")
    col.add(
        documents=["warfarin is an anticoagulant"],
        ids=["doc1"],
    )
    res = col.query(query_texts=["blood thinner"], n_results=1)
    hit = res["ids"][0][0] if res["ids"] and res["ids"][0] else None
    record("ChromaDB add + query", hit == "doc1", f"nearest id={hit!r}")
    client.delete_collection("setup_check_tmp")

except Exception as exc:
    record("ChromaDB initialised", False, str(exc))


# ── 4. sentence-transformers (embedding model download) ──────────────────────
print("\n--- sentence-transformers ---")
try:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    vec = model.encode("test sentence")
    record(
        "sentence-transformers encode",
        len(vec) > 0,
        f"embedding dim={len(vec)}",
    )
except Exception as exc:
    record("sentence-transformers encode", False, str(exc))


# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
passed = sum(1 for _, ok, _ in results if ok)
total = len(results)
print(f"  Result: {passed}/{total} checks passed")
if passed == total:
    print(f"  [{PASS}] All systems go — ready to build!\n")
    sys.exit(0)
else:
    failed = [name for name, ok, _ in results if not ok]
    print(f"  [{FAIL}] Fix the above failures before continuing.")
    print(f"         Failed checks: {', '.join(failed)}\n")
    sys.exit(1)
