"""
ChromaDB-backed knowledge store for drug interaction literature.

Ingests data/knowledge/fda_interactions.txt on first init, splits into
~200-word chunks, embeds with sentence-transformers, and exposes semantic
search via query().
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List

import chromadb
from sentence_transformers import SentenceTransformer

# ── Constants ──────────────────────────────────────────────────────────────────

COLLECTION_NAME   = "drug_interactions"
EMBED_MODEL       = "all-MiniLM-L6-v2"
CHUNK_SIZE_WORDS  = 200
MAX_RAG_RESULTS   = 3

_DEFAULT_TEXT_PATH = (
    Path(__file__).parent.parent / "data" / "knowledge" / "fda_interactions.txt"
)
_DEFAULT_PERSIST_DIR = (
    Path(__file__).parent.parent / "data" / "chromadb"
)


# ── Chunker ────────────────────────────────────────────────────────────────────

def _chunk_text(text: str, words_per_chunk: int = CHUNK_SIZE_WORDS) -> List[str]:
    """Split text into ~words_per_chunk chunks at paragraph boundaries."""
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: List[str] = []
    current_words: List[str] = []

    for para in paragraphs:
        para_words = para.split()
        if current_words and len(current_words) + len(para_words) > words_per_chunk:
            chunks.append(" ".join(current_words))
            current_words = []
        current_words.extend(para_words)
        # Flush immediately when a single paragraph exceeds the limit.
        if len(current_words) >= words_per_chunk:
            chunks.append(" ".join(current_words))
            current_words = []

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


# ── KnowledgeStore ─────────────────────────────────────────────────────────────

class KnowledgeStore:
    """
    Semantic store backed by ChromaDB and sentence-transformers.

    Usage::

        store = KnowledgeStore()
        store.init()                       # idempotent — ingests only if needed
        results = store.query("warfarin aspirin bleeding", n_results=3)
    """

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        text_path: str | Path | None = None,
        embed_model: str = EMBED_MODEL,
    ) -> None:
        self._persist_dir = Path(
            persist_dir or os.environ.get("CHROMA_PERSIST_DIR", _DEFAULT_PERSIST_DIR)
        )
        self._text_path = Path(text_path or _DEFAULT_TEXT_PATH)
        self._persist_dir.mkdir(parents=True, exist_ok=True)

        chroma_host = os.environ.get("CHROMA_HOST", "")
        if chroma_host:
            chroma_port = int(os.environ.get("CHROMA_PORT", 8000))
            self._client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        else:
            self._client = chromadb.PersistentClient(path=str(self._persist_dir))

        self._embedder = SentenceTransformer(embed_model)
        self._collection: chromadb.Collection | None = None

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Return sentence-transformer embeddings for *texts*."""
        return self._embedder.encode(texts, show_progress_bar=False).tolist()

    # --- Write operations -----------------------------------------------------

    def init(self) -> None:
        """Ensure the ChromaDB collection exists and is populated."""
        existing = [c.name for c in self._client.list_collections()]

        if COLLECTION_NAME in existing:
            self._collection = self._client.get_collection(COLLECTION_NAME)
            if self._collection.count() > 0:
                return  # already populated — nothing to do

        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        if not self._text_path.exists():
            raise FileNotFoundError(
                f"Knowledge source not found: {self._text_path}\n"
                "Create data/knowledge/fda_interactions.txt before calling init()."
            )

        text   = self._text_path.read_text(encoding="utf-8")
        chunks = _chunk_text(text)
        ids    = [f"chunk-{i:04d}" for i in range(len(chunks))]

        self._collection.add(
            ids=ids,
            documents=chunks,
            embeddings=self._embed(chunks),
            metadatas=[
                {"source": self._text_path.name, "chunk_index": i}
                for i in range(len(chunks))
            ],
        )

    # --- Read operations ------------------------------------------------------

    def query(self, text: str, n_results: int = MAX_RAG_RESULTS) -> List[str]:
        """Return up to *n_results* chunks ranked by cosine similarity to *text*."""
        if self._collection is None:
            raise RuntimeError("KnowledgeStore not initialised — call init() first.")

        query_embedding = self._embed([text])[0]
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self._collection.count()),
            include=["documents", "distances"],
        )

        return results["documents"][0] if results["documents"] else []

    # --- Delete / TTL operations ----------------------------------------------

    @property
    def document_count(self) -> int:
        """Number of chunks currently stored in the collection."""
        if self._collection is None:
            return 0
        return self._collection.count()


# ── __main__ test block ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")

    print("\n=== KnowledgeStore smoke-test ===\n")

    store = KnowledgeStore()

    print("Initialising (embedding model load + ingestion on first run)…")
    try:
        store.init()
        print(f"[PASS] init()  —  {store.document_count} chunks in collection\n")
    except Exception as exc:
        print(f"[FAIL] init() — {exc}")
        sys.exit(1)

    queries = [
        "methotrexate folic acid interaction",
        "warfarin bleeding risk aspirin",
        "QT prolongation hydroxychloroquine",
        "NSAID ACE inhibitor kidney",
    ]

    for q in queries:
        print(f'Query: "{q}"')
        try:
            results = store.query(q, n_results=2)
            for i, chunk in enumerate(results, 1):
                preview = chunk[:120].replace("\n", " ")
                print(f"  [{i}] {preview}…")
            print(f"  [PASS] returned {len(results)} result(s)\n")
        except Exception as exc:
            print(f"  [FAIL] {exc}\n")
            sys.exit(1)

    print("All KnowledgeStore checks passed.")
