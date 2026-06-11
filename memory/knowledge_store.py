"""
ChromaDB-backed knowledge store for drug interaction literature.

Ingests data/knowledge/fda_interactions.txt on first init, splits it into
~200-word chunks, embeds with sentence-transformers (all-MiniLM-L6-v2), and
exposes semantic search via query().
"""

from __future__ import annotations

import os
import re
import textwrap
from pathlib import Path
from typing import List

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ── Constants ──────────────────────────────────────────────────────────────────

_COLLECTION_NAME = "drug_interactions"
_EMBED_MODEL = "all-MiniLM-L6-v2"
_CHUNK_WORDS = 200
_DEFAULT_TEXT_PATH = (
    Path(__file__).parent.parent / "data" / "knowledge" / "fda_interactions.txt"
)
_DEFAULT_PERSIST_DIR = (
    Path(__file__).parent.parent / "data" / "chromadb"
)


# ── Chunker ────────────────────────────────────────────────────────────────────

def _chunk_text(text: str, words_per_chunk: int = _CHUNK_WORDS) -> List[str]:
    """
    Split *text* into chunks of approximately *words_per_chunk* words, breaking
    at paragraph boundaries where possible to preserve semantic coherence.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    chunks: List[str] = []
    current_words: List[str] = []

    for para in paragraphs:
        para_words = para.split()
        if current_words and len(current_words) + len(para_words) > words_per_chunk:
            chunks.append(" ".join(current_words))
            current_words = []
        current_words.extend(para_words)
        # If a single paragraph already exceeds the limit, flush it immediately
        if len(current_words) >= words_per_chunk:
            chunks.append(" ".join(current_words))
            current_words = []

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


# ── KnowledgeStore ─────────────────────────────────────────────────────────────

class KnowledgeStore:
    """
    Semantic knowledge store backed by ChromaDB and sentence-transformers.

    Usage::

        store = KnowledgeStore()
        store.init()                       # idempotent — ingests only if needed
        results = store.query("warfarin aspirin bleeding", n_results=3)
    """

    def __init__(
        self,
        persist_dir: str | Path | None = None,
        text_path: str | Path | None = None,
        embed_model: str = _EMBED_MODEL,
    ) -> None:
        self._persist_dir = Path(persist_dir or os.environ.get("CHROMA_PERSIST_DIR", _DEFAULT_PERSIST_DIR))
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

    # ── Embedding function wrapper for ChromaDB ────────────────────────────────

    def _embed(self, texts: List[str]) -> List[List[float]]:
        return self._embedder.encode(texts, show_progress_bar=False).tolist()

    # ── init ───────────────────────────────────────────────────────────────────

    def init(self) -> None:
        """
        Ensure the ChromaDB collection exists and is populated.
        Safe to call multiple times — will skip ingestion if the collection
        already contains documents.
        """
        existing = [c.name for c in self._client.list_collections()]

        if _COLLECTION_NAME in existing:
            self._collection = self._client.get_collection(_COLLECTION_NAME)
            count = self._collection.count()
            if count > 0:
                return  # already populated — nothing to do

        # Create (or re-open empty) collection
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        # Ingest
        if not self._text_path.exists():
            raise FileNotFoundError(
                f"Knowledge source not found: {self._text_path}\n"
                "Create data/knowledge/fda_interactions.txt before calling init()."
            )

        text = self._text_path.read_text(encoding="utf-8")
        chunks = _chunk_text(text)

        ids = [f"chunk-{i:04d}" for i in range(len(chunks))]
        embeddings = self._embed(chunks)

        self._collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"source": self._text_path.name, "chunk_index": i} for i in range(len(chunks))],
        )

    # ── query ──────────────────────────────────────────────────────────────────

    def query(self, text: str, n_results: int = 3) -> List[str]:
        """
        Semantic search over the knowledge base.

        Returns up to *n_results* document chunks ranked by cosine similarity
        to *text*.  Raises RuntimeError if init() has not been called.
        """
        if self._collection is None:
            raise RuntimeError("KnowledgeStore not initialised — call init() first.")

        query_embedding = self._embed([text])[0]
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, self._collection.count()),
            include=["documents", "distances"],
        )

        docs: List[str] = results["documents"][0] if results["documents"] else []
        return docs

    # ── convenience ────────────────────────────────────────────────────────────

    @property
    def document_count(self) -> int:
        """Number of chunks currently stored."""
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
        print(f"Query: \"{q}\"")
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
