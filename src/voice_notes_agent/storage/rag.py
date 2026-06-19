"""Local vector index + retrieval over notes (§5.5, FR-R2/R3/R4).

Transcript chunks and summaries are embedded and indexed in a local ChromaDB collection
(FR-R2). ``search`` returns only the most relevant chunks for a query (FR-R3), so the
LLM never receives the entire note history — each call stays small (FR-R4).

Embeddings default to a local sentence-transformers model (Open Decision O-1), keeping
note text fully on-device. Set ``providers.embeddings.vendor: cloud`` to use a cloud
embedding model instead.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

log = logging.getLogger(__name__)

_COLLECTION = "notes"


def chunk_text(text: str, *, max_chars: int = 800, overlap: int = 120) -> list[str]:
    """Split a transcript into overlapping chunks on sentence-ish boundaries."""
    text = text.strip()
    if not text:
        return []
    # Split on sentence enders but keep them; then greedily pack into windows.
    pieces = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    cur = ""
    for piece in pieces:
        if len(cur) + len(piece) + 1 <= max_chars:
            cur = f"{cur} {piece}".strip()
        else:
            if cur:
                chunks.append(cur)
            # Start the next chunk with a tail overlap for context continuity.
            tail = cur[-overlap:] if overlap and cur else ""
            cur = f"{tail} {piece}".strip()
    if cur:
        chunks.append(cur)
    return chunks


@dataclass
class SearchHit:
    text: str
    session_id: str
    timestamp: str
    score: float


def _local_embedding_fn(model_name: str):  # pragma: no cover - heavy optional dep
    from chromadb.utils import embedding_functions

    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)


class NotesIndex:
    """Thin wrapper over a persistent ChromaDB collection."""

    def __init__(self, cfg, paths, *, client=None, embedding_fn=None) -> None:
        self._cfg = cfg
        self._paths = paths
        self._client = client
        self._embedding_fn = embedding_fn
        self._collection = None

    def _get_collection(self):  # pragma: no cover - requires chromadb
        if self._collection is not None:
            return self._collection
        if self._client is None:
            import chromadb

            self._client = chromadb.PersistentClient(path=str(self._paths.chroma))
        if self._embedding_fn is None and self._cfg.providers.embeddings.vendor == "local":
            self._embedding_fn = _local_embedding_fn(self._cfg.providers.embeddings.model)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    def index_session(self, *, session_id: str, started: str, transcript: str, summary: str) -> int:
        """Embed and index a session's chunks + summary. Returns chunk count (FR-R2)."""
        col = self._get_collection()
        docs: list[str] = []
        ids: list[str] = []
        metas: list[dict] = []

        for i, chunk in enumerate(chunk_text(transcript)):
            docs.append(chunk)
            ids.append(f"{session_id}:chunk:{i}")
            metas.append({"session_id": session_id, "timestamp": started, "kind": "transcript"})

        if summary.strip():
            docs.append(summary)
            ids.append(f"{session_id}:summary")
            metas.append({"session_id": session_id, "timestamp": started, "kind": "summary"})

        if docs:
            col.upsert(documents=docs, ids=ids, metadatas=metas)  # pragma: no cover
        return len(docs)

    def search(self, query: str, k: int = 5) -> list[SearchHit]:
        """Return the top-k most relevant chunks for ``query`` (FR-R3)."""
        col = self._get_collection()
        res = col.query(query_texts=[query], n_results=k)  # pragma: no cover
        hits: list[SearchHit] = []  # pragma: no cover
        documents = res.get("documents", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0]
        distances = res.get("distances", [[]])[0]
        for doc, meta, dist in zip(documents, metadatas, distances):
            hits.append(
                SearchHit(
                    text=doc,
                    session_id=meta.get("session_id", ""),
                    timestamp=meta.get("timestamp", ""),
                    score=1.0 - float(dist),  # cosine distance -> similarity
                )
            )
        return hits
