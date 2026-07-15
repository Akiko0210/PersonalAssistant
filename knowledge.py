"""Trading knowledge base: one-time PDF ingestion + semantic search.

Reference material (books/PDFs/text) the user drops into the project-root
``knowledge/`` folder is chunked, embedded once, and stored in a persistent Chroma
collection (under ``data/chroma``) separate from notes.
The agent queries it on demand via the ``search_knowledge`` tool, so the content
is never pasted into the conversation.

Ingestion is idempotent: each PDF is identified by the SHA-256 of its bytes and
recorded in a manifest. A file whose hash is already known is skipped before any
text extraction or embedding happens, so re-scanning an unchanged folder costs
only a few hashes (no embedding-model load). This lets the scan run cheaply at
every boot while a given book is embedded exactly once.
"""

import hashlib
import json
import logging
from datetime import datetime

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader

import config as cfg
from atomic_io import write_json_atomic

log = logging.getLogger("knowledge")

_UPSERT_BATCH = 256  # chunks per Chroma upsert call (keeps memory/latency bounded)


class KnowledgeStore:
    def __init__(self):
        cfg.ensure_dirs()
        self._col = None  # Chroma collection, loaded lazily on first real use

    # --- chroma --------------------------------------------------------------
    def _ensure_chroma(self):
        if self._col is not None:
            return
        log.info("loading embedding model + chroma (knowledge, first use)...")
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=cfg.EMBED_MODEL
        )
        client = chromadb.PersistentClient(path=str(cfg.CHROMA_DIR))
        self._col = client.get_or_create_collection(
            name=cfg.KNOWLEDGE_COLLECTION, embedding_function=ef
        )
        log.info("knowledge collection ready")

    # --- manifest ------------------------------------------------------------
    def _load_manifest(self) -> dict:
        if cfg.KNOWLEDGE_MANIFEST.exists():
            try:
                return json.loads(cfg.KNOWLEDGE_MANIFEST.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                log.warning("knowledge manifest unreadable; treating as empty")
        return {}

    def _save_manifest(self, manifest: dict):
        write_json_atomic(cfg.KNOWLEDGE_MANIFEST, manifest)

    # --- ingestion -----------------------------------------------------------
    @staticmethod
    def _chunk(text: str):
        """Split one page's text into overlapping character windows. Whitespace is
        collapsed first so chunk sizes reflect real content, not PDF layout gaps."""
        text = " ".join((text or "").split())
        if not text:
            return
        size = cfg.KB_CHUNK_CHARS
        step = max(1, size - cfg.KB_CHUNK_OVERLAP)
        for start in range(0, len(text), step):
            chunk = text[start:start + size].strip()
            if chunk:
                yield chunk
            if start + size >= len(text):
                break

    def _embed_sections(self, path, file_hash: str, title: str, sections) -> int:
        """Chunk and upsert a sequence of ``(text, page)`` sections (``page`` may be
        None for sourceless formats like plain text). Returns the chunk count."""
        ids, docs, metas = [], [], []
        n_chunks = 0

        def flush():
            nonlocal ids, docs, metas
            if ids:
                self._col.upsert(ids=ids, documents=docs, metadatas=metas)
                ids, docs, metas = [], [], []

        for text, page in sections:
            for chunk in self._chunk(text):
                meta = {"source": path.name, "title": title}
                if page is not None:
                    meta["page"] = page
                ids.append(f"kb_{file_hash[:12]}_{n_chunks}")
                docs.append(chunk)
                metas.append(meta)
                n_chunks += 1
                if len(ids) >= _UPSERT_BATCH:
                    flush()
        flush()
        return n_chunks

    def _ingest_pdf(self, path, file_hash: str) -> dict:
        """Extract, chunk, and embed one PDF (page by page). Returns a manifest entry."""
        reader = PdfReader(str(path))
        title = path.stem
        try:
            meta_title = (reader.metadata or {}).title
            if meta_title and meta_title.strip():
                title = meta_title.strip()
        except Exception:  # some PDFs have malformed/encrypted metadata
            pass

        def pages():
            for page_no, page in enumerate(reader.pages, start=1):
                try:
                    yield (page.extract_text() or "", page_no)
                except Exception as e:
                    log.warning("%s p.%d: extract failed: %s", path.name, page_no, e)

        n_chunks = self._embed_sections(path, file_hash, title, pages())
        return {
            "source": path.name,
            "title": title,
            "pages": len(reader.pages),
            "chunks": n_chunks,
            "ingested": datetime.now().isoformat(timespec="seconds"),
        }

    def _ingest_text(self, path, file_hash: str) -> dict:
        """Chunk and embed a plain-text or markdown file (no page structure)."""
        title = path.stem
        text = path.read_text(encoding="utf-8", errors="replace")
        n_chunks = self._embed_sections(path, file_hash, title, [(text, None)])
        return {
            "source": path.name,
            "title": title,
            "chunks": n_chunks,
            "ingested": datetime.now().isoformat(timespec="seconds"),
        }

    def ingest_folder(self) -> str:
        """Scan the knowledge folder and embed any PDF not already ingested.

        Idempotent and cheap when nothing is new: files whose content hash is
        already in the manifest are skipped before Chroma is even loaded. Returns a
        short human-readable summary suitable for logging."""
        files = sorted(
            p for ext in ("*.pdf", "*.txt", "*.md")
            for p in cfg.KNOWLEDGE_DIR.glob(ext)
        )
        if not files:
            return (f"No PDF or text files found in {cfg.KNOWLEDGE_DIR} — add some "
                    "and run this again.")

        manifest = self._load_manifest()
        known_hashes = set(manifest)
        added, skipped, failed = [], 0, 0

        for path in files:
            try:
                file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as e:
                log.warning("could not read %s: %s", path.name, e)
                failed += 1
                continue
            if file_hash in known_hashes:
                skipped += 1
                continue
            # Only now (a genuinely new file) do we pay the Chroma/model cost.
            self._ensure_chroma()
            try:
                if path.suffix.lower() == ".pdf":
                    entry = self._ingest_pdf(path, file_hash)
                else:
                    entry = self._ingest_text(path, file_hash)
            except Exception as e:
                log.warning("failed to ingest %s: %s", path.name, e)
                failed += 1
                continue
            manifest[file_hash] = entry
            self._save_manifest(manifest)  # persist per-file so a crash keeps progress
            known_hashes.add(file_hash)
            added.append(entry)
            log.info("ingested %d chunks from '%s' (%s)",
                     entry["chunks"], entry["title"], path.name)

        if not added:
            if skipped and not failed:
                return f"Knowledge base up to date — {skipped} file(s) already ingested."
            parts = []
            if skipped:
                parts.append(f"{skipped} already ingested")
            if failed:
                parts.append(f"{failed} failed")
            return "Nothing new ingested" + (f" ({', '.join(parts)})." if parts else ".")

        titles = ", ".join(f"'{e['title']}'" for e in added)
        total_chunks = sum(e["chunks"] for e in added)
        summary = f"Ingested {total_chunks} chunk(s) from {len(added)} new file(s): {titles}."
        if skipped:
            summary += f" Skipped {skipped} already-ingested file(s)."
        if failed:
            summary += f" {failed} file(s) failed."
        return summary

    # --- retrieval (used as a Claude tool) -----------------------------------
    def search(self, query: str, n: int = None) -> str:
        n = n or cfg.KB_SEARCH_RESULTS
        manifest = self._load_manifest()
        if not manifest:
            return ("No trading knowledge has been ingested yet. Add PDFs or text "
                    "files to the knowledge folder and run python voice_agent.py --ingest.")
        self._ensure_chroma()
        count = self._col.count()
        if count == 0:
            return "No trading knowledge has been ingested yet."
        res = self._col.query(query_texts=[query], n_results=min(n, count))
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        if not docs:
            return "I couldn't find anything about that in your trading knowledge."
        out = []
        for doc, meta in zip(docs, metas):
            meta = meta or {}
            title = meta.get("title", meta.get("source", "source"))
            page = meta.get("page")
            cite = f"{title}, p.{page}" if page else title
            snippet = " ".join(doc.split())[:400]
            out.append(f"[{cite}] {snippet}")
        return "\n\n".join(out)

    def list_sources(self) -> str:
        manifest = self._load_manifest()
        if not manifest:
            return f"No knowledge sources ingested yet. Add PDFs or text files to {cfg.KNOWLEDGE_DIR}."
        lines = []
        for entry in sorted(manifest.values(), key=lambda e: e.get("ingested", "")):
            pages = entry.get("pages")
            loc = f"{pages} pages, " if pages else ""
            lines.append(
                f"{entry.get('title', entry.get('source'))} "
                f"({loc}{entry.get('chunks', '?')} chunks, "
                f"ingested {entry.get('ingested', 'unknown')})"
            )
        return "Ingested knowledge sources:\n" + "\n".join(lines)

    def forget(self, name: str) -> str:
        """Remove an ingested source (its chunks + manifest entry) by title or
        filename, so a corrected file can be re-ingested cleanly."""
        name = (name or "").strip().lower()
        if not name:
            return "Which source should I forget? Give a title or filename."
        manifest = self._load_manifest()
        match = None
        for h, entry in manifest.items():
            if name in (entry.get("title", "").lower(), entry.get("source", "").lower()) \
                    or name in entry.get("title", "").lower():
                match = (h, entry)
                break
        if match is None:
            return f"No ingested source matches '{name}'."
        h, entry = match
        self._ensure_chroma()
        try:
            self._col.delete(where={"source": entry["source"]})
        except Exception as e:
            log.warning("chroma delete for %s: %s", entry["source"], e)
        manifest.pop(h, None)
        self._save_manifest(manifest)
        log.info("forgot knowledge source '%s'", entry.get("title"))
        return f"Removed '{entry.get('title', entry.get('source'))}' from the knowledge base."
