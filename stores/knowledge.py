"""Trading knowledge base: one-time document/video ingestion + semantic search.

Reference material the user drops into the project-root ``knowledge/`` folder is
chunked, embedded once, and stored in a persistent Chroma collection (under
``data/chroma``) separate from notes. PDFs and text are read directly; video and
audio (``.mp4`` and friends) are transcribed by Whisper first, and their chunks
carry the timestamp they were spoken at so a citation points at the moment to
rewatch. The agent queries all of it on demand via the ``search_knowledge`` tool,
so the content is never pasted into the conversation.

Ingestion is idempotent: each file is identified by the SHA-256 of its bytes and
recorded in a manifest. A file whose hash is already known is skipped before any
extraction or embedding happens, so re-scanning an unchanged folder costs only a
few hashes (no embedding-model load). This lets the scan run cheaply at every
boot while a given book is embedded exactly once.

Media is the one exception to "scan everything at boot": transcribing an hour of
video takes minutes, not the seconds a PDF costs, so the boot scan lists new
media and leaves it for an explicit ``--ingest`` run rather than holding the
agent's startup open. See ``ingest_folder(include_media=...)``.
"""

import hashlib
import json
import logging
from datetime import datetime

from stores import chroma_store
from pypdf import PdfReader

import config as cfg
from lib.atomic_io import read_json, write_json_atomic

log = logging.getLogger("knowledge")

_UPSERT_BATCH = 256  # chunks per Chroma upsert call (keeps memory/latency bounded)
_HASH_BLOCK = 1 << 20  # 1 MiB; video is far too big to slurp whole for hashing


def _hms(seconds) -> str:
    """Format a transcript offset as M:SS, or H:MM:SS once it passes an hour."""
    total = max(0, int(seconds or 0))
    hours, minutes, secs = total // 3600, (total % 3600) // 60, total % 60
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


class KnowledgeStore:
    def __init__(self):
        cfg.ensure_dirs()
        self._col = None  # Chroma collection, loaded lazily on first real use
        self._whisper = None  # transcription model, loaded only if media shows up

    # --- chroma --------------------------------------------------------------
    def _ensure_chroma(self):
        if self._col is not None:
            return
        log.info("loading embedding model + chroma (knowledge, first use)...")
        self._col = chroma_store.collection(cfg.KNOWLEDGE_COLLECTION)
        log.info("knowledge collection ready")

    # --- whisper -------------------------------------------------------------
    def _ensure_whisper(self):
        """Load the transcription model on first media file. Kept lazy and local
        to this module so a text-only knowledge base never pays for it, and so
        the live agent (which has its own dictation model) only ever holds a
        second one in memory during an explicit ingest run."""
        if self._whisper is not None:
            return
        from faster_whisper import WhisperModel

        log.info("loading whisper model %s for transcription...", cfg.KB_MEDIA_MODEL)
        self._whisper = WhisperModel(
            cfg.KB_MEDIA_MODEL,
            device=cfg.WHISPER_DEVICE,
            compute_type=cfg.WHISPER_COMPUTE,
        )
        log.info("transcription model ready")

    # --- manifest ------------------------------------------------------------
    def _load_manifest(self) -> dict:
        return read_json(cfg.KNOWLEDGE_MANIFEST, {},
                         warn=lambda e: log.warning(
                             "knowledge manifest unreadable; treating as empty"))

    def _save_manifest(self, manifest: dict):
        write_json_atomic(cfg.KNOWLEDGE_MANIFEST, manifest)

    # --- ingestion -----------------------------------------------------------
    @staticmethod
    def _file_hash(path) -> str:
        """SHA-256 of a file's bytes, read in blocks. Video runs to gigabytes, so
        unlike a PDF it can't safely be slurped whole just to identify it."""
        digest = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(_HASH_BLOCK), b""):
                digest.update(block)
        return digest.hexdigest()

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
        """Chunk and upsert a sequence of ``(text, locator)`` sections. ``locator``
        is extra metadata pinning the text to a place in its source -- ``{"page": n}``
        for PDFs, ``{"t": seconds}`` for transcripts, ``None`` for formats like
        plain text that have no such structure. Returns the chunk count."""
        ids, docs, metas = [], [], []
        n_chunks = 0

        def flush():
            nonlocal ids, docs, metas
            if ids:
                self._col.upsert(ids=ids, documents=docs, metadatas=metas)
                ids, docs, metas = [], [], []

        for text, locator in sections:
            for chunk in self._chunk(text):
                meta = {"source": path.name, "title": title}
                if locator:
                    meta.update(locator)
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
                    yield (page.extract_text() or "", {"page": page_no})
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

    @staticmethod
    def _media_windows(segments):
        """Group Whisper segments into chunk-sized windows, each tagged with the
        start time of its first segment.

        Whisper emits a few seconds of speech at a time -- far too granular to
        embed one vector per segment, which would strand every sentence from its
        context. We accumulate up to ``KB_CHUNK_CHARS`` and carry a tail of about
        ``KB_CHUNK_OVERLAP`` characters into the next window, mirroring the
        overlap the character chunker gives PDFs so a passage straddling a window
        boundary is still retrievable whole."""
        buf, size = [], 0

        def window():
            return (" ".join(text for _, text in buf), {"t": buf[0][0]})

        for seg in segments:
            text = (seg.text or "").strip()
            if not text:
                continue
            if buf and size + len(text) + 1 > cfg.KB_CHUNK_CHARS:
                yield window()
                tail, kept = [], 0
                for start, prev in reversed(buf):
                    if kept >= cfg.KB_CHUNK_OVERLAP:
                        break
                    tail.insert(0, (start, prev))
                    kept += len(prev) + 1
                buf, size = tail, kept
            buf.append((float(seg.start or 0.0), text))
            size += len(text) + 1
        if buf:
            yield window()

    @staticmethod
    def _with_progress(segments, total, name, on_progress):
        """Pass segments through, reporting how far into the recording we are.

        Without this a long transcription is a single log line followed by tens
        of minutes of silence, which is indistinguishable from a hang -- and this
        is the one operation slow enough for that to matter."""
        next_mark = 5
        for seg in segments:
            if total:
                pct = min(100, int(seg.end / total * 100))
                if pct >= next_mark:
                    next_mark = pct - (pct % 5) + 5
                    log.info("  %s: %d%% (%s of %s)",
                             name, pct, _hms(seg.end), _hms(total))
                    if on_progress:
                        on_progress(name, pct)
            yield seg

    def _ingest_media(self, path, file_hash: str, on_progress=None) -> dict:
        """Transcribe one audio/video file and embed the transcript with
        timestamps. This is the slow path -- minutes per hour of material -- but
        the manifest means it is paid exactly once per file."""
        self._ensure_whisper()
        log.info("transcribing '%s' (slow, but only ever once)...", path.name)
        # PyAV decodes the container, so mp4/mkv/mov work without ffmpeg on PATH.
        segments, info = self._whisper.transcribe(
            str(path), language="en", vad_filter=True, beam_size=5,
        )
        duration = getattr(info, "duration", None)
        if duration:
            log.info("  %s is %s long", path.name, _hms(duration))
        # `segments` is lazy: the transcription actually runs as it's consumed.
        n_chunks = self._embed_sections(
            path, file_hash, path.stem,
            self._media_windows(
                self._with_progress(segments, duration, path.name, on_progress)),
        )
        entry = {
            "source": path.name,
            "title": path.stem,
            "chunks": n_chunks,
            "ingested": datetime.now().isoformat(timespec="seconds"),
        }
        if duration:
            entry["duration"] = _hms(duration)
        return entry

    def _pending_media(self, manifest: dict) -> list:
        """Names of media files in the folder that aren't in the manifest yet.

        Matched by filename rather than content hash on purpose: this runs on
        every boot, and hashing a multi-gigabyte video just to discover we
        already have it would cost more than the check is worth."""
        known = {entry.get("source") for entry in manifest.values()}
        return sorted(
            p.name for ext in cfg.KB_MEDIA_EXTS
            for p in cfg.KNOWLEDGE_DIR.glob(f"*{ext}")
            if p.name not in known
        )

    def ingest_folder(self, include_media: bool = True, on_progress=None) -> str:
        """Scan the knowledge folder and embed anything not already ingested.

        Idempotent and cheap when nothing is new: files whose content hash is
        already in the manifest are skipped before Chroma is even loaded. Returns a
        short human-readable summary suitable for logging.

        ``include_media`` is False on the agent's boot scan: transcribing video
        would hold startup open for minutes, so new media is reported and left
        for an explicit ``--ingest`` run instead.

        ``on_progress``, if given, is called as ``(name, pct=None)``: once with
        just the name as each new file is picked up — the only warning a caller
        gets that it is about to spend minutes on a transcription — and then
        repeatedly with a percentage while a recording is being transcribed. The
        dashboard uses it to drive its status line."""
        exts = ["*.pdf", "*.txt", "*.md"]
        if include_media:
            exts += [f"*{ext}" for ext in cfg.KB_MEDIA_EXTS]
        files = sorted(
            p for ext in exts for p in cfg.KNOWLEDGE_DIR.glob(ext)
        )
        manifest = self._load_manifest()
        # Media the boot scan is deliberately walking past, so it can say so
        # rather than looking like the file was ignored.
        deferred = [] if include_media else self._pending_media(manifest)

        if not files:
            if deferred:
                return (f"{len(deferred)} new video/audio file(s) waiting in "
                        f"{cfg.KNOWLEDGE_DIR}: {', '.join(deferred)}. "
                        "Run --ingest to transcribe them.")
            return (f"No PDF, text, or video files found in {cfg.KNOWLEDGE_DIR} — "
                    "add some and run this again.")

        known_hashes = set(manifest)
        added, skipped, failed = [], 0, 0

        for path in files:
            try:
                file_hash = self._file_hash(path)
            except OSError as e:
                log.warning("could not read %s: %s", path.name, e)
                failed += 1
                continue
            if file_hash in known_hashes:
                skipped += 1
                continue
            # Only now (a genuinely new file) do we pay the Chroma/model cost.
            if on_progress:
                on_progress(path.name)
            self._ensure_chroma()
            try:
                suffix = path.suffix.lower()
                if suffix == ".pdf":
                    entry = self._ingest_pdf(path, file_hash)
                elif suffix in cfg.KB_MEDIA_EXTS:
                    entry = self._ingest_media(path, file_hash, on_progress)
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

        note = ""
        if deferred:
            note = (f" {len(deferred)} new video/audio file(s) not transcribed yet "
                    f"({', '.join(deferred)}) — run --ingest for those.")

        if not added:
            if skipped and not failed:
                return (f"Knowledge base up to date — {skipped} file(s) already "
                        f"ingested.{note}")
            parts = []
            if skipped:
                parts.append(f"{skipped} already ingested")
            if failed:
                parts.append(f"{failed} failed")
            return ("Nothing new ingested"
                    + (f" ({', '.join(parts)})." if parts else ".") + note)

        titles = ", ".join(f"'{e['title']}'" for e in added)
        total_chunks = sum(e["chunks"] for e in added)
        summary = f"Ingested {total_chunks} chunk(s) from {len(added)} new file(s): {titles}."
        if skipped:
            summary += f" Skipped {skipped} already-ingested file(s)."
        if failed:
            summary += f" {failed} file(s) failed."
        return summary + note

    # --- retrieval (used as a Claude tool) -----------------------------------
    def search(self, query: str, n: int = None) -> str:
        n = n or cfg.KB_SEARCH_RESULTS
        manifest = self._load_manifest()
        if not manifest:
            return ("No trading knowledge has been ingested yet. Add PDFs, text, "
                    "or video files to the knowledge folder and run "
                    "python voice_agent.py --ingest.")
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
            page, at = meta.get("page"), meta.get("t")
            if page:
                cite = f"{title}, p.{page}"
            elif at is not None:
                cite = f"{title}, {_hms(at)}"  # a moment to rewatch
            else:
                cite = title
            snippet = " ".join(doc.split())[:400]
            out.append(f"[{cite}] {snippet}")
        return "\n\n".join(out)

    def list_sources(self) -> str:
        manifest = self._load_manifest()
        if not manifest:
            return (f"No knowledge sources ingested yet. Add PDFs, text, or video "
                    f"files to {cfg.KNOWLEDGE_DIR}.")
        lines = []
        for entry in sorted(manifest.values(), key=lambda e: e.get("ingested", "")):
            pages, duration = entry.get("pages"), entry.get("duration")
            loc = f"{pages} pages, " if pages else (f"{duration}, " if duration else "")
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
