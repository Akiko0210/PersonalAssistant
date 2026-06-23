"""Note storage, retrieval, and semantic search.

Transcripts are written live (one append per recognised utterance) so nothing is
lost if a session is interrupted. Summaries are saved with YAML frontmatter and
indexed into a persistent Chroma collection for semantic search. A small
`index.json` keeps an ordered record for fast "recent notes" lookups.
"""

import json
import logging
from datetime import datetime

import chromadb
from chromadb.utils import embedding_functions

import config as cfg

log = logging.getLogger("notes")


class NoteStore:
    def __init__(self):
        cfg.ensure_dirs()
        self.index = self._load_index()
        self._col = None
        log.info("note store ready (%d notes on disk)", len(self.index))

    def _ensure_chroma(self):
        if self._col is not None:
            return
        log.info("loading embedding model + chroma (first use)...")
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=cfg.EMBED_MODEL
        )
        client = chromadb.PersistentClient(path=str(cfg.CHROMA_DIR))
        self._col = client.get_or_create_collection(
            name="notes", embedding_function=ef
        )
        log.info("chroma ready")

    # --- index helpers -------------------------------------------------------
    def _load_index(self):
        if cfg.INDEX_PATH.exists():
            return json.loads(cfg.INDEX_PATH.read_text(encoding="utf-8"))
        return {}

    def _save_index(self):
        cfg.INDEX_PATH.write_text(
            json.dumps(self.index, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # --- live transcript -----------------------------------------------------
    def new_session(self) -> str:
        note_id = datetime.now().strftime("note_%Y-%m-%d_%H%M%S")
        path = cfg.TRANSCRIPT_DIR / f"{note_id}.md"
        path.write_text(
            f"# Transcript {note_id}\n\nStarted {datetime.now().isoformat(timespec='seconds')}\n\n",
            encoding="utf-8",
        )
        return note_id

    def append_transcript(self, note_id: str, text: str):
        path = cfg.TRANSCRIPT_DIR / f"{note_id}.md"
        with path.open("a", encoding="utf-8") as f:
            f.write(text.strip() + "\n")

    def read_transcript(self, note_id: str) -> str:
        path = cfg.TRANSCRIPT_DIR / f"{note_id}.md"
        if not path.exists():
            return ""
        lines = path.read_text(encoding="utf-8").splitlines()
        # drop the header block (first blank-line separated section)
        body = []
        seen_blank = 0
        for ln in lines:
            if seen_blank >= 2:
                body.append(ln)
            elif ln.strip() == "":
                seen_blank += 1
        return "\n".join(body).strip()

    # --- summaries -----------------------------------------------------------
    def save_summary(self, note_id: str, title: str, full_markdown: str):
        date = datetime.now().isoformat(timespec="seconds")
        path = cfg.SUMMARY_DIR / f"{note_id}.md"
        frontmatter = f"---\ntitle: {title}\ndate: {date}\nid: {note_id}\n---\n\n"
        path.write_text(frontmatter + full_markdown.strip() + "\n", encoding="utf-8")

        self.index[note_id] = {"title": title, "date": date}
        self._save_index()

        self._ensure_chroma()
        self._col.upsert(
            ids=[note_id],
            documents=[full_markdown],
            metadatas=[{"title": title, "date": date}],
        )
        log.info("saved summary %s (%s)", note_id, title)
        return path

    # --- retrieval (used as Claude tools) ------------------------------------
    def search_notes(self, query: str, n: int = None) -> str:
        n = n or cfg.SEARCH_RESULTS
        if not self.index:
            return "No notes have been saved yet."
        self._ensure_chroma()
        res = self._col.query(query_texts=[query], n_results=min(n, len(self.index)))
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        if not ids:
            return "No matching notes found."
        out = []
        for nid, doc, meta in zip(ids, docs, metas):
            title = (meta or {}).get("title", nid)
            date = (meta or {}).get("date", "")
            snippet = " ".join(doc.split())[:400]
            out.append(f"[{nid}] {title} ({date})\n{snippet}")
        return "\n\n".join(out)

    def list_recent_notes(self, n: int = 5) -> str:
        if not self.index:
            return "No notes have been saved yet."
        recent = sorted(self.index.items(), key=lambda kv: kv[0], reverse=True)[:n]
        return "\n".join(
            f"[{nid}] {info['title']} ({info['date']})" for nid, info in recent
        )

    def read_note(self, note_id: str) -> str:
        path = cfg.SUMMARY_DIR / f"{note_id}.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        transcript = self.read_transcript(note_id)
        return transcript or f"No note found with id {note_id}."
