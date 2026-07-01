"""Note storage, retrieval, and semantic search.

Transcripts are written live (one append per recognised utterance) so nothing is
lost if a session is interrupted. Summaries are saved with YAML frontmatter and
indexed into a persistent Chroma collection for semantic search. A small
`index.json` keeps an ordered record for fast "recent notes" lookups.
"""

import collections
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
        self._migrate_legacy()
        self._col = None
        log.info("note store ready (%d notes on disk)", len(self.index))

    def _migrate_legacy(self):
        """One-time move of pre-category notes (flat summaries/transcripts) into the
        General folder, tagging them so the storage model stays uniform. Runs before
        Chroma loads (keeps startup fast) and is idempotent — a no-op once every
        index entry has a category."""
        legacy = [nid for nid, info in self.index.items() if "category" not in info]
        if not legacy:
            return
        dest = cfg.category_dir(cfg.DEFAULT_CATEGORY)
        dest.mkdir(parents=True, exist_ok=True)
        for nid in legacy:
            old_summary = cfg.SUMMARY_DIR / f"{nid}.md"
            if old_summary.exists():
                try:
                    old_summary.replace(dest / f"{nid}.md")
                except OSError as e:
                    log.warning("migrate: summary %s: %s", nid, e)
            old_tx = cfg.TRANSCRIPT_DIR / f"{nid}.md"
            if old_tx.exists():
                try:
                    old_tx.replace(dest / f"{nid}.transcript.md")
                except OSError as e:
                    log.warning("migrate: transcript %s: %s", nid, e)
            self.index[nid]["category"] = cfg.DEFAULT_CATEGORY
        self._save_index()
        log.info("migrated %d legacy note(s) into %s", len(legacy), cfg.DEFAULT_CATEGORY)
        for d in (cfg.SUMMARY_DIR, cfg.TRANSCRIPT_DIR):  # drop now-empty legacy dirs
            try:
                if d.exists() and not any(d.iterdir()):
                    d.rmdir()
            except OSError:
                pass

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
        path = cfg.PENDING_DIR / f"{note_id}.md"
        path.write_text(
            f"# Transcript {note_id}\n\nStarted {datetime.now().isoformat(timespec='seconds')}\n\n",
            encoding="utf-8",
        )
        return note_id

    def append_transcript(self, note_id: str, text: str):
        path = cfg.PENDING_DIR / f"{note_id}.md"
        with path.open("a", encoding="utf-8") as f:
            f.write(text.strip() + "\n")

    def read_transcript(self, note_id: str) -> str:
        # A saved note's transcript lives in its category folder; an in-progress
        # one is still in the pending staging dir.
        cat = (self.index.get(note_id) or {}).get("category")
        candidates = []
        if cat:
            candidates.append(cfg.category_dir(cat) / f"{note_id}.transcript.md")
        candidates.append(cfg.PENDING_DIR / f"{note_id}.md")
        path = next((p for p in candidates if p.exists()), None)
        if path is None:
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
    def save_summary(self, note_id: str, title: str, full_markdown: str, category: str = None):
        category = category if category in cfg.NOTE_CATEGORIES else cfg.DEFAULT_CATEGORY
        date = datetime.now().isoformat(timespec="seconds")
        cdir = cfg.category_dir(category)
        cdir.mkdir(parents=True, exist_ok=True)
        path = cdir / f"{note_id}.md"
        frontmatter = (
            f"---\ntitle: {title}\ndate: {date}\nid: {note_id}\ncategory: {category}\n---\n\n"
        )
        path.write_text(frontmatter + full_markdown.strip() + "\n", encoding="utf-8")

        self.index[note_id] = {"title": title, "date": date, "category": category}
        self._save_index()

        # Move the live transcript out of staging into the chosen category folder.
        pending = cfg.PENDING_DIR / f"{note_id}.md"
        if pending.exists():
            try:
                pending.replace(cdir / f"{note_id}.transcript.md")
            except OSError as e:
                log.warning("could not move transcript for %s: %s", note_id, e)
        else:
            log.warning("no pending transcript found for %s", note_id)

        self._ensure_chroma()
        self._col.upsert(
            ids=[note_id],
            documents=[full_markdown],
            metadatas=[{"title": title, "date": date, "category": category}],
        )
        log.info("saved summary %s (%s) -> %s", note_id, title, category)
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
            out.append(f"[{nid}] {title} ({date}) — {self._category_label(nid)}\n{snippet}")
        return "\n\n".join(out)

    def list_recent_notes(self, n: int = 5) -> str:
        if not self.index:
            return "No notes have been saved yet."
        recent = sorted(self.index.items(), key=lambda kv: kv[0], reverse=True)[:n]
        return "\n".join(
            f"[{nid}] {info['title']} ({info['date']}) — {self._category_label(nid)}"
            for nid, info in recent
        )

    def read_note(self, note_id: str) -> str:
        cat = (self.index.get(note_id) or {}).get("category", cfg.DEFAULT_CATEGORY)
        path = cfg.category_dir(cat) / f"{note_id}.md"
        if path.exists():
            return path.read_text(encoding="utf-8")
        transcript = self.read_transcript(note_id)
        return transcript or f"No note found with id {note_id}."

    def _category_label(self, note_id: str) -> str:
        cat = (self.index.get(note_id) or {}).get("category", cfg.DEFAULT_CATEGORY)
        return cfg.NOTE_CATEGORIES.get(cat, {}).get("display", cat)

    @staticmethod
    def _match_category(name: str):
        """Resolve a folder name (slug, display, or folder) to a category slug,
        tolerating loose phrasing. Returns None if nothing matches."""
        name = (name or "").strip().lower()
        if not name:
            return None
        for slug, meta in cfg.NOTE_CATEGORIES.items():
            if name in (slug.lower(), meta["display"].lower(), meta["folder"].lower()):
                return slug
        for slug, meta in cfg.NOTE_CATEGORIES.items():
            label = meta["display"].lower()
            if name in slug.lower() or name in label or label in name:
                return slug
        return None

    def list_folders(self) -> str:
        lines = [
            f"{meta['display']}: {meta['description']}"
            for meta in cfg.NOTE_CATEGORIES.values()
        ]
        return "You can file notes into these folders. " + " ".join(lines)

    def count_notes(self, folder: str = None) -> str:
        counts = collections.Counter(
            (info.get("category") or cfg.DEFAULT_CATEGORY) for info in self.index.values()
        )
        if not folder:
            total = sum(counts.values())
            if total == 0:
                return "You have no notes yet."
            parts = [
                f"{counts.get(slug, 0)} in {meta['display']}"
                for slug, meta in cfg.NOTE_CATEGORIES.items()
            ]
            return f"You have {total} notes total: " + ", ".join(parts) + "."
        slug = self._match_category(folder)
        if slug is None:
            known = ", ".join(m["display"] for m in cfg.NOTE_CATEGORIES.values())
            return f"I don't have a folder matching '{folder}'. Folders are: {known}."
        n = counts.get(slug, 0)
        display = cfg.NOTE_CATEGORIES[slug]["display"]
        return f"You have {n} note{'' if n == 1 else 's'} in {display}."
