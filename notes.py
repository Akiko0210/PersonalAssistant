"""Note storage, retrieval, and semantic search.

Transcripts are written live (one append per recognised utterance) so nothing is
lost if a session is interrupted. Summaries are saved with YAML frontmatter and
indexed into a persistent Chroma collection for semantic search. A small
`index.json` keeps an ordered record for fast "recent notes" lookups.
"""

import collections
import json
import logging
import re
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
    def _resolve_scope(self, folder):
        """Resolve an optional spoken folder name into a slug for scoped queries.
        Returns (slug, error_message) — slug None means unscoped; error set means
        the name didn't match any folder."""
        if not folder:
            return None, None
        slug = self._match_category(folder)
        if slug is None:
            known = ", ".join(m["display"] for m in cfg.NOTE_CATEGORIES.values())
            return None, f"I don't have a folder matching '{folder}'. Folders are: {known}."
        return slug, None

    def _note_category(self, note_id: str) -> str:
        return (self.index.get(note_id) or {}).get("category") or cfg.DEFAULT_CATEGORY

    def search_notes(self, query: str, n: int = None, folder: str = None) -> str:
        n = n or cfg.SEARCH_RESULTS
        if not self.index:
            return "No notes have been saved yet."
        slug, err = self._resolve_scope(folder)
        if err:
            return err
        self._ensure_chroma()
        # When scoped, over-fetch and filter by the index — the index is the
        # authoritative record of each note's category (Chroma metadata could lag
        # for notes moved before metadata syncing was fixed).
        fetch = len(self.index) if slug else min(n, len(self.index))
        res = self._col.query(query_texts=[query], n_results=fetch)
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        out = []
        for nid, doc, meta in zip(ids, docs, metas):
            if slug and self._note_category(nid) != slug:
                continue
            title = (meta or {}).get("title", nid)
            date = (meta or {}).get("date", "")
            snippet = " ".join(doc.split())[:400]
            out.append(f"[{nid}] {title} ({date}) — {self._category_label(nid)}\n{snippet}")
            if len(out) >= n:
                break
        if not out:
            if slug:
                disp = cfg.NOTE_CATEGORIES[slug]["display"]
                return f"No matching notes found in {disp}."
            return "No matching notes found."
        return "\n\n".join(out)

    def list_recent_notes(self, n: int = 5, folder: str = None) -> str:
        if not self.index:
            return "No notes have been saved yet."
        slug, err = self._resolve_scope(folder)
        if err:
            return err
        items = self.index.items()
        if slug:
            items = [(nid, info) for nid, info in items
                     if self._note_category(nid) == slug]
            if not items:
                disp = cfg.NOTE_CATEGORIES[slug]["display"]
                return f"There are no notes in {disp} yet."
        recent = sorted(items, key=lambda kv: kv[0], reverse=True)[:n]
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

    def create_folder(self, name: str, description: str = None) -> str:
        """Create a new note folder from a spoken name. Rejects a name that already
        maps to an existing folder so we don't create confusing near-duplicates."""
        name = (name or "").strip()
        if not name:
            return "What would you like to call the new folder?"
        existing = self._match_category(name)
        if existing is not None:
            return f"You already have a folder called {cfg.NOTE_CATEGORIES[existing]['display']}."
        slug = cfg.add_category(name, description)
        log.info("created folder %s (%s)", slug, cfg.NOTE_CATEGORIES[slug]["display"])
        return f"Created a new folder called {cfg.NOTE_CATEGORIES[slug]['display']}."

    def rename_folder(self, current: str, new_name: str) -> str:
        """Rename an existing folder. Existing notes stay filed under it (the slug
        is preserved); only the display name and its on-disk directory change."""
        slug = self._match_category(current)
        if slug is None:
            known = ", ".join(m["display"] for m in cfg.NOTE_CATEGORIES.values())
            return f"I couldn't find a folder called '{current}'. Your folders are: {known}."
        new_name = (new_name or "").strip()
        if not new_name:
            return "What should I rename it to?"
        clash = self._match_category(new_name)
        if clash is not None and clash != slug:
            return f"There's already a folder called {cfg.NOTE_CATEGORIES[clash]['display']}."
        old_display = cfg.NOTE_CATEGORIES[slug]["display"]
        cfg.rename_category(slug, new_name)
        new_display = cfg.NOTE_CATEGORIES[slug]["display"]
        log.info("renamed folder %s: %s -> %s", slug, old_display, new_display)
        return f"Renamed {old_display} to {new_display}."

    # --- moving notes / deleting folders -------------------------------------
    def _move_note_files(self, note_id: str, from_slug: str, to_slug: str):
        """Relocate a note's summary and transcript files between category dirs."""
        src, dst = cfg.category_dir(from_slug), cfg.category_dir(to_slug)
        dst.mkdir(parents=True, exist_ok=True)
        for suffix in (".md", ".transcript.md"):
            p = src / f"{note_id}{suffix}"
            if p.exists():
                try:
                    p.replace(dst / f"{note_id}{suffix}")
                except OSError as e:
                    log.warning("move %s: %s", p, e)

    def _rewrite_category(self, note_id: str, slug: str):
        """Rewrite the summary file's frontmatter so its `category:` line matches
        the folder it now lives in — the file itself must never disagree with the
        index, or anything reading the note directly sees a stale category."""
        path = cfg.category_dir(slug) / f"{note_id}.md"
        if not path.exists():
            return
        try:
            text = path.read_text(encoding="utf-8")
            new, n = re.subn(r"(?m)^category:\s*.*$", f"category: {slug}", text, count=1)
            if n:
                path.write_text(new, encoding="utf-8")
        except OSError as e:
            log.warning("frontmatter update for %s: %s", note_id, e)

    def _reassign(self, note_id: str, to_slug: str):
        """Move one note into another category, keeping every copy of its category
        consistent: files on disk, frontmatter, index, and Chroma metadata."""
        info = self.index[note_id]
        self._move_note_files(note_id, info.get("category", cfg.DEFAULT_CATEGORY), to_slug)
        info["category"] = to_slug
        self._rewrite_category(note_id, to_slug)
        # Always sync Chroma too (loading it if needed): folder-scoped queries and
        # search results read this metadata, so it must not lag behind the index.
        self._ensure_chroma()
        try:
            self._col.update(
                ids=[note_id],
                metadatas=[{"title": info.get("title", ""),
                            "date": info.get("date", ""),
                            "category": to_slug}],
            )
        except Exception as e:  # never let a metadata hiccup fail the move
            log.warning("chroma metadata update for %s: %s", note_id, e)

    def move_note(self, note_id: str, to_folder: str) -> str:
        """Move a single saved note into another folder. `note_id` is the id shown
        in brackets by search_notes / list_recent_notes."""
        note_id = (note_id or "").strip()
        info = self.index.get(note_id)
        if not info:
            return f"I couldn't find a note with id {note_id}."
        to_slug = self._match_category(to_folder)
        if to_slug is None:
            known = ", ".join(m["display"] for m in cfg.NOTE_CATEGORIES.values())
            return f"I don't have a folder called '{to_folder}'. Your folders are: {known}."
        if info.get("category", cfg.DEFAULT_CATEGORY) == to_slug:
            return f"That note is already in {cfg.NOTE_CATEGORIES[to_slug]['display']}."
        title = info.get("title", note_id)
        self._reassign(note_id, to_slug)
        self._save_index()
        log.info("moved note %s -> %s", note_id, to_slug)
        return f"Moved '{title}' to {cfg.NOTE_CATEGORIES[to_slug]['display']}."

    def resync(self) -> str:
        """One-pass consistency repair: make every note's index entry, on-disk
        location, frontmatter, and Chroma metadata agree. The index is
        authoritative when its slug exists; when it doesn't (e.g. a slug that was
        renamed in config), the note adopts the folder its file actually lives in.
        Safe to run repeatedly."""
        self._ensure_chroma()
        index_fixed = moved = fm_fixed = chroma_fixed = 0
        problems = []

        for nid, info in self.index.items():
            slug = info.get("category")
            disk_slug = next(
                (s for s, m in cfg.NOTE_CATEGORIES.items()
                 if (cfg.DATA_DIR / m["folder"] / f"{nid}.md").exists()),
                None,
            )
            if slug not in cfg.NOTE_CATEGORIES:
                if disk_slug is None:
                    problems.append(f"{nid}: unknown category '{slug}' and no file found")
                    continue
                log.info("resync %s: dead slug '%s' -> '%s' (from disk)", nid, slug, disk_slug)
                info["category"] = slug = disk_slug
                index_fixed += 1
            elif disk_slug is None:
                problems.append(f"{nid}: summary file missing from every folder")
                continue
            elif disk_slug != slug:
                log.info("resync %s: file in '%s' but index says '%s' — moving file",
                         nid, disk_slug, slug)
                self._move_note_files(nid, disk_slug, slug)
                moved += 1

            # Frontmatter: rewrite only when it actually disagrees.
            path = cfg.category_dir(slug) / f"{nid}.md"
            try:
                text = path.read_text(encoding="utf-8")
                m = re.search(r"(?m)^category:\s*(.+)$", text)
                if m and m.group(1).strip() != slug:
                    log.info("resync %s: frontmatter '%s' -> '%s'",
                             nid, m.group(1).strip(), slug)
                    self._rewrite_category(nid, slug)
                    fm_fixed += 1
            except OSError as e:
                problems.append(f"{nid}: could not read summary: {e}")

            # Chroma metadata.
            try:
                got = self._col.get(ids=[nid])
                if not got["ids"]:
                    problems.append(f"{nid}: not embedded in Chroma")
                elif (got["metadatas"][0] or {}).get("category") != slug:
                    log.info("resync %s: chroma '%s' -> '%s'", nid,
                             (got["metadatas"][0] or {}).get("category"), slug)
                    self._col.update(
                        ids=[nid],
                        metadatas=[{"title": info.get("title", ""),
                                    "date": info.get("date", ""),
                                    "category": slug}],
                    )
                    chroma_fixed += 1
            except Exception as e:
                problems.append(f"{nid}: chroma update failed: {e}")

        self._save_index()
        report = (f"Resync done: {index_fixed} index slug(s) fixed, {moved} file(s) "
                  f"moved, {fm_fixed} frontmatter(s) rewritten, {chroma_fixed} "
                  f"chroma record(s) updated.")
        if problems:
            report += "\nUnresolved:\n" + "\n".join(f"  - {p}" for p in problems)
        return report

    def delete_folder(self, name: str, move_notes_to: str = None) -> str:
        """Delete a folder. Any notes in it are relocated first (never destroyed) —
        to `move_notes_to` if given, otherwise to the default General folder. The
        default folder itself can't be deleted."""
        slug = self._match_category(name)
        if slug is None:
            known = ", ".join(m["display"] for m in cfg.NOTE_CATEGORIES.values())
            return f"I couldn't find a folder called '{name}'. Your folders are: {known}."
        if slug == cfg.DEFAULT_CATEGORY:
            disp = cfg.NOTE_CATEGORIES[slug]["display"]
            return f"I can't delete the {disp} folder — it's the default that catches everything else."
        if move_notes_to:
            dest = self._match_category(move_notes_to)
            if dest is None:
                known = ", ".join(m["display"] for m in cfg.NOTE_CATEGORIES.values())
                return (f"I don't have a folder called '{move_notes_to}' to move the notes into. "
                        f"Your folders are: {known}.")
            if dest == slug:
                return "That's the folder you're deleting — tell me a different one for the notes."
        else:
            dest = cfg.DEFAULT_CATEGORY

        note_ids = [nid for nid, info in self.index.items()
                    if info.get("category", cfg.DEFAULT_CATEGORY) == slug]
        for nid in note_ids:
            self._reassign(nid, dest)
        if note_ids:
            self._save_index()

        disp = cfg.NOTE_CATEGORIES[slug]["display"]
        cfg.delete_category(slug)
        log.info("deleted folder %s (moved %d note(s) to %s)", slug, len(note_ids), dest)
        if note_ids:
            n = len(note_ids)
            dest_disp = cfg.NOTE_CATEGORIES[dest]["display"]
            return (f"Deleted the {disp} folder and moved its {n} "
                    f"note{'' if n == 1 else 's'} to {dest_disp}.")
        return f"Deleted the {disp} folder."

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
