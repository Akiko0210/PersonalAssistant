"""Long-term conversation memory.

The live chat history is a rolling window (cfg.HISTORY_MAX_MESSAGES); anything
older would be lost. Instead, messages that fall off the window are staged here
(their plain text — tool chatter is skipped) and later *consolidated*: one cheap
model call summarises the staged excerpt into a dense memory record, which is
embedded into a persistent Chroma collection. The agent can then answer "what
did we talk about last month?" via the search_past_conversations tool.

Staging is free (a JSON append, no model call), so it happens inline whenever
the window trims. Consolidation runs at boot, and only once enough material has
accumulated (cfg.MEMORY_MIN_MESSAGES), so most boots skip it entirely. If the
model call fails (offline, etc.) the staged text is kept and retried next boot —
nothing is dropped.
"""

import json
import logging
import re
from datetime import datetime

import chromadb
from chromadb.utils import embedding_functions

import config as cfg
from atomic_io import write_json_atomic

log = logging.getLogger("memory")

CONSOLIDATE_PROMPT = """You are archiving part of a voice-assistant conversation \
into long-term memory. Summarise the excerpt below into one compact memory record: \
what was discussed, concrete facts, numbers, names, decisions, and preferences — \
anything the user might ask about weeks later. Dense plain prose, no preamble, no \
markdown.

Excerpt:
"""

RECALL_PROMPT = """You are the memory-recall subsystem of a voice assistant. \
Below are verbatim conversation lines that scrolled out of the assistant's \
recent window earlier in THIS session (oldest first, timestamped). Extract \
ONLY material that directly bears on the query: the relevant statements, \
decisions, and numbers, keeping exact figures and quoting or closely \
paraphrasing the lines, with their timestamps. Plain prose, no markdown, no \
preamble. This is strict: if the lines below contain nothing that answers the \
query, reply with exactly NOTHING_RELEVANT — never substitute loosely related \
or recent-but-off-topic material, and never answer from your own knowledge. \
An honest NOTHING_RELEVANT lets the assistant check its other memory stores; \
an off-topic answer misleads it.

Query: {query}

Staged conversation:
"""


class ConversationMemory:
    def __init__(self):
        cfg.ensure_dirs()
        self._col = None  # Chroma collection, loaded lazily on first real use

    def _ensure_chroma(self):
        if self._col is not None:
            return
        log.info("loading embedding model + chroma (memory, first use)...")
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=cfg.EMBED_MODEL
        )
        client = chromadb.PersistentClient(path=str(cfg.CHROMA_DIR))
        self._col = client.get_or_create_collection(
            name=cfg.MEMORY_COLLECTION, embedding_function=ef
        )
        log.info("memory collection ready")

    # --- staging (free — no model call) ---------------------------------------
    @staticmethod
    def _message_text(msg) -> str | None:
        """Flatten one history message to 'role: text'. Tool results and tool-use
        blocks are skipped — the spoken conversation is what's worth remembering."""
        role = msg.get("role", "")
        content = msg.get("content")
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            parts = [
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            text = " ".join(p for p in parts if p).strip()
        else:
            text = ""
        return f"{role}: {text}" if text else None

    def _load_pending(self) -> list:
        if cfg.MEMORY_PENDING_PATH.exists():
            try:
                data = json.loads(cfg.MEMORY_PENDING_PATH.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data
            except (OSError, ValueError):
                log.warning("memory staging file unreadable; starting fresh")
        return []

    def _save_pending(self, pending: list):
        # Atomic (temp + rename) so a power loss mid-save can't corrupt the
        # staging file and lose not-yet-consolidated memory.
        write_json_atomic(cfg.MEMORY_PENDING_PATH, pending)

    def record_dropped(self, messages) -> int:
        """Stage messages that fell off the history window. Returns how many
        lines were kept (plain user/assistant text; tool traffic is dropped)."""
        lines = [t for m in messages if (t := self._message_text(m))]
        if not lines:
            return 0
        pending = self._load_pending()
        pending.append(
            {"ts": datetime.now().isoformat(timespec="seconds"), "lines": lines}
        )
        self._save_pending(pending)
        log.info("staged %d line(s) for long-term memory", len(lines))
        return len(lines)

    # --- consolidation (one model call, run at boot) ---------------------------
    def consolidate(self, client) -> str | None:
        """Summarise staged text into one memory record and embed it. Returns a
        status line, or None when there wasn't enough staged to bother."""
        pending = self._load_pending()
        n_lines = sum(len(batch.get("lines", [])) for batch in pending)
        if n_lines < cfg.MEMORY_MIN_MESSAGES:
            return None

        blocks = []
        for batch in pending:
            ts = batch.get("ts", "")
            day = ts[:10] if ts else "unknown date"
            blocks.append(f"[{day}]\n" + "\n".join(batch.get("lines", [])))
        transcript = "\n\n".join(blocks)

        resp = client.messages.create(
            model=cfg.CONVO_MODEL,
            max_tokens=cfg.MEMORY_MAX_TOKENS,
            thinking={"type": "disabled"},
            messages=[{"role": "user", "content": CONSOLIDATE_PROMPT + transcript}],
        )
        summary = "".join(b.text for b in resp.content if b.type == "text").strip()
        if not summary:
            log.warning("memory consolidation returned no text; keeping staged")
            return None

        first = (pending[0].get("ts") or "")[:10]
        last = (pending[-1].get("ts") or "")[:10]
        date = first if first == last else f"{first} to {last}"

        self._ensure_chroma()
        doc_id = datetime.now().strftime("conv_%Y-%m-%d_%H%M%S")
        self._col.upsert(
            ids=[doc_id],
            documents=[summary],
            metadatas=[{"date": date, "messages": n_lines}],
        )
        # Clear staging only after the embed succeeded — a failure above keeps
        # everything staged for the next boot.
        self._save_pending([])
        return f"archived {n_lines} message(s) into long-term memory ({date})"

    # --- retrieval (used as a Claude tool) -------------------------------------
    def recall_staged(self, client, query: str) -> str | None:
        """LLM read over the staged, not-yet-consolidated lines: hand the
        WHOLE staged text to a cheap model with the query and let it extract
        what's relevant. No retrieval step means no retrieval misses — this is
        what makes complex queries ('what did we decide about sizing?') work,
        where a keyword scan can only match literal words. Runs only when the
        memory tool is invoked, so it costs nothing per turn; a full session's
        staging is a few thousand tokens, well under a cent on the default
        model.

        Returns the extracted answer; "" when the model read everything and
        found nothing relevant (a real answer — don't fall back); None when
        there is nothing staged or the call failed (caller should fall back
        to the offline keyword scan)."""
        batches = self._load_pending()
        lines = []
        for batch in batches:
            ts = (batch.get("ts") or "")[:16].replace("T", " ")
            for line in batch.get("lines", []):
                lines.append(f"[{ts}] {line}")
        if not lines:
            return None
        # Budget the staged text, newest lines kept — 100k chars ≈ 25k tokens.
        text, budget = [], 100_000
        for line in reversed(lines):
            budget -= len(line) + 1
            if budget < 0:
                break
            text.append(line)
        staged = "\n".join(reversed(text))
        try:
            resp = client.messages.create(
                model=cfg.CONVO_MODEL,
                max_tokens=cfg.MEMORY_MAX_TOKENS,
                thinking={"type": "disabled"},
                messages=[{"role": "user",
                           "content": RECALL_PROMPT.format(query=query) + staged}],
            )
        except Exception as e:  # offline etc. — degrade to the keyword scan
            log.warning("staged-memory recall failed (%s); using keyword scan", e)
            return None
        answer = "".join(b.text for b in resp.content if b.type == "text").strip()
        if not answer:
            return None
        if "NOTHING_RELEVANT" in answer:
            return ""
        return answer

    def search_staged(self, query: str, max_lines: int = 12) -> list:
        """Keyword scan over the staged, NOT-yet-consolidated lines — the
        verbatim text of messages that fell off the window since the last
        boot. Consolidation only runs at startup, so without this a long
        session has a blind spot: something said two hours ago is neither in
        the live window nor searchable in the archive (exactly how Cobe lost
        a trade structure mid-session, 2026-07-20 21:07). No model call, no
        embeddings — the lines are already on disk; just read them."""
        words = {w for w in re.findall(r"[a-z0-9']+", (query or "").lower())
                 if len(w) > 2}
        if not words:
            return []
        hits = []
        for batch in self._load_pending():
            ts = (batch.get("ts") or "")[:16].replace("T", " ")
            for line in batch.get("lines", []):
                if any(w in line.lower() for w in words):
                    hits.append(f"[{ts}] {' '.join(line.split())[:300]}")
        return hits[-max_lines:]  # most recent matches win the budget

    def search(self, query: str, n: int = None, client=None) -> str:
        n = n or cfg.MEMORY_SEARCH_RESULTS
        sections = []

        # Same-session recall first: staged lines are newer than any archive
        # record and verbatim, so when both match, these are the better answer.
        # Preferred path is the LLM read (handles queries the literal words
        # can't match); the keyword scan is the offline/no-client fallback.
        # recalled == "" means the model read everything and found nothing —
        # that's an answer, not a failure, so no fallback then.
        recalled = self.recall_staged(client, query) if client is not None else None
        if recalled:
            sections.append("From earlier in this session (not yet archived):\n"
                            + recalled)
        elif recalled is None:
            staged = self.search_staged(query)
            if staged:
                sections.append("From earlier in this session (not yet archived):\n"
                                + "\n".join(staged))

        self._ensure_chroma()
        count = self._col.count()
        if count:
            res = self._col.query(query_texts=[query], n_results=min(n, count))
            docs = res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0]
            archived = [f"[{(meta or {}).get('date', 'unknown date')}] "
                        f"{' '.join(doc.split())[:800]}"
                        for doc, meta in zip(docs, metas)]
            if archived:
                sections.append("From archived conversations:\n"
                                + "\n\n".join(archived))

        if sections:
            return "\n\n".join(sections)
        if count == 0:
            return ("No archived conversations yet — long-term memory only fills "
                    "up as older conversations age out of the recent window.")
        return "Nothing in past conversations matches that."
