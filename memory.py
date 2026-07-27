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

# The staged read is a needle-in-a-haystack job over the entire staged buffer
# (~90k characters in a long session), and the conversation default is picked
# for latency, not for that. Measured against a real staged file, Haiku 4.5
# answered NOTHING_RELEVANT for every one of three topics that were plainly in
# the text (the keyword scan found 10-12 matching lines for each); Sonnet found
# all three on the same prompt and still correctly rejected two control queries
# about things never discussed. Softening the prompt did not rescue Haiku — the
# model is the variable that matters here, so recall pins its own rather than
# following the conversation's. It runs only when the user explicitly asks
# about past conversations, so the cost lands on the turns that need it.
RECALL_MODEL = cfg.CONVO_MODELS["sonnet"]


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
                model=RECALL_MODEL,
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
        the live window nor searchable in the archive (exactly how Tom lost
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

    def _archive_section(self, query: str, n: int):
        """(section text or None, record count or None, error text or None)
        from the Chroma archive. Never raises.

        A broken archive must not cost the caller the staged results it has
        already gathered: this used to be one bare query() whose exception
        unwound search() entirely, so a Chroma fault came back to the model as
        a lone "Tool error: ..." with every same-session hit discarded
        (2026-07-27 08:12, "Error creating hnsw segment reader: Nothing found
        on disk" against index files that were demonstrably present).

        The first failure drops the cached collection and retries once, since
        a long-lived process can be left holding a handle the store no longer
        honours while a freshly built client reads the very same files. That
        is best-effort — Chroma may hand back the same underlying system — so
        the caller still gets a clean error to report if it doesn't take."""
        error = None
        for attempt in (1, 2):
            try:
                self._ensure_chroma()
                count = self._col.count()
                if not count:
                    return None, 0, None
                res = self._col.query(query_texts=[query],
                                      n_results=min(n, count))
                docs = res.get("documents", [[]])[0]
                metas = res.get("metadatas", [[]])[0]
                archived = [f"[{(meta or {}).get('date', 'unknown date')}] "
                            f"{' '.join(doc.split())[:800]}"
                            for doc, meta in zip(docs, metas)]
                if not archived:
                    return None, count, None
                return ("From archived conversations:\n"
                        + "\n\n".join(archived)), count, None
            except Exception as e:  # noqa: BLE001 - reported, never raised
                error = str(e)
                if attempt == 1:
                    log.warning("archive search failed (%s); rebuilding the "
                                "collection and retrying once", e)
                    self._col = None
                else:
                    log.error("archive search still failing after a "
                              "reconnect: %s", e)
        return None, None, error

    def search(self, query: str, n: int = None, client=None) -> str:
        n = n or cfg.MEMORY_SEARCH_RESULTS
        sections = []

        # Same-session recall first: staged lines are newer than any archive
        # record and verbatim, so when both match, these are the better answer.
        # Preferred path is the LLM read (handles queries the literal words
        # can't match); the keyword scan backs it up.
        recalled = self.recall_staged(client, query) if client is not None else None
        if recalled:
            sections.append("From earlier in this session (not yet archived):\n"
                            + recalled)
        else:
            # No content from the read — the call failed (None) or it reported
            # nothing relevant (""). Neither verdict ends the search any more.
            # "Nothing relevant" used to be trusted as final, which made a
            # false negative fatal: it suppressed this scan even though the
            # literal words were sitting in the staged lines. The scan is free
            # (no model call), so it runs either way and its hits are labelled
            # by how much they can be trusted.
            staged = self.search_staged(query)
            if staged:
                header = ("From earlier in this session (not yet archived)"
                          if recalled is None else
                          "Lines from earlier in this session that mention "
                          "this (keyword matches — a closer read judged them "
                          "irrelevant, so weigh them yourself)")
                sections.append(header + ":\n" + "\n".join(staged))

        archive, count, error = self._archive_section(query, n)
        if archive:
            sections.append(archive)

        if sections:
            if error:
                sections.append("(The archived-conversation index could not be "
                                "read just now, so only same-session memory was "
                                "searched.)")
            return "\n\n".join(sections)
        if error:
            return ("The archived-conversation index could not be read "
                    f"({error}). Nothing in same-session memory matched "
                    "either, so this isn't proof the topic never came up — "
                    "the archive simply wasn't searchable.")
        if count == 0:
            return ("No archived conversations yet — long-term memory only fills "
                    "up as older conversations age out of the recent window.")
        return "Nothing in past conversations matches that."
