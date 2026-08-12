"""Long-term conversation memory — per persona.

Each persona's live thread is a rolling window (cfg.HISTORY_MAX_MESSAGES);
anything older would be lost. Instead, messages that fall off the window are
staged here tagged with the persona that owned them, and later *consolidated*:
one cheap model call per persona summarises its staged excerpt into a dense
memory record, embedded into that persona's own Chroma collection
(conversations_<key>). Retrieval is scoped the same way — a caller searches
its own staging, its own archive, and its own saved live window (plus any
registry `reads` grants), and structurally cannot touch another persona's.
The pre-isolation `conversations` collection and untagged staged batches are
legacy: genuinely shared history, readable by all, labelled as such.

Staging is free (a JSON append, no model call), so it happens inline whenever
the window trims. Consolidation runs at boot, and only for personas with
enough material (cfg.MEMORY_MIN_MESSAGES each), so most boots skip it. If the
model call fails (offline, etc.) that persona's staged text is kept and
retried next boot — nothing is dropped.
"""

import json
import logging
import re
from datetime import datetime

from stores import chroma_store

import config as cfg
from brain import agents
from lib.atomic_io import read_json, write_json_atomic

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
        self._cols = {}  # collection name -> Chroma collection, loaded lazily

    def _col_for(self, name: str):
        col = self._cols.get(name)
        if col is None:
            log.info("loading chroma collection '%s' (first use)...", name)
            col = self._cols[name] = chroma_store.collection(name)
        return col

    @staticmethod
    def _readable(caller):
        """Staged-batch owners `caller` may read. None (legacy, pre-isolation
        batches) is always readable — that history was genuinely shared."""
        return {None, *agents.readable_owners(caller)}

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
        return read_json(cfg.MEMORY_PENDING_PATH, [], expect=list,
                         warn=lambda e: log.warning(
                             "memory staging file unreadable; starting fresh"))

    def _save_pending(self, pending: list):
        # Atomic (temp + rename) so a power loss mid-save can't corrupt the
        # staging file and lose not-yet-consolidated memory.
        write_json_atomic(cfg.MEMORY_PENDING_PATH, pending)

    def record_dropped(self, messages, owner) -> int:
        """Stage messages that fell off `owner`'s history window. The owner is
        captured HERE, at drop time — by the boot-time consolidate the active
        persona is long gone, so a batch without its owner could only ever be
        legacy. Returns how many lines were kept (plain user/assistant text;
        tool traffic is dropped)."""
        lines = [t for m in messages if (t := self._message_text(m))]
        if not lines:
            return 0
        pending = self._load_pending()
        pending.append(
            {"ts": datetime.now().isoformat(timespec="seconds"),
             "agent": owner, "lines": lines}
        )
        self._save_pending(pending)
        log.info("staged %d line(s) for %s's long-term memory", len(lines), owner)
        return len(lines)

    # --- consolidation (one model call per persona, run at boot) ---------------
    def consolidate(self, client) -> str | None:
        """Summarise each persona's staged text into its own archive. The
        MEMORY_MIN_MESSAGES threshold applies PER persona — two Tom lines must
        not get consolidated just because Alice staged twenty. At most one
        model call per persona with enough material (bounded and rare — most
        boots stage nothing at all). Returns a combined status line, or None
        when no group had enough."""
        pending = self._load_pending()
        groups = {}  # owner key (None = legacy) -> [batch, ...] in order
        for batch in pending:
            owner = batch.get("agent")
            if owner is not None and owner not in agents.AGENTS:
                owner = None  # a renamed/removed persona's past is legacy now
            groups.setdefault(owner, []).append(batch)

        done, statuses = set(), []  # id()s of batches already embedded
        for owner, batches in groups.items():
            n_lines = sum(len(b.get("lines", [])) for b in batches)
            if n_lines < cfg.MEMORY_MIN_MESSAGES:
                continue

            blocks = []
            for batch in batches:
                ts = batch.get("ts", "")
                day = ts[:10] if ts else "unknown date"
                blocks.append(f"[{day}]\n" + "\n".join(batch.get("lines", [])))
            transcript = "\n\n".join(blocks)

            resp = client.messages.create(
                model=cfg.CONVO_MODEL,
                max_tokens=cfg.MEMORY_MAX_TOKENS,
                thinking={"type": "disabled"},
                messages=[{"role": "user",
                           "content": CONSOLIDATE_PROMPT + transcript}],
            )
            summary = "".join(b.text for b in resp.content
                              if b.type == "text").strip()
            if not summary:
                log.warning("consolidation for %s returned no text; keeping "
                            "staged", owner or "legacy")
                continue

            first = (batches[0].get("ts") or "")[:10]
            last = (batches[-1].get("ts") or "")[:10]
            date = first if first == last else f"{first} to {last}"

            name = (cfg.agent_memory_collection(owner) if owner
                    else cfg.MEMORY_COLLECTION)
            doc_id = datetime.now().strftime("conv_%Y-%m-%d_%H%M%S")
            self._col_for(name).upsert(
                ids=[doc_id],
                documents=[summary],
                metadatas=[{"date": date, "messages": n_lines}],
            )
            # Clear THIS group from staging as soon as its embed lands: a
            # failure in a later group must keep only that group staged, not
            # re-archive this one as a duplicate next boot.
            done.update(id(b) for b in batches)
            self._save_pending([b for b in pending if id(b) not in done])
            statuses.append(f"{n_lines} message(s) for "
                            f"{owner or 'the shared archive'} ({date})")

        if not statuses:
            return None
        return "archived " + "; ".join(statuses)

    # --- retrieval (used as a Claude tool) -------------------------------------
    def _staged_batches(self, caller) -> list:
        """Staged batches `caller` may read, plus a synthetic batch per
        readable persona's SAVED live thread. The live read is what makes
        "ask Alice what we discussed" work at all: a delegate starts with an
        empty message list (llm.run_delegated_task), so without it anything
        still inside Alice's 40-message window would be invisible to her.
        The files are on disk and saved every turn — this is a read, not IO
        machinery."""
        readable = self._readable(caller)
        batches = [b for b in self._load_pending()
                   if (b.get("agent") if b.get("agent") in agents.AGENTS
                       else None) in readable]
        for k in agents.readable_owners(caller):
            msgs = read_json(cfg.history_path(k), [], expect=list)
            lines = [t for m in msgs
                     if isinstance(m, dict) and (t := self._message_text(m))]
            if lines:
                batches.append({"ts": "current session", "lines": lines})
        return batches

    def recall_staged(self, client, query: str, caller=None) -> str | None:
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
        batches = self._staged_batches(caller)
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

    def search_staged(self, query: str, max_lines: int = 12,
                      caller=None) -> list:
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
        for batch in self._staged_batches(caller):
            ts = (batch.get("ts") or "")[:16].replace("T", " ")
            for line in batch.get("lines", []):
                if any(w in line.lower() for w in words):
                    hits.append(f"[{ts}] {' '.join(line.split())[:300]}")
        return hits[-max_lines:]  # most recent matches win the budget

    def _query_archive(self, name: str, query: str, n: int):
        """One collection's hits as (distance, doc, meta) rows, plus its record
        count and error text. Never raises.

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
                col = self._col_for(name)
                count = col.count()
                if not count:
                    return [], 0, None
                res = col.query(query_texts=[query], n_results=min(n, count))
                docs = res.get("documents", [[]])[0]
                metas = res.get("metadatas", [[]])[0]
                dists = res.get("distances", [[]])[0]
                return list(zip(dists, docs, metas)), count, None
            except Exception as e:  # noqa: BLE001 - reported, never raised
                error = str(e)
                if attempt == 1:
                    log.warning("archive search of %s failed (%s); rebuilding "
                                "the collection and retrying once", name, e)
                    self._cols.pop(name, None)
                else:
                    log.error("archive search of %s still failing after a "
                              "reconnect: %s", name, e)
        return [], None, error

    def _archive_section(self, query: str, n: int, caller=None):
        """(section text or None, record count or None, error text or None)
        across every archive `caller` may read: its own conversations_<key>
        (plus grants) and the legacy shared collection, whose hits are
        labelled so the model knows they predate per-agent memory. One
        embedding space, so distances merge honestly."""
        sources = [(cfg.agent_memory_collection(k), False)
                   for k in agents.readable_owners(caller)]
        sources.append((cfg.MEMORY_COLLECTION, True))
        hits, total, error = [], 0, None
        for name, legacy in sources:
            rows, count, err = self._query_archive(name, query, n)
            hits.extend((dist, doc, meta, legacy) for dist, doc, meta in rows)
            if count is None:
                error = error or err
            else:
                total += count
        if not hits:
            return None, (None if error and not total else total), error
        hits.sort(key=lambda h: h[0])
        archived = []
        for _, doc, meta, legacy in hits[:n]:
            tag = " (from the shared archive, before per-agent memory)" \
                if legacy else ""
            archived.append(f"[{(meta or {}).get('date', 'unknown date')}{tag}] "
                            f"{' '.join(doc.split())[:800]}")
        return ("From archived conversations:\n"
                + "\n\n".join(archived)), total, error

    def search(self, query: str, n: int = None, client=None,
               caller=None) -> str:
        n = n or cfg.MEMORY_SEARCH_RESULTS
        sections = []

        # Same-session recall first: staged lines are newer than any archive
        # record and verbatim, so when both match, these are the better answer.
        # Preferred path is the LLM read (handles queries the literal words
        # can't match); the keyword scan backs it up.
        recalled = (self.recall_staged(client, query, caller=caller)
                    if client is not None else None)
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
            staged = self.search_staged(query, caller=caller)
            if staged:
                header = ("From earlier in this session (not yet archived)"
                          if recalled is None else
                          "Lines from earlier in this session that mention "
                          "this (keyword matches — a closer read judged them "
                          "irrelevant, so weigh them yourself)")
                sections.append(header + ":\n" + "\n".join(staged))

        archive, count, error = self._archive_section(query, n, caller=caller)
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
