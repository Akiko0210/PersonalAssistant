"""Conversation memory — one record per exchange, per persona.

Every user↔assistant exchange is embedded into the persona's own Chroma
collection (conversations_<key>) the moment its turn ends, and every later
turn retrieves from it (brain/context.py). Retrieval is hybrid: dense
similarity plus a BM25 index over the same records, because MiniLM embeddings
are weak on exactly the tokens spoken follow-ups hinge on — tickers, names,
numbers. Records are round-level (one exchange, not a session summary) with
their time in metadata: LongMemEval found round-level values read best and
that facts-as-values lose information, and recency is a ranking signal here,
not a filter.

Isolation is structural: a caller reads its own collection (plus registry
`reads` grants) and the pre-isolation shared `conversations` archive, whose
summary records predate per-agent memory and are labelled as legacy. There is
no parameter through which a persona can name another's collection.
"""

import hashlib
import logging
import math
import re
from collections import Counter, namedtuple
from datetime import datetime

from rank_bm25 import BM25Okapi

from stores import chroma_store

import config as cfg
from brain import agents, context
from lib.atomic_io import park, read_json

log = logging.getLogger("memory")

# Dense and lexical hits side by side. `count` is the number of records in the
# readable collections — None when unknown because every query failed — and
# `error` the last Chroma error text, so search() can tell the model
# "unsearchable" apart from "nothing there".
Rows = namedtuple("Rows", "dense lexical count error")

_WORD = re.compile(r"[a-z0-9']+")


def tokens(text) -> list:
    """BM25 tokens: lowercase words of three or more characters. One tokenizer
    for the index and the query, in one place."""
    return [w for w in _WORD.findall((text or "").lower()) if len(w) > 2]


def _bm25(corpus):
    """BM25Okapi with Lucene's IDF, log(1 + (N - n + 0.5) / (n + 0.5)), which
    is positive for every term. rank_bm25's own IDF is zero (or negative, then
    patched) for a term found in half the records — and a personal index is
    tiny at first, so "SPX" in one of two exchanges scored 0 and looked like
    no match at all. Positive IDF keeps "score > 0" meaning "matched"."""
    bm25 = BM25Okapi(corpus)
    df = Counter(word for doc in bm25.doc_freqs for word in doc)
    total = bm25.corpus_size
    bm25.idf = {w: math.log(1 + (total - n + 0.5) / (n + 0.5))
                for w, n in df.items()}
    return bm25


def exchange_id(owner, user_msg) -> str:
    """Deterministic id for the exchange that starts at `user_msg`, so
    re-indexing (the boot backfill, a deferred self-note landing after the
    turn) overwrites instead of duplicating. Live turns always carry a `ts`
    (converse stamps it at append); a pre-ts message hashes on text alone."""
    key = f"{user_msg.get('ts', '')}|{user_msg.get('content', '')}"
    return f"xc_{owner}_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:16]}"


COMPANION = ":a"  # id suffix of an exchange's assistant-half key record


def _approx(ts) -> bool:
    """True for a naive stamp — the shape the old staging file wrote, at flush
    time rather than speech time (off by minutes to a day; 388 records on
    2026-09-18). Live stamps carry an offset (history.now_iso) and are exact.
    Set on every exchange record: a where= equality never matches a record
    that lacks the key, so the time window relies on the False being there."""
    try:
        return datetime.fromisoformat(ts).tzinfo is None
    except (TypeError, ValueError):
        return False


def _reply_half(doc) -> str:
    """The assistant's part of a stored exchange document, from its first
    'assistant:' line on. Replies span lines (a read-back list), so this is
    one split, not a per-line filter; a user turn never follows an assistant
    one inside a single exchange."""
    i = doc.find("\nassistant:")
    return doc[i + 1:] if i >= 0 else ""


def _exchange_hit(h):
    """A companion row read back as its exchange: the parent's id (what the
    fusion and exclude_ids key on) and the whole text (what the model reads).
    Parents and legacy summaries pass through untouched."""
    parent = h.meta.get("exchange")
    if not parent:
        return h
    meta = {k: v for k, v in h.meta.items() if k not in ("exchange", "part", "text")}
    return h._replace(id=parent, doc=h.meta.get("text", h.doc), meta=meta)


def _time_where(window):
    """Chroma where= for an epoch window, exact stamps only — a flush-time
    stamp inside the window says nothing about when the words were said."""
    lo, hi = window
    return {"$and": [{"approx": False},
                     {"epoch": {"$gte": lo}}, {"epoch": {"$lte": hi}}]}


def _in_window(meta, window) -> bool:
    """The same test as _time_where, for the lexical list (BM25 runs over the
    in-memory corpus, not through Chroma)."""
    lo, hi = window
    epoch = meta.get("epoch")
    return (meta.get("approx") is False and isinstance(epoch, (int, float))
            and lo <= epoch <= hi)


class ConversationMemory:
    def __init__(self):
        cfg.ensure_dirs()
        self._cols = {}     # collection name -> Chroma collection, loaded lazily
        self._lexical = {}  # collection name -> (BM25Okapi, ids, docs, metas)

    def _col_for(self, name: str):
        col = self._cols.get(name)
        if col is None:
            log.info("loading chroma collection '%s' (first use)...", name)
            col = self._cols[name] = chroma_store.collection(name)
        return col

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

    # --- indexing -------------------------------------------------------------
    @staticmethod
    def _exchanges(messages):
        """Split a message list at plain-string user messages: one group per
        exchange, [(user_msg, [user_msg, ...its replies and tool traffic]), ...].
        Anything before the first plain user message has no exchange to
        belong to and is skipped."""
        groups = []
        for m in messages:
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                groups.append((m, [m]))
            elif groups:
                groups[-1][1].append(m)
        return groups

    @staticmethod
    def _records_for(xid, doc, meta):
        """One exchange as its records: the parent (id = the exchange id,
        document = the whole exchange — a retrieval key and the value both)
        and the companion (id + ':a', document = the LEAD LINE of the reply,
        metadata carrying the parent id and the full text). Two keys, one
        value. The lead line is the sentence that frames the exchange ("Got
        it — that's its own item. Your list now:"); everything after it is
        the exchange's substance, which dilutes the key the way the user's
        turn does. Measured 2026-09-18 over 259 exchanges, for "what is my
        to-do list": the read-back ranked 93rd keyed on the whole exchange,
        69th on the whole reply, 9th on its lead line — and the substance
        stays reachable through the parent key. The live path and the boot
        backfill both build records here, so the two shapes cannot drift."""
        parent = {**meta, "approx": _approx(meta.get("ts"))}
        ids, docs, metas = [xid], [doc], [parent]
        lead = _reply_half(doc).split("\n", 1)[0]
        if lead:
            ids.append(xid + COMPANION)
            docs.append(lead)
            metas.append({**parent, "exchange": xid, "part": "a", "text": doc})
        return ids, docs, metas

    def index_exchanges(self, messages, owner, *, skip_existing=True) -> int:
        """Embed each exchange in `messages` into `owner`'s collection. The ONE
        write path: the live turn (skip_existing=False, so a self-note that
        lands after the reply overwrites its exchange), the boot backfill of
        the saved threads, the staging-file migration, and the log seeder.
        Tool traffic is skipped; tool NAMES go into metadata, not the text —
        identifiers would shift a short document's embedding. Returns how
        many exchanges were written (two records each, _records_for)."""
        batch = []  # (exchange id, ids, docs, metas)
        for user_msg, group in self._exchanges(messages):
            lines = [t for m in group if (t := self._message_text(m))]
            if len(lines) < 2:
                continue  # no assistant text: an abandoned turn, nothing to remember
            ts = user_msg.get("ts") or ""
            meta = {"agent": owner, "ts": ts, "kind": "exchange"}
            try:
                meta["epoch"] = datetime.fromisoformat(ts).timestamp()
            except ValueError:
                pass  # pre-ts message: retrievable, just never boosted as recent
            used = sorted({b.get("name") for m in group
                           if isinstance(m.get("content"), list)
                           for b in m["content"]
                           if isinstance(b, dict) and b.get("type") == "tool_use"
                           and b.get("name")})
            if used:
                meta["tools"] = ",".join(used)
            xid = exchange_id(owner, user_msg)
            batch.append((xid, *self._records_for(xid, "\n".join(lines), meta)))
        if not batch:
            return 0
        name = cfg.agent_memory_collection(owner)
        col = self._col_for(name)
        if skip_existing:
            # Presence is judged by the parent; a parent whose companion is
            # missing is backfill_keys' job, not this path's.
            have = set(col.get(ids=[b[0] for b in batch], include=[]).get("ids") or [])
            batch = [b for b in batch if b[0] not in have]
            if not batch:
                return 0
        col.upsert(ids=[i for b in batch for i in b[1]],
                   documents=[d for b in batch for d in b[2]],
                   metadatas=[m for b in batch for m in b[3]])
        self._lexical.pop(name, None)  # rebuilt from the store on the next query
        return len(batch)

    def migrate_pending(self) -> int:
        """One-shot: fold the pre-retrieval staging file — batches of
        'role: text' lines that fell off the old window, awaiting a boot-time
        summary — into the exchange index, then park it as .bak (never
        delete; a wrong parse must stay recoverable). Untagged batches
        predate personas and have no collection to go to. Returns how many
        exchanges were indexed; 0 when there is no file."""
        path = cfg.MEMORY_PENDING_PATH
        if not path.exists():
            return 0
        n = 0
        for batch in read_json(path, [], expect=list):
            owner = batch.get("agent")
            if owner not in agents.AGENTS:
                continue
            ts = batch.get("ts", "")
            msgs = []
            for line in batch.get("lines", []):
                role, sep, text = str(line).partition(": ")
                if sep and role in ("user", "assistant") and text:
                    msgs.append({"role": role, "content": text, "ts": ts})
            n += self.index_exchanges(msgs, owner)
        parked = park(path)
        log.info("staging file: %d exchange(s) folded into the index; parked as %s",
                 n, parked.name)
        return n

    def backfill_keys(self, owner) -> tuple:
        """One-shot, idempotent boot pass over `owner`'s collection: give every
        exchange written before the companion keys existed (2026-09-18) its
        assistant-half record, and stamp `approx` on every record that lacks
        it — a metadata-only update, no re-embedding. No deletes, so a pass
        interrupted halfway leaves a working store and the next boot finishes
        it. Returns (companions added, records stamped): (0, 0) once the
        collection is current, which is every boot but one."""
        name = cfg.agent_memory_collection(owner)
        col = self._col_for(name)
        got = col.get(include=["documents", "metadatas"])
        ids = got.get("ids") or []
        docs = got.get("documents") or []
        metas = [m or {} for m in (got.get("metadatas") or [])]
        present = set(ids)
        add_ids, add_docs, add_metas, stamp_ids, stamp_metas = [], [], [], [], []
        for xid, doc, meta in zip(ids, docs, metas):
            if meta.get("kind") != "exchange" or meta.get("part"):
                continue
            if "approx" not in meta:
                stamp_ids.append(xid)
                stamp_metas.append({**meta, "approx": _approx(meta.get("ts"))})
            if xid + COMPANION not in present:
                rec_ids, rec_docs, rec_metas = self._records_for(xid, doc, meta)
                if len(rec_ids) > 1:
                    add_ids.append(rec_ids[1])
                    add_docs.append(rec_docs[1])
                    add_metas.append(rec_metas[1])
        if stamp_ids:
            col.update(ids=stamp_ids, metadatas=stamp_metas)
        for i in range(0, len(add_ids), 256):  # bounded batches; each one embeds
            col.upsert(ids=add_ids[i:i + 256], documents=add_docs[i:i + 256],
                       metadatas=add_metas[i:i + 256])
        if stamp_ids or add_ids:
            self._lexical.pop(name, None)
        return len(add_ids), len(stamp_ids)

    # --- retrieval ------------------------------------------------------------
    def _query_archive(self, name: str, query: str, rows: int, where=None):
        """One collection's nearest `rows` records as chroma_store.Hit rows —
        companion rows already folded back to their exchange (_exchange_hit),
        so two rows may share an id and collapse in the fusion — plus its
        record count and error text. Never raises.

        A broken collection must not cost the caller the others' results:
        this used to be one bare query() whose exception unwound search()
        entirely, so a Chroma fault came back to the model as a lone "Tool
        error: ..." with every other hit discarded (2026-07-27 08:12, "Error
        creating hnsw segment reader: Nothing found on disk" against index
        files that were demonstrably present).

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
                kwargs = {"query_texts": [query], "n_results": min(rows, count)}
                if where is not None:
                    kwargs["where"] = where
                res = col.query(**kwargs)
                return ([_exchange_hit(h) for h in chroma_store.hits(res)],
                        count, None)
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

    def _lexical_index(self, name: str):
        """BM25 over every exchange in collection `name` — parents only, since
        a companion repeats its parent's text and would count the exchange
        twice — built lazily from the store and dropped whenever a write
        lands (a few thousand short records rebuild in tens of milliseconds)."""
        cached = self._lexical.get(name)
        if cached is None:
            got = self._col_for(name).get(include=["documents", "metadatas"])
            rows = [(i, d, m or {})
                    for i, d, m in zip(got.get("ids") or [], got.get("documents") or [],
                                       got.get("metadatas") or [])
                    if not (m or {}).get("part")]
            ids = [r[0] for r in rows]
            docs = [r[1] for r in rows]
            metas = [r[2] for r in rows]
            bm25 = _bm25([tokens(d) for d in docs]) if docs else None
            cached = self._lexical[name] = (bm25, ids, docs, metas)
        return cached

    def _lexical_rows(self, name: str, query: str, n: int, window=None) -> list:
        bm25, ids, docs, metas = self._lexical_index(name)
        words = tokens(query)
        if bm25 is None or not words:
            return []
        scores = bm25.get_scores(words)
        order = sorted(range(len(ids)), key=lambda i: -scores[i])
        rows = [chroma_store.Hit(float(scores[i]), docs[i], metas[i], ids[i])
                for i in order
                if scores[i] > 0 and (window is None or _in_window(metas[i], window))]
        return rows[:n]

    def _vectors(self, name: str, ids) -> dict:
        """Stored embeddings for `ids` in collection `name`, {id: vector}, so
        the fill can measure redundancy between candidates (context._fill).
        {} on any failure — a missing vector costs a candidate its penalty,
        never the turn."""
        if not ids:
            return {}
        try:
            got = self._col_for(name).get(ids=list(ids), include=["embeddings"])
            embs = got.get("embeddings")
            return {} if embs is None else dict(zip(got.get("ids") or [], embs))
        except Exception as e:  # noqa: BLE001
            log.warning("could not read embeddings from %s: %s", name, e)
            return {}

    def query_rows(self, query: str, n: int = None, caller=None, *,
                   window=None) -> Rows:
        """Dense and lexical hits from every archive `caller` may read: its
        own conversations_<key> (plus grants) and the legacy shared
        collection, whose rows are tagged legacy so the reader knows they
        predate per-agent memory. One embedding space, so dense distances
        merge honestly; BM25 scores do not merge across collections, which
        is why the fusion normalises them per turn. `n` counts exchanges per
        collection and retriever. `window` — (lo, hi) epoch seconds —
        restricts both lists to exactly-timed exchanges inside it: the time
        filter behind search_past_conversations, which the per-turn
        Background never passes. Never raises: a broken collection reports
        through `error` and the others still answer."""
        n = n or cfg.MEMORY_SEARCH_RESULTS
        where = _time_where(window) if window is not None else None
        sources = [(cfg.agent_memory_collection(k), False)
                   for k in agents.readable_owners(caller)]
        sources.append((cfg.MEMORY_COLLECTION, True))
        dense, lexical, total, error = [], [], 0, None
        for name, legacy in sources:
            # A persona's collection holds two key records per exchange, so
            # twice the rows keep `n` meaning exchanges once they collapse;
            # the legacy archive has one record per summary.
            rows, count, err = self._query_archive(name, query,
                                                   n if legacy else 2 * n, where)
            if count is None:
                error = error or err
                continue
            total += count
            if not count:
                continue
            try:
                lex = self._lexical_rows(name, query, n, window)
            except Exception as e:  # the dense hits are still worth returning
                log.warning("lexical search of %s failed: %s", name, e)
                lex = []
            vecs = self._vectors(name, {h.id for h in rows + lex})
            rows = [h._replace(vec=vecs.get(h.id)) for h in rows]
            lex = [h._replace(vec=vecs.get(h.id)) for h in lex]
            if legacy:
                rows = [h._replace(meta={**h.meta, "legacy": True}) for h in rows]
                lex = [h._replace(meta={**h.meta, "legacy": True}) for h in lex]
            dense.extend(rows)
            lexical.extend(lex)
        dense.sort(key=lambda h: h.score)
        lexical.sort(key=lambda h: -h.score)
        return Rows(dense, lexical, None if error and not total else total, error)

    def rank(self, query: str, *, caller, window=None, exclude_ids=(), now=None):
        """The ranking behind search_past_conversations and the eval harness:
        CONTEXT_CANDIDATES deep, fused exactly as the Background is
        (context.fuse), so the tool can never disagree with what the model
        was already shown — it goes deeper and, with a window, elsewhere.
        Without a window the relevance gate applies. With one, the window IS
        the relevance test and the gate is off: the exchanges from "this
        morning at 9:41" sit at cosine 0.02–0.09 against that question and
        would never clear it (2026-09-17 12:10). Returns (candidates, rows)."""
        rows = self.query_rows(query, cfg.CONTEXT_CANDIDATES + len(exclude_ids),
                               caller, window=window)
        ranked = context.fuse(rows.dense, rows.lexical, now=now,
                              exclude_ids=exclude_ids)
        if window is None:
            ranked = [c for c in ranked if c.gate]
        return ranked, rows

    def search(self, query: str, n: int = None, caller=None, *, window=None,
               exclude_ids=()) -> str:
        """The search_past_conversations tool's text. A windowed result reads
        oldest-first — a recalled stretch of conversation is read in order —
        and says when it was cut, so the model narrows the window or adds a
        topic rather than taking the first `n` for the whole story."""
        n = n or cfg.MEMORY_SEARCH_RESULTS
        ranked, rows = self.rank(query, caller=caller, window=window,
                                 exclude_ids=exclude_ids)
        shown = ranked[:n]
        span = ""
        if window is not None:
            shown.sort(key=lambda c: c.meta.get("epoch") or 0)
            span = f" between {_clock(window[0])} and {_clock(window[1])}"
        if shown:
            out = []
            for c in shown:
                tag = (" (from the shared archive, before per-agent memory)"
                       if c.meta.get("legacy") else "")
                out.append(f"[{context.when(c.meta)}{tag}] "
                           f"{' '.join(c.doc.split())[:800]}")
            order = " (oldest first)" if window is not None else ""
            text = f"From past conversations{span}{order}:\n" + "\n\n".join(out)
            if window is not None and len(ranked) > n:
                text += (f"\n\n({n} of {len(ranked)} in that window — narrow it "
                         "or add a topic.)")
            if rows.error:
                text += ("\n\n(Part of the conversation index could not be read "
                         "just now, so this may be incomplete.)")
            return text
        if rows.error:
            return ("The conversation index could not be read "
                    f"({rows.error}), so this isn't proof the topic never came "
                    "up — it simply wasn't searchable.")
        if rows.count == 0:
            return ("No past conversations are indexed yet — memory fills as "
                    "we talk.")
        if window is not None:
            return f"Nothing{span} among exactly-timed exchanges."
        return "Nothing in past conversations matches that."


def _clock(epoch) -> str:
    """An epoch as the label the Background puts on exchanges ("9:00am
    9/17/2026"), so a window's bounds read like the stamps inside it. The
    open lower bound (0, from a window with no `since`) has no clock — and
    Windows cannot even convert it (OSError 22 for anything before 1970
    local time)."""
    if epoch <= 0:
        return "the beginning"
    stamp = datetime.fromtimestamp(epoch).astimezone().isoformat(timespec="seconds")
    return context.when({"ts": stamp})
