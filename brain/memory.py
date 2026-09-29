"""Conversation memory — one record per exchange, per persona.

Every user↔assistant exchange is embedded into the persona's own Chroma
collection (conversations_<key>) when its turn ends, and every later turn
retrieves from it (brain/context.py). Retrieval is hybrid: dense similarity
plus a BM25 index over the same records, because MiniLM embeddings are weak
on exactly the tokens spoken follow-ups hinge on — tickers, names, numbers.
Records are round-level (one exchange, not a session summary) with their
time in metadata: LongMemEval found round-level values read best, and
recency is a ranking signal here, not a filter.

This is the EPISODIC store — what was said, verbatim, the source of truth.
What is TRUE NOW (a list's current contents, a decision, a preference) is a
different question that top-k over exchanges answers badly; brain/facts.py
derives it from these records and can be rebuilt from them. Both stores read
through HybridStore below.

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


def and_where(*conds):
    """A Chroma where= over the given conditions (None ones dropped): one
    condition stands alone, since Chroma rejects a one-element $and."""
    conds = [c for c in conds if c]
    if not conds:
        return None
    return conds[0] if len(conds) == 1 else {"$and": conds}


def time_where(window):
    """Chroma where= for an epoch window, exact stamps only — a flush-time
    stamp inside the window says nothing about when the words were said."""
    lo, hi = window
    return {"$and": [{"approx": False},
                     {"epoch": {"$gte": lo}}, {"epoch": {"$lte": hi}}]}


def in_window(meta, window) -> bool:
    """The same test as time_where, for the lexical list (BM25 runs over the
    in-memory corpus, not through Chroma)."""
    lo, hi = window
    epoch = meta.get("epoch")
    return (meta.get("approx") is False and isinstance(epoch, (int, float))
            and lo <= epoch <= hi)


class HybridStore:
    """A Chroma collection read two ways — dense through Chroma, lexical
    through a BM25 index over the same records — with the failure handling
    both readers share. The exchange index below and the fact store
    (brain/facts.py) are its two shapes; each says how a stored row reads
    back (`_row`) and which rows BM25 indexes (`_indexable`)."""

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

    def _touch(self, name: str):
        """A write landed: the BM25 index is rebuilt from the store on the
        next query (a few thousand short records take tens of milliseconds)."""
        self._lexical.pop(name, None)

    @staticmethod
    def _row(h):
        return h

    @staticmethod
    def _indexable(meta) -> bool:
        return True

    def _query_archive(self, name: str, query: str, rows: int, where=None):
        """One collection's nearest `rows` records as chroma_store.Hit rows
        (through `_row`, so two rows may share an id and collapse in the
        fusion) plus its record count and error text. An empty query with a
        where= is a plain filtered read — "everything between 9 and 10 this
        morning" has no topic to embed. Never raises.

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
                if not query.strip() and where is not None:
                    got = col.get(where=where, include=["documents", "metadatas"])
                    hits = [chroma_store.Hit(1.0, d, m or {}, i)  # distance 1: no topic, no similarity
                            for i, d, m in zip(got.get("ids") or [],
                                               got.get("documents") or [],
                                               got.get("metadatas") or [])]
                else:
                    kwargs = {"query_texts": [query], "n_results": min(rows, count)}
                    if where is not None:
                        kwargs["where"] = where
                    hits = chroma_store.hits(col.query(**kwargs))
                return [self._row(h) for h in hits], count, None
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
        """BM25 over the indexable records of collection `name`, built lazily
        from the store and dropped whenever a write lands (_touch)."""
        cached = self._lexical.get(name)
        if cached is None:
            got = self._col_for(name).get(include=["documents", "metadatas"])
            rows = [(i, d, m or {})
                    for i, d, m in zip(got.get("ids") or [], got.get("documents") or [],
                                       got.get("metadatas") or [])
                    if self._indexable(m or {})]
            ids = [r[0] for r in rows]
            docs = [r[1] for r in rows]
            metas = [r[2] for r in rows]
            bm25 = _bm25([tokens(d) for d in docs]) if docs else None
            cached = self._lexical[name] = (bm25, ids, docs, metas)
        return cached

    def _lexical_rows(self, name: str, query: str, n: int, keep=None) -> list:
        """The `n` best BM25 matches whose metadata passes `keep` (the same
        predicate the dense side's where= expresses)."""
        bm25, ids, docs, metas = self._lexical_index(name)
        words = tokens(query)
        if bm25 is None or not words:
            return []
        scores = bm25.get_scores(words)
        order = sorted(range(len(ids)), key=lambda i: -scores[i])
        rows = [chroma_store.Hit(float(scores[i]), docs[i], metas[i], ids[i])
                for i in order
                if scores[i] > 0 and (keep is None or keep(metas[i]))]
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

    def _collect(self, sources, query: str, n: int, where=None, keep=None) -> Rows:
        """Dense and lexical hits over `sources` — [(collection name, legacy?,
        dense rows per `n`)] — merged into one Rows. One embedding space, so
        dense distances merge honestly; BM25 scores do not merge across
        collections, which is why the fusion normalises them per turn. Never
        raises: a broken collection reports through `error` and the others
        still answer."""
        dense, lexical, total, error = [], [], 0, None
        for name, legacy, per in sources:
            rows, count, err = self._query_archive(name, query, per * n, where)
            if count is None:
                error = error or err
                continue
            total += count
            if not count:
                continue
            try:
                lex = self._lexical_rows(name, query, n, keep)
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


class ConversationMemory(HybridStore):
    _row = staticmethod(_exchange_hit)

    @staticmethod
    def _indexable(meta) -> bool:
        # Parents only: a companion repeats its parent's text and would count
        # the exchange twice.
        return not meta.get("part")

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

    def exchanges_of(self, messages, owner) -> list:
        """The exchanges in `messages` as (id, document, metadata) — what the
        index writes and what the fact extractor reads, from ONE derivation
        so they can never disagree about what an exchange is. Tool traffic is
        skipped; tool NAMES go into metadata, not the text — identifiers
        would shift a short document's embedding. A turn with no assistant
        text (abandoned) is not an exchange."""
        out = []
        for user_msg, group in self._exchanges(messages):
            lines = [t for m in group if (t := self._message_text(m))]
            if len(lines) < 2:
                continue
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
            out.append((exchange_id(owner, user_msg), "\n".join(lines), meta))
        return out

    def index_exchanges(self, messages, owner, *, skip_existing=True) -> int:
        """Embed each exchange in `messages` into `owner`'s collection. The ONE
        write path: the live turn (skip_existing=False, so a self-note that
        lands after the reply overwrites its exchange), the boot backfill of
        the saved threads, and the log seeder. Returns how many exchanges
        were written (two records each, _records_for)."""
        batch = [(xid, *self._records_for(xid, doc, meta))
                 for xid, doc, meta in self.exchanges_of(messages, owner)]
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
        self._touch(name)
        return len(batch)

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
            self._touch(name)
        return len(add_ids), len(stamp_ids)

    def unextracted(self, owner) -> list:
        """(id, document, metadata) of every exchange the fact extractor has
        not read yet, oldest first — the order facts must be learned in, since
        "add item four" only reads as an update to a list the extractor has
        already seen. The marker is `extracted` on the parent record
        (mark_extracted), so a boot picks up exactly what the worker had not
        reached when the process last exited."""
        col = self._col_for(cfg.agent_memory_collection(owner))
        got = col.get(include=["documents", "metadatas"])
        rows = [(i, d, m or {})
                for i, d, m in zip(got.get("ids") or [], got.get("documents") or [],
                                   got.get("metadatas") or [])
                if (m or {}).get("kind") == "exchange" and not (m or {}).get("part")
                and not (m or {}).get("extracted")]
        rows.sort(key=lambda r: r[2].get("epoch") or 0)
        return rows

    def mark_extracted(self, owner, xid):
        """Metadata-only: chromadb merges an update into the stored metadata
        (1.5.9, probed 2026-09-22), so the one key is all that is written."""
        self._col_for(cfg.agent_memory_collection(owner)).update(
            ids=[xid], metadatas=[{"extracted": True}])

    # --- retrieval ------------------------------------------------------------
    def query_rows(self, query: str, n: int = None, caller=None, *,
                   window=None) -> Rows:
        """Dense and lexical hits from every archive `caller` may read: its
        own conversations_<key> (plus grants) and the legacy shared
        collection, whose rows are tagged legacy so the reader knows they
        predate per-agent memory. `n` counts exchanges per collection and
        retriever; a persona's collection holds two key records per exchange,
        so it is read twice as deep to keep `n` meaning exchanges once they
        collapse. `window` — (lo, hi) epoch seconds — restricts both lists to
        exactly-timed exchanges inside it."""
        n = n or cfg.MEMORY_SEARCH_RESULTS
        where = time_where(window) if window is not None else None
        keep = (lambda m: in_window(m, window)) if window is not None else None
        sources = [(cfg.agent_memory_collection(k), False, 2)
                   for k in agents.readable_owners(caller)]
        sources.append((cfg.MEMORY_COLLECTION, True, 1))
        return self._collect(sources, query, n, where, keep)

    def rank(self, query: str, *, caller, window=None, exclude_ids=(), now=None):
        """The ranking behind the recall tool and the eval harness:
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
                              exclude_ids=exclude_ids, gated=window is None,
                              by_recency=window is None)
        if window is None:
            ranked = [c for c in ranked if c.gate]
        return ranked, rows

    def search(self, query: str, n: int = None, caller=None, *, window=None,
               exclude_ids=()) -> str:
        """The recall tool's text for a topic and/or a time window. A windowed
        result reads oldest-first — a recalled stretch of conversation is
        read in order — and says when it was cut, so the model narrows the
        window or adds a topic rather than taking the first `n` for the whole
        story. With a window and no topic the oldest `n` are shown."""
        n = n or cfg.MEMORY_SEARCH_RESULTS
        ranked, rows = self.rank(query, caller=caller, window=window,
                                 exclude_ids=exclude_ids)
        span = ""
        if window is not None:
            if not query.strip():
                ranked.sort(key=lambda c: c.meta.get("epoch") or 0)
            span = f" between {_clock(window[0])} and {_clock(window[1])}"
        shown = ranked[:n]
        if shown:
            if window is not None:
                shown.sort(key=lambda c: c.meta.get("epoch") or 0)
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

    def fetch(self, ids, caller) -> str:
        """The recall tool's read-by-id: the named exchanges, oldest first,
        from the collections `caller` may read — how the model follows a
        fact's `from` ids back to the words behind it."""
        want = [i for i in ids if isinstance(i, str)]
        found = {}
        for k in agents.readable_owners(caller):
            try:
                got = self._col_for(cfg.agent_memory_collection(k)).get(
                    ids=want, include=["documents", "metadatas"])
            except Exception as e:  # noqa: BLE001
                log.warning("could not fetch exchanges from %s's memory: %s", k, e)
                continue
            for i, d, m in zip(got.get("ids") or [], got.get("documents") or [],
                               got.get("metadatas") or []):
                found[i] = (d, m or {})
        if not found:
            return "None of those exchange ids are in your memory."
        rows = sorted(found.values(), key=lambda dm: dm[1].get("epoch") or 0)
        text = "Exchanges, oldest first:\n" + "\n\n".join(
            f"[{context.when(m)}] {' '.join(d.split())[:1200]}" for d, m in rows)
        missing = len([i for i in want if i not in found])
        return text + (f"\n\n({missing} id(s) not found.)" if missing else "")


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
