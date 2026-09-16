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

    def index_exchanges(self, messages, owner, *, skip_existing=True) -> int:
        """Embed each exchange in `messages` into `owner`'s collection. The ONE
        write path: the live turn (skip_existing=False, so a self-note that
        lands after the reply overwrites its exchange), the boot backfill of
        the saved threads, the staging-file migration, and the log seeder.
        Tool traffic is skipped; tool NAMES go into metadata, not the text —
        identifiers would shift a short document's embedding. Returns how
        many records were written."""
        ids, docs, metas = [], [], []
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
            ids.append(exchange_id(owner, user_msg))
            docs.append("\n".join(lines))
            metas.append(meta)
        if not ids:
            return 0
        name = cfg.agent_memory_collection(owner)
        col = self._col_for(name)
        if skip_existing:
            have = set(col.get(ids=ids, include=[]).get("ids") or [])
            keep = [i for i, x in enumerate(ids) if x not in have]
            ids = [ids[i] for i in keep]
            docs = [docs[i] for i in keep]
            metas = [metas[i] for i in keep]
            if not ids:
                return 0
        col.upsert(ids=ids, documents=docs, metadatas=metas)
        self._lexical.pop(name, None)  # rebuilt from the store on the next query
        return len(ids)

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

    # --- retrieval ------------------------------------------------------------
    def _query_archive(self, name: str, query: str, n: int):
        """One collection's dense hits as chroma_store.Hit rows, plus its record
        count and error text. Never raises.

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
                res = col.query(query_texts=[query], n_results=min(n, count))
                return chroma_store.hits(res), count, None
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
        """BM25 over every record in collection `name`, built lazily from the
        store and dropped whenever index_exchanges writes to it (a few
        thousand short records rebuild in tens of milliseconds)."""
        cached = self._lexical.get(name)
        if cached is None:
            got = self._col_for(name).get(include=["documents", "metadatas"])
            ids = got.get("ids") or []
            docs = got.get("documents") or []
            metas = [m or {} for m in (got.get("metadatas") or [])]
            bm25 = _bm25([tokens(d) for d in docs]) if docs else None
            cached = self._lexical[name] = (bm25, ids, docs, metas)
        return cached

    def _lexical_rows(self, name: str, query: str, n: int) -> list:
        bm25, ids, docs, metas = self._lexical_index(name)
        words = tokens(query)
        if bm25 is None or not words:
            return []
        scores = bm25.get_scores(words)
        order = sorted(range(len(ids)), key=lambda i: -scores[i])[:n]
        return [chroma_store.Hit(float(scores[i]), docs[i], metas[i], ids[i])
                for i in order if scores[i] > 0]

    def query_rows(self, query: str, n: int = None, caller=None) -> Rows:
        """Dense and lexical hits from every archive `caller` may read: its
        own conversations_<key> (plus grants) and the legacy shared
        collection, whose rows are tagged legacy so the reader knows they
        predate per-agent memory. One embedding space, so dense distances
        merge honestly; BM25 scores do not merge across collections, which
        is why the fusion normalises them per turn. Never raises: a broken
        collection reports through `error` and the others still answer."""
        n = n or cfg.MEMORY_SEARCH_RESULTS
        sources = [(cfg.agent_memory_collection(k), False)
                   for k in agents.readable_owners(caller)]
        sources.append((cfg.MEMORY_COLLECTION, True))
        dense, lexical, total, error = [], [], 0, None
        for name, legacy in sources:
            rows, count, err = self._query_archive(name, query, n)
            if count is None:
                error = error or err
                continue
            total += count
            if not count:
                continue
            try:
                lex = self._lexical_rows(name, query, n)
            except Exception as e:  # the dense hits are still worth returning
                log.warning("lexical search of %s failed: %s", name, e)
                lex = []
            if legacy:
                rows = [h._replace(meta={**h.meta, "legacy": True}) for h in rows]
                lex = [h._replace(meta={**h.meta, "legacy": True}) for h in lex]
            dense.extend(rows)
            lexical.extend(lex)
        dense.sort(key=lambda h: h.score)
        lexical.sort(key=lambda h: -h.score)
        return Rows(dense, lexical, None if error and not total else total, error)

    def search(self, query: str, n: int = None, caller=None) -> str:
        """The search_past_conversations tool. Ranks with the same fusion the
        per-turn Background uses (context.fuse), so the tool can never
        disagree with what the model was already shown — it only goes deeper
        than the budget allowed."""
        n = n or cfg.MEMORY_SEARCH_RESULTS
        rows = self.query_rows(query, n, caller)
        ranked = [c for c in context.fuse(rows.dense, rows.lexical) if c.gate][:n]
        if ranked:
            out = []
            for c in ranked:
                tag = (" (from the shared archive, before per-agent memory)"
                       if c.meta.get("legacy") else "")
                out.append(f"[{context.when(c.meta)}{tag}] "
                           f"{' '.join(c.doc.split())[:800]}")
            text = "From past conversations:\n" + "\n\n".join(out)
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
        return "Nothing in past conversations matches that."
