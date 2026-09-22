"""The fact store — what is TRUE NOW, derived from the exchanges.

Retrieval over raw exchanges answers "what was said"; it answers "what is
the current state of X" badly by construction, because the state is spread
over every exchange that changed it and top-k under a budget has to find all
of them. On 2026-09-18 six to-do items arrived in six exchanges and the
persona could recall one. A fact is one claim about one entity, in the
extractor's words, kept CURRENT: adding an item to a list is an UPDATE of
the list's one fact, not a seventh loose record (the Mem0 pattern — add /
update / supersede against the facts already known).

Facts are a derived index, never a second truth. Every fact names the
exchanges it came from (`source_ids`), so the model can read the words
behind it (the recall tool), and the whole store can be rebuilt from the
exchange index by re-running the extractor (scripts/rebuild_facts.py) — an
extractor bug is fixed by fixing the prompt, never by editing facts. Nothing
is deleted: a fact that stops being true is marked superseded and drops out
of retrieval; "what was on my list before I removed X" is answered from the
exchanges (recall, by time), which the superseded record still points at.

Records live in facts_<key>, one collection per persona, in the same
embedding space as the exchanges, read through memory.HybridStore (dense +
BM25). Document = "entity: value"; metadata = entity, the ts/epoch/approx
of the exchange that established it, source_ids, superseded_by ("" while
current) and superseded_epoch.
"""

import hashlib
import logging
import re

from stores import chroma_store

import config as cfg
from brain import agents
from brain.llm.plain import ask
from brain.memory import HybridStore, Rows, and_where, in_window, time_where
from lib.llm_json import first_object

log = logging.getLogger("facts")

_SLUG = re.compile(r"[^a-z0-9:_-]+")


def slug(key) -> str:
    """An entity key as the extractor is told to write them: lowercase,
    kind:name, no spaces — so "Todo: 2026-09-18" and "todo:2026-09-18" are
    one key."""
    return _SLUG.sub("-", str(key or "").strip().lower()).strip("-")


def fact_id(owner, xid, entity, value) -> str:
    """Deterministic, so re-learning an exchange (a rebuild, a boot that
    repeats one the worker had not marked) overwrites instead of duplicating."""
    key = f"{xid}|{entity}|{value}"
    return f"f_{owner}_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:16]}"


# The extractor's call shape. Adaptive thinking shares max_tokens with the
# answer, and the hard cases — a list update, where the value carries every
# item — are exactly where the model thinks longest: at 1500 and again at
# 4000, updates of the 2026-09-18 list were cut off, read as no answer, and
# items went missing (harness, 2026-09-22, 4 of 303 exchanges). Billed as
# used, so the headroom costs nothing on the ordinary short reply.
EXTRACT_MAX_TOKENS = 16000
EXTRACT_EFFORT = "medium"


def extractor(client):
    """The extractor's model call, bound to `client` (brain/llm/plain.ask)."""
    return lambda system, prompt: ask(client, cfg.FACTS_MODEL, system, prompt,
                                      max_tokens=EXTRACT_MAX_TOKENS,
                                      effort=EXTRACT_EFFORT)


class FactStore(HybridStore):
    def __init__(self):
        super().__init__()
        self._entities = {}  # owner -> current entity keys, most recent first

    @staticmethod
    def _current_where():
        """Chroma where= for facts still in force. The eval harness overrides
        this to read the store as of a replayed moment."""
        return {"superseded_by": ""}

    @staticmethod
    def _is_current(meta) -> bool:
        return meta.get("superseded_by") == ""

    # --- reading --------------------------------------------------------------
    def query_rows(self, query: str, n: int = None, owner=None, *,
                   window=None) -> Rows:
        """Current facts of `owner` (and grants) nearest `query`, dense and
        lexical, optionally inside a time window — the same shape the
        exchange index returns, so brain/context.py fuses both alike."""
        n = n or cfg.CONTEXT_CANDIDATES
        where = and_where(self._current_where(),
                          time_where(window) if window is not None else None)

        def keep(meta):
            return self._is_current(meta) and (window is None or in_window(meta, window))
        sources = [(cfg.agent_facts_collection(k), False, 1)
                   for k in agents.readable_owners(owner)]
        return self._collect(sources, query, n, where, keep)

    def current(self, owner, entities) -> list:
        """The facts in force about `entities`, oldest first, as Hit rows —
        read by key, not by rank: the path behind "what's on my list". [] on
        any failure (the ranked path still runs)."""
        keys = [slug(e) for e in entities if slug(e)]
        if not keys:
            return []
        by_entity = {"entity": keys[0]} if len(keys) == 1 else {"entity": {"$in": keys}}
        out = []
        for k in agents.readable_owners(owner):
            name = cfg.agent_facts_collection(k)
            try:
                col = self._col_for(name)
                if not col.count():
                    continue
                got = col.get(where=and_where(by_entity, self._current_where()),
                              include=["documents", "metadatas"])
            except Exception as e:  # noqa: BLE001
                log.warning("could not read facts from %s: %s", name, e)
                continue
            ids = got.get("ids") or []
            vecs = self._vectors(name, ids)
            out += [chroma_store.Hit(0.0, d, m or {}, i, vecs.get(i))
                    for i, d, m in zip(ids, got.get("documents") or [],
                                       got.get("metadatas") or [])]
        out.sort(key=lambda h: h.meta.get("epoch") or 0)
        return out

    def entities(self, owner) -> dict:
        """`owner`'s current entity keys → the stamp of each one's latest
        fact, most recently established first, capped at
        FACTS_KNOWN_ENTITIES — what the query rewrite chooses from (dated, so
        "my list" resolves to the latest list, not the first one) and the
        extractor reuses. Cached per owner; apply() invalidates. {} on
        failure."""
        if owner not in self._entities:
            seen = {}  # entity -> (epoch, ts)
            try:
                col = self._col_for(cfg.agent_facts_collection(owner))
                got = (col.get(where=self._current_where(), include=["metadatas"])
                       if col.count() else {})
                for m in got.get("metadatas") or []:
                    m = m or {}
                    e, epoch = m.get("entity"), m.get("epoch") or 0
                    if e and epoch >= seen.get(e, (-1, ""))[0]:
                        seen[e] = (epoch, m.get("ts") or "")
            except Exception as e:  # noqa: BLE001
                log.warning("could not list %s's entities: %s", owner, e)
                return {}
            self._entities[owner] = dict(sorted(seen.items(), key=lambda kv: -kv[1][0]))
        items = list(self._entities[owner].items())[:cfg.FACTS_KNOWN_ENTITIES]
        return {k: ts for k, (_, ts) in items}

    def recent(self, owner, n) -> list:
        """The `n` most recently established current facts, as Hit rows: the
        state being built right now. The extractor must see the list it is
        adding to even when the new item's words are nothing like the list's
        — item six, "meditate with an eyeshade", against a list about
        dashboards and email, was added as a second fact because the search
        on the exchange text never surfaced the first (2026-09-18). [] on
        failure."""
        name = cfg.agent_facts_collection(owner)
        try:
            col = self._col_for(name)
            if not col.count():
                return []
            got = col.get(where=self._current_where(), include=["documents", "metadatas"])
        except Exception as e:  # noqa: BLE001
            log.warning("could not read %s's recent facts: %s", owner, e)
            return []
        rows = sorted(zip(got.get("ids") or [], got.get("documents") or [],
                          got.get("metadatas") or []),
                      key=lambda r: -((r[2] or {}).get("epoch") or 0))[:n]
        return [chroma_store.Hit(0.0, d, m or {}, i) for i, d, m in rows]

    # --- writing --------------------------------------------------------------
    def apply(self, owner, ops, *, xid, ts, epoch=None, approx=False) -> tuple:
        """Write the extractor's operations for exchange `xid`. add: a new
        record. update: a new record for the entity (provenance accumulates:
        the old fact's sources plus this exchange) and the old one marked
        superseded by it. supersede: the old one marked superseded by the
        exchange. Marks are metadata-only updates (chromadb merges them).
        Returns (added, updated, superseded)."""
        name = cfg.agent_facts_collection(owner)
        col = self._col_for(name)
        old_ids = [op["id"] for op in ops if op["op"] in ("update", "supersede")]
        old = {}
        if old_ids:
            got = col.get(ids=old_ids, include=["metadatas"])
            old = dict(zip(got.get("ids") or [], got.get("metadatas") or []))
        stamp = {"superseded_epoch": epoch} if epoch is not None else {}
        ids, docs, metas, marks = [], [], [], []
        added = updated = superseded = 0
        for op in ops:
            if op["op"] == "supersede":
                if op["id"] in old:
                    marks.append((op["id"], xid))
                    superseded += 1
                continue
            if op["op"] == "update":
                prev = old.get(op["id"])
                if prev is None:
                    continue
                entity = prev.get("entity") or slug(op.get("entity"))
                sources = f"{prev.get('source_ids', '')},{xid}".strip(",")
            else:
                entity, sources = slug(op.get("entity")), xid
            if not entity:
                continue
            fid = fact_id(owner, xid, entity, op["value"])
            ids.append(fid)
            docs.append(f"{entity}: {op['value']}")
            metas.append({"agent": owner, "kind": "fact", "entity": entity,
                          "ts": ts, "approx": bool(approx),
                          **({"epoch": epoch} if epoch is not None else {}),
                          "source_ids": sources, "superseded_by": ""})
            if op["op"] == "update":
                marks.append((op["id"], fid))
                updated += 1
            else:
                added += 1
        if ids:
            col.upsert(ids=ids, documents=docs, metadatas=metas)
        if marks:
            col.update(ids=[m[0] for m in marks],
                       metadatas=[{"superseded_by": m[1], **stamp} for m in marks])
        if ids or marks:
            self._touch(name)
            self._entities.pop(owner, None)
        return added, updated, superseded


# --- the extractor ------------------------------------------------------------
SYSTEM = (
    "You maintain a voice assistant's long-term memory of one user. You read "
    "one finished exchange and decide what durable facts it establishes: "
    "things the user would expect the assistant to still know days later — "
    "preferences, decisions, plans and commitments, names and relationships, "
    "the contents and state of lists and ongoing matters, corrections to "
    "earlier facts. Not: greetings, questions that add nothing new, the "
    "assistant's own explanations, opinions, or diagnoses of its own memory "
    "and behaviour, what the assistant said about its tools or abilities, "
    "settings that change by the hour (which model is answering), transient "
    "states (a mood, a passing calculation), or the fact that a tool was "
    "used. The assistant's own answers about what it remembers, does not "
    "remember, or cannot find establish nothing.\n\n"
    "You are shown the facts already known that may be related, each with an "
    "id and an entity key. Reply with JSON only, on one line, no line breaks "
    "inside strings: {\"ops\": [...]}, each op one of\n"
    '  {"op": "add", "entity": "<key>", "value": "<fact>"}   a new fact\n'
    '  {"op": "update", "id": "<id>", "value": "<fact>"}     a known fact whose '
    "value changed; the value is the COMPLETE current fact\n"
    '  {"op": "supersede", "id": "<id>"}                    a known fact no '
    "longer true, with no replacement\n"
    "An empty list means the exchange establishes nothing durable — the "
    "common case.\n\n"
    "Rules. Entity keys are short stable slugs, kind:name — todo:2026-09-18, "
    "person:tom, project:voice-agent, preference:coffee; reuse a listed key "
    "whenever it is the same thing and never invent a variant of one. A thing "
    "that belongs to a day carries its date in the key (todo:2026-09-18), "
    "never 'today' or 'this week', which stop being true; a new day's list "
    "is a new entity, not an update of yesterday's. A list "
    "or collection is ONE fact whose value holds its complete current "
    "contents, so adding or removing an item is an update carrying every "
    "remaining item. If two known facts describe the same thing, update one "
    "with the whole truth and supersede the other. A value is one to three "
    "plain sentences, complete on its "
    "own, no pronouns, in the user's own terms; dates in it are absolute (the "
    "exchange's time is given). Write only what the exchange actually "
    "establishes; when the user and the assistant disagree, the user's word "
    "stands; never guess. The assistant has a name of its own (given with the "
    "exchange); the user is the person it serves and is never that name — "
    "write 'the user' unless the user states their own name. Never record "
    "which model, voice or persona is answering."
)


def extract(ask, doc, ts, candidates, known, *, assistant="the assistant"):
    """The extractor's operations for one exchange, validated: an update or
    supersede must name a candidate id, an add needs an entity and a value.
    `assistant` is the persona's name, so the extractor never takes it for
    the user's (a round of extraction once named the user "Alice"). None
    when the model could not be asked or gave no ops list at all — that is
    "not done", and the exchange stays unmarked for a later pass; [] is
    "nothing durable", which is done."""
    shown = "\n".join(
        f"[{h.id}] {h.meta.get('entity', '?')} — {h.doc.split(': ', 1)[-1]}"
        f" (as of {h.meta.get('ts') or 'unknown'})"
        for h in candidates) or "(none)"
    prompt = (f"Exchange at {ts or 'an unknown time'} (the assistant is {assistant}):"
              f"\n{doc}\n\n"
              f"Known facts that may be related:\n{shown}\n\n"
              f"Other entity keys in use: {', '.join(known) or '(none)'}")
    try:
        text = ask(SYSTEM, prompt)
    except Exception as e:  # noqa: BLE001
        log.warning("fact extraction failed: %s", e)
        return None
    ops = first_object(text).get("ops")
    if not isinstance(ops, list):
        log.warning("fact extraction gave no ops list: %r ... %r",
                    str(text)[:160], str(text)[-120:])
        return None
    ids = {h.id for h in candidates}
    clean = []
    for op in ops:
        if not isinstance(op, dict):
            continue
        kind, value = op.get("op"), " ".join(str(op.get("value") or "").split())
        if kind == "add" and value and slug(op.get("entity")):
            clean.append({"op": "add", "entity": slug(op["entity"]), "value": value})
        elif kind == "update" and value and op.get("id") in ids:
            clean.append({"op": "update", "id": op["id"], "value": value})
        elif kind == "supersede" and op.get("id") in ids:
            clean.append({"op": "supersede", "id": op["id"]})
    return clean


def learn(ask, memory, facts, owner, xid, doc, meta):
    """Read one exchange into the fact store: the related current facts are
    fetched (hybrid search on the exchange text, plus the most recently
    established facts and every current fact of the entities those name, so
    an addition meets the list it belongs to), the extractor decides, the
    store applies, and the
    exchange is marked read. Returns the apply counts, or None when the
    extractor could not run (the mark is withheld, so a later pass retries)."""
    rows = facts.query_rows(doc, cfg.FACTS_CANDIDATES, owner)
    candidates = {}
    for h in rows.dense + rows.lexical + facts.recent(owner, cfg.FACTS_RECENT):
        candidates.setdefault(h.id, h)
    # Every current fact of an entity in play, so one list cannot fork into
    # two: an update has to see all of it.
    for h in facts.current(owner, {h.meta.get("entity") for h in candidates.values()
                                   if h.meta.get("entity")}):
        candidates.setdefault(h.id, h)
    ops = extract(ask, doc, meta.get("ts"), list(candidates.values()),
                  facts.entities(owner),
                  assistant=agents.AGENTS.get(owner, {}).get("name", owner))
    if ops is None:
        return None
    counts = facts.apply(owner, ops, xid=xid, ts=meta.get("ts") or "",
                         epoch=meta.get("epoch"), approx=meta.get("approx", False))
    memory.mark_extracted(owner, xid)
    return counts


def catch_up(ask, memory, facts, owner, *, progress=None) -> tuple:
    """Learn every exchange of `owner` the extractor has not read, oldest
    first — the boot pass, the rebuild script and the eval harness all run
    this. Returns (read, failed); `progress(done, total)` every 25 and at
    the end."""
    pending = memory.unextracted(owner)
    read = failed = 0
    for i, (xid, doc, meta) in enumerate(pending, 1):
        if learn(ask, memory, facts, owner, xid, doc, meta) is None:
            failed += 1
        else:
            read += 1
        if progress and (i % 25 == 0 or i == len(pending)):
            progress(i, len(pending))
    return read, failed
