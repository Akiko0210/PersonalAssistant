"""Shared fakes for the Chroma-backed stores (see llm_fixtures.py and
agent_fixtures.py for the other layers). One FakeCol for every test that
touches a collection — it used to be a copy per test file, and the copies
disagreed about what query() returned.
"""

from brain.facts import FactStore
from brain.memory import ConversationMemory
from stores.knowledge import KnowledgeStore


def _match(meta, where):
    """Enough of Chroma's where= for the stores' own filters: equality, $in,
    the comparison operators, $and / $or — and a missing key never matches,
    which is the rule the `approx` and `superseded_by` stamps lean on."""
    if not where:
        return True
    if "$and" in where:
        return all(_match(meta, w) for w in where["$and"])
    if "$or" in where:
        return any(_match(meta, w) for w in where["$or"])
    ((key, cond),) = where.items()
    if key not in meta:
        return False
    val = meta[key]
    if not isinstance(cond, dict):
        return val == cond
    ops = {"$in": lambda a, b: a in b, "$nin": lambda a, b: a not in b,
           "$gt": lambda a, b: a > b, "$gte": lambda a, b: a >= b,
           "$lt": lambda a, b: a < b, "$lte": lambda a, b: a <= b,
           "$eq": lambda a, b: a == b, "$ne": lambda a, b: a != b}
    return all(ops[op](val, b) for op, b in cond.items())


class FakeCol:
    """A Chroma collection double: records upserts, serves canned hits from
    query(), and answers get() from what was upserted — the exchange index's
    skip-existing check, its BM25 corpus and the fact store's by-key reads
    all read a collection back that way. Keeps the tests loading neither
    Chroma nor the embedding model."""

    def __init__(self, hits=()):
        self.upserts = []      # (ids, documents, metadatas)
        self.updates = []      # (ids, metadatas) — metadata-only writes
        self.hits = list(hits)  # (doc, meta, distance[, id])
        self.queries = 0
        self.wheres = []       # the where= of every query, None when absent
        self.rows = {}         # id -> (doc, meta), from upserts

    def upsert(self, ids, documents, metadatas):
        self.upserts.append((list(ids), list(documents), list(metadatas)))
        for id_, doc, meta in zip(ids, documents, metadatas):
            self.rows[id_] = (doc, meta)

    def update(self, ids, metadatas):
        # Merges, as chromadb 1.5.9 does (probed 2026-09-22): the stores
        # write single-key marks and rely on the rest surviving.
        self.updates.append((list(ids), list(metadatas)))
        for id_, meta in zip(ids, metadatas):
            if id_ in self.rows:
                doc, old = self.rows[id_]
                self.rows[id_] = (doc, {**old, **meta})

    def count(self):
        return len(self.hits) or len(self.rows)

    def query(self, query_texts, n_results, where=None, **kwargs):
        self.queries += 1
        self.wheres.append(where)
        rows = self.hits[:n_results]
        return {"ids": [[r[3] if len(r) > 3 else f"hit{i}"
                         for i, r in enumerate(rows)]],
                "documents": [[r[0] for r in rows]],
                "metadatas": [[r[1] for r in rows]],
                "distances": [[r[2] for r in rows]]}

    def get(self, ids=None, where=None, include=None, **kwargs):
        keys = [i for i in (ids if ids is not None else list(self.rows))
                if i in self.rows and _match(self.rows[i][1], where)]
        return {"ids": keys,
                "documents": [self.rows[i][0] for i in keys],
                "metadatas": [self.rows[i][1] for i in keys]}


def fake_store(cols):
    """A KnowledgeStore whose _col_for serves from `cols` (name -> FakeCol),
    creating on demand so tests can also assert which names were touched."""
    store = KnowledgeStore.__new__(KnowledgeStore)
    store._whisper = None
    store._cols = {}
    store._col_for = lambda name: cols.setdefault(name, FakeCol())
    return store


def memory_over(cols):
    """A ConversationMemory over `cols` (name -> FakeCol), created on demand."""
    mem = ConversationMemory.__new__(ConversationMemory)
    mem._cols, mem._lexical = {}, {}
    mem._col_for = lambda name: cols.setdefault(name, FakeCol())
    return mem


def facts_over(cols):
    """A FactStore over the same kind of `cols`."""
    fs = FactStore.__new__(FactStore)
    fs._cols, fs._lexical, fs._entities = {}, {}, {}
    fs._col_for = lambda name: cols.setdefault(name, FakeCol())
    return fs
