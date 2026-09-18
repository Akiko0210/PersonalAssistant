"""The one Chroma bootstrap behind the three stores (notes, memory,
knowledge). Each lazily loads the same embedding model against the same
persistent directory and differs only in collection name — this was the same
ten lines three times. Imports chromadb at module level, so only the stores
(never the dashboard) may import this.
"""

import logging
from collections import namedtuple

import chromadb
from chromadb.utils import embedding_functions

import config as cfg

log = logging.getLogger("chroma")

# One retrieval row. `score` is the retriever's native number — Chroma's
# distance for a dense query, BM25 for a lexical one — so the caller fusing
# the two lists (brain/context.py) knows which scale it holds by which list
# the row came from. Ids ride along because the per-turn Background excludes
# the exchanges the model already sees verbatim, and needs them to do so.
# `vec` is the record's stored embedding when the caller fetched it (the
# memory store does, so the Background's fill can measure redundancy between
# candidates); None otherwise.
Hit = namedtuple("Hit", "score doc meta id vec", defaults=(None,))

# The distance space the collections query in, which decides how a distance
# becomes a similarity (brain/context.similarity). chromadb >= 1.0 lets the
# embedding function choose the space of a collection created without one,
# and the sentence-transformer function chooses cosine — so these collections
# are cosine, not the squared-L2 similarity() assumed until 2026-09-18, when
# every sim read 0.5 + cos/2 and the relevance gate never fired. Read from the
# first collection opened; a later one in a different space is logged, since
# it would silently break the gate again.
SPACE = "cosine"
_spaces = {}  # collection name -> space seen


def space_of(col) -> str:
    """The space a collection was created with: its stored configuration
    (chromadb >= 1.0), else the legacy hnsw:space metadata, else Chroma's
    old default of l2."""
    try:
        conf = col.configuration_json or {}
    except Exception:  # noqa: BLE001 - older chromadb keeps no configuration
        conf = {}
    for section in ("hnsw", "spann"):
        space = (conf.get(section) or {}).get("space")
        if space:
            return space
    return (getattr(col, "metadata", None) or {}).get("hnsw:space") or "l2"


def hits(res) -> list:
    """The first row-set of one collection.query() result as Hit rows. Chroma
    returns parallel lists per query text; every store used to unzip them by
    hand, and one copy forgot the `meta or {}` guard."""
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    ids = res.get("ids", [[]])[0] or [None] * len(docs)
    return [Hit(dist, doc, meta or {}, id_)
            for dist, doc, meta, id_ in zip(dists, docs, metas, ids)]


def collection(name):
    """A ready collection in the shared persistent store, embedding with
    cfg.EMBED_MODEL. First call in a process pays the model load; the caller
    owns announcing that (each store logs its own "first use" line)."""
    global SPACE
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=cfg.EMBED_MODEL
    )
    client = chromadb.PersistentClient(path=str(cfg.CHROMA_DIR))
    col = client.get_or_create_collection(name=name, embedding_function=ef)
    space = space_of(col)
    if not _spaces:
        SPACE = space
        log.info("chroma distance space: %s", space)
    elif space != SPACE:
        log.warning("collection %s is in %s space; similarity() assumes %s",
                    name, space, SPACE)
    _spaces[name] = space
    return col
