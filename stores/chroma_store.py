"""The one Chroma bootstrap behind the three stores (notes, memory,
knowledge). Each lazily loads the same embedding model against the same
persistent directory and differs only in collection name — this was the same
ten lines three times. Imports chromadb at module level, so only the stores
(never the dashboard) may import this.
"""

from collections import namedtuple

import chromadb
from chromadb.utils import embedding_functions

import config as cfg

# One retrieval row. `score` is the retriever's native number — Chroma's
# distance for a dense query, BM25 for a lexical one — so the caller fusing
# the two lists (brain/context.py) knows which scale it holds by which list
# the row came from. Ids ride along because the per-turn Background excludes
# the exchanges the model already sees verbatim, and needs them to do so.
Hit = namedtuple("Hit", "score doc meta id")


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
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=cfg.EMBED_MODEL
    )
    client = chromadb.PersistentClient(path=str(cfg.CHROMA_DIR))
    return client.get_or_create_collection(name=name, embedding_function=ef)
