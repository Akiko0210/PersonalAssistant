"""The one Chroma bootstrap behind the three stores (notes, memory,
knowledge). Each lazily loads the same embedding model against the same
persistent directory and differs only in collection name — this was the same
ten lines three times. Imports chromadb at module level, so only the stores
(never the dashboard) may import this.
"""

import chromadb
from chromadb.utils import embedding_functions

import config as cfg


def collection(name):
    """A ready collection in the shared persistent store, embedding with
    cfg.EMBED_MODEL. First call in a process pays the model load; the caller
    owns announcing that (each store logs its own "first use" line)."""
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=cfg.EMBED_MODEL
    )
    client = chromadb.PersistentClient(path=str(cfg.CHROMA_DIR))
    return client.get_or_create_collection(name=name, embedding_function=ef)
