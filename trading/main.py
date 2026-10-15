"""The broker registry: which variant module answers for which broker.

Two brokers, two variant files — tastytrade.py (Bonny) and schwab.py (Tom)
— each exposing the same module-level surface (listed in tastytrade.py's
docstring). Everything broker-neutral asks here for the module by key and
never imports a variant directly; a persona's `brokers` tuple in
brain/agents.py is what maps a voice to a key.

Imports stay lazy so `trading` keeps its side-effect-free package import:
the dashboard must not pay for `requests` until a Trading page is opened.
"""

import importlib

# key -> (import path, label shown to the user). Order = dashboard order.
BROKERS = {
    "schwab": ("trading.schwab", "Charles Schwab"),
    "tastytrade": ("trading.tastytrade", "tastytrade"),
}


def keys():
    return list(BROKERS)


def label(key):
    return BROKERS[key][1]


def module(key):
    """The variant module for `key`; raises KeyError for an unknown broker
    (the tools and routes turn that into a sentence)."""
    path, _ = BROKERS[key]
    return importlib.import_module(path)


def normalize(key):
    """Spoken/typed broker names -> registry key, or None."""
    k = (key or "").strip().lower().replace(" ", "")
    aliases = {"schwab": "schwab", "charlesschwab": "schwab",
               "thinkorswim": "schwab", "tos": "schwab",
               "tastytrade": "tastytrade", "tasty": "tastytrade",
               "tastyworks": "tastytrade"}
    return aliases.get(k)
