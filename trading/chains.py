"""The option-chain cache, shared by both brokers.

Each broker variant fetches and normalizes its own API shape into ChainData
(tastytrade.fetch_chain, schwab.fetch_chain); this module only remembers
the result per (broker, underlying) for CHAIN_CACHE_TTL_S — strikes and
expirations don't change intraday, and the chain is the one big call.
Strategy building and leg validation work off ChainData only — they never
see raw API shapes.
"""

import threading
import time

from trading import config as tcfg
from trading import symbols as tsym

_cache = {}
_cache_lock = threading.Lock()


def fetch_chain(broker, client, underlying, force=False):
    """ChainData for an underlying ('SPX', 'AAPL', '/ES') on `broker` (a
    variant module), cached."""
    underlying = tsym.normalize_underlying(underlying)
    key = (broker.KEY, underlying)
    now = time.time()
    with _cache_lock:
        hit = _cache.get(key)
        if hit and not force and now - hit.fetched_at < tcfg.CHAIN_CACHE_TTL_S:
            return hit
    chain = broker.fetch_chain(client, underlying)
    if not chain.expirations:
        raise ValueError(f"No option chain found for {underlying}.")
    with _cache_lock:
        _cache[key] = chain
    return chain


def extend_chain(broker, client, chain, start, end, every_strike=False):
    """Fetch the expirations start..end (every strike of them with
    every_strike) and merge them into `chain` — the cached object, so the
    stretch stays for the cache's life. Schwab's normal load is one week at
    80 strikes (#32); this is how a later date or a far strike is reached."""
    extra = broker.fetch_chain(client, chain.underlying, start=start, end=end,
                               every_strike=every_strike)
    with _cache_lock:
        for exp, strikes in extra.strikes.items():
            chain.strikes[exp] = sorted(set(chain.strikes.get(exp, [])) | set(strikes))
        chain.options.update(extra.options)
        chain.expiration_contract.update(extra.expiration_contract)
        chain.expirations = sorted(chain.strikes)
    return chain


def clear_cache():
    with _cache_lock:
        _cache.clear()
