"""Option-chain fetch + normalization + cache.

Both chain families (equity/index via /option-chains/{sym}/nested, futures
via /futures-option-chains/{code}/nested) normalize into the same
ChainData: expirations, strikes-per-expiration, and an OptionInfo (OCC +
streamer symbol) per (expiration, strike, C/P). Strategy building and leg
validation work off ChainData only — they never see raw API shapes.

Where the same expiration date appears under two roots (SPX AM-settled
monthly vs SPXW PM weekly), the duplicate is resolved by expiration-type
priority — Regular > Quarterly > End-Of-Month > Weekly — the same rule
Tasty-Web applies.

Chains are cached for CHAIN_CACHE_TTL_S (strikes don't change intraday).
"""

import threading
import time

from trading import config as tcfg
from trading import symbols as tsym
from trading.models import ChainData, OptionInfo

_TYPE_PRIORITY = {"Regular": 0, "Quarterly": 1, "End-of-Month": 2,
                  "End-Of-Month": 2, "Weekly": 3}

_cache = {}
_cache_lock = threading.Lock()


def _num(x):
    """API strike prices arrive as strings like '6000.0'."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _add_strikes(chain, exp, strike_rows):
    strikes = chain.strikes.setdefault(exp, [])
    for row in strike_rows:
        sp = _num(row.get("strike-price"))
        if sp is None:
            continue
        if sp not in strikes:
            strikes.append(sp)
        if row.get("call"):
            chain.options[(exp, sp, "C")] = OptionInfo(
                occ=row["call"], streamer=row.get("call-streamer-symbol", ""))
        if row.get("put"):
            chain.options[(exp, sp, "P")] = OptionInfo(
                occ=row["put"], streamer=row.get("put-streamer-symbol", ""))


def _normalize_equity(underlying, data):
    chain = ChainData(
        underlying=underlying,
        underlying_streamer=tcfg.UNDERLYING_QUOTE_SYMBOL.get(underlying, underlying),
        expirations=[], strikes={}, options={}, fetched_at=time.time())
    # Prefer the Standard chain; collect expirations across items, resolving
    # same-date duplicates by expiration-type priority.
    best_type = {}  # exp-date -> (priority, expiration-dict)
    for item in data.get("items", []):
        if item.get("option-chain-type") not in (None, "Standard"):
            continue
        for exp in item.get("expirations", []):
            date = exp.get("expiration-date")
            if not date:
                continue
            prio = _TYPE_PRIORITY.get(exp.get("expiration-type"), 9)
            if date not in best_type or prio < best_type[date][0]:
                best_type[date] = (prio, exp)
    for date in sorted(best_type):
        chain.expirations.append(date)
        _add_strikes(chain, date, best_type[date][1].get("strikes", []))
        chain.strikes[date].sort()
    return chain


def _normalize_futures(underlying, data, futures_items):
    chain = ChainData(
        underlying=underlying,
        underlying_streamer=tcfg.UNDERLYING_QUOTE_SYMBOL.get(underlying, underlying),
        expirations=[], strikes={}, options={}, is_futures=True,
        fetched_at=time.time())
    # Front month: the active contract's streamer symbol beats the
    # continuous-symbol default when we have it.
    for fut in futures_items or []:
        if fut.get("active-month") and fut.get("streamer-symbol"):
            chain.underlying_streamer = fut["streamer-symbol"]
            break
    by_date = {}
    for oc in data.get("option-chains", []):
        for exp in oc.get("expirations", []):
            date = exp.get("expiration-date")
            if not date:
                continue
            prio = _TYPE_PRIORITY.get(exp.get("expiration-type"), 9)
            if date not in by_date or prio < by_date[date][0]:
                by_date[date] = (prio, exp)
    for date in sorted(by_date):
        exp = by_date[date][1]
        chain.expirations.append(date)
        chain.expiration_contract[date] = exp.get("underlying-symbol", "")
        _add_strikes(chain, date, exp.get("strikes", []))
        chain.strikes[date].sort()
    return chain


def fetch_chain(client, underlying, force=False):
    """ChainData for an underlying ('SPX', 'AAPL', '/ES'), cached."""
    underlying = tsym.normalize_underlying(underlying)
    now = time.time()
    with _cache_lock:
        hit = _cache.get(underlying)
        if hit and not force and now - hit.fetched_at < tcfg.CHAIN_CACHE_TTL_S:
            return hit
    if tsym.is_futures(underlying):
        code = tsym.product_code(underlying)
        data = client.nested_futures_option_chain(code)
        futs = data.get("futures") or []
        if not futs:
            try:
                futs = client.futures(code).get("items", [])
            except Exception:
                futs = []  # front-month streamer stays the continuous default
        chain = _normalize_futures(underlying, data, futs)
    else:
        chain = _normalize_equity(underlying, client.nested_option_chain(underlying))
    if not chain.expirations:
        raise ValueError(f"No option chain found for {underlying}.")
    with _cache_lock:
        _cache[underlying] = chain
    return chain


def clear_cache():
    with _cache_lock:
        _cache.clear()
