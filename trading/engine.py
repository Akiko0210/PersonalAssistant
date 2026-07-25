"""The trading engine: one lazily built object wiring the package together.

Both surfaces (voice tools, dashboard routes) call get_engine() — the
first call constructs the client/quote service/order manager from config;
if credentials are missing it returns (None, friendly-problem-string)
instead of raising, so an unconfigured machine degrades to a clear
message rather than a stack trace. Nothing in the agent's startup path
constructs this: a user who never says a trading word never pays for it.
"""

import threading
from datetime import date, timedelta

from trading import chains as tchains
from trading import config as tcfg
from trading import pnl as tpnl
from trading import symbols as tsym
from trading.orders import OrderManager
from trading.quotes import QuoteService
from trading.tasty_client import TastyClient

_engine = None
_engine_lock = threading.Lock()


class TradingEngine:
    def __init__(self):
        self.client = TastyClient()
        self.quotes = QuoteService(self.client)
        self.orders = OrderManager(self.client)

    # --- market data -----------------------------------------------------

    def chain(self, underlying, force=False):
        return tchains.fetch_chain(self.client, underlying, force=force)

    def _snapshot_fallback(self, api_symbol, streamer_symbol, group):
        """REST snapshot for one symbol into the quote cache (live only;
        sandbox has no REST quotes — errors are swallowed, caller copes
        with a None quote)."""
        try:
            data = self.client.market_data_by_type(**{group: api_symbol})
            items = data.get("items", []) if isinstance(data, dict) else []
            self.quotes.snapshot_merge(items, {api_symbol: streamer_symbol})
        except Exception:
            pass

    def underlying_quote(self, underlying, wait_s=2.5):
        """Live Quote for an underlying ('SPX', '/ES', 'AAPL')."""
        underlying = tsym.normalize_underlying(underlying)
        chain = self.chain(underlying)
        streamer = chain.underlying_streamer
        if self.quotes.start():
            self.quotes.subscribe([streamer], trade=True)
            q = self.quotes.get(streamer, wait_s=wait_s)
            if q and (q.bid is not None or q.last is not None):
                return q
        group = ("future" if chain.is_futures
                 else "index" if underlying in ("SPX", "RUT", "NDX", "VIX", "XSP")
                 else "equity")
        api_symbol = (chain.expiration_contract.get(chain.expirations[0])
                      if chain.is_futures else underlying)
        self._snapshot_fallback(api_symbol or underlying, streamer, group)
        return self.quotes.get(streamer)

    def leg_quote_fn(self, chain):
        """get_quote(streamer)->Quote for ticket pricing: subscribes all of
        the chain's currently relevant leg symbols on demand."""
        def get_quote(streamer):
            if not streamer:
                return None
            if self.quotes.start():
                self.quotes.subscribe([streamer])
                return self.quotes.get(streamer, wait_s=2.0)
            return self.quotes.get(streamer)
        return get_quote

    # --- positions & P&L -------------------------------------------------

    def positions(self, underlying=None):
        items = self.client.positions()
        pos = tpnl.parse_positions(items)
        if underlying:
            u = tsym.normalize_underlying(underlying)
            pos = [p for p in pos if p.underlying == u]
        return pos

    def mark(self, position, wait_s=1.5):
        """Best available mark: live DXLink mid, then REST snapshot, then
        yesterday's close (stale but honest — flagged by the caller)."""
        streamer = position.streamer_symbol
        if not streamer and "Option" in position.instrument_type:
            parsed = tsym.parse_occ(position.symbol)
            if parsed:
                try:
                    chain = self.chain(position.underlying)
                    info = chain.option(parsed["expiration"], parsed["strike"],
                                        parsed["cp"])
                    streamer = info.streamer if info else None
                except Exception:
                    streamer = None
        elif not streamer:
            streamer = tcfg.UNDERLYING_QUOTE_SYMBOL.get(
                position.symbol, position.symbol)
        if streamer and self.quotes.start():
            self.quotes.subscribe([streamer])
            q = self.quotes.get(streamer, wait_s=wait_s)
            if q and q.mid is not None:
                return q.mid
        if streamer:
            group = ("future-option" if position.symbol.startswith("./")
                     else "equity-option" if "Option" in position.instrument_type
                     else "future" if position.symbol.startswith("/")
                     else "equity")
            self._snapshot_fallback(position.symbol, streamer, group)
            q = self.quotes.get(streamer)
            if q and q.mid is not None:
                return q.mid
        return position.close_price

    def pnl_report(self, start, end, underlying=None):
        """PnLReport for [start, end] (ISO dates), optionally one
        underlying. Fetches transactions with the open-matching lookback."""
        fetch_start = None
        if start:
            fetch_start = (date.fromisoformat(start) -
                           timedelta(days=tcfg.PNL_OPEN_LOOKBACK_DAYS)).isoformat()
        filters = {"start_date": fetch_start, "end_date": end,
                   "types": ["Trade", "Receive Deliver", "Money Movement"]}
        if underlying:
            filters["underlying_symbol"] = tsym.normalize_underlying(underlying)
        txs = self.client.transactions_all(**filters)
        positions = self.positions(underlying)
        return tpnl.report(txs, start, end, positions=positions,
                           mark_fn=self.mark)

    # --- status ----------------------------------------------------------

    def status(self):
        return {
            "env": "live" if tcfg.is_live() else "sandbox",
            "base_url": tcfg.base_url(),
            "account": self.client.account_id,
            "stream": ("running" if self.quotes.start()
                       else f"unavailable ({self.quotes.last_error})"),
        }


def get_engine():
    """(engine, None) or (None, why-not) — never raises for missing config."""
    global _engine
    problem = tcfg.credentials_problem()
    if problem:
        return None, problem
    with _engine_lock:
        if _engine is None:
            _engine = TradingEngine()
        return _engine, None
