"""The trading engine: one lazily built object per broker wiring the
package together.

Both surfaces (voice tools, dashboard routes) call get_engine(broker) — the
first call constructs that broker's client/quotes/order manager from its
variant module (trading/main.py); if credentials are missing it returns
(None, friendly-problem-string) instead of raising, so an unconfigured
broker degrades to a clear message rather than a stack trace. Nothing in
the agent's startup path constructs this: a user who never says a trading
word never pays for it. The two engines never share state: each has its
own ticket file, audit log, chain cache key and quote cache.
"""

import threading
from datetime import date, timedelta

from trading import accounts as taccounts
from trading import chains as tchains
from trading import config as tcfg
from trading import main as tmain
from trading import pnl as tpnl
from trading import symbols as tsym
from trading.orders import OrderManager

_engines = {}
_engine_lock = threading.Lock()


class TradingEngine:
    def __init__(self, broker):
        self.broker = broker                      # the variant module
        self.key = broker.KEY
        tcfg.ensure_dirs(self.key)   # now, so a legacy audit log is read, not just written
        self.client = broker.Client()
        self.quotes = broker.make_quotes(self.client)
        self.orders = OrderManager(self.client, broker)

    # --- market data -----------------------------------------------------

    def chain(self, underlying, force=False):
        return tchains.fetch_chain(self.broker, self.client, underlying, force=force)

    def extend_chain(self, chain, start, end, every_strike=False):
        return tchains.extend_chain(self.broker, self.client, chain, start, end,
                                    every_strike=every_strike)

    def underlying_quote(self, underlying, wait_s=2.5):
        """Live Quote for an underlying ('SPX', '/ES', 'AAPL')."""
        underlying = tsym.normalize_underlying(underlying)
        direct = self.broker.chainless_quote_symbol(underlying)
        if direct:   # quotable but not tradable here: no chain to load (#56)
            return self.quotes.get(direct)
        chain = self.chain(underlying)
        streamer = chain.underlying_streamer
        if self.quotes.start():
            self.quotes.subscribe([streamer], trade=True)
            q = self.quotes.get(streamer, wait_s=wait_s)
            if q and (q.bid is not None or q.last is not None):
                return q
        # No tick (sandbox has none; Schwab never streams): REST snapshot.
        self.quotes.refresh([(self.broker.underlying_api_symbol(chain), streamer)])
        return self.quotes.get(streamer)

    def leg_quote_fn(self, chain, greeks=False):
        """get_quote(streamer)->Quote for ticket pricing: subscribes all of
        the chain's currently relevant leg symbols on demand. greeks=True
        also streams and waits for each leg's greeks (stream only: the REST
        snapshot fallback carries none)."""
        def get_quote(streamer):
            if not streamer:
                return None
            if self.quotes.start():
                self.quotes.subscribe([streamer], greeks=greeks)
                return self.quotes.get(streamer, wait_s=2.0, greeks=greeks)
            return self.quotes.get(streamer)
        prefetch = getattr(self.quotes, "prefetch", None)
        if prefetch:
            get_quote.prefetch = prefetch   # ticket.batch_quotes batches on it
        else:
            # A stream: subscribe the whole batch at once, so forty strikes
            # wait for their first ticks together, not one by one (#59).
            def get_quote_prefetch(streamers):
                if self.quotes.start():
                    self.quotes.subscribe(streamers, greeks=greeks)
            get_quote.prefetch = get_quote_prefetch
        return get_quote

    # --- accounts ----------------------------------------------------------

    def accounts(self):
        """Every account number this login reaches (discovered, cached)."""
        return self.client.account_numbers()

    def default_account(self, persona):
        """The account `persona` works on when the user didn't say one, or
        None when that is ambiguous (several accounts, no focus)."""
        return taccounts.default(self.key, persona, self.accounts(),
                                 self.client.default_account)

    def label(self, account):
        return taccounts.label(self.key, account)

    def balance(self, account):
        return self.broker.parse_balance(self.client.balances(account))

    # --- positions & P&L -------------------------------------------------

    def positions(self, account, underlying=None):
        pos = self.broker.parse_positions(self.client.positions(account))
        if underlying:
            u = tsym.normalize_underlying(underlying)
            pos = [p for p in pos if p.underlying == u]
        return pos

    def mark(self, position, wait_s=1.5):
        """Best available mark: live mid, then REST snapshot, then the
        broker's own close/mark (stale but honest — flagged by the caller)."""
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
            self.quotes.refresh([(position.symbol, streamer)])
            q = self.quotes.get(streamer)
            if q and q.mid is not None:
                return q.mid
        return position.close_price

    def pnl_report(self, account, start, end, underlying=None):
        """PnLReport for [start, end] (ISO dates), optionally one
        underlying. Fetches transactions with the open-matching lookback."""
        fetch_start = None
        if start:
            fetch_start = (date.fromisoformat(start) -
                           timedelta(days=tcfg.PNL_OPEN_LOOKBACK_DAYS)).isoformat()
        u = tsym.normalize_underlying(underlying) if underlying else None
        txs = self.broker.transactions(self.client, account, fetch_start, end, u)
        positions = self.positions(account, underlying)
        return tpnl.report(txs, start, end, positions=positions,
                           mark_fn=self.mark)

    # --- status ----------------------------------------------------------

    def status(self):
        s = {"broker": self.key, "label": tmain.label(self.key)}
        s.update(self.broker.status(self.client))
        s["stream"] = (("running" if self.quotes.start()
                        else f"unavailable ({self.quotes.last_error})")
                       if self.key == "tastytrade" else "REST snapshots (no stream)")
        return s


def get_engine(broker):
    """(engine, None) or (None, why-not) — never raises for missing config
    or an unknown broker key."""
    try:
        mod = tmain.module(broker)
    except KeyError:
        return None, f"Unknown broker '{broker}'."
    problem = mod.credentials_problem()
    if problem:
        return None, problem
    with _engine_lock:
        if broker not in _engines:
            _engines[broker] = TradingEngine(mod)
        return _engines[broker], None
