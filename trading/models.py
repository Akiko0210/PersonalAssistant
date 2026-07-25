"""Dataclasses shared across the trading package.

Pure data, no I/O. Everything is constructed either from API responses
(by tasty_client/chains/orders/pnl) or from user intent (by strategies/
ticket), and consumed by the voice tools and dashboard adapters.
"""

from dataclasses import dataclass, field, asdict


@dataclass
class Quote:
    """Latest known market data for one streamer symbol."""
    symbol: str
    bid: float = None
    ask: float = None
    last: float = None
    updated_at: float = 0.0  # time.time() of the newest tick

    @property
    def mid(self):
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return self.last


@dataclass
class OptionInfo:
    """One tradable option contract, as found in a chain."""
    occ: str            # order symbol (OCC / future-option format)
    streamer: str       # dxFeed symbol for quotes


@dataclass
class ChainData:
    """A normalized option chain for one underlying.

    strikes maps expiration -> sorted list of strike prices;
    options maps (expiration, strike, "C"|"P") -> OptionInfo.
    """
    underlying: str                       # as spoken/typed: "SPX", "/ES"
    underlying_streamer: str              # dxFeed symbol for the underlying
    expirations: list                     # sorted ISO dates, nearest first
    strikes: dict                         # exp -> [float, ...]
    options: dict                         # (exp, strike, cp) -> OptionInfo
    is_futures: bool = False
    # Futures only: option expiration -> futures contract symbol (/ESU6),
    # and the front-month contract's streamer symbol.
    expiration_contract: dict = field(default_factory=dict)
    fetched_at: float = 0.0

    def option(self, expiration, strike, cp):
        return self.options.get((expiration, float(strike), cp))


@dataclass
class Leg:
    """One leg of a strategy: what the user is expressing.

    strike/expiration must exist in the chain to be orderable; `cp` is
    "Call"/"Put"; `side` is "Long"/"Short"; open_close "Open"/"Close".
    """
    strike: float
    cp: str
    expiration: str
    side: str
    size: int = 1
    open_close: str = "Open"

    @property
    def action(self):
        close = self.open_close == "Close"
        if self.side == "Long":
            return "Buy to Close" if close else "Buy to Open"
        return "Sell to Close" if close else "Sell to Open"

    @property
    def code(self):
        """BTO/STO/BTC/STC for terse display."""
        return {"Buy to Open": "BTO", "Sell to Open": "STO",
                "Buy to Close": "BTC", "Sell to Close": "STC"}[self.action]


@dataclass
class DryRunResult:
    """What the broker says the order would do (from /orders/dry-run)."""
    ok: bool
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    buying_power_change: float = None
    new_buying_power: float = None
    margin_requirement: float = None
    total_fees: float = None
    raw: dict = field(default_factory=dict, repr=False)


@dataclass
class PlacedOrder:
    """A live or historical order as returned by the orders endpoints."""
    id: str
    status: str
    underlying: str
    price: float = None
    price_effect: str = None
    time_in_force: str = None
    legs: list = field(default_factory=list)   # [{symbol, action, quantity, filled}]
    created_at: str = None
    raw: dict = field(default_factory=dict, repr=False)

    TERMINAL = {"Filled", "Cancelled", "Expired", "Rejected",
                "Removed", "Partially Removed"}

    @property
    def is_working(self):
        return self.status not in self.TERMINAL


@dataclass
class Position:
    """One open position with enough fields for marking and P&L."""
    symbol: str                 # instrument symbol (OCC for options)
    underlying: str
    instrument_type: str
    quantity: float             # signed: negative = short
    average_open_price: float
    multiplier: float
    close_price: float = None   # yesterday's close (no live mark in API)
    streamer_symbol: str = None
    realized_today: float = 0.0
    raw: dict = field(default_factory=dict, repr=False)

    def market_value(self, mark):
        return mark * self.multiplier * self.quantity

    def unrealized(self, mark):
        if mark is None:
            return None
        return (mark - self.average_open_price) * self.quantity * self.multiplier


@dataclass
class RealizedEvent:
    """One realized-P&L event: a close matched (FIFO) against its opens."""
    underlying: str
    symbol: str
    closed_at: str              # ISO timestamp of the closing execution
    quantity: float
    proceeds: float             # net credit received on close (after fees)
    cost: float                 # net debit paid to open the matched lots
    pnl: float                  # proceeds - cost


@dataclass
class PnLReport:
    start: str
    end: str
    realized: float = 0.0
    fees: float = 0.0
    events: list = field(default_factory=list)       # [RealizedEvent]
    by_underlying: dict = field(default_factory=dict)  # underlying -> realized
    unrealized: float = None                          # None when no marks
    unrealized_by_position: list = field(default_factory=list)
    note: str = ""

    def to_dict(self):
        return asdict(self)
