"""Tom's trade log: the user's thinkorswim/Schwab fills, in SQLite.

Two writers feed it. Email fill alerts arrive one order at a time with no
fees (log_trade). Account statements are authoritative: they carry exact
fees and expirations, and they replace the alert-logged rows of the span
they cover (import_statement). There is one row per broker order, keyed
(account, ref); a statement's split fills of one order are merged before
they get here. Legs live in their own table, as signed per-contract
quantities, so no reader ever re-parses text.

The fill-line grammar is thinkorswim's, proven on all 110 fills of the
2026-09-20 statement against that statement's own per-leg trade history.
Anything outside it — iron condors, straddles, futures options — parses to
None and is refused rather than guessed: one wrong leg silently corrupts
every position P&L after it.
"""

import re
import sqlite3
from contextlib import contextmanager

from trading import config as tcfg
from trading import symbols as tsym

# --- Fill-line grammar -----------------------------------------------------

# Spread keyword -> (strategy, leg ratios in the order the line lists
# strikes). The first leg takes the order's sign; "" is a single option.
_SPREADS = {
    "": ("single", (1,)),
    "VERTICAL": ("vertical", (1, -1)),
    "CALENDAR": ("calendar", (1, -1)),
    "DIAGONAL": ("diagonal", (1, -1)),
    "BUTTERFLY": ("butterfly", (1, -2, 1)),
    "CONDOR": ("condor", (1, -1, -1, 1)),
    "DBL DIAG": ("double_diagonal", (1, 1, -1, -1)),
}
STRATEGIES = tuple(s for s, _ in _SPREADS.values())

_MONTHS = {m: i for i, m in enumerate(
    "JAN FEB MAR APR MAY JUN JUL AUG SEP OCT NOV DEC".split(), 1)}
_EXP = r"\d{1,2} [A-Z]{3} \d{2}"
_STRIKE = r"\d+(?:\.\d+)?"
# "BOT +2 CALENDAR SPX 100 (Weeklys) 3 FEB 26/30 JAN 26 6860 PUT @7.00 CBOE".
# It ends at the price: statements append a venue, while alerts glue market
# data straight on ("@1.85LAST=7715.14 BID=...").
_FILL_RE = re.compile(
    r"(?P<side>BOT|SOLD) (?P<qty>[+-]\d+) "
    r"(?:(?P<spread>" + "|".join(k for k in _SPREADS if k) + r") )?"
    r"(?P<symbol>[A-Z]+) (?P<mult>\d+) (?:\(\w+\) )?"
    rf"(?P<exps>{_EXP}(?:/{_EXP})?) "
    rf"(?P<strikes>{_STRIKE}(?:/{_STRIKE})*) "
    r"(?P<types>(?:CALL|PUT)(?:/(?:CALL|PUT))*) "
    r"@(?P<price>\d*\.?\d+)")


def _iso(exp):
    """'3 FEB 26' -> '2026-02-03'."""
    day, mon, yy = exp.split()
    return f"20{yy}-{_MONTHS[mon]:02d}-{int(day):02d}"


def parse_fill(line):
    """A thinkorswim fill line -> {text, side, quantity, strategy, symbol,
    multiplier, price, amount, legs}, or None outside the known grammar.
    `legs` are (expiration, strike, 'C'|'P', signed contracts); `amount` is
    the cash, credit positive — exactly the statement's AMOUNT column."""
    m = _FILL_RE.match(line or "")
    if not m:
        return None
    qty, price, mult = int(m["qty"]), float(m["price"]), int(m["mult"])
    if (m["side"] == "BOT") != (qty > 0):
        return None
    strategy, ratio = _SPREADS[m["spread"] or ""]
    n = len(ratio)
    exps = [_iso(e) for e in m["exps"].split("/")]
    strikes = [float(k) for k in m["strikes"].split("/")]
    types = [t[0] for t in m["types"].split("/")]
    if strategy == "calendar":              # one strike, two expirations
        strikes *= 2
    if len(types) == 1:
        types *= n
    if len(exps) == 1:
        exps *= n
    elif strategy == "double_diagonal":     # first expiry's pair, then second's
        exps = [exps[0]] * 2 + [exps[1]] * 2
    if not len(exps) == len(strikes) == len(types) == n:
        return None
    return {"text": m.group(0), "side": m["side"], "quantity": qty,
            "strategy": strategy,
            "symbol": tsym.normalize_underlying(m["symbol"]),
            "multiplier": mult, "price": price,
            "amount": round(-qty * price * mult, 2),
            "legs": [(exps[i], strikes[i], types[i], qty * ratio[i])
                     for i in range(n)]}


def order(fill, **fields):
    """A trade-log row from a parsed fill plus its account, ref, time, cash,
    fees and source."""
    return {"symbol": fill["symbol"], "strategy": fill["strategy"],
            "quantity": fill["quantity"], "price": fill["price"],
            "legs": fill["legs"], "fills": 1, **fields}


# --- Store -----------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    account     TEXT NOT NULL,      -- '*****699SCHW', as the broker prints it
    ref         TEXT NOT NULL,      -- statement REF # or alert #
    executed_at TEXT NOT NULL,      -- local time, ISO seconds
    symbol      TEXT NOT NULL,      -- underlying
    strategy    TEXT NOT NULL,      -- a STRATEGIES word, or 'expiration'
    quantity    INTEGER,            -- signed spread count; NULL: expiration
    price       REAL,               -- net price per spread
    amount      REAL NOT NULL,      -- cash, credit positive
    fees        REAL,               -- <= 0; NULL: not known yet (alert)
    fills       INTEGER NOT NULL DEFAULT 1,  -- broker rows merged in
    description TEXT NOT NULL,      -- the broker's line
    source      TEXT NOT NULL,      -- 'statement' | 'email'
    PRIMARY KEY (account, ref));
CREATE TABLE IF NOT EXISTS legs (
    account     TEXT NOT NULL,
    ref         TEXT NOT NULL,
    expiration  TEXT NOT NULL,
    strike      REAL NOT NULL,
    option_type TEXT NOT NULL,      -- 'C' | 'P'
    quantity    INTEGER NOT NULL,   -- signed contracts
    FOREIGN KEY (account, ref) REFERENCES trades ON DELETE CASCADE);
CREATE INDEX IF NOT EXISTS trades_time ON trades (account, executed_at);
"""

_COLS = ("account", "ref", "executed_at", "symbol", "strategy", "quantity",
         "price", "amount", "fees", "fills", "description", "source")


@contextmanager
def _db():
    """One transaction on the log (committed, or rolled back on error). The
    path is read per call so tests can repoint it."""
    path = tcfg.TRADE_LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    try:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")   # legs follow their trade
        db.executescript(_SCHEMA)
        with db:
            yield db
    finally:
        db.close()


def save(trades, *, replace=False, supersede=None):
    """Write orders in one transaction; returns (written, superseded).

    Without `replace`, an order already logged is left alone, so re-reading
    an alert is a no-op. A statement passes `replace` so its exact row wins.
    `supersede` = (account, start, end) first deletes the alert-logged rows
    in that span, because a statement is the complete record of its span and
    an alert # is not the statement's REF # for the same fill."""
    written = superseded = 0
    with _db() as db:
        if supersede:
            superseded = db.execute(
                "DELETE FROM trades WHERE source = 'email' AND account = ? "
                "AND executed_at >= ? AND executed_at <= ?", supersede).rowcount
        for t in trades:
            key = (t["account"], t["ref"])
            if db.execute("SELECT 1 FROM trades WHERE account = ? AND ref = ?",
                          key).fetchone():
                if not replace:
                    continue
                db.execute("DELETE FROM trades WHERE account = ? AND ref = ?",
                           key)
            db.execute(f"INSERT INTO trades ({', '.join(_COLS)}) "
                       f"VALUES ({', '.join('?' * len(_COLS))})",
                       [t.get(c) for c in _COLS])
            db.executemany("INSERT INTO legs VALUES (?, ?, ?, ?, ?, ?)",
                           [(*key, *leg) for leg in t["legs"]])
            written += 1
    return written, superseded


def query(account=None, start=None, end=None, symbol=None, strategy=None,
          refs=None):
    """Orders oldest first, each a dict of the trade's columns plus `legs`
    as (expiration, strike, type, quantity). `account` is exact (resolve a
    spoken one against accounts() first); start and end are ISO dates,
    inclusive."""
    where, args = [], []
    for sql, value in (("t.account = ?", account),
                       ("substr(t.executed_at, 1, 10) >= ?", start),
                       ("substr(t.executed_at, 1, 10) <= ?", end),
                       ("t.symbol = ?", symbol and tsym.normalize_underlying(symbol)),
                       ("t.strategy = ?", strategy)):
        if value:
            where.append(sql)
            args.append(value)
    if refs:
        where.append(f"t.ref IN ({', '.join('?' * len(refs))})")
        args.extend(refs)
    cond = (" WHERE " + " AND ".join(where)) if where else ""
    with _db() as db:
        trades = [dict(r) | {"legs": []} for r in db.execute(
            f"SELECT t.* FROM trades t{cond} ORDER BY t.executed_at, t.ref",
            args)]
        by_key = {(t["account"], t["ref"]): t for t in trades}
        for r in db.execute(
                "SELECT l.* FROM legs l JOIN trades t "
                f"ON l.account = t.account AND l.ref = t.ref{cond} "
                "ORDER BY l.rowid", args):
            by_key[(r["account"], r["ref"])]["legs"].append(
                (r["expiration"], r["strike"], r["option_type"], r["quantity"]))
    return trades


def accounts():
    """[(account, trades, first date, last date)] — what the log holds, for
    resolving a spoken account or asking which one."""
    with _db() as db:
        return [tuple(r) for r in db.execute(
            "SELECT account, COUNT(*), MIN(substr(executed_at, 1, 10)), "
            "MAX(substr(executed_at, 1, 10)) FROM trades "
            "GROUP BY account ORDER BY account")]
