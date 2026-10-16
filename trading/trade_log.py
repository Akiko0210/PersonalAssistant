"""Tom's trade log: the user's thinkorswim/Schwab fills, in SQLite.

Three writers feed it. Email fill alerts arrive one order at a time with no
fees (log_trade). Account statements are authoritative: they carry exact
fees and expirations, and they replace the alert-logged rows of the span
they cover (import_statement). The Schwab API sync (sync_trades) writes the
statement's own rows — same refs, times, cash and fees — straight from the
broker, with the same authority. There is one row per broker order, keyed
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
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime

from trading import config as tcfg
from trading import symbols as tsym
from trading import trade_positions as tpos

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
# "custom": an API-synced order of a shape the fill grammar has no word for
# (an iron condor). Its legs come exact from the broker, so it is held, not
# refused — the refusal above guards against misreading text, not shapes.
STRATEGIES = tuple(s for s, _ in _SPREADS.values()) + ("custom",)

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
    source      TEXT NOT NULL,      -- 'statement' | 'api' | 'email'
    position_id INTEGER,            -- positions.id; NULL: not assigned
    PRIMARY KEY (account, ref));
-- An explicit position: what the user means by "the diagonal". Its legs are
-- never stored — they are the net of its trades, so they cannot go stale
-- the way a hand-kept leg list did (a campaign body a week out of date).
CREATE TABLE IF NOT EXISTS positions (
    id          INTEGER PRIMARY KEY,
    account     TEXT NOT NULL,
    name        TEXT NOT NULL,
    record_id   INTEGER,            -- the campaign record it serves (working_state)
    created     TEXT NOT NULL);
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
         "price", "amount", "fees", "fills", "description", "source",
         "position_id")


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
        # Logs made before positions existed lack the column.
        if "position_id" not in {r[1] for r in db.execute(
                "PRAGMA table_info(trades)")}:
            db.execute("ALTER TABLE trades ADD COLUMN position_id INTEGER")
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
    an alert # is not the statement's REF # for the same fill.

    A replaced row hands its position to its replacement: the same ref
    directly, a superseded alert row through the statement row with its
    legs nearest in time. Without that, every import would silently
    ungroup the positions it covers."""
    written = superseded = 0
    with _db() as db:
        carried = []    # (signature, time, position) of superseded alert rows
        if supersede:
            carried = [(_signature(t), t["executed_at"], t["position_id"])
                       for t in _read(db, "t.source = 'email' AND t.account = ? "
                                      "AND t.executed_at >= ? AND "
                                      "t.executed_at <= ? AND "
                                      "t.position_id IS NOT NULL", supersede)]
            superseded = db.execute(
                "DELETE FROM trades WHERE source = 'email' AND account = ? "
                "AND executed_at >= ? AND executed_at <= ?", supersede).rowcount
        for t in trades:
            key = (t["account"], t["ref"])
            old = db.execute("SELECT position_id FROM trades WHERE account = ? "
                             "AND ref = ?", key).fetchone()
            if old:
                if not replace:
                    continue
                db.execute("DELETE FROM trades WHERE account = ? AND ref = ?",
                           key)
            pid = (old and old["position_id"]) or _take(carried, t)
            db.execute(f"INSERT INTO trades ({', '.join(_COLS)}) "
                       f"VALUES ({', '.join('?' * len(_COLS))})",
                       [pid if c == "position_id" else t.get(c) for c in _COLS])
            db.executemany("INSERT INTO legs VALUES (?, ?, ?, ?, ?, ?)",
                           [(*key, *leg) for leg in t["legs"]])
            written += 1
    return written, superseded


def _signature(t):
    # Contracts and direction, not size: a statement merges an order's split
    # fills, so its row can be bigger than the alert that announced it.
    return (t["account"], frozenset((e, k, cp, q > 0) for e, k, cp, q in t["legs"]))


def _take(carried, t):
    """The position of the carried alert row that is `t`'s fill (same legs,
    nearest in time), removing it so one alert hands over only once."""
    same = [c for c in carried if c[0] == _signature(t)]
    if not same:
        return None
    when = datetime.fromisoformat(t["executed_at"])
    best = min(same, key=lambda c: abs(datetime.fromisoformat(c[1]) - when))
    carried.remove(best)
    return best[2]


def _read(db, where="", args=()):
    cond = f" WHERE {where}" if where else ""
    trades = [dict(r) | {"legs": []} for r in db.execute(
        f"SELECT t.* FROM trades t{cond} ORDER BY t.executed_at, t.ref", args)]
    by_key = {(t["account"], t["ref"]): t for t in trades}
    for r in db.execute(
            "SELECT l.* FROM legs l JOIN trades t "
            f"ON l.account = t.account AND l.ref = t.ref{cond} "
            "ORDER BY l.rowid", args):
        by_key[(r["account"], r["ref"])]["legs"].append(
            (r["expiration"], r["strike"], r["option_type"], r["quantity"]))
    return trades


def query(account=None, start=None, end=None, symbol=None, strategy=None,
          refs=None, positions=None):
    """Orders oldest first, each a dict of the trade's columns plus `legs`
    as (expiration, strike, type, quantity). `account` is exact (resolve a
    spoken one against accounts() first); start and end are ISO dates,
    inclusive; `positions` is a list of position ids."""
    where, args = [], []
    for sql, value in (("t.account = ?", account),
                       ("substr(t.executed_at, 1, 10) >= ?", start),
                       ("substr(t.executed_at, 1, 10) <= ?", end),
                       ("t.symbol = ?", symbol and tsym.normalize_underlying(symbol)),
                       ("t.strategy = ?", strategy)):
        if value:
            where.append(sql)
            args.append(value)
    for column, values in (("t.ref", refs or None),
                           ("t.position_id", positions)):
        if values is not None:
            where.append(f"{column} IN ({', '.join('?' * len(values))})")
            args.extend(values)
    with _db() as db:
        return _read(db, " AND ".join(where), args)


# --- Positions -----------------------------------------------------------------

def create_position(account, name, record_id=None):
    with _db() as db:
        return db.execute(
            "INSERT INTO positions (account, name, record_id, created) "
            "VALUES (?, ?, ?, ?)", (account, name.strip(), record_id,
                                    datetime.now().isoformat(timespec="seconds"))
        ).lastrowid


def positions(account=None, ids=None, record_id=None):
    """Position rows, oldest first, filtered by any of the arguments."""
    where, args = [], []
    if account:
        where.append("account = ?")
        args.append(account)
    if record_id is not None:
        where.append("record_id = ?")
        args.append(record_id)
    if ids is not None:
        where.append(f"id IN ({', '.join('?' * len(ids))})")
        args.extend(ids)
    cond = f" WHERE {' AND '.join(where)}" if where else ""
    with _db() as db:
        return [dict(r) for r in db.execute(
            f"SELECT * FROM positions{cond} ORDER BY id", args)]


def set_record(position_id, record_id):
    with _db() as db:
        db.execute("UPDATE positions SET record_id = ? WHERE id = ?",
                   (record_id, position_id))


def assign(account, refs, position_id):
    """Put these trades in a position — the user's answer to an ambiguous
    fill. Returns how many changed."""
    with _db() as db:
        return db.execute(
            f"UPDATE trades SET position_id = ? WHERE account = ? AND ref IN "
            f"({', '.join('?' * len(refs))})",
            (position_id, account, *refs)).rowcount


def auto_assign(account):
    """Apply the one-match rule to `account`'s unassigned trades, oldest
    first, each against the positions' legs as of that trade. Returns {ref:
    [position ids]}: one id means it joined that position now (or earlier),
    none or several mean the user must say. Walking the whole account makes
    the order fills were logged in irrelevant — alerts arrive out of order."""
    with _db() as db:
        trades = _read(db, "t.account = ?", (account,))
        net = defaultdict(Counter)       # position -> its legs so far
        result, joined = {}, []
        for t in trades:
            if t["position_id"] is None:
                hits = tpos.closing(t, net)
                result[t["ref"]] = hits
                if len(hits) == 1:
                    t["position_id"] = hits[0]
                    joined.append((hits[0], t["ref"]))
            else:
                result[t["ref"]] = [t["position_id"]]
            if t["position_id"] is not None:
                for e, k, cp, q in t["legs"]:
                    net[t["position_id"]][(t["symbol"], e, k, cp)] += q
        db.executemany("UPDATE trades SET position_id = ? WHERE account = ? "
                       "AND ref = ?", [(p, account, r) for p, r in joined])
    return result


def accounts():
    """[(account, trades, first date, last date)] — what the log holds, for
    resolving a spoken account or asking which one."""
    with _db() as db:
        return [tuple(r) for r in db.execute(
            "SELECT account, COUNT(*), MIN(substr(executed_at, 1, 10)), "
            "MAX(substr(executed_at, 1, 10)) FROM trades "
            "GROUP BY account ORDER BY account")]
