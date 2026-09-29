"""Tools over Tom's trade log: the user's thinkorswim/Schwab fills in
SQLite (trading/trade_log.py).

The writers are Tom's: log_trade takes email fill alerts, import_statement
takes account statements. The two readers are Linda's too, so her
post-trade reviews see real executions instead of the user's retelling.

Every number a reader returns is summed here, from the log. The model
interprets positions; it never adds the cash up itself.
"""

import re
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

from lib.dates import PERIODS, period_range
from tools import tool
from trading import config as tcfg
from trading import statement as tstatement
from trading import symbols as tsym
from trading import trade_log
from trading import trade_positions as tpos

# One alert: "#131646602945 BOT +1 SPX 100 (Weeklys) 25 SEP 26 7600 PUT
# @1.85LAST=7715.14 BID=... , ACCOUNT *****868SCHW". The market data between
# price and account is noise, and parse_fill stops at the price.
_ALERT_RE = re.compile(
    r"#(?P<ref>\d+)\s+(?P<line>(?:BOT|SOLD)\s.*?),\s*ACCOUNT\s+(?P<account>\S+)",
    re.S)

_FILTERS = {
    "account": {"type": "string", "description": (
        "Account, or any part of it ('699'); 'all' combines accounts. Omit "
        "when the user didn't say: with several accounts the tool asks.")},
    "period": {"type": "string", "enum": list(PERIODS)},
    "start_date": {"type": "string", "description": "ISO date, inclusive."},
    "end_date": {"type": "string", "description": "ISO date, inclusive."},
    "symbol": {"type": "string", "description": "Underlying, e.g. 'SPX'."},
    "strategy": {"type": "string", "enum": list(trade_log.STRATEGIES)},
}


# --- Formatting --------------------------------------------------------------

def _legs(legs):
    """Net legs {(symbol, exp, strike, type): qty} as one phrase."""
    return ", ".join(f"{q:+d} {s} {e} {k:g}{cp}"
                     for (s, e, k, cp), q in sorted(legs.items()))


def _trade_line(t, with_account=False):
    parts = [t["ref"], t["executed_at"][:16].replace("T", " "),
             t["symbol"], t["strategy"]]
    if t["quantity"] is not None:
        parts.append(f"{t['quantity']:+d} @{t['price']:.2f}")
    if t["fills"] > 1:
        parts.append(f"({t['fills']} fills)")
    if t["strategy"] != "expiration":
        fees = "pending" if t["fees"] is None else f"{t['fees']:,.2f}"
        parts.append(f"cash {t['amount']:+,.2f} fees {fees}")
    if with_account:
        parts.insert(0, t["account"])
    legs = ", ".join(f"{q:+d} {e} {k:g}{cp}" for e, k, cp, q in t["legs"])
    return " ".join(parts) + " | " + legs


def _totals_line(what, t):
    line = (f"{what}: cash {t['cash']:+,.2f}, fees {t['fees']:,.2f}, "
            f"net {t['net']:+,.2f}")
    if t["pending"]:
        line += (f" (fees still pending on {t['pending']} alert-logged fills "
                 "until the next statement import)")
    return line


# --- Filters -------------------------------------------------------------------

def _account(arg):
    """(account, None), (None, None) for all accounts, or (None, reply):
    the reply asks which account instead of guessing."""
    known = trade_log.accounts()
    if not known:
        return None, ("The trade log is empty: import a statement or log a "
                      "fill first.")
    listing = "; ".join(f"{a} ({n} trades, {lo} to {hi})"
                        for a, n, lo, hi in known)
    arg = (arg or "").strip()
    if arg.lower() == "all":
        return None, None
    hits = [a for a, *_ in known if arg.upper() in a.upper()]
    if len(hits) == 1:
        return hits[0], None
    if not arg:
        return None, (f"The trade log holds several accounts: {listing}. Ask "
                      "the user which one they mean, or pass account='all'.")
    what = "matches several accounts" if hits else "matches no account"
    return None, f"'{arg}' {what} in the trade log. Accounts: {listing}."


def _range(args):
    """(start, end) ISO dates; either may be None (open-ended)."""
    if args.get("period"):
        return period_range(args["period"])
    return args.get("start_date"), args.get("end_date")


# --- Writers (Tom) -------------------------------------------------------------

@tool({
    "name": "log_trade",
    "description": (
        "Record thinkorswim/Schwab fills in the trade log from a fill-alert "
        "email (alerts@thinkorswim.com). Pass the alert text and the email's "
        "Date header exactly as get_email_thread returned them; the tool "
        "reads the order number, fill line and account itself. Safe to "
        "repeat: an order already in the log isn't logged twice."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "alert": {"type": "string",
                      "description": "The alert text, verbatim."},
            "email_date": {"type": "string",
                           "description": "The email's Date header, verbatim."},
        },
        "required": ["alert", "email_date"],
    },
})
def log_trade(ctx, args):
    try:
        when = parsedate_to_datetime(args.get("email_date") or "")
    except (TypeError, ValueError):
        return (f"Couldn't read the date '{args.get('email_date')}'. Nothing "
                "logged.")
    # Statement times are the user's local clock; an alert's zoned send time
    # is converted in code (hand conversion has slipped before, 2026-09-18).
    if when.tzinfo:
        when = when.astimezone().replace(tzinfo=None)
    when = when.isoformat(timespec="seconds")
    found = list(_ALERT_RE.finditer(args.get("alert") or ""))
    if not found:
        return ("No fill alert ('#number BOT/SOLD ..., ACCOUNT ...') in that "
                "text. Nothing logged.")
    out, new = [], 0
    for m in found:
        line = " ".join(m["line"].split())
        fill = trade_log.parse_fill(line)
        if not fill:
            out.append(f"#{m['ref']}: can't read '{line[:100]}', so nothing "
                       "was logged for it (only single, vertical, calendar, "
                       "diagonal, butterfly, condor and double-diagonal "
                       "option lines are understood)")
            continue
        trade = trade_log.order(
            fill, account=m["account"], ref=m["ref"], executed_at=when,
            amount=fill["amount"], fees=None, description=fill["text"],
            source="email")
        written, _ = trade_log.save([trade])
        new += written
        verb = "Logged" if written else "Already in the log"
        out.append(f"{verb} on {m['account']}: {_trade_line(trade)}")
    if new:
        out.append("Fees stay pending until the next statement import")
    return ". ".join(out) + "."


def _newest_statement():
    # Statement names start with their date, so the last name is the newest.
    found = sorted(tcfg.STATEMENTS_DIR.glob("*AccountStatement*.csv"))
    return found[-1] if found else None


@tool({
    "name": "import_statement",
    "description": (
        "Import a thinkorswim/Schwab account-statement CSV into the trade "
        "log. The statement is the authoritative record: it adds exact fees "
        "and expirations, and replaces the alert-logged fills of the span it "
        "covers. With no path it imports the newest statement in the user's "
        "statements folder. Safe to repeat."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {"type": "string",
                     "description": "CSV path; omit for the newest statement."},
        },
    },
})
def import_statement(ctx, args):
    path = Path(args["path"]) if args.get("path") else _newest_statement()
    if path is None:
        return f"No *AccountStatement*.csv in {tcfg.STATEMENTS_DIR}."
    try:
        st = tstatement.parse(path.read_text(encoding="utf-8-sig"))
    except OSError as e:
        return f"Couldn't read {path}: {e}"
    if st["problems"]:
        return (f"Nothing imported from {path.name}: "
                + "; ".join(st["problems"]) + ".")
    if not st["orders"]:
        return f"{path.name} has no trades or expirations. Nothing imported."
    grace = (datetime.fromisoformat(st["last"])
             + timedelta(minutes=tcfg.STATEMENT_EMAIL_GRACE_MIN))
    _, superseded = trade_log.save(
        st["orders"], replace=True,
        supersede=(st["account"], st["since"],
                   grace.isoformat(timespec="seconds")))
    fills = [o for o in st["orders"] if o["strategy"] != "expiration"]
    merged = sum(o["fills"] - 1 for o in fills)
    reply = (f"Imported {path.name} for {st['account']}, {st['since']} to "
             f"{st['last'][:10]}: {len(fills)} orders ({merged} split fills "
             f"merged) and {len(st['orders']) - len(fills)} expirations. It "
             "ties out to the statement's TOTAL row. "
             + _totals_line("Net of the span", tpos.totals(st["orders"])) + ".")
    if superseded:
        reply += (f" Replaced {superseded} alert-logged fills in that span "
                  "with the statement's rows.")
    if st["skipped"]:
        reply += f" Skipped {st['skipped']} non-trade cash rows."
    return reply


# --- Readers (Tom and Linda) ---------------------------------------------------

@tool({
    "name": "query_trade_log",
    "description": (
        "The user's thinkorswim/Schwab trades from the trade log, one line "
        "each (ref, time, strategy, quantity @ price, cash, fees | legs), then "
        "exact totals and the legs the selection leaves open. Filter by "
        "period or dates, account, symbol, strategy, or a list of refs. Pass "
        "refs to check any grouping of trades: no open legs means they form "
        "a closed set, and its net is the exact P&L. Quote these totals "
        "rather than adding numbers yourself. tastytrade trades aren't here "
        "(get_pnl)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {**_FILTERS,
                       "refs": {"type": "array", "items": {"type": "string"},
                                "description": "Refs of the trades to select."}},
    },
})
def query_trade_log(ctx, args):
    account, ask = _account(args.get("account"))
    if ask:
        return ask
    start, end = _range(args)
    trades = trade_log.query(account=account, start=start, end=end,
                             symbol=args.get("symbol"),
                             strategy=args.get("strategy"),
                             refs=args.get("refs"))
    if not trades:
        return "No trades in the log match that."
    shown = trades[-tcfg.TRADE_QUERY_MAX_ROWS:]
    lines = [_trade_line(t, with_account=account is None) for t in shown]
    if len(shown) < len(trades):
        lines.insert(0, f"(The latest {len(shown)} of {len(trades)} trades; "
                        "narrow the dates for the rest. Totals cover all.)")
    lines.append(_totals_line(f"{len(trades)} trades", tpos.totals(trades)))
    legs = tpos.net_legs(trades)
    lines.append("Open legs of this selection: "
                 + (_legs(legs) if legs else "none; every leg nets to zero."))
    return "\n".join(lines)


def _position_line(p, account=None):
    ts = p["trades"]
    first, last = ts[0]["executed_at"][:10], ts[-1]["executed_at"][:10]
    kinds = list(dict.fromkeys(t["strategy"] for t in ts))
    head = (f"#{p['number']} {p['status']} {first} to {last} {ts[0]['symbol']} "
            f"{'/'.join(kinds)}, {len(ts)} trade{'s' * (len(ts) > 1)} "
            f"[{' '.join(t['ref'] for t in ts)}]")
    if account:
        head = f"{account} {head}"
    if p["status"] == "closed":
        body = (f"P&L {p['net']:+,.2f} (cash {p['cash']:+,.2f}, "
                f"fees {p['fees']:,.2f})")
    else:
        body = (f"net cash so far {p['net']:+,.2f}; "
                f"open legs {_legs(p['open_legs'])}")
    if p["pending"]:
        body += f"; fees pending on {p['pending']} fills"
    if p["tangled"]:
        body += "; too tangled to split exactly, shown as one group"
    return f"{head}: {body}"


@tool({
    "name": "trade_log_positions",
    "description": (
        "The user's thinkorswim/Schwab positions and their P&L, computed "
        "from the trade log. A position follows the user's rule: the minimal "
        "set of trades whose legs net to zero. Each is closed; open; or "
        "unresolved (its leftover legs have already expired: opened before "
        "the log starts, or the expiration isn't logged yet). Each line gives "
        "dates, strategies, refs, and exact P&L (or net cash so far). A "
        "period selects positions whose latest trade falls in it, which for "
        "closed ones is the close date, i.e. realized P&L. Use for "
        "thinkorswim P&L questions; get_pnl covers tastytrade only."
    ),
    "input_schema": {
        "type": "object",
        "properties": {**_FILTERS,
                       "status": {"type": "string",
                                  "enum": ["all", "closed", "open"],
                                  "description": "'open' includes unresolved."}},
    },
})
def trade_log_positions(ctx, args):
    account, ask = _account(args.get("account"))
    if ask:
        return ask
    names = [account] if account else [a for a, *_ in trade_log.accounts()]
    start, end = _range(args)
    symbol = tsym.normalize_underlying(args.get("symbol") or "")
    status = args.get("status") or "all"

    def keep(p):
        ts, last = p["trades"], p["trades"][-1]["executed_at"][:10]
        if (start and last < start) or (end and last > end):
            return False
        if symbol and ts[0]["symbol"] != symbol:
            return False
        if args.get("strategy") and args["strategy"] not in {
                t["strategy"] for t in ts}:
            return False
        return status == "all" or (status == "closed") == (
            p["status"] == "closed")

    # Grouping always sees the whole history: a position closed in the
    # period may have opened long before it.
    picked = [(name, p) for name in names
              for p in tpos.group_positions(trade_log.query(account=name))
              if keep(p)]
    if not picked:
        return "No positions in the log match that."
    shown = picked[-tcfg.TRADE_QUERY_MAX_ROWS:]
    lines = [_position_line(p, name if account is None else None)
             for name, p in shown]
    if len(shown) < len(picked):
        lines.insert(0, f"(The latest {len(shown)} of {len(picked)} "
                        "positions; narrow the dates for the rest. Totals "
                        "cover all.)")
    closed = [p for _, p in picked if p["status"] == "closed"]
    if closed:
        won = sum(p["net"] > 0 for p in closed)
        lost = sum(p["net"] < 0 for p in closed)
        lines.append(f"Closed: {len(closed)} positions, P&L "
                     f"{sum(p['net'] for p in closed):+,.2f} ({won} winners, "
                     f"{lost} losers)")
    for kind in ("open", "unresolved"):
        some = [p for _, p in picked if p["status"] == kind]
        if some:
            lines.append(f"{kind.capitalize()}: {len(some)} positions, net "
                         f"cash so far {sum(p['net'] for p in some):+,.2f} "
                         "(not P&L: legs remain)")
    return "\n".join(lines)
