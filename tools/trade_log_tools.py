"""Tools over Tom's trade log: the user's thinkorswim/Schwab fills in
SQLite (trading/trade_log.py).

The writers are Tom's: log_trade takes email fill alerts, import_statement
takes account statements. The two readers are Linda's too, so her
post-trade reviews see real executions instead of the user's retelling.

Every number a reader returns is summed here, from the log. The model
interprets positions; it never adds the cash up itself.

Positions are explicit (trade_log.positions): a new fill joins the one
position whose open leg it closes, and the user decides every other case —
no match, or several. One position per thing the user names, so the same
question gets the same answer however it is asked (2026-10-15: three
queries over one day's data gave three different short legs).
"""

import logging
import re
from datetime import date, datetime, timedelta
from email.utils import parsedate_to_datetime
from pathlib import Path

from lib.dates import PERIODS, period_range, with_weekday
from stores import working_state
from tools import tool
from tools.trading_tools import display_heading
from trading import accounts as taccounts
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

log = logging.getLogger("trading")

_FILTERS = {
    "account": {"type": "string", "description": (
        "Account: its nickname ('IRA', 'red'), or any part of it ('699'); "
        "'all' combines accounts. Omit when the user didn't say: the account "
        "chosen with set_account is used, and with none the tool asks.")},
    "period": {"type": "string", "enum": list(PERIODS)},
    "start_date": {"type": "string", "description": "ISO date, inclusive."},
    "end_date": {"type": "string", "description": "ISO date, inclusive."},
    "symbol": {"type": "string", "description": "Underlying, e.g. 'SPX'."},
    "strategy": {"type": "string", "enum": list(trade_log.STRATEGIES)},
}


# --- Formatting --------------------------------------------------------------

def _legs(legs):
    """Net legs {(symbol, exp, strike, type): qty} as one phrase. An expired
    leftover says so: unmarked, it was read aloud as a live short leg."""
    today = date.today().isoformat()
    return ", ".join(f"{q:+d} {s} {with_weekday(e)} {k:g}{cp}"
                     + (" (expired, not yet cleared)" if e < today else "")
                     for (s, e, k, cp), q in sorted(legs.items()))


def _trade_line(t, with_account=False):
    when = t["executed_at"]
    parts = [t["ref"], f"{with_weekday(when[:10])} {when[11:16]}",
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
    legs = ", ".join(f"{q:+d} {with_weekday(e)} {k:g}{cp}"
                     for e, k, cp, q in t["legs"])
    return " ".join(parts) + " | " + legs


def _totals_line(what, t):
    line = (f"{what}: cash {t['cash']:+,.2f}, fees {t['fees']:,.2f}, "
            f"net {t['net']:+,.2f}")
    if t["pending"]:
        line += (f" (fees still pending on {t['pending']} alert-logged fills "
                 "until the next statement import)")
    return line


# --- Filters -------------------------------------------------------------------

def _account(arg, persona=None):
    """(account, None), (None, None) for all accounts, or (None, reply):
    the reply asks which account instead of guessing. A nickname ('IRA') and,
    when nothing was said, the persona's chosen Schwab account
    (trading/accounts.py) both turn into the digits the log's masked ids
    keep ('*****699SCHW'), so "red" means the same account as on the live
    tools."""
    known = trade_log.accounts()
    arg = (arg or "").strip()
    named = taccounts.digits_for("schwab", arg) if arg else None
    if named:
        arg = named[-3:]
    elif not arg and persona:
        chosen = taccounts.focus("schwab", persona)
        if chosen:
            arg = chosen[-3:]
    if not known:
        return None, ("The trade log is empty: import a statement or log a "
                      "fill first.")
    listing = "; ".join(f"{a} ({n} trades, {lo} to {hi})"
                        for a, n, lo, hi in known)
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


# --- Positions -----------------------------------------------------------------

def _label(p):
    return f"P{p['id']} '{p['name']}'"


def _placed(account, refs):
    """One line per ref after the one-match rule has run: where it joined,
    or what to ask the user."""
    result = trade_log.auto_assign(account)
    names = {p["id"]: _label(p) for p in trade_log.positions(account=account)}
    by_position = {}
    for t in trade_log.query(account=account):
        if t["position_id"] is not None:
            by_position.setdefault(t["position_id"], []).append(t)
    open_ones = [names[pid] for pid, ts in by_position.items()
                 if tpos.net_legs(ts)]
    out = []
    for ref in refs:
        hits = result.get(ref, [])
        if len(hits) == 1:
            out.append(f"#{ref} is in {names[hits[0]]}")
        elif hits:
            out.append(f"#{ref} closes legs of " + " and ".join(
                names[h] for h in hits) + ": ask the user which it belongs "
                "to, then assign_trades")
        else:
            out.append(f"#{ref} closes no open position's leg: ask the user "
                       "whether it starts a new position (and its name) or "
                       "adds to one ("
                       + (", ".join(open_ones) or "none open") + "), then "
                       "assign_trades")
    return out


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
    out, logged = [], []
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
        if written:
            logged.append(trade)
        verb = "Logged" if written else "Already in the log"
        out.append(f"{verb} on {m['account']}: {_trade_line(trade)}")
    if logged:
        out.append("Fees stay pending until the next statement import")
        for account in dict.fromkeys(t["account"] for t in logged):
            out += _placed(account, [t["ref"] for t in logged
                                     if t["account"] == account])
    return ". ".join(out) + "."


def _record(ctx, args):
    """(record id or None, error or None) for an optional `record` arg."""
    if args.get("record") is None:
        return None, None
    rid = int(args["record"])
    if working_state.title(ctx.active_agent, rid) is None:
        return None, f"You have no record #{rid} you can read."
    return rid, None


@tool({
    "name": "assign_trades",
    "description": (
        "Put trade-log fills in a position: the user's answer when log_trade "
        "says a fill closes no open position's leg, or legs of several. Pass "
        "the refs and either position=<id> (an existing one) or "
        "new_position=<the name the user gives it>; record=<campaign id> "
        "ties the position to its campaign record. Only on the user's yes. "
        "Fills that roll this position's legs afterwards join it by "
        "themselves."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "refs": {"type": "array", "items": {"type": "string"},
                     "description": "Refs as log_trade or query_trade_log "
                                    "printed them."},
            "position": {"type": "integer",
                         "description": "An existing position's id."},
            "new_position": {"type": "string",
                             "description": "Name for a new position."},
            "record": {"type": "integer",
                       "description": "The campaign record it serves."},
        },
        "required": ["refs"],
    },
})
def assign_trades(ctx, args):
    refs = [str(r).strip() for r in args.get("refs") or [] if str(r).strip()]
    if not refs:
        return "Which fills? Pass their refs."
    trades = trade_log.query(refs=refs)
    if not trades:
        return "None of those refs is in the trade log."
    accounts = {t["account"] for t in trades}
    if len(accounts) > 1:
        return ("Those fills are in different accounts; a position lives in "
                "one. Assign each account's fills separately.")
    account = accounts.pop()
    rid, err = _record(ctx, args)
    if err:
        return err
    name = (args.get("new_position") or "").strip()
    if name:
        pid = trade_log.create_position(account, name, rid)
    elif args.get("position") is not None:
        found = trade_log.positions(ids=[int(args["position"])])
        if not found or found[0]["account"] != account:
            return (f"No position P{args['position']} in {account}; "
                    "trade_log_positions lists them.")
        pid = found[0]["id"]
        if rid is not None:
            trade_log.set_record(pid, rid)
    else:
        return "Which position? Pass position=<id> or new_position=<name>."
    trade_log.assign(account, [t["ref"] for t in trades], pid)
    # Rolls logged after these fills, still waiting, can join now.
    pending = [t["ref"] for t in trade_log.query(account=account)
               if t["position_id"] is None]
    joined = [line for line in _placed(account, pending) if " is in " in line]
    reply = f"Assigned {len(trades)} fills to P{pid}."
    if joined:
        reply += " Then: " + "; ".join(joined) + "."
    missing = sorted(set(refs) - {t["ref"] for t in trades})
    if missing:
        reply += " Not in the log: " + ", ".join(missing) + "."
    return reply


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
        text = path.read_text(encoding="utf-8-sig")
    except OSError as e:
        return f"Couldn't read {path}: {e}"
    return import_statement_text(path.name, text)[1]


def import_statement_text(name, text):
    """(imported, reply) for one statement's text. Shared by the tool and the
    dashboard's drop zone (web/server.py), which keeps a copy only when
    `imported` — so a refused file never becomes "the latest statement"."""
    st = tstatement.parse(text)
    if st["problems"]:
        return False, (f"Nothing imported from {name}: "
                       + "; ".join(st["problems"]) + ".")
    if not st["orders"]:
        return False, f"{name} has no trades or expirations. Nothing imported."
    grace = (datetime.fromisoformat(st["last"])
             + timedelta(minutes=tcfg.STATEMENT_EMAIL_GRACE_MIN))
    _, superseded = trade_log.save(
        st["orders"], replace=True,
        supersede=(st["account"], st["since"],
                   grace.isoformat(timespec="seconds")))
    # Its expirations close position legs, and any row the carry-over missed
    # is re-placed by the same one-match rule.
    trade_log.auto_assign(st["account"])
    fills = [o for o in st["orders"] if o["strategy"] != "expiration"]
    merged = sum(o["fills"] - 1 for o in fills)
    reply = (f"Imported {name} for {st['account']}, {st['since']} to "
             f"{st['last'][:10]}: {len(fills)} orders ({merged} split fills "
             f"merged) and {len(st['orders']) - len(fills)} expirations. It "
             "ties out to the statement's TOTAL row. "
             + _totals_line("Net of the span", tpos.totals(st["orders"])) + ".")
    if superseded:
        reply += (f" Replaced {superseded} alert-logged fills in that span "
                  "with the statement's rows.")
    if st["skipped"]:
        reply += f" Skipped {st['skipped']} non-trade cash rows."
    return True, reply


@tool({
    "name": "sync_trades",
    "description": (
        "Update the trade log straight from Charles Schwab: every fill (with "
        "its exact fees) and expiration since the log's newest synced or "
        "imported row, for each account shared with the app, or one. These "
        "are the statement's own rows, so no CSV export is needed, and "
        "alert-logged fills in the span are replaced by their exact rows. "
        "Safe to repeat. Use it when the user asks to update the log, and "
        "before answering from the log if it may be behind (today's fills "
        "not in yet). A statement is still needed only for history older "
        "than a year, or an account Schwab doesn't share with the app."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "account": {"type": "string", "description": (
                "Nickname or last digits; omit for every shared account.")},
            "start_date": {"type": "string", "description": (
                "ISO date to re-sync from; omit to resume where the log ends.")},
        },
    },
})
def sync_trades(ctx, args):
    try:
        from trading.engine import get_engine
        eng, err = get_engine("schwab")
        if err:
            return err
        numbers = eng.accounts()
    except Exception as e:
        return f"Couldn't reach Schwab: {e}"
    said = (args.get("account") or "").strip()
    if said and said.lower() not in ("all", "all accounts"):
        number, problem = taccounts.resolve("schwab", said, numbers)
        if problem:
            return problem
        numbers = [number]
    return " ".join(_sync_account(eng, n, args.get("start_date")) for n in numbers)


def _sync_account(eng, number, start=None):
    """One account's sync, as one spoken sentence. Each account is its own
    save() transaction, so a failure leaves the others written and this one
    untouched."""
    from trading import schwab
    acct, label = schwab.log_account(number), eng.label(number)
    today = date.today()
    floor = today - timedelta(days=tcfg.SCHWAB_TRANSACTIONS_MAX_DAYS)
    logged = trade_log.query(account=acct)
    if start:
        begin = date.fromisoformat(start)
    else:
        newest = max((t["executed_at"] for t in logged if t["source"] != "email"),
                     default=None)
        begin = (date.fromisoformat(newest[:10]) - timedelta(days=1) if newest
                 else today - timedelta(days=tcfg.SCHWAB_SYNC_FIRST_DAYS))
    begin = max(begin, floor)
    try:
        rows, skipped = schwab.trade_log_rows(eng.client, number, begin.isoformat(),
                                              today.isoformat())
    except Exception as e:
        return f"{label}: couldn't sync ({e})."
    if not rows:
        return f"{label}: nothing to sync since {begin}."
    before = {t["ref"] for t in logged}
    last = datetime.fromisoformat(max(r["executed_at"] for r in rows))
    grace = last + timedelta(seconds=tcfg.SCHWAB_SYNC_ALERT_GRACE_S)
    _, superseded = trade_log.save(rows, replace=True, supersede=(
        acct, begin.isoformat(), grace.isoformat(timespec="seconds")))
    new = [r for r in rows if r["ref"] not in before]
    fills = [r for r in new if r["strategy"] != "expiration"]
    reply = (f"{label}: synced {begin} to today — {len(new)} new rows "
             f"({len(fills)} orders, {len(new) - len(fills)} expirations), "
             f"{len(rows) - len(new)} already in the log.")
    if superseded:
        reply += f" Replaced {superseded} alert-logged fills with their exact rows."
    if skipped:
        reply += f" Skipped {skipped} stock orders (the log holds options)."
    # _placed runs the one-match rule over the whole account (so synced
    # expirations close their legs too) and says which fills need the user.
    asks = [line for line in _placed(acct, [r["ref"] for r in fills])
            if "ask the user" in line]
    if asks:
        reply += " Positions to settle: " + "; ".join(asks) + "."
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
        "a closed set, and its net is the exact P&L. Pass position=<id> for "
        "one position's trades — the way to answer 'what's open in X' — or "
        "record=<campaign id> for the positions serving that campaign. "
        "Quote these totals rather than adding numbers yourself. tastytrade "
        "trades aren't here (get_pnl)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {**_FILTERS,
                       "refs": {"type": "array", "items": {"type": "string"},
                                "description": "Refs of the trades to select."},
                       "position": {"type": "integer",
                                    "description": "A position's id."},
                       "record": {"type": "integer",
                                  "description": "A campaign record's id."}},
    },
}, display=display_heading("Trades from the trade log", broker="schwab"))
def query_trade_log(ctx, args):
    chosen = None       # the positions selected, when a position/record is
    rid, err = _record(ctx, args)
    if err:
        return err
    if rid is not None:
        chosen = trade_log.positions(record_id=rid)
        if not chosen:
            return (f"Record #{rid} has no positions yet — assign_trades "
                    "with record= ties one to it.")
    elif args.get("position") is not None:
        chosen = trade_log.positions(ids=[int(args["position"])])
        if not chosen:
            return (f"No position P{args['position']}; trade_log_positions "
                    "lists them.")
    # A position names its own account; otherwise ask rather than guess.
    account, ask = ((None, None) if chosen and not args.get("account")
                    else _account(args.get("account"), ctx.active_agent))
    if ask:
        return ask
    start, end = _range(args)
    trades = trade_log.query(account=account, start=start, end=end,
                             symbol=args.get("symbol"),
                             strategy=args.get("strategy"),
                             refs=args.get("refs"),
                             positions=chosen and [p["id"] for p in chosen])
    if not trades:
        return "No trades in the log match that."
    shown = trades[-tcfg.TRADE_QUERY_MAX_ROWS:]
    lines = [_trade_line(t, with_account=account is None) for t in shown]
    if chosen:
        lines.insert(0, "Positions: " + ", ".join(
            f"{_label(p)} on {p['account']}" for p in chosen))
    if len(shown) < len(trades):
        lines.insert(0, f"(The latest {len(shown)} of {len(trades)} trades; "
                        "narrow the dates for the rest. Totals cover all.)")
    lines.append(_totals_line(f"{len(trades)} trades", tpos.totals(trades)))
    legs = tpos.net_legs(trades)
    lines.append("Open legs of this selection: "
                 + (_legs(legs) if legs else "none; every leg nets to zero."))
    return "\n".join(lines)


def _all_positions(account):
    """One account's named positions, then its unassigned trades grouped by
    the user's zero-net rule."""
    trades = trade_log.query(account=account)
    named = []
    for row in trade_log.positions(account=account):
        ts = [t for t in trades if t["position_id"] == row["id"]]
        if ts:
            label = _label(row) + (f" (campaign #{row['record_id']})"
                                   if row["record_id"] else "")
            named.append({"label": label, "trades": ts, "tangled": False,
                          **tpos.describe(ts)})
    rest = tpos.group_positions([t for t in trades if t["position_id"] is None])
    # By latest trade, so the row cap (which keeps the newest) never drops
    # a live named position behind months of old rule-grouped ones.
    return sorted(named + [p | {"label": f"#{p['number']}"} for p in rest],
                  key=lambda p: p["trades"][-1]["executed_at"])


def _position_line(p, account=None):
    ts = p["trades"]
    first, last = ts[0]["executed_at"][:10], ts[-1]["executed_at"][:10]
    kinds = list(dict.fromkeys(t["strategy"] for t in ts))
    head = (f"{p['label']} {p['status']} {first} to {last} {ts[0]['symbol']} "
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
        "from the trade log. Named positions are P<id> 'name' (with their "
        "campaign record); trades not assigned to one follow the user's rule: the minimal set of trades whose legs net to zero "
        "(#n). Each is closed; open; or "
        "unresolved (its leftover legs have already expired: opened before "
        "the log starts, or the expiration isn't logged yet). Each line gives "
        "dates, strategies, refs, and exact P&L (or net cash so far). A "
        "period selects positions whose latest trade falls in it, which for "
        "closed ones is the close date, i.e. realized P&L. Use for "
        "thinkorswim P&L by the user's position rule; get_pnl is the live "
        "broker account's own figures."
    ),
    "input_schema": {
        "type": "object",
        "properties": {**_FILTERS,
                       "status": {"type": "string",
                                  "enum": ["all", "closed", "open"],
                                  "description": "'open' includes unresolved."}},
    },
}, display=display_heading("Positions from the trade log", broker="schwab"))
def trade_log_positions(ctx, args):
    account, ask = _account(args.get("account"), ctx.active_agent)
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
    picked = [(name, p) for name in names for p in _all_positions(name)
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
