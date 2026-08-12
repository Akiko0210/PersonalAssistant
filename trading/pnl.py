"""Realized and unrealized P&L.

Realized: from /transactions. Every Trade (and expiring/assigned
Receive Deliver) is run through a per-instrument FIFO matcher: opens push
lots, closes pop them, and the realized event is the signed sum of the
matched money (net of fees — the API's net-value already is). Direction
is inferred from the action when it says so ("Buy to Open") and from
inventory sign otherwise (futures use bare Buy/Sell); a close larger than
the inventory splits, with the remainder opening the other way. Futures
daily cash settlement ("Futures Settlement" / "Mark to Market" money
movements) is added per-underlying without FIFO — that's how futures
P&L actually realizes.

The period filter applies to *closes*: transactions are fetched with a
lookback before the period start (PNL_OPEN_LOOKBACK_DAYS) so opens that
precede the period still match. An open older than the lookback shows up
as an unmatched close (flagged in the report note) rather than silently
wrong numbers.

Unrealized: positions have no live mark (close-price is yesterday's), so
the caller supplies mark_fn(position) -> price from streaming quotes or a
REST snapshot; positions without a mark are listed with unrealized None.

Pure module: fully testable with fixture transactions/positions.
"""

from collections import deque
from datetime import datetime, timezone

from trading.models import PnLReport, Position, RealizedEvent

SETTLEMENT_SUBTYPES = {"Futures Settlement", "Mark to Market"}
CLOSING_RD_SUBTYPES = {"Expiration", "Assignment", "Exercise",
                       "Cash Settled Assignment", "Cash Settled Exercise"}


def _num(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def _signed(tx, key):
    """Signed money: Credit = +, Debit = -."""
    v = _num(tx.get(key), None)
    if v is None:
        return 0.0
    return -v if tx.get(f"{key}-effect") == "Debit" else v


def _ts(tx):
    return tx.get("executed-at") or tx.get("transaction-date") or ""


def _in_period(iso, start, end):
    d = (iso or "")[:10]
    return (not start or d >= start) and (not end or d <= end)


def parse_positions(items):
    out = []
    for p in items:
        qty = _num(p.get("quantity"))
        if p.get("quantity-direction") == "Short":
            qty = -qty
        if qty == 0:
            continue
        out.append(Position(
            symbol=p.get("symbol", ""),
            underlying=p.get("underlying-symbol", p.get("symbol", "")),
            instrument_type=p.get("instrument-type", ""),
            quantity=qty,
            average_open_price=_num(p.get("average-open-price")),
            multiplier=_num(p.get("multiplier"), 1.0) or 1.0,
            close_price=_num(p.get("close-price"), None),
            streamer_symbol=p.get("streamer-symbol") or None,
            realized_today=_num(p.get("realized-today")),
            raw=p))
    return out


def _direction(tx, inventory_qty):
    """(+1 buy / -1 sell, opening?) for a transaction."""
    action = (tx.get("action") or "").lower()
    buy = action.startswith("buy")
    sign = 1 if buy else -1
    if "to open" in action:
        return sign, True
    if "to close" in action:
        return sign, False
    # Bare Buy/Sell (futures): same direction as inventory (or flat) opens.
    if inventory_qty == 0 or (inventory_qty > 0) == buy:
        return sign, True
    return sign, False


def realized_events(transactions):
    """FIFO-match closes to opens.

    Returns (events, fees_total, settlements, unmatched_count) where
    settlements is {underlying: [(iso_ts, amount), ...]} of futures cash
    settlement, and unmatched_count counts closes with no known open
    (opens older than the fetch window).
    """
    events, settlements = [], {}
    fees_total, unmatched = 0.0, 0
    lots = {}  # instrument symbol -> deque of [signed_qty, signed_money]

    for tx in sorted(transactions, key=_ts):
        ttype = tx.get("transaction-type")
        sub = tx.get("transaction-sub-type", "")
        underlying = tx.get("underlying-symbol") or tx.get("symbol", "")
        if ttype == "Money Movement":
            if sub in SETTLEMENT_SUBTYPES:
                settlements.setdefault(underlying, []).append(
                    (_ts(tx), _signed(tx, "net-value")))
            continue
        if ttype == "Receive Deliver" and sub not in CLOSING_RD_SUBTYPES:
            continue
        if ttype not in ("Trade", "Receive Deliver"):
            continue

        symbol = tx.get("symbol", "")
        qty = abs(_num(tx.get("quantity")))
        money = _signed(tx, "net-value")
        fees_total += abs(_signed(tx, "value") - money)
        if qty == 0:
            continue
        q = lots.setdefault(symbol, deque())
        inventory = sum(l[0] for l in q)

        if ttype == "Receive Deliver":
            # Expiration/assignment closes whatever is held.
            sign, opening = (-1 if inventory > 0 else 1), False
        else:
            sign, opening = _direction(tx, inventory)

        remaining_qty, remaining_money = qty, money
        if not opening:
            matched_money = 0.0
            matched_qty = 0.0
            while remaining_qty > 1e-9 and q:
                lot = q[0]
                lot_qty = abs(lot[0])
                take = min(lot_qty, remaining_qty)
                share = lot[1] * (take / lot_qty)
                matched_money += share
                lot[0] -= take * (1 if lot[0] > 0 else -1)
                lot[1] -= share
                matched_qty += take
                remaining_qty -= take
                if abs(lot[0]) < 1e-9:
                    q.popleft()
            if matched_qty > 0:
                close_money = money * (matched_qty / qty)
                events.append(RealizedEvent(
                    underlying=underlying, symbol=symbol,
                    closed_at=_ts(tx), quantity=matched_qty,
                    proceeds=round(close_money, 2),
                    cost=round(-matched_money, 2),
                    pnl=round(close_money + matched_money, 2)))
                remaining_money = money - close_money
            if remaining_qty > 1e-9:
                if matched_qty == 0:
                    unmatched += 1
                else:
                    # Close overshot the inventory: remainder opens the
                    # other way.
                    q.append([remaining_qty * sign, remaining_money])
                continue
        else:
            q.append([qty * sign, money])
    return events, round(fees_total, 2), settlements, unmatched


def report(transactions, start, end, positions=None, mark_fn=None):
    """PnLReport for closes (and settlements) inside [start, end]
    (ISO dates, inclusive), plus a current unrealized snapshot when
    positions are given."""
    events, fees, settlements, unmatched = realized_events(transactions)
    rep = PnLReport(start=start or "", end=end or "", fees=fees)

    for ev in events:
        if not _in_period(ev.closed_at, start, end):
            continue
        rep.events.append(ev)
        rep.realized += ev.pnl
        rep.by_underlying[ev.underlying] = round(
            rep.by_underlying.get(ev.underlying, 0.0) + ev.pnl, 2)
    for underlying, entries in settlements.items():
        amount = round(sum(a for ts, a in entries
                           if _in_period(ts, start, end)), 2)
        if amount:
            rep.realized += amount
            rep.by_underlying[underlying] = round(
                rep.by_underlying.get(underlying, 0.0) + amount, 2)
    rep.realized = round(rep.realized, 2)

    if positions is not None:
        total, missing = 0.0, 0
        for pos in positions:
            mark = mark_fn(pos) if mark_fn else None
            u = pos.unrealized(mark)
            rep.unrealized_by_position.append({
                "symbol": pos.symbol, "underlying": pos.underlying,
                "quantity": pos.quantity, "mark": mark,
                "unrealized": round(u, 2) if u is not None else None})
            if u is None:
                missing += 1
            else:
                total += u
        rep.unrealized = round(total, 2)
        if missing:
            rep.note += (f"{missing} position(s) had no live mark and are "
                         "not in the unrealized total. ")
    if unmatched:
        rep.note += (f"{unmatched} closing trade(s) had no matching open "
                     "inside the fetched history; their P&L is excluded. ")
    return rep
