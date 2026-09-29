"""Schwab/thinkorswim account-statement CSV -> trade-log orders.

Only the Cash Balance section is read. It lists every cash event in the
account, one row per fill (TRD) or expiration (RAD), each with its REF #,
fees and amount. The statement's later sections are filtered by symbol or
carry no REF #.

Before anything reaches the log, the rows are checked against the
statement's own TOTAL row: a misread amount must refuse the import, not
skew every P&L after it.
"""

import csv
import re

from trading import symbols as tsym
from trading import trade_log

_HEADER_RE = re.compile(
    r"Account Statement for (?P<account>\S+) .*?since (?P<since>\d+/\d+/\d+)")
# "Removed due to Expiration PUT ... EXP: -1.0 .SPXW260410P6500": the signed
# change to the position, then the option's symbol.
_EXPIRY_RE = re.compile(
    r"EXP: (?P<qty>-?\d+(?:\.\d+)?) \.(?P<root>[A-Z]+)(?P<ymd>\d{6})"
    r"(?P<cp>[CP])(?P<strike>\d+(?:\.\d+)?)")


def _money(text):
    """'1,400.00', '-2.28', '($217.64)', '$12,440.00', '' -> float."""
    s = text.replace(",", "").replace("$", "").strip()
    if s.startswith("(") and s.endswith(")"):
        return -float(s[1:-1])
    return float(s) if s else 0.0


def _date(mdy):
    """'1/23/26' -> '2026-01-23'."""
    m, d, y = mdy.split("/")
    return f"20{y[-2:]}-{int(m):02d}-{int(d):02d}"


def _merge_legs(a, b):
    net = {}
    for e, k, cp, q in a + b:
        net[(e, k, cp)] = net.get((e, k, cp), 0) + q
    return [(*c, q) for c, q in net.items()]


def _row(account, when, desc):
    """One TRD or RAD row -> its order fields and legs, or None when the
    line can't be read."""
    fill = trade_log.parse_fill(desc)
    if fill:
        return trade_log.order(fill, account=account, executed_at=when,
                               description=desc, source="statement")
    m = _EXPIRY_RE.search(desc)
    if not m:
        return None
    y = m["ymd"]
    return {"account": account, "executed_at": when,
            "symbol": tsym.normalize_underlying(m["root"]),
            "strategy": "expiration", "quantity": None, "price": None,
            "fills": 1, "description": desc, "source": "statement",
            "legs": [(f"20{y[:2]}-{y[2:4]}-{y[4:]}", float(m["strike"]),
                      m["cp"], round(float(m["qty"])))]}


def parse(text):
    """Statement text -> {account, since, last, orders, skipped, problems}.
    `orders` are trade-log rows oldest first, with split fills merged by
    REF #. `skipped` counts non-trade cash rows (interest, journals).
    Any `problems` means import nothing."""
    lines = text.splitlines()
    head = _HEADER_RE.match(lines[0]) if lines else None
    if not head:
        return {"problems": ["it has no 'Account Statement for' header, so "
                             "it isn't a thinkorswim account statement"]}
    account = head["account"]
    if "Cash Balance" not in lines:
        return {"problems": ["it has no Cash Balance section"]}
    orders, problems, skipped = {}, [], 0
    sums, total = [0.0, 0.0, 0.0], None      # misc fees, commissions, amount
    for row in csv.reader(lines[lines.index("Cash Balance") + 2:]):
        if len(row) < 8:
            break
        if row[4] == "TOTAL":
            total = [_money(x) for x in row[5:8]]
            break
        day, clock, kind, ref, desc = row[:5]
        cash = [_money(x) for x in row[5:8]]
        sums = [a + b for a, b in zip(sums, cash)]
        if kind not in ("TRD", "RAD"):       # cash, but no position change
            skipped += 1
            continue
        o = _row(account, f"{_date(day)}T{clock}", desc)
        if o is None:
            problems.append(f"can't read the {kind} line '{desc}'")
            continue
        ref = ref.strip('="')
        fees, amount = cash[0] + cash[1], cash[2]
        if ref not in orders:
            orders[ref] = o | {"ref": ref, "amount": amount, "fees": fees}
            continue
        prev = orders[ref]                  # a split fill of the same order
        prev["amount"] += amount
        prev["fees"] += fees
        prev["fills"] += 1
        if prev["quantity"] is not None:
            prev["quantity"] += o["quantity"]
        prev["legs"] = _merge_legs(prev["legs"], o["legs"])
    if total is None:
        problems.append("there's no TOTAL row to check the rows against")
    elif any(abs(a - b) > 0.005 for a, b in zip(sums, total)):
        problems.append(
            "its rows don't add up to its TOTAL row (fees, commissions, "
            f"amount: rows {', '.join(f'{x:,.2f}' for x in sums)}; "
            f"TOTAL {', '.join(f'{x:,.2f}' for x in total)})")
    rows = sorted(orders.values(), key=lambda o: (o["executed_at"], o["ref"]))
    for o in rows:
        o["amount"], o["fees"] = round(o["amount"], 2), round(o["fees"], 2)
    return {"account": account, "since": _date(head["since"]),
            "last": rows[-1]["executed_at"] if rows else None,
            "orders": rows, "skipped": skipped, "problems": problems}
