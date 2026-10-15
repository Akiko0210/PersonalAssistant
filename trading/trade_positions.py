"""Group logged trades into positions and price them. Pure: no I/O.

Two kinds of position. An ASSIGNED one is explicit: its trades carry its
id (trade_log.positions), and `closing` is the rule that grows it — a fill
joins the one position whose open leg it closes, so a roll chain follows
its diagonal instead of splitting off as its own "debit spread"
(2026-10-15). An unassigned trade still falls under the user's rule below.

The user's rule: the minimal set of trades whose legs net to
zero, meaning no smaller subset of it also nets to zero. Trades are walked
oldest first. Each new trade closes the smallest zero-sum subset of the
open trades that includes it; ties go to the earliest trades (FIFO).

Every set closed this way is minimal. Before the new trade arrives, no open
subset nets to zero (by induction), so a zero-sum part of the closed set
without the new trade can't exist, and a part with it would be smaller than
the smallest.

Whatever is left open is grouped by shared contracts:
- "open" while any leg still has time to run;
- "unresolved" once every leftover leg has expired. Either the position was
  opened before the log starts, or its expiration isn't logged yet (alerts
  don't report expirations; the next statement import does).
"""

from collections import Counter
from datetime import date
from itertools import combinations

from trading import config as tcfg


def net_legs(trades):
    """{(symbol, expiration, strike, type): signed quantity} for the legs
    that don't net to zero across `trades`."""
    net = Counter()
    for t in trades:
        for e, k, cp, q in t["legs"]:
            net[(t["symbol"], e, k, cp)] += q
    return {c: q for c, q in net.items() if q}


def totals(trades):
    """Cash, fees and net of a set of trades. Fees not yet known (alert
    fills) are counted in `pending`, never guessed."""
    cash = round(sum(t["amount"] for t in trades), 2)
    fees = round(sum(t["fees"] or 0.0 for t in trades), 2)
    return {"cash": cash, "fees": fees, "net": round(cash + fees, 2),
            "pending": sum(t["fees"] is None for t in trades)}


def closing(trade, open_by_position):
    """Ids of the positions in {id: net legs} that `trade` closes or partly
    closes a leg of: a leg of the opposite sign on the same contract. The
    caller passes one account's positions, so another account never
    matches."""
    return [pid for pid, net in open_by_position.items()
            if any(net.get((trade["symbol"], e, k, cp), 0) * q < 0
                   for e, k, cp, q in trade["legs"])]


def describe(trades, today=None):
    """{status, open_legs, totals} of one position's trades: "closed" when
    every leg nets to zero, "open" while a leftover leg has time to run,
    "unresolved" once every leftover has expired."""
    today = (today or date.today()).isoformat()
    legs = net_legs(trades)
    status = ("closed" if not legs else
              "open" if any(e >= today for _, e, _, _ in legs) else
              "unresolved")
    return {"status": status, "open_legs": legs, **totals(trades)}


def _linked(seed, pool, contracts):
    """Indices in `pool` that share a contract with trade `seed`, directly or
    through each other, in time order."""
    keys, rest, found, grew = set(contracts[seed]), list(pool), [], True
    while grew:
        grew = False
        for i in list(rest):
            if keys & contracts[i]:
                keys |= contracts[i]
                found.append(i)
                rest.remove(i)
                grew = True
    return sorted(found)


def _smallest_zero(i, others, trades):
    """The smallest subset of `others` that nets to zero together with trade
    i, or None. `others` is in time order, so combinations() yields the
    earliest trades first among subsets of equal size: FIFO on ties."""
    for size in range(len(others) + 1):
        for combo in combinations(others, size):
            if not net_legs([trades[j] for j in (i, *combo)]):
                return [i, *combo]
    return None


def group_positions(trades, today=None):
    """One account's trades, oldest first (as trade_log.query returns them),
    -> positions numbered by first trade. Each position is {number, status,
    trades, open_legs, tangled} plus totals(). `tangled` means the search
    hit its cap and left these trades grouped by shared contracts instead."""
    contracts = [{(t["symbol"], e, k, cp) for e, k, cp, _ in t["legs"]}
                 for t in trades]
    still_open, closed, tangled = [], [], set()
    for i in range(len(trades)):
        others = _linked(i, still_open, contracts)
        still_open.append(i)
        if len(others) > tcfg.POSITION_SEARCH_MAX_TRADES:
            tangled.update(others + [i])
            continue
        hit = _smallest_zero(i, others, trades)
        if hit:
            closed.append(sorted(hit))
            still_open = [j for j in still_open if j not in hit]
    groups = closed
    while still_open:
        seed = still_open.pop(0)
        g = [seed] + _linked(seed, still_open, contracts)
        still_open = [j for j in still_open if j not in g]
        groups.append(sorted(g))
    groups.sort()
    positions = []
    for n, g in enumerate(groups, 1):
        ts = [trades[j] for j in g]
        positions.append({"number": n, "trades": ts,
                          "tangled": bool(tangled & set(g)),
                          **describe(ts, today)})
    return positions
