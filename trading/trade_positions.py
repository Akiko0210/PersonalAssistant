"""Group logged trades into positions and price them. Pure: no I/O.

Two kinds of position. An ASSIGNED one is explicit: its trades carry its
id (trade_log.positions), and `leg_effects` is the rule that grows it — a
fill joins the one position whose open leg it closes, so a roll chain
follows its diagonal instead of splitting off as its own "debit spread"
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

from collections import Counter, defaultdict
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


def leg_effects(trades):
    """One account's trades, oldest first -> {ref: [(leg, closed, holders)]}
    per leg: `closed` is the signed part of the leg that closes contracts
    earlier trades left open (the rest opens new ones), `holders` the refs of
    every earlier trade still holding the opposite side of that contract.
    Measured against the whole history, never a lookup's rows: on 2026-10-21
    a buyback whose short sat a day before the window read as a new long."""
    lots = defaultdict(list)     # contract -> [[ref, signed qty still open]]
    out = {}
    for t in trades:
        effects = []
        for e, k, cp, q in t["legs"]:
            held = lots[(t["symbol"], e, k, cp)]
            holders = [ref for ref, n in held if n * q < 0]
            left = q
            while held and held[0][1] * left < 0:    # FIFO, oldest lot first
                take = min(abs(left), abs(held[0][1]))
                held[0][1] += take if held[0][1] < 0 else -take
                left += -take if left > 0 else take
                if not held[0][1]:
                    held.pop(0)
            if left:
                held.append([t["ref"], left])
            effects.append(((e, k, cp, q), q - left, holders))
        out[t["ref"]] = effects
    return out


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
