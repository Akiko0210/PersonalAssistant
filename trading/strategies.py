"""The 19 strategy leg templates, ported from Tasty-Web lib/utils.ts.

Given a normalized chain and the live underlying price, build_legs()
places a named strategy's legs on real strikes: the ATM strike is located
by price, wings are offset by a "wing step" derived from actual strike
spacing (~5 points wide), far expirations snap to real dates, and every
strike is snapped to that expiration's own strike list.

Also ported: which strategies allow per-leg expirations
(MULTI_EXPIRY_STRATEGIES) and which legs must share a strike
(STRIKE_LOCKED_GROUPS) — ticket.py enforces both.

Pure module: no I/O, fully unit-testable with a fixture chain.
"""

from trading.models import Leg

STRATEGIES = [
    "Long Call", "Long Put", "Short Call", "Short Put",
    "Bull Call Spread", "Bear Put Spread", "Bull Put Spread", "Bear Call Spread",
    "Long Straddle", "Short Straddle", "Long Strangle", "Short Strangle",
    "Calendar Spread", "Diagonal Spread",
    "Iron Condor", "Iron Butterfly", "Long Butterfly", "Condor",
    "Double Diagonal",
]

MULTI_EXPIRY_STRATEGIES = {"Calendar Spread", "Diagonal Spread", "Double Diagonal"}

# Leg indices (in build_legs order) locked to one strike; editing any leg
# in a group moves the others with it.
STRIKE_LOCKED_GROUPS = {
    "Long Straddle": [[0, 1]],
    "Short Straddle": [[0, 1]],
    "Calendar Spread": [[0, 1]],
    "Iron Butterfly": [[1, 2]],
}


def match_strategy(name):
    """Resolve a spoken name to the canonical strategy name, or None.
    'iron condor', 'an Iron  Condor' -> 'Iron Condor'."""
    key = " ".join((name or "").lower().replace("-", " ").split())
    for s in STRATEGIES:
        if key == s.lower():
            return s
    # Forgiving containment match ("put credit spread" isn't here, but
    # "condor" alone shouldn't guess between Condor and Iron Condor).
    hits = [s for s in STRATEGIES if key and key in s.lower()]
    return hits[0] if len(hits) == 1 else None


def closest(values, target):
    """The member of `values` closest to target (values non-empty)."""
    return min(values, key=lambda v: abs(v - target))


def _wing_step(strikes, atm_idx):
    """Index steps that correspond to roughly a 5-point-wide spread."""
    if len(strikes) < 2:
        return 1
    i = max(1, min(len(strikes) - 1, atm_idx))
    spacing = strikes[i] - strikes[i - 1]
    if spacing <= 0:
        return 1
    return max(1, round(5 / spacing))


def build_legs(strategy, chain, current_price, expiration=None, size=1):
    """Legs for `strategy` on `chain`, centred on `current_price`.

    expiration picks the near expiration (defaults to the nearest); far
    legs (calendars/diagonals) go ~4 expirations further out, as in
    Tasty-Web. Returns [] for an unknown strategy.
    """
    if strategy not in STRATEGIES or not chain.expirations:
        return []
    near = expiration if expiration in chain.expirations else chain.expirations[0]
    near_strikes = chain.strikes.get(near) or []
    if not near_strikes:
        return []

    price = current_price if current_price is not None else \
        near_strikes[len(near_strikes) // 2]
    atm_idx = near_strikes.index(closest(near_strikes, price))
    ws = _wing_step(near_strikes, atm_idx)

    def s(offset):
        return near_strikes[max(0, min(len(near_strikes) - 1, atm_idx + offset))]

    # Far expiration: ~4 out from the chosen near one (or the last).
    ni = chain.expirations.index(near)
    far = chain.expirations[min(ni + 3, len(chain.expirations) - 1)]

    def in_exp(exp, target):
        exp_strikes = chain.strikes.get(exp) or []
        return closest(exp_strikes, target) if exp_strikes else target

    def leg(strike, cp, exp, side, qty=None):
        return Leg(strike=strike, cp=cp, expiration=exp, side=side,
                   size=qty if qty is not None else size)

    t = {
        "Long Call":   lambda: [leg(s(0), "Call", near, "Long")],
        "Long Put":    lambda: [leg(s(0), "Put", near, "Long")],
        "Short Call":  lambda: [leg(s(ws), "Call", near, "Short")],
        "Short Put":   lambda: [leg(s(-ws), "Put", near, "Short")],
        "Bull Call Spread": lambda: [leg(s(0), "Call", near, "Long"),
                                     leg(s(ws), "Call", near, "Short")],
        "Bear Put Spread":  lambda: [leg(s(0), "Put", near, "Long"),
                                     leg(s(-ws), "Put", near, "Short")],
        "Bull Put Spread":  lambda: [leg(s(-ws), "Put", near, "Short"),
                                     leg(s(-2 * ws), "Put", near, "Long")],
        "Bear Call Spread": lambda: [leg(s(ws), "Call", near, "Short"),
                                     leg(s(2 * ws), "Call", near, "Long")],
        "Long Straddle":  lambda: [leg(s(0), "Call", near, "Long"),
                                   leg(s(0), "Put", near, "Long")],
        "Short Straddle": lambda: [leg(s(0), "Call", near, "Short"),
                                   leg(s(0), "Put", near, "Short")],
        "Long Strangle":  lambda: [leg(s(ws), "Call", near, "Long"),
                                   leg(s(-ws), "Put", near, "Long")],
        "Short Strangle": lambda: [leg(s(ws), "Call", near, "Short"),
                                   leg(s(-ws), "Put", near, "Short")],
        "Calendar Spread": lambda: [leg(s(0), "Call", near, "Short"),
                                    leg(in_exp(far, s(0)), "Call", far, "Long")],
        "Diagonal Spread": lambda: [leg(s(ws), "Call", near, "Short"),
                                    leg(in_exp(far, s(0)), "Call", far, "Long")],
        "Iron Condor": lambda: [leg(s(-ws), "Put", near, "Short"),
                                leg(s(-2 * ws), "Put", near, "Long"),
                                leg(s(ws), "Call", near, "Short"),
                                leg(s(2 * ws), "Call", near, "Long")],
        "Iron Butterfly": lambda: [leg(s(-ws), "Put", near, "Long"),
                                   leg(s(0), "Put", near, "Short"),
                                   leg(s(0), "Call", near, "Short"),
                                   leg(s(ws), "Call", near, "Long")],
        "Long Butterfly": lambda: [leg(s(-ws), "Call", near, "Long", size),
                                   leg(s(0), "Call", near, "Short", 2 * size),
                                   leg(s(ws), "Call", near, "Long", size)],
        "Condor": lambda: [leg(s(-2 * ws), "Call", near, "Long"),
                           leg(s(-ws), "Call", near, "Short"),
                           leg(s(ws), "Call", near, "Short"),
                           leg(s(2 * ws), "Call", near, "Long")],
        "Double Diagonal": lambda: [leg(in_exp(far, s(-2 * ws)), "Put", far, "Long"),
                                    leg(s(-ws), "Put", near, "Short"),
                                    leg(s(ws), "Call", near, "Short"),
                                    leg(in_exp(far, s(2 * ws)), "Call", far, "Long")],
    }
    return t[strategy]()
