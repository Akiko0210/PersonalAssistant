"""The display window's building blocks: every card is made of these, so
every card looks the same and the user knows where to look (#28).

    card      a heading's worth of content: a total first, then sections
    section   one account (or one group): its heading, rows, notes
    row       one thing: an optional BADGE word (LONG, SHORT, FILLED), a
              label, a value on the right, and smaller detail lines
    total     the one figure that matters most, shown big at the top

A value's `kind` says what it is. Only PNL is coloured (gain or loss, and
the sign stays); cash, prices and quantities stay white, because a "-1"
short leg or "-9,000.00" paid is not a loss. Tools build their card in the
same loop as their spoken sentence, from the same formatted figures, so the
screen can't disagree with the voice. Plain dicts: they go straight to JSON.
"""

from datetime import date

from trading import symbols as tsym

PNL, MONEY = "pnl", "money"


def card(sections=(), total=None, notes=()):
    sections = list(sections)
    # The total goes first: with one section its total *is* the card's.
    if total is None and len(sections) == 1:
        total = sections[0].pop("total", None)
    return {"total": total, "sections": sections,
            "notes": [n for n in notes if n]}


def section(heading=None, rows=(), total=None, notes=()):
    return {"heading": heading, "rows": list(rows), "total": total,
            "notes": [n for n in notes if n]}


def row(label, value=None, kind=None, badge=None, lines=()):
    return {"label": label, "value": value, "kind": kind, "badge": badge,
            "lines": [x for x in lines if x]}


def total(label, value, kind=PNL):
    return {"label": label, "value": value, "kind": kind}


def contract(root, expiration, strike, cp):
    """'SPX 7750 Put · Fri Nov 6' — the order the eye wants: what, then when.
    The weekday is written out (the model got it wrong when left to work it
    out, #20) and the year only when it isn't this one."""
    try:
        d = date.fromisoformat(expiration)
        when = f"{d:%a %b} {d.day}" + (f" {d.year}" if d.year != date.today().year else "")
    except (TypeError, ValueError):
        when = expiration
    kind = {"C": "Call", "P": "Put"}.get(cp, cp)
    return f"{tsym.normalize_underlying(root)} {strike:g} {kind} · {when}"


def symbol_name(symbol):
    """A broker's order symbol ('SPXW  261106P07750000') as a contract name;
    anything that isn't an option comes back as it was."""
    o = tsym.parse_occ(symbol or "")
    return contract(o["root"], o["expiration"], o["strike"], o["cp"]) if o else symbol
