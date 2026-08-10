"""The working order ticket: one draft multi-leg order being built.

The ticket is what the user is *saying* ("SPX iron condor, move the put
side down 10"): legs with strike/type/expiration/side/size, a limit
price, and a time-in-force. It is:

- validated against the live chain (every strike must exist for its
  expiration, expirations must be tradable, strike-locked strategies keep
  their locked legs together);
- priced from live quotes (net mid, credit positive — Tasty-Web's sign
  convention);
- serialized to the exact API payload (port of buildOrderPayload);
- persisted atomically to data/trading/ticket.json so the voice agent and
  the dashboard edit the same draft;
- fingerprinted, so the review->submit gate in orders.py can refuse to
  submit anything that changed after its dry-run review.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict

from lib.atomic_io import write_json_atomic
from trading import config as tcfg
from trading import strategies as tstrat
from trading.models import Leg


@dataclass
class Ticket:
    underlying: str
    strategy: str = None          # canonical name, or None for freeform
    legs: list = field(default_factory=list)     # [Leg]
    tif: str = "Day"              # "Day" | "GTC"
    limit_price: float = None     # absolute; None until set/priced
    price_effect: str = None      # "Credit" | "Debit"
    updated_at: float = 0.0
    review: dict = None           # stamped by orders.review(); see fingerprint

    # --- integrity -------------------------------------------------------

    def fingerprint(self):
        """Hash of everything that defines the order. Any change after a
        review invalidates the review (orders.py compares)."""
        content = {
            "underlying": self.underlying, "tif": self.tif,
            "price": self.limit_price, "effect": self.price_effect,
            "legs": [[l.strike, l.cp, l.expiration, l.side, l.size,
                      l.open_close] for l in self.legs],
        }
        return hashlib.sha1(
            json.dumps(content, sort_keys=True).encode()).hexdigest()[:16]

    def validate(self, chain):
        """List of human-readable problems; empty means orderable
        (price still checked separately by payload())."""
        problems = []
        if not self.legs:
            problems.append("The ticket has no legs.")
        if len(self.legs) > 4:
            problems.append("Option orders allow at most 4 legs.")
        expirations = set()
        for i, leg in enumerate(self.legs, 1):
            where = f"Leg {i} ({leg.code} {leg.strike} {leg.cp})"
            if leg.cp not in ("Call", "Put"):
                problems.append(f"{where}: type must be Call or Put.")
            if leg.side not in ("Long", "Short"):
                problems.append(f"{where}: side must be Long or Short.")
            if not isinstance(leg.size, int) or leg.size < 1:
                problems.append(f"{where}: size must be a whole number >= 1.")
            if leg.expiration not in chain.expirations:
                problems.append(
                    f"{where}: {leg.expiration} is not a listed expiration "
                    f"for {chain.underlying}.")
                continue
            expirations.add(leg.expiration)
            strikes = chain.strikes.get(leg.expiration, [])
            if leg.strike not in strikes:
                near = tstrat.closest(strikes, leg.strike) if strikes else None
                hint = f" Closest available is {near:g}." if near is not None else ""
                problems.append(
                    f"{where}: strike {leg.strike:g} does not exist for "
                    f"{leg.expiration}.{hint}")
            elif chain.option(leg.expiration, leg.strike, leg.cp[0]) is None:
                problems.append(f"{where}: no such contract in the chain.")
        if (len(expirations) > 1 and self.strategy
                and self.strategy not in tstrat.MULTI_EXPIRY_STRATEGIES):
            problems.append(
                f"A {self.strategy} uses a single expiration; these legs "
                f"span {len(expirations)}.")
        return problems

    # --- editing ---------------------------------------------------------

    def set_strike(self, leg_index, strike, chain):
        """Set a leg's strike (snapped to its expiration's chain), moving
        any strike-locked partners with it. Returns the affected indices."""
        groups = tstrat.STRIKE_LOCKED_GROUPS.get(self.strategy, [])
        moved = [leg_index]
        for group in groups:
            if leg_index in group:
                moved = list(group)
                break
        for i in moved:
            leg = self.legs[i]
            strikes = chain.strikes.get(leg.expiration) or []
            leg.strike = tstrat.closest(strikes, strike) if strikes else strike
        self.touch()
        return moved

    def touch(self):
        self.updated_at = time.time()
        # Any edit invalidates a pending review (submit re-checks anyway;
        # clearing here keeps what the dashboard shows honest).
        if self.review and self.review.get("fingerprint") != self.fingerprint():
            self.review = None

    # --- pricing ---------------------------------------------------------

    def leg_quotes(self, chain, get_quote):
        """[(leg, Quote-or-None)] via each leg's streamer symbol."""
        out = []
        for leg in self.legs:
            info = chain.option(leg.expiration, leg.strike, leg.cp[0])
            out.append((leg, get_quote(info.streamer) if info else None))
        return out

    def net_mid(self, chain, get_quote):
        """Net mid across legs, per contract: positive = credit to you.
        None if any leg has no usable quote."""
        total = 0.0
        for leg, q in self.leg_quotes(chain, get_quote):
            mid = q.mid if q else None
            if mid is None:
                return None
            total += (mid if leg.side == "Short" else -mid) * leg.size
        return total

    def apply_net_price(self, net):
        """Store a signed net price using the credit-positive convention."""
        self.limit_price = round(abs(net), 2)
        self.price_effect = "Credit" if net > 0 else "Debit"
        self.touch()

    # --- API payload -----------------------------------------------------

    def payload(self, chain):
        """The POST /orders body. Raises ValueError if not orderable."""
        problems = self.validate(chain)
        if self.limit_price is None or self.limit_price <= 0:
            problems.append("No limit price set.")
        if self.price_effect not in ("Credit", "Debit"):
            problems.append("No price effect (credit/debit) set.")
        if problems:
            raise ValueError(" ".join(problems))
        legs = []
        for leg in self.legs:
            occ = chain.option(leg.expiration, leg.strike, leg.cp[0]).occ
            legs.append({
                "instrument-type": "Future Option" if occ.startswith("./")
                                   else "Equity Option",
                "symbol": occ,
                "quantity": leg.size,
                "action": leg.action,
            })
        return {
            "time-in-force": self.tif,
            "order-type": "Limit",
            "price": f"{self.limit_price:.2f}",
            "price-effect": self.price_effect,
            "source": "voice-agent",
            "legs": legs,
        }

    # --- description -----------------------------------------------------

    def describe(self):
        """Compact spoken/inline summary of the ticket."""
        if not self.legs:
            return f"Empty ticket on {self.underlying}."
        legs = "; ".join(
            f"{l.code} {l.size} {l.expiration} {l.strike:g} {l.cp}"
            for l in self.legs)
        price = (f" at {self.limit_price:.2f} {self.price_effect}"
                 if self.limit_price else ", no price set")
        name = self.strategy or "custom"
        return f"{self.underlying} {name}: {legs}{price}, {self.tif}."

    # --- persistence -----------------------------------------------------

    def to_dict(self):
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d):
        legs = [Leg(**l) for l in d.get("legs", [])]
        return cls(underlying=d.get("underlying", ""),
                   strategy=d.get("strategy"), legs=legs,
                   tif=d.get("tif", "Day"),
                   limit_price=d.get("limit_price"),
                   price_effect=d.get("price_effect"),
                   updated_at=d.get("updated_at", 0.0),
                   review=d.get("review"))


def save(ticket):
    tcfg.ensure_dirs()
    write_json_atomic(tcfg.TICKET_PATH, ticket.to_dict())


def load():
    """The persisted ticket, or None."""
    try:
        with open(tcfg.TICKET_PATH, encoding="utf-8") as f:
            return Ticket.from_dict(json.load(f))
    except (OSError, ValueError, TypeError, KeyError):
        return None


def clear():
    try:
        tcfg.TICKET_PATH.unlink()
    except OSError:
        pass
