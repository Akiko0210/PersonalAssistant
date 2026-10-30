"""The working order ticket: one draft multi-leg order being built.

The ticket is what the user is *saying* ("SPX iron condor, move the put
side down 10"): legs with strike/type/expiration/side/size, a limit
price, and a time-in-force. It is:

- validated against the live chain (every strike must exist for its
  expiration, expirations must be tradable, strike-locked strategies keep
  their locked legs together);
- priced from live quotes (net mid, credit positive — Tasty-Web's sign
  convention);
- reduced to orderable legs (order_legs) that each broker variant turns
  into its own payload;
- persisted atomically to data/trading/<broker>/ticket.json so the voice
  agent and the dashboard edit the same draft, and a draft on one broker
  can never reach the other;
- fingerprinted, so the review->submit gate in orders.py can refuse to
  submit anything that changed after its dry-run review.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict

from lib.atomic_io import write_json_atomic
from lib.dates import with_weekday
from trading import config as tcfg
from trading import strategies as tstrat
from trading.models import Leg


def batch_quotes(get_quote, streamers):
    """[Quote-or-None] per streamer symbol (a None stays None). A get_quote
    with a `prefetch` attribute is handed every symbol first, so a strip of
    strikes or a 4-leg ticket is one Schwab request, one stream subscribe."""
    prefetch = getattr(get_quote, "prefetch", None)
    if prefetch:
        prefetch([s for s in streamers if s])
    return [get_quote(s) if s else None for s in streamers]


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
    account: str = None           # the account it is for; set at build/review

    # --- integrity -------------------------------------------------------

    def fingerprint(self):
        """Hash of everything that defines the order. Any change after a
        review invalidates the review (orders.py compares)."""
        content = {
            # The account is in the hash: a ticket reviewed on the IRA and
            # then moved to another account must be reviewed again there.
            "account": self.account,
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

    def locked_with(self, leg_index):
        """The legs that move with this one's strike: its strike-locked
        group, or itself."""
        for group in tstrat.STRIKE_LOCKED_GROUPS.get(self.strategy, []):
            if leg_index in group:
                return list(group)
        return [leg_index]

    def set_strike(self, leg_index, strike, chain):
        """Set a leg's strike (snapped to its expiration's chain), moving
        any strike-locked partners with it. Returns the affected indices."""
        moved = self.locked_with(leg_index)
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
        """[(leg, Quote-or-None)] via each leg's streamer symbol. A
        get_quote with a `prefetch` attribute (Schwab's REST cache) is
        handed every symbol first, so a 4-leg ticket is one request."""
        infos = [chain.option(leg.expiration, leg.strike, leg.cp[0])
                 for leg in self.legs]
        return list(zip(self.legs, batch_quotes(
            get_quote, [i.streamer if i else None for i in infos])))

    def net_mid(self, chain, get_quote):
        """Net mid across legs, per contract: positive = credit to you.
        None if any leg has no usable quote."""
        return self.net_of(self.leg_quotes(chain, get_quote))

    @staticmethod
    def net_of(quotes):
        """net_mid over leg_quotes() output already in hand."""
        total = 0.0
        for leg, q in quotes:
            mid = q.mid if q else None
            if mid is None:
                return None
            total += (mid if leg.side == "Short" else -mid) * leg.size
        return total

    @staticmethod
    def net_greeks(quotes):
        """The whole ticket's greeks over leg_quotes() output: each leg's
        per-share greek times its size, short legs negated (a long holder's
        sign). A greek any leg lacks is None — a sum over the legs that
        happened to arrive would be a wrong number said with confidence."""
        out = {}
        for name in ("delta", "gamma", "theta", "vega"):
            vals = [getattr(q, name) if q else None for _, q in quotes]
            out[name] = (None if None in vals else
                         sum((v if leg.side == "Long" else -v) * leg.size
                             for (leg, _), v in zip(quotes, vals)))
        return out

    def apply_net_price(self, net):
        """Store a signed net price using the credit-positive convention."""
        self.limit_price = round(abs(net), 2)
        self.price_effect = "Credit" if net > 0 else "Debit"
        self.touch()

    # --- orderable legs --------------------------------------------------

    def order_legs(self, chain):
        """[(occ_symbol, leg)] for an orderable ticket — what a broker's
        order_payload builds from. Raises ValueError if not orderable."""
        problems = self.validate(chain)
        if self.limit_price is None or self.limit_price <= 0:
            problems.append("No limit price set.")
        if self.price_effect not in ("Credit", "Debit"):
            problems.append("No price effect (credit/debit) set.")
        if problems:
            raise ValueError(" ".join(problems))
        return [(chain.option(leg.expiration, leg.strike, leg.cp[0]).occ, leg)
                for leg in self.legs]

    # --- description -----------------------------------------------------

    def describe(self):
        """Compact spoken/inline summary of the ticket."""
        if not self.legs:
            return f"Empty ticket on {self.underlying}."
        legs = "; ".join(
            f"{l.code} {l.size} {with_weekday(l.expiration)} {l.strike:g} {l.cp}"
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
                   review=d.get("review"), account=d.get("account"))


def save(ticket, broker):
    tcfg.ensure_dirs(broker)
    write_json_atomic(tcfg.ticket_path(broker), ticket.to_dict())


def load(broker):
    """The persisted ticket for `broker`, or None."""
    try:
        with open(tcfg.ticket_path(broker), encoding="utf-8") as f:
            return Ticket.from_dict(json.load(f))
    except (OSError, ValueError, TypeError, KeyError):
        return None


def clear(broker):
    try:
        tcfg.ticket_path(broker).unlink()
    except OSError:
        pass
