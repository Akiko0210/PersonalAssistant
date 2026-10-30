"""Voice tools for real trading on two brokers. See TRADING_PLAN.md.

Design rules:
- Every handler degrades to a friendly spoken string — a missing .env key,
  a sandbox without quotes, an expired Schwab sign-in, or a broker rejection
  must never crash the conversation loop (dispatch would catch it, but a
  clear sentence beats "Tool error: ...").
- The broker is the persona's, never the model's choice: Tom's hat says
  Schwab, Bonny's says tastytrade (brain/agents.py `brokers`). A tool asked
  for the other broker refuses and names who has it. Linda reads both; her
  read-only tools take `broker` to focus on one and otherwise cover each
  separately — numbers are only ever added in the sentence, never in storage.
- The dangerous direction is gated twice: submit_order requires a prior
  review (dry run) of the exact current ticket — enforced by fingerprint
  in trading.orders, shared with the dashboard — and its `confirmed` flag,
  which the model is instructed to set only after the user has heard the
  review and explicitly said to go ahead.
- The working ticket is persisted per broker (data/trading/<broker>/
  ticket.json), so a draft built by voice is the same draft shown in the
  dashboard's Trading page, and a Schwab draft can never reach tastytrade.
- Accounts work like brokers, one level down: a login may reach several
  (the IRA and "red"), the persona works on the one the user chose with
  set_account (trading/accounts.py, persisted), every answer names its
  account, and `account` reaches another one — or "all" — without moving
  the choice. Orders act on exactly one account, named in the review.
"""

import time
from datetime import date, datetime, timedelta

from brain import agents
from display import card as blocks
from lib.dates import PERIODS, period_range, with_weekday, zone_name
from tools import tool
from trading import accounts as taccounts
from trading import config as tcfg
from trading import main as tmain
from trading import strategies as tstrat
from trading import symbols as tsym
from trading import ticket as tticket

BROKER_ARG = {"type": "string", "enum": tmain.keys(),
              "description": "Only when the user names a broker; otherwise "
                             "your own broker (or, for a reader of both, each)."}
ACCOUNT_ARG = {"type": "string",
               "description": "Only when the user names an account for this "
                              "question: its nickname ('IRA', 'red'), last "
                              "digits ('699'), or 'all'. Omit to use the account "
                              "chosen with set_account."}
_ALL = {"all", "all accounts", "every account", "both", "both accounts"}


def _fmt(x, none="unknown"):
    return f"{x:,.2f}" if isinstance(x, (int, float)) else none


def _pnl(x, none="unknown"):
    """A P&L figure, always signed: the display window colours it, and the
    sign is what says gain or loss to someone who can't tell the colours."""
    return f"{x:+,.2f}" if isinstance(x, (int, float)) else none


def display_heading(what, broker=None):
    """A tool's display= heading: `what`, then who answered on which broker
    ('Open positions — Tom, Schwab'). `broker` pins a fixed one (the trade
    log is Schwab's); otherwise the one asked for, or the persona's own."""
    def heading(ctx, args):
        hat = agents.AGENTS.get(ctx.active_agent or "") or {}
        keys = ([broker] if broker else
                [tmain.normalize(args["broker"])] if args.get("broker") else
                _mine(ctx))
        brokers = " and ".join(tmain.label(k) for k in keys if k)
        who = ", ".join(x for x in (hat.get("name"), brokers) if x)
        return f"{what} — {who}" if who else what
    return heading


# --- which broker ----------------------------------------------------------

def _mine(ctx):
    hat = agents.AGENTS.get(ctx.active_agent or "") or {}
    return tuple(hat.get("brokers", ()))


def _owner_of(broker):
    """The persona that trades `broker` alone — who to send the user to."""
    for key, hat in agents.AGENTS.items():
        if tuple(hat.get("brokers", ())) == (broker,):
            return hat["name"]
    return None


def _brokers(ctx, args):
    """([broker_key, ...], None) or (None, spoken-problem): every broker a
    read-only call covers — the one asked for, or all of the persona's."""
    mine = _mine(ctx)
    if not mine:
        return None, "This assistant doesn't trade — ask Tom (Schwab) or Bonny (tastytrade)."
    asked = args.get("broker")
    if not asked:
        return list(mine), None
    key = tmain.normalize(asked)
    if not key:
        return None, f"I don't know a broker called '{asked}'."
    if key not in mine:
        me = agents.AGENTS[ctx.active_agent]["name"]
        owner = _owner_of(key)
        return None, (f"{me} works on {tmain.label(mine[0])} only; "
                      f"{tmain.label(key)} is {owner}'s — ask {owner}." if owner else
                      f"{me} doesn't trade on {tmain.label(key)}.")
    return [key], None


def _broker(ctx, args):
    """(broker_key, None) or (None, problem): the ONE broker a mutating call
    acts on. A persona with two brokers must be told which."""
    keys, problem = _brokers(ctx, args)
    if problem:
        return None, problem
    if len(keys) > 1:
        return None, "Which broker — Schwab or tastytrade?"
    return keys[0], None


def _engine_for(broker):
    """(engine, None) or (None, spoken-problem). Import inside the call so
    a missing `requests` dependency degrades to a sentence too."""
    try:
        from trading.engine import get_engine
        return get_engine(broker)
    except Exception as e:
        return None, f"Trading engine failed to start: {e}"


def _engine(ctx, args):
    broker, problem = _broker(ctx, args)
    if problem:
        return None, problem
    return _engine_for(broker)


def _engines(ctx, args):
    """[(broker, engine-or-None, problem-or-None)] for a read across brokers."""
    keys, problem = _brokers(ctx, args)
    if problem:
        return None, problem
    return [(k, *_engine_for(k)) for k in keys], None


def _load_ticket(broker):
    tk = tticket.load(broker)
    return tk if tk and tk.legs is not None else None


def _ticket_chain(eng):
    """(ticket, chain, None) or (None, None, spoken-problem). The chain
    reaches every leg: one built on a fetched stretch outlives the cache."""
    tk = _load_ticket(eng.key)
    if tk is None:
        return None, None, ("There is no order ticket yet — build a "
                            "strategy first.")
    try:
        chain = eng.chain(tk.underlying)
    except Exception as e:
        return None, None, f"Couldn't load the {tk.underlying} chain: {e}"
    for leg in tk.legs:
        if leg.strike not in (chain.strikes.get(leg.expiration) or []):
            _strike_listed(eng, chain, leg.expiration, leg.strike)  # validate() says what's off
    return tk, chain, None


# --- reaching past the loaded chain ------------------------------------------
# Schwab's chain is loaded a week deep at 80 strikes (#32): the whole one is
# too big for its API. A later date or a far strike is fetched when asked
# for; only one that isn't listed at all is refused — never snapped to the
# edge of whatever happens to be loaded.

def _expiration(eng, chain, said):
    """(expiration, None) or (None, spoken-problem): the listed expiration
    closest to the ISO date `said`, no further than CHAIN_REACH_DAYS."""
    target = date.fromisoformat(said)
    reach = timedelta(days=tcfg.CHAIN_REACH_DAYS)
    if not chain.expirations or said > chain.expirations[-1]:
        # Never from before today: Schwab answers 400 "Check Param Values" to
        # a window that starts in the past, which broke every date 8-13 days
        # out (2026-10-27, #42) — and nothing before today can be traded.
        start = max(target - reach, date.today())
        try:
            eng.extend_chain(chain, start.isoformat(),
                             (target + reach).isoformat())
        except Exception as e:
            return None, f"Couldn't load {chain.underlying} expirations near {said}: {e}"
    near = min(chain.expirations, default=None,
               key=lambda d: abs(date.fromisoformat(d) - target))
    # No "nearest" offered: only the loaded week and this stretch were
    # looked at, so a listed date between them would go unmentioned.
    if near is None or abs(date.fromisoformat(near) - target) > reach:
        return None, (f"{chain.underlying} has no expiration listed within "
                      f"{tcfg.CHAIN_REACH_DAYS} days of {with_weekday(said)}. "
                      "Name another date.")
    return near, None


def _strike_listed(eng, chain, expiration, strike):
    """None, or why `strike` can't be had on `expiration`. Outside the
    loaded strikes, that expiration's every strike is fetched first."""
    strikes = chain.strikes.get(expiration) or []
    if not strikes or not strikes[0] <= strike <= strikes[-1]:
        try:
            eng.extend_chain(chain, expiration, expiration, every_strike=True)
        except Exception as e:
            return f"Couldn't load the {chain.underlying} {expiration} strikes: {e}"
        strikes = chain.strikes.get(expiration) or []
    if not strikes:
        return f"{chain.underlying} has no expiration on {with_weekday(expiration)}."
    if not strikes[0] <= strike <= strikes[-1]:
        return (f"{chain.underlying} {with_weekday(expiration)} strikes run from "
                f"{strikes[0]:g} to {strikes[-1]:g}; {strike:g} isn't listed.")
    return None


def _ticket_card(tk, notes=(), extra=()):
    """The ticket on the display: the limit price first, then every leg by
    number, side and contract — the number is what the user says to adjust
    it ("move leg 2 down"). `extra` rows follow the legs (a review's fees)."""
    price = (f"{tk.limit_price:.2f} {tk.price_effect}" if tk.limit_price
             else "not set")
    legs = [blocks.row(blocks.contract(tk.underlying, l.expiration, l.strike, l.cp),
                       badge=l.side.upper(),
                       lines=[f"Leg {i}: {l.action}, {l.size} "
                              f"contract{'s' * (l.size > 1)}"])
            for i, l in enumerate(tk.legs, 1)]
    return blocks.card([blocks.section(
        f"{tk.underlying} {tk.strategy or 'custom'} · {tk.tif}",
        legs + [r for r in extra if r],
        total=blocks.total("Limit price", price, blocks.MONEY))], notes=notes)


def _engine_and_ticket(ctx, args=None):
    """(eng, ticket, chain, None) or (None, None, None, spoken-problem) —
    the engine + current-ticket preamble the ticket-mutating tools share."""
    eng, err = _engine(ctx, args or {})
    if err:
        return None, None, None, err
    tk, chain, problem = _ticket_chain(eng)
    if problem:
        return None, None, None, problem
    return eng, tk, chain, None


def _accounts(ctx, eng, args):
    """([account, ...], None) or (None, spoken-problem): the accounts one
    question covers — the one named, all of them, or the persona's chosen
    one. With several accounts and no choice yet it asks, once."""
    try:
        numbers = eng.accounts()
    except Exception as e:
        return None, f"Couldn't list the {tmain.label(eng.key)} accounts: {e}"
    said = (args.get("account") or "").strip()
    if said.lower() in _ALL:
        return numbers, None
    if said:
        n, problem = taccounts.resolve(eng.key, said, numbers)
        return ([n], None) if n else (None, problem)
    n = eng.default_account(ctx.active_agent)
    if n:
        return [n], None
    listing = ", ".join(eng.label(x) for x in numbers)
    return None, (f"Which {tmain.label(eng.key)} account — {listing}? Say which "
                  "to work on ('move to my IRA') or ask about all of them.")


def _one_account(ctx, eng, args=None):
    """(account, None) or (None, problem) for a call that acts on exactly
    one account: an order, a cancel."""
    if ((args or {}).get("account") or "").strip().lower() in _ALL:
        return None, "Orders go to one account at a time — which one?"
    found, problem = _accounts(ctx, eng, args or {})
    return (found[0], None) if found else (None, problem)


def _per_account(ctx, args, one, total=None):
    """Run one(engine, account) -> (sentence, number-or-None, section-or-None)
    for every account a read covers, across the persona's brokers. Each
    sentence is prefixed with its account (and broker, when there are two),
    and each account is one section of the display card (a sentence with no
    section becomes the section's note). With more than one account, total
    = (name, format, kind) adds a combined figure — added up in the sentence
    and on the card only, never in storage."""
    engs, problem = _engines(ctx, args)
    if problem:
        return problem
    parts, values, sections = [], [], []
    for broker, eng, err in engs:
        where = f"{tmain.label(broker)}, " if len(engs) > 1 else ""
        if err:
            parts.append(f"{tmain.label(broker)}: {err}")
            sections.append(blocks.section(tmain.label(broker), notes=[err]))
            continue
        accts, problem = _accounts(ctx, eng, args)
        if problem:
            parts.append(where + problem)
            sections.append(blocks.section(where.rstrip(", ") or None,
                                           notes=[problem]))
            continue
        for account in accts:
            text, value, sec = one(eng, account)
            parts.append(f"{where}{eng.label(account)}: {text}")
            values.append(value)
            sections.append({**(sec or blocks.section(notes=[text])),
                             "heading": f"{where}{eng.label(account)}"})
    combined = None
    if total and len(values) > 1 and None not in values:
        name, fmt, kind = total
        combined = blocks.total(f"All accounts, {name}", fmt(sum(values)), kind)
        parts.append(f"All of them together, added here only: {name} "
                     f"{combined['value']}.")
    ctx.card = blocks.card(sections, total=combined)
    return " ".join(parts)


@tool({
    "name": "trading_status",
    "description": (
        "Report the trading setup: broker, sandbox or live environment, "
        "account, sign-in state, quote state, and the current order ticket "
        "if any. Use when the user asks about trading configuration, "
        "whether trading is in sandbox or live mode, or why something "
        "trading-related failed."
    ),
    "input_schema": {"type": "object", "properties": {"broker": BROKER_ARG}},
})
def trading_status(ctx, args):
    engs, problem = _engines(ctx, args)
    if problem:
        return problem
    out = []
    for broker, eng, err in engs:
        text = err if err else _status_line(ctx, broker, eng)
        out.append(f"{tmain.label(broker)}: {text}" if len(engs) > 1 else text)
    return " ".join(out)


def _status_line(ctx, broker, eng):
    s = eng.status()
    accts = s.get("accounts") or []
    chosen = eng.default_account(ctx.active_agent) if accts else None
    parts = [f"Trading is in {s['env']} mode."]
    if accts:
        parts.append("Accounts: " + ", ".join(eng.label(a) for a in accts) + ".")
        parts.append(f"Working on {eng.label(chosen)}." if chosen else
                     "No account chosen yet — the user picks one ('move to my IRA').")
    parts.append(f"Quotes: {s['stream']}.")
    if s.get("problem"):
        parts.append(s["problem"])
    if s.get("warning"):
        parts.append(s["warning"])
    tk = _load_ticket(broker)
    parts.append(f"Current ticket: {tk.describe()}" if tk
                 else "No order ticket is open.")
    if s["env"] == "sandbox":
        parts.append("Orders go to the tastytrade sandbox, not real money; "
                     "set TASTY_ENV=live to trade for real.")
    return " ".join(parts)


@tool({
    "name": "set_account",
    "description": (
        "Choose which account you work on, or name an account. Without "
        "nickname: switch to that account ('move to my IRA', 'use red', "
        "'switch to 868') — every later balance, position, P&L, order and "
        "review question goes to it, in this and later sessions, until the "
        "user switches again. With nickname: remember the user's name for an "
        "account ('call 868 red', 'the 699 one is my IRA') without switching."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "account": {"type": "string",
                        "description": "Nickname or last digits of the account."},
            "nickname": {"type": "string",
                         "description": "Only to name the account, e.g. 'IRA'."},
            "broker": BROKER_ARG,
        },
        "required": ["account"],
    },
})
def set_account(ctx, args):
    eng, err = _engine(ctx, args)
    if err:
        return err
    try:
        numbers = eng.accounts()
    except Exception as e:
        return f"Couldn't list the {tmain.label(eng.key)} accounts: {e}"
    said = (args.get("account") or "").strip()
    number, problem = taccounts.resolve(eng.key, said, numbers)
    if problem:
        return problem
    if args.get("nickname"):
        digits = "".join(c for c in said if c.isdigit()) or number[-3:]
        taccounts.set_name(eng.key, digits, args["nickname"])
        return f"Got it: account …{number[-3:]} is {args['nickname']}."
    taccounts.set_focus(eng.key, ctx.active_agent, number)
    reply = (f"Now working on {eng.label(number)}. Balance, positions, P&L "
             "and orders go to it until you switch.")
    tk = _load_ticket(eng.key)
    if tk and tk.account and tk.account != number:
        reply += (f" The open ticket was for {eng.label(tk.account)}; it will "
                  "be reviewed again on this account before it can be placed.")
    return reply


@tool({
    "name": "get_balance",
    "description": (
        "Account balance: net liquidating value, cash, and buying power. Use "
        "for 'check my balance', 'how much cash do I have', 'what's my "
        "buying power', 'what's the account worth'."
    ),
    "input_schema": {"type": "object",
                     "properties": {"account": ACCOUNT_ARG, "broker": BROKER_ARG}},
})
def get_balance(ctx, args):
    def one(eng, account):
        try:
            b = eng.balance(account)
        except Exception as e:
            return f"couldn't fetch the balance: {e}.", None, None
        return (f"net liquidating value {_fmt(b.net_liq)}, cash {_fmt(b.cash)}, "
                f"buying power {_fmt(b.buying_power)}."), b.net_liq, None
    return _per_account(ctx, args, one, total=(
        "net liquidating value", _fmt, blocks.MONEY))


@tool({
    "name": "get_quote",
    "description": (
        "Real-time market quote (bid, ask, last) for an underlying symbol — "
        "SPX, any stock ticker like AAPL, or any futures root like /ES ('the "
        "e-mini'), /MES, /NQ, /CL (crude), /GC (gold), /ZB, /6E. Use "
        "whenever the user asks where an underlying is trading. Options are "
        "priced another way: build them as a ticket and call price_order."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "symbol": {"type": "string",
                       "description": "Underlying, e.g. 'SPX', '/ES', 'AAPL'."},
            "broker": BROKER_ARG,
        },
        "required": ["symbol"],
    },
})
def get_quote(ctx, args):
    eng, err = _engine(ctx, args)
    if err:
        return err
    symbol = tsym.normalize_underlying(args.get("symbol", ""))
    if not symbol:
        return "Which symbol?"
    try:
        q = eng.underlying_quote(symbol)
    except Exception as e:
        return f"Couldn't get a quote for {symbol}: {e}"
    if q is None or (q.bid is None and q.last is None):
        why = eng.quotes.last_error
        return (f"No quote available for {symbol} right now" +
                (f" ({why})" if why else "") + "." +
                (" In sandbox mode real-time data is unavailable."
                 if eng.broker.env_label() == "sandbox" else ""))
    def px(x):
        # Under 10, two decimals erase the price: /6E 1.12345 read as 1.12,
        # /NG 2.903 as 2.90 (#56). Up to five there, trailing zeros dropped.
        if abs(x) >= 10:
            return _fmt(x)
        s = f"{x:.5f}".rstrip("0")
        return s + "0" * max(0, 2 - len(s.split(".")[1]))
    bits = []
    if q.bid is not None and q.ask is not None:
        bits.append(f"bid {px(q.bid)}, ask {px(q.ask)}, mid {px(q.mid)}")
    if q.last is not None:
        bits.append(f"last {px(q.last)}")
    # A futures quote names its contract ('/ES (/ESZ26)'): the root alone
    # doesn't say which month was priced.
    contract = (q.symbol or "").split(":")[0]
    name = (f"{symbol} ({contract})" if tsym.is_futures(symbol) and contract
            and contract != symbol else symbol)
    return f"{name}: " + "; ".join(bits) + "." + _as_of(q)


def _as_of(q):
    """When the price is from, in words. Without it Tom read the 4pm close as
    "right now" forty minutes after the bell, then invented a feed lag to
    explain the user's screen (2026-10-29, #55)."""
    def clock(ts):
        dt = datetime.fromtimestamp(ts).astimezone()
        return dt.strftime("%I:%M %p ").lstrip("0") + zone_name(dt)
    if q.closed:
        return (" Market closed: this is the last price, not a live one" +
                (f" (last update {clock(q.as_of)})" if q.as_of else "") + ".")
    if not q.as_of:
        return " The broker gave no time for this price."
    age = time.time() - q.as_of
    if age > tcfg.QUOTE_STALE_S:
        return (f" Last update {clock(q.as_of)}, {round(age / 60)} minutes ago"
                " — may not be live.")
    return f" As of {clock(q.as_of)}."


@tool({
    "name": "list_expirations",
    "description": (
        "List the nearest option expiration dates for an underlying, so the "
        "user can pick one while building a strategy. Use when they ask "
        "what expirations are available."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "symbol": {"type": "string"},
            "count": {"type": "integer",
                      "description": "How many to list (default 6)."},
        },
        "required": ["symbol"],
    },
})
def list_expirations(ctx, args):
    eng, err = _engine(ctx, args)
    if err:
        return err
    symbol = tsym.normalize_underlying(args.get("symbol", ""))
    try:
        chain = eng.chain(symbol)
    except Exception as e:
        return f"Couldn't load the {symbol} chain: {e}"
    n = max(1, min(int(args.get("count") or 6), 20))
    exps = chain.expirations[:n]
    # Not a total: Schwab's chain is loaded a week deep (#32).
    return (f"Nearest {symbol} expirations: " +
            ", ".join(map(with_weekday, exps)) +
            ". Later ones are looked up when the user names a date.")


@tool({
    "name": "build_strategy",
    "description": (
        "Start a new order ticket from a named options strategy on an "
        "underlying, with legs auto-placed at sensible strikes around the "
        "current price from the real option chain. Strategies: " +
        ", ".join(tstrat.STRATEGIES) + ". Replaces any existing ticket. "
        "After building, read the legs back to the user; they can then "
        "adjust legs, set a price, review, and submit. Use for 'set up an "
        "iron condor on SPX', 'buy a call on the e-mini', etc. The ticket "
        "goes to your own broker; if the user names the other broker, pass "
        "it and the tool says who handles it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "strategy": {"type": "string",
                         "description": "One of the named strategies."},
            "symbol": {"type": "string",
                       "description": "Underlying, e.g. 'SPX', '/ES'."},
            "expiration": {"type": "string",
                           "description": "Optional ISO date for the near "
                                          "expiration; snapped to the "
                                          "closest real one. Omit for the "
                                          "nearest."},
            "size": {"type": "integer",
                     "description": "Contracts per leg (default 1)."},
            "broker": BROKER_ARG,
        },
        "required": ["strategy", "symbol"],
    },
}, display=display_heading("Order ticket"))
def build_strategy(ctx, args):
    eng, err = _engine(ctx, args)
    if err:
        return err
    strategy = tstrat.match_strategy(args.get("strategy", ""))
    if not strategy:
        return (f"I don't know the strategy '{args.get('strategy')}'. "
                "Options: " + ", ".join(tstrat.STRATEGIES) + ".")
    symbol = tsym.normalize_underlying(args.get("symbol", ""))
    try:
        chain = eng.chain(symbol)
    except Exception as e:
        return f"Couldn't load the {symbol} option chain: {e}"
    expiration = None
    if args.get("expiration"):
        expiration, problem = _expiration(eng, chain, args["expiration"])
        if problem:
            return problem
    q = eng.underlying_quote(symbol)
    price = q.mid if q and q.mid is not None else (q.last if q else None)
    size = max(1, int(args.get("size") or 1))
    legs = tstrat.build_legs(strategy, chain, price, expiration=expiration,
                             size=size)
    if not legs:
        return f"Couldn't build a {strategy} on {symbol} from the chain."
    account = eng.default_account(ctx.active_agent)
    tk = tticket.Ticket(underlying=symbol, strategy=strategy, legs=legs,
                        account=account)
    tk.touch()
    tticket.save(tk, eng.key)
    at = f" (underlying at {_fmt(price)})" if price is not None else \
         " (no live underlying price — strikes centred on the chain)"
    where = f" for {eng.label(account)}" if account else ""
    built = f"Built on {tmain.label(eng.key)}{where}{at}"
    ctx.card = _ticket_card(tk, notes=[built + "."])
    return f"{built}: {tk.describe()} Say adjustments, or ask me to review it."


@tool({
    "name": "adjust_leg",
    "description": (
        "Change one leg of the current order ticket: strike, expiration, "
        "size, side (Long/Short), or option type (Call/Put). Legs are "
        "numbered from 1 in the order they were read back. Strikes and "
        "expirations snap to the closest real ones in the chain; strike-"
        "locked strategies (straddles, calendars, iron butterfly body) "
        "move their partner legs together. Use for 'move the put side "
        "down 10 points', 'make it two contracts', 'push the long leg a "
        "week out'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "leg_number": {"type": "integer", "description": "1-based."},
            "strike": {"type": "number"},
            "expiration": {"type": "string", "description": "ISO date."},
            "size": {"type": "integer"},
            "side": {"type": "string", "enum": ["Long", "Short"]},
            "option_type": {"type": "string", "enum": ["Call", "Put"]},
        },
        "required": ["leg_number"],
    },
}, display=display_heading("Order ticket"))
def adjust_leg(ctx, args):
    eng, tk, chain, problem = _engine_and_ticket(ctx)
    if problem:
        return problem
    i = int(args.get("leg_number", 0)) - 1
    if not 0 <= i < len(tk.legs):
        return f"The ticket has {len(tk.legs)} legs — leg {i + 1} doesn't exist."
    leg = tk.legs[i]
    changes = []
    if args.get("expiration"):
        snapped, problem = _expiration(eng, chain, args["expiration"])
        problem = problem or _strike_listed(eng, chain, snapped, leg.strike)
        if problem:
            return problem
        leg.expiration = snapped
        strikes = chain.strikes.get(snapped) or []
        if strikes and leg.strike not in strikes:
            leg.strike = tstrat.closest(strikes, leg.strike)
        changes.append(f"expiration {snapped}")
    if args.get("strike") is not None:
        strike = float(args["strike"])
        for j in tk.locked_with(i):
            problem = _strike_listed(eng, chain, tk.legs[j].expiration, strike)
            if problem:
                return problem
        moved = tk.set_strike(i, strike, chain)
        changes.append(f"strike {leg.strike:g}" +
                       (f" (moved legs {', '.join(str(m + 1) for m in moved)} "
                        "together)" if len(moved) > 1 else ""))
    if args.get("size") is not None:
        leg.size = max(1, int(args["size"]))
        changes.append(f"size {leg.size}")
    if args.get("side"):
        leg.side = args["side"]
        changes.append(f"now {leg.side}")
    if args.get("option_type"):
        leg.cp = args["option_type"]
        changes.append(f"now a {leg.cp}")
    if not changes:
        return "Nothing to change — say a strike, expiration, size, side, or type."
    tk.touch()
    tticket.save(tk, eng.key)
    problems = tk.validate(chain)
    tail = (" Note: " + " ".join(problems)) if problems else ""
    updated = f"Leg {i + 1} updated ({', '.join(changes)})."
    ctx.card = _ticket_card(tk, notes=[updated, *problems])
    return f"{updated} Ticket now: {tk.describe()}{tail}"


@tool({
    "name": "set_order_terms",
    "description": (
        "Set the current ticket's limit price and/or time-in-force. Price "
        "can be an explicit amount with credit/debit, or 'mid' to price at "
        "the live net mid of the legs. Use for 'make it a 1.20 credit', "
        "'price it at mid', 'good till cancelled'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "price": {"type": "number",
                      "description": "Absolute net price per spread."},
            "price_effect": {"type": "string", "enum": ["Credit", "Debit"],
                             "description": "Required with price."},
            "use_mid": {"type": "boolean",
                        "description": "Price at the live net mid instead."},
            "tif": {"type": "string", "enum": ["Day", "GTC"]},
        },
    },
}, display=display_heading("Order ticket"))
def set_order_terms(ctx, args):
    eng, tk, chain, problem = _engine_and_ticket(ctx)
    if problem:
        return problem
    said = []
    if args.get("use_mid"):
        net = tk.net_mid(chain, eng.leg_quote_fn(chain))
        if net is None:
            return ("Couldn't price at mid — not all legs have live quotes. "
                    "In sandbox mode real-time option quotes are unavailable; "
                    "set an explicit price instead.")
        tk.apply_net_price(net)
        said.append(f"priced at mid: {tk.limit_price:.2f} {tk.price_effect}")
    elif args.get("price") is not None:
        effect = args.get("price_effect")
        if effect not in ("Credit", "Debit"):
            return "Is that price a credit or a debit?"
        tk.limit_price = round(abs(float(args["price"])), 2)
        tk.price_effect = effect
        tk.touch()
        said.append(f"limit {tk.limit_price:.2f} {tk.price_effect}")
    if args.get("tif"):
        tk.tif = args["tif"]
        tk.touch()
        said.append(f"time-in-force {tk.tif}")
    if not said:
        return "Say a price (with credit or debit), 'mid', or a time-in-force."
    tticket.save(tk, eng.key)
    ctx.card = _ticket_card(tk, notes=[f"Done: {', '.join(said)}."])
    return f"Done: {', '.join(said)}. {tk.describe()}"


@tool({
    "name": "price_order",
    "description": (
        "Price the current ticket from live market quotes: each leg's bid, "
        "ask and mid, implied volatility and greeks (delta, gamma, theta, "
        "vega), and the whole ticket's net mid and net greeks. Quotes only — "
        "no broker check, so any size prices whatever the account's buying "
        "power, and nothing is placed or changed. Use it whenever the user "
        "asks what an option or spread is worth or for its greeks, including "
        "a position held on thinkorswim: build it as a ticket, then price "
        "it. review_order is the broker's dry run before submitting, not a "
        "price check."
    ),
    "input_schema": {"type": "object", "properties": {}},
}, display=display_heading("Ticket pricing"))
def price_order(ctx, args):
    eng, tk, chain, problem = _engine_and_ticket(ctx)
    if problem:
        return problem
    problems = tk.validate(chain)
    if problems:
        return "The ticket can't be priced yet: " + " ".join(problems)
    quotes = tk.leg_quotes(chain, eng.leg_quote_fn(chain, greeks=True))

    def g(x, spec):
        return format(x, spec) if x is not None else "n/a"
    lines, rows = [], []
    for i, (leg, q) in enumerate(quotes, 1):
        name = (f"{leg.code} {leg.size} {with_weekday(leg.expiration)} "
                f"{leg.strike:g} {leg.cp}")
        contract = blocks.contract(tk.underlying, leg.expiration, leg.strike, leg.cp)
        if q is None or q.mid is None:
            lines.append(f"{name}: no live quote")
            rows.append(blocks.row(contract, "no live quote",
                                   badge=leg.side.upper()))
            continue
        greeks = ("greeks not available" if q.delta is None else
                  f"IV {g(q.iv and q.iv * 100, '.1f')}%, delta {g(q.delta, '+.3f')}, "
                  f"gamma {g(q.gamma, '.4f')}, theta {g(q.theta, '+.2f')}, "
                  f"vega {g(q.vega, '.2f')}")
        lines.append(f"{name}: bid {_fmt(q.bid)}, ask {_fmt(q.ask)}, "
                     f"mid {_fmt(q.mid)}; {greeks}")
        rows.append(blocks.row(contract, f"mid {_fmt(q.mid)}", blocks.MONEY,
                               badge=leg.side.upper(), lines=[
                                   f"Leg {i}: {leg.action}, {leg.size}",
                                   f"Bid {_fmt(q.bid)} · ask {_fmt(q.ask)}",
                                   greeks.capitalize()]))
    net = tk.net_of(quotes)
    net_price = (f"{abs(net):.2f} {'credit' if net > 0 else 'debit'}"
                 if net is not None else None)
    lines.append(f"Net mid {net_price} for the ticket as built" if net_price else
                 "No net price: a leg has no live quote (the sandbox has no "
                 "real-time option quotes)")
    ng = tk.net_greeks(quotes)
    lines.append("Net greeks per share (times the multiplier, 100 for SPX, "
                 f"for dollars): delta {g(ng['delta'], '+.3f')}, gamma "
                 f"{g(ng['gamma'], '+.4f')}, theta {g(ng['theta'], '+.2f')} a "
                 f"day, vega {g(ng['vega'], '+.2f')}"
                 if None not in ng.values() else
                 "No net greeks: the quotes carried none for at least one leg")
    ctx.card = blocks.card(
        [blocks.section(f"{tk.underlying} {tk.strategy or 'custom'}", rows)],
        total=net_price and blocks.total("Net mid, ticket as built", net_price,
                                         blocks.MONEY),
        notes=[lines[-1] + ".", "Quotes only: nothing placed, buying power "
               "not checked.", "" if net_price else lines[-2] + "."])
    return ("Live pricing, quotes only (nothing placed, buying power not "
            "checked). " + ". ".join(lines) + ".")


@tool({
    "name": "review_order",
    "description": (
        "Review the current ticket before submitting: validates every leg "
        "against the chain, runs the broker's dry run, and reports "
        "buying-power effect, fees, and any warnings. MUST be run (and "
        "heard by the user) before submit_order will work. Use for 'review "
        "it', 'check the order'. Not for prices or greeks — that is "
        "price_order, which works even when this dry run is rejected."
    ),
    "input_schema": {"type": "object", "properties": {}},
}, display=display_heading("Order review"))
def review_order(ctx, args):
    eng, tk, chain, problem = _engine_and_ticket(ctx)
    if problem:
        return problem
    problems = tk.validate(chain)
    if problems:
        return "The ticket isn't orderable yet: " + " ".join(problems)
    if tk.limit_price is None:
        net = tk.net_mid(chain, eng.leg_quote_fn(chain))
        if net is not None:
            tk.apply_net_price(net)
            tticket.save(tk, eng.key)
        else:
            return ("No limit price set, and mid pricing has no live "
                    "quotes. Set a price first ('make it a 1.20 credit').")
    account, problem = _one_account(ctx, eng)
    if problem:
        return problem
    if tk.account != account:
        tk.account = account     # the fingerprint changes: any old review is void
        tk.touch()
        tticket.save(tk, eng.key)
    result = eng.orders.review(tk, chain)
    if not result.ok:
        # Says nothing about price: on 2026-10-05 a buying-power rejection
        # here led Tom to conclude he could not price option legs at all, a
        # belief the user then spent a day correcting.
        return ("The broker rejected the dry run (this account can't place "
                "it; the price is unaffected — price_order gives it): " +
                " ".join(result.errors))
    reviewed = f"Reviewed for {eng.label(account)} on {tmain.label(eng.key)}"
    parts = [f"{reviewed}: {tk.describe()}"]
    bp = None
    if result.buying_power_change is not None:
        bp = ("Buying power effect", _fmt(result.buying_power_change))
    elif result.new_buying_power is not None:   # Schwab reports the projection
        bp = ("Projected buying power", _fmt(result.new_buying_power))
    if bp:
        parts.append(" ".join(bp))
    fees = _fmt(result.total_fees)
    parts.append(f"fees about {fees}")
    warnings = "Warnings: " + "; ".join(result.warnings) if result.warnings else ""
    if warnings:
        parts.append(warnings)
    env = (f"This is the {eng.broker.env_label()} environment. Say "
           "'submit it' to place the order.")
    parts.append(env)
    ctx.card = _ticket_card(tk, notes=[reviewed + ".", warnings, env], extra=[
        bp and blocks.row(*bp, blocks.MONEY),
        blocks.row("Fees, about", fees, blocks.MONEY)])
    return ". ".join(parts)


@tool({
    "name": "submit_order",
    "description": (
        "Place the reviewed order with the broker. HARD RULE: set "
        "confirmed=true ONLY when the user has already heard the "
        "review_order summary for this exact ticket and has then "
        "explicitly told you to submit/place it ('yes, submit', 'send it', "
        "'place the order'). Never set it in the same turn as building or "
        "changing the ticket, and never infer it. If the ticket changed "
        "since the review, this fails and the review must be repeated."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "confirmed": {"type": "boolean",
                          "description": "True only after explicit user "
                                         "go-ahead post-review."},
        },
        "required": ["confirmed"],
    },
}, display=display_heading("Order submission"))
def submit_order(ctx, args):
    eng, tk, chain, problem = _engine_and_ticket(ctx)
    if problem:
        return problem
    account, problem = _one_account(ctx, eng)
    if problem:
        return problem
    if tk.account and tk.account != account:
        return (f"Not placed: this order was reviewed for {eng.label(tk.account)}, "
                f"but you're now working on {eng.label(account)}. Review it "
                "again for this account first.")
    order, reason = eng.orders.submit(tk, chain,
                                      confirmed=bool(args.get("confirmed")))
    if reason:
        return reason
    where = eng.label(tk.account)
    ctx.record_event(f"Order {order.id} submitted on {tmain.label(eng.key)}, "
                     f"{where}: {tk.describe()}")
    env = eng.broker.env_label()
    return (f"Order {order.id} submitted for {where} on {tmain.label(eng.key)} "
            f"({env}): status {order.status}. Ask me for the order status any "
            "time, or say cancel to pull it.")


@tool({
    "name": "list_orders",
    "description": (
        "List recent orders and their statuses (working, filled, "
        "cancelled...). Use for 'any working orders?', 'did my order "
        "fill?', 'what did I trade today?'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "start_date": {"type": "string",
                           "description": ("ISO date to search from. Omit for "
                                           "the default recent window, which "
                                           "covers orders still working from "
                                           "earlier days.")},
            "working_only": {"type": "boolean"},
            "account": ACCOUNT_ARG,
            "broker": BROKER_ARG,
        },
    },
}, display=display_heading("Orders"))
def list_orders(ctx, args):
    # Defaulting to today made "any working orders?" answer "no" while orders
    # were live at the broker — they were simply older than midnight.
    start = args.get("start_date") or tcfg.default_orders_start()

    def one(eng, account):
        try:
            orders = eng.orders.list_orders(account, start_date=start)
        except Exception as e:
            return f"couldn't fetch orders: {e}.", None, None
        if args.get("working_only"):
            orders = [o for o in orders if o.is_working]
        if not orders:
            return f"no orders since {start}.", None, None
        lines, rows = [], []
        for o in orders[:10]:
            legs = "; ".join(f"{l['action']} {l['quantity']} {l['symbol']}"
                             for l in o.legs)
            price = f"{o.price:g} {o.price_effect}" if o.price else None
            lines.append(f"Order {o.id} on {o.underlying}: {o.status}"
                         + (f" at {price}" if price else "") + f" — {legs}")
            rows.append(blocks.row(
                f"{o.underlying} · order {o.id}", price, blocks.MONEY,
                badge=o.status.upper(), lines=[
                    f"{l['action'].replace('_', ' ').capitalize()} "
                    f"{l['quantity']} {blocks.symbol_name(l['symbol'])}"
                    for l in o.legs]))
        more = f"+{len(orders) - 10} more" if len(orders) > 10 else ""
        return (". ".join(lines) + (f" ({more})" if more else "") + ".", None,
                blocks.section(rows=rows, notes=[more]))
    return _per_account(ctx, args, one)


@tool({
    "name": "cancel_order",
    "description": (
        "Cancel a working order. With an order id, cancels that one; "
        "without, cancels the single working order if there is exactly "
        "one, otherwise lists them to pick from. Cancelling is always "
        "safe and needs no review."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "order_id": {"type": "string"},
            "account": ACCOUNT_ARG,
            "broker": BROKER_ARG,
        },
    },
})
def cancel_order(ctx, args):
    eng, err = _engine(ctx, args)
    if err:
        return err
    account, problem = _one_account(ctx, eng, args)
    if problem:
        return problem
    order_id = (args.get("order_id") or "").strip()
    if not order_id:
        try:
            working = eng.orders.working_orders(
                account, start_date=(date.today() - timedelta(days=30)).isoformat())
        except Exception as e:
            return f"Couldn't fetch working orders: {e}"
        if not working:
            return f"There are no working orders to cancel in {eng.label(account)}."
        if len(working) > 1:
            names = "; ".join(f"{o.id} ({o.underlying}, {o.status})"
                              for o in working)
            return f"There are {len(working)} working orders: {names}. Which one?"
        order_id = working[0].id
    order, reason = eng.orders.cancel(account, order_id)
    if reason:
        return f"Couldn't cancel {order_id} in {eng.label(account)}: {reason}"
    return f"{eng.label(account)}: order {order.id} on {order.underlying}: {order.status}."


@tool({
    "name": "get_positions",
    "description": (
        "Open positions with live marks and unrealized P&L, optionally "
        "for one underlying. Use for 'what are my positions?', 'how is my "
        "SPX trade doing?'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "underlying": {"type": "string"},
            "account": ACCOUNT_ARG,
            "broker": BROKER_ARG,
        },
    },
}, display=display_heading("Open positions"))
def get_positions(ctx, args):
    def one(eng, account):
        try:
            positions = eng.positions(account, args.get("underlying"))
        except Exception as e:
            return f"couldn't fetch positions: {e}.", None, None
        if not positions:
            scope = f" in {tsym.normalize_underlying(args['underlying'])}" \
                if args.get("underlying") else ""
            return f"no open positions{scope}.", 0.0, None
        lines, rows, total, unmarked = [], [], 0.0, 0
        for p in positions[:12]:
            mark = eng.mark(p)
            u = p.unrealized(mark)
            side = "long" if p.quantity > 0 else "short"
            upnl = f"unrealized {_pnl(u)}" if u is not None else "no mark"
            if u is None:
                unmarked += 1
            else:
                total += u
            avg = _fmt(p.average_open_price)
            lines.append(f"{side} {abs(p.quantity):g} {p.symbol} "
                         f"(avg {avg}, {upnl})")
            rows.append(blocks.row(
                blocks.symbol_name(p.symbol),
                _pnl(u) if u is not None else "no mark", blocks.PNL,
                badge=side.upper(),
                lines=[f"Quantity {abs(p.quantity):g} · avg {avg}"]))
        summary = f"Total unrealized {_pnl(total)}"
        unmarked_note = f"{unmarked} position(s) unmarked" if unmarked else ""
        if unmarked:
            summary += f" ({unmarked_note})"
        return ". ".join(lines) + f". {summary}.", total, blocks.section(
            rows=rows, total=blocks.total("Total unrealized", _pnl(total)),
            notes=[unmarked_note])
    return _per_account(ctx, args, one, total=("unrealized", _pnl, blocks.PNL))


@tool({
    "name": "get_pnl",
    "description": (
        "Profit and loss report: realized P&L over a period (from "
        "transaction history, net of fees) plus current unrealized, "
        "optionally for one underlying. Period can be a named one or "
        "explicit dates. Use for 'how much did I make this week?', 'P&L "
        "on SPX for June', 'what's my realized today?'. Each broker's "
        "account is reported on its own; a combined figure is only a sum "
        "spoken at the end."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "period": {"type": "string", "enum": list(PERIODS),
                       "description": "Named period (preferred)."},
            "start_date": {"type": "string", "description": "ISO date."},
            "end_date": {"type": "string", "description": "ISO date."},
            "underlying": {"type": "string"},
            "account": ACCOUNT_ARG,
            "broker": BROKER_ARG,
        },
    },
}, display=display_heading("Profit and loss"))
def get_pnl(ctx, args):
    start, end = period_range(args.get("period"), args.get("start_date"),
                              args.get("end_date"))
    scope = f" on {tsym.normalize_underlying(args['underlying'])}" \
        if args.get("underlying") else ""

    def one(eng, account):
        try:
            rep = eng.pnl_report(account, start, end, underlying=args.get("underlying"))
        except Exception as e:
            return f"couldn't build the P&L report: {e}.", None, None
        realized, fees = _pnl(rep.realized), _fmt(rep.fees)
        parts = [f"from {rep.start} to {rep.end}{scope}: realized "
                 f"{realized} net of {fees} in fees "
                 f"across {len(rep.events)} closed trades"]
        rows = [blocks.row("Fees", fees, blocks.MONEY),
                blocks.row("Closed trades", str(len(rep.events)))]
        if rep.by_underlying and not args.get("underlying"):
            tops = [(u, _pnl(v)) for u, v in sorted(
                rep.by_underlying.items(), key=lambda kv: -abs(kv[1]))[:4]]
            parts.append("by underlying: " +
                         ", ".join(f"{u} {v}" for u, v in tops))
            rows += [blocks.row(f"{u} realized", v, blocks.PNL) for u, v in tops]
        if rep.unrealized is not None:
            parts.append(f"current unrealized {_pnl(rep.unrealized)}")
            rows.append(blocks.row("Unrealized now", _pnl(rep.unrealized),
                                   blocks.PNL))
        note = (rep.note or "").strip()
        if note:
            parts.append(note)
        return ". ".join(parts) + ".", rep.realized, blocks.section(
            rows=rows, notes=[note],
            total=blocks.total(f"Realized{scope}, {rep.start} to {rep.end}",
                               realized))
    return _per_account(ctx, args, one, total=("realized", _pnl, blocks.PNL))


@tool({
    "name": "clear_ticket",
    "description": (
        "Discard the current order ticket and start fresh. Use for "
        "'scrap that', 'clear the ticket', 'start over'. Does not touch "
        "submitted orders."
    ),
    "input_schema": {"type": "object", "properties": {}},
}, display=display_heading("Order ticket"))
def clear_ticket(ctx, args):
    broker, problem = _broker(ctx, args)
    if problem:
        return problem
    tk = _load_ticket(broker)
    if tk is None:
        return "There was no ticket to clear."
    tticket.clear(broker)
    return f"Cleared the ticket ({tk.describe()})"
