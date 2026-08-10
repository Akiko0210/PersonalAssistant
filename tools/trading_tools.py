"""Voice tools for real trading (tastytrade). See TRADING_PLAN.md.

Design rules:
- Every handler degrades to a friendly spoken string — a missing .env key,
  a sandbox without quotes, or a broker rejection must never crash the
  conversation loop (dispatch would catch it, but a clear sentence beats
  "Tool error: ...").
- The dangerous direction is gated twice: submit_order requires a prior
  review (dry run) of the exact current ticket — enforced by fingerprint
  in trading.orders, shared with the dashboard — and its `confirmed` flag,
  which the model is instructed to set only after the user has heard the
  review and explicitly said to go ahead.
- The working ticket is persisted (data/trading/ticket.json), so a draft
  built by voice is the same draft shown in the dashboard's Trading page.
"""

from datetime import date, timedelta

from tools import tool
from trading import config as tcfg
from trading import strategies as tstrat
from trading import symbols as tsym
from trading import ticket as tticket


def _engine():
    """(engine, None) or (None, spoken-problem). Import inside the call so
    a missing `requests` dependency degrades to a sentence too."""
    try:
        from trading.engine import get_engine
        return get_engine()
    except Exception as e:
        return None, f"Trading engine failed to start: {e}"


def _fmt(x, none="unknown"):
    return f"{x:,.2f}" if isinstance(x, (int, float)) else none


def _load_ticket():
    tk = tticket.load()
    return tk if tk and tk.legs is not None else None


def _ticket_chain(eng):
    """(ticket, chain, None) or (None, None, spoken-problem)."""
    tk = _load_ticket()
    if tk is None:
        return None, None, ("There is no order ticket yet — build a "
                            "strategy first.")
    try:
        return tk, eng.chain(tk.underlying), None
    except Exception as e:
        return None, None, f"Couldn't load the {tk.underlying} chain: {e}"


@tool({
    "name": "trading_status",
    "description": (
        "Report the trading setup: sandbox or live environment, account, "
        "quote-stream state, and the current order ticket if any. Use when "
        "the user asks about trading configuration, whether trading is in "
        "sandbox or live mode, or why something trading-related failed."
    ),
    "input_schema": {"type": "object", "properties": {}},
})
def trading_status(ctx, args):
    eng, err = _engine()
    if err:
        return err
    s = eng.status()
    tk = _load_ticket()
    parts = [f"Trading is in {s['env']} mode on account {s['account']}.",
             f"Quote stream: {s['stream']}."]
    parts.append(f"Current ticket: {tk.describe()}" if tk
                 else "No order ticket is open.")
    if s["env"] == "sandbox":
        parts.append("Orders go to the tastytrade sandbox, not real money; "
                     "set TASTY_ENV=live to trade for real.")
    return " ".join(parts)


@tool({
    "name": "get_quote",
    "description": (
        "Real-time market quote (bid, ask, last) for an underlying symbol — "
        "SPX, /ES or 'the e-mini', /MES, or any stock ticker like AAPL. Use "
        "whenever the user asks for a price, a quote, or where something is "
        "trading."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "symbol": {"type": "string",
                       "description": "Underlying, e.g. 'SPX', '/ES', 'AAPL'."},
        },
        "required": ["symbol"],
    },
})
def get_quote(ctx, args):
    eng, err = _engine()
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
                (f" ({why})" if why else "") +
                ". In sandbox mode real-time data is unavailable.")
    bits = []
    if q.bid is not None and q.ask is not None:
        bits.append(f"bid {_fmt(q.bid)}, ask {_fmt(q.ask)}, mid {_fmt(q.mid)}")
    if q.last is not None:
        bits.append(f"last {_fmt(q.last)}")
    return f"{symbol}: " + "; ".join(bits) + "."


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
    eng, err = _engine()
    if err:
        return err
    symbol = tsym.normalize_underlying(args.get("symbol", ""))
    try:
        chain = eng.chain(symbol)
    except Exception as e:
        return f"Couldn't load the {symbol} chain: {e}"
    n = max(1, min(int(args.get("count") or 6), 20))
    exps = chain.expirations[:n]
    return (f"Nearest {symbol} expirations: " + ", ".join(exps) +
            f". ({len(chain.expirations)} total.)")


@tool({
    "name": "build_strategy",
    "description": (
        "Start a new order ticket from a named options strategy on an "
        "underlying, with legs auto-placed at sensible strikes around the "
        "current price from the real option chain. Strategies: " +
        ", ".join(tstrat.STRATEGIES) + ". Replaces any existing ticket. "
        "After building, read the legs back to the user; they can then "
        "adjust legs, set a price, review, and submit. Use for 'set up an "
        "iron condor on SPX', 'buy a call on the e-mini', etc."
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
        },
        "required": ["strategy", "symbol"],
    },
})
def build_strategy(ctx, args):
    eng, err = _engine()
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
        expiration = min(chain.expirations,
                         key=lambda d: abs(date.fromisoformat(d).toordinal() -
                                           date.fromisoformat(args["expiration"]).toordinal()))
    q = eng.underlying_quote(symbol)
    price = q.mid if q and q.mid is not None else (q.last if q else None)
    size = max(1, int(args.get("size") or 1))
    legs = tstrat.build_legs(strategy, chain, price, expiration=expiration,
                             size=size)
    if not legs:
        return f"Couldn't build a {strategy} on {symbol} from the chain."
    tk = tticket.Ticket(underlying=symbol, strategy=strategy, legs=legs)
    tk.touch()
    tticket.save(tk)
    at = f" (underlying at {_fmt(price)})" if price is not None else \
         " (no live underlying price — strikes centred on the chain)"
    return f"Built{at}: {tk.describe()} Say adjustments, or ask me to review it."


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
})
def adjust_leg(ctx, args):
    eng, err = _engine()
    if err:
        return err
    tk, chain, problem = _ticket_chain(eng)
    if problem:
        return problem
    i = int(args.get("leg_number", 0)) - 1
    if not 0 <= i < len(tk.legs):
        return f"The ticket has {len(tk.legs)} legs — leg {i + 1} doesn't exist."
    leg = tk.legs[i]
    changes = []
    if args.get("expiration"):
        target = args["expiration"]
        snapped = min(chain.expirations,
                      key=lambda d: abs(date.fromisoformat(d).toordinal() -
                                        date.fromisoformat(target).toordinal()))
        leg.expiration = snapped
        strikes = chain.strikes.get(snapped) or []
        if strikes and leg.strike not in strikes:
            leg.strike = tstrat.closest(strikes, leg.strike)
        changes.append(f"expiration {snapped}")
    if args.get("strike") is not None:
        moved = tk.set_strike(i, float(args["strike"]), chain)
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
    tticket.save(tk)
    problems = tk.validate(chain)
    tail = (" Note: " + " ".join(problems)) if problems else ""
    return (f"Leg {i + 1} updated ({', '.join(changes)}). "
            f"Ticket now: {tk.describe()}{tail}")


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
})
def set_order_terms(ctx, args):
    eng, err = _engine()
    if err:
        return err
    tk, chain, problem = _ticket_chain(eng)
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
    tticket.save(tk)
    return f"Done: {', '.join(said)}. {tk.describe()}"


@tool({
    "name": "review_order",
    "description": (
        "Review the current ticket before submitting: validates every leg "
        "against the chain, runs the broker's dry run, and reports cost, "
        "buying-power effect, fees, and any warnings. MUST be run (and "
        "heard by the user) before submit_order will work. Use for 'review "
        "it', 'what would this cost', 'check the order'."
    ),
    "input_schema": {"type": "object", "properties": {}},
})
def review_order(ctx, args):
    eng, err = _engine()
    if err:
        return err
    tk, chain, problem = _ticket_chain(eng)
    if problem:
        return problem
    problems = tk.validate(chain)
    if problems:
        return "The ticket isn't orderable yet: " + " ".join(problems)
    if tk.limit_price is None:
        net = tk.net_mid(chain, eng.leg_quote_fn(chain))
        if net is not None:
            tk.apply_net_price(net)
            tticket.save(tk)
        else:
            return ("No limit price set, and mid pricing has no live "
                    "quotes. Set a price first ('make it a 1.20 credit').")
    result = eng.orders.review(tk, chain)
    if not result.ok:
        return "The broker rejected the dry run: " + " ".join(result.errors)
    parts = [f"Reviewed: {tk.describe()}",
             f"Buying power effect {_fmt(result.buying_power_change)}",
             f"fees about {_fmt(result.total_fees)}"]
    if result.warnings:
        parts.append("Warnings: " + "; ".join(result.warnings))
    env = eng.status()["env"]
    parts.append(f"This is the {env} environment. Say 'submit it' to place "
                 "the order.")
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
})
def submit_order(ctx, args):
    eng, err = _engine()
    if err:
        return err
    tk, chain, problem = _ticket_chain(eng)
    if problem:
        return problem
    order, reason = eng.orders.submit(tk, chain,
                                      confirmed=bool(args.get("confirmed")))
    if reason:
        return reason
    ctx.record_event(f"Order {order.id} submitted: {tk.describe()}")
    env = eng.status()["env"]
    return (f"Order {order.id} submitted ({env}): status {order.status}. "
            "Ask me for the order status any time, or say cancel to pull it.")


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
        },
    },
})
def list_orders(ctx, args):
    eng, err = _engine()
    if err:
        return err
    # Defaulting to today made "any working orders?" answer "no" while orders
    # were live at the broker — they were simply older than midnight.
    start = args.get("start_date") or tcfg.default_orders_start()
    try:
        orders = eng.orders.list_orders(start_date=start)
    except Exception as e:
        return f"Couldn't fetch orders: {e}"
    if args.get("working_only"):
        orders = [o for o in orders if o.is_working]
    if not orders:
        return f"No orders since {start}."
    lines = []
    for o in orders[:10]:
        legs = "; ".join(f"{l['action']} {l['quantity']} {l['symbol']}"
                         for l in o.legs)
        price = f" at {o.price:g} {o.price_effect}" if o.price else ""
        lines.append(f"Order {o.id} on {o.underlying}: {o.status}{price} — {legs}")
    more = f" (+{len(orders) - 10} more)" if len(orders) > 10 else ""
    return ". ".join(lines) + more


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
        },
    },
})
def cancel_order(ctx, args):
    eng, err = _engine()
    if err:
        return err
    order_id = (args.get("order_id") or "").strip()
    if not order_id:
        try:
            working = eng.orders.working_orders(
                start_date=(date.today() - timedelta(days=30)).isoformat())
        except Exception as e:
            return f"Couldn't fetch working orders: {e}"
        if not working:
            return "There are no working orders to cancel."
        if len(working) > 1:
            names = "; ".join(f"{o.id} ({o.underlying}, {o.status})"
                              for o in working)
            return f"There are {len(working)} working orders: {names}. Which one?"
        order_id = working[0].id
    order, reason = eng.orders.cancel(order_id)
    if reason:
        return f"Couldn't cancel {order_id}: {reason}"
    return f"Order {order.id} on {order.underlying}: {order.status}."


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
        },
    },
})
def get_positions(ctx, args):
    eng, err = _engine()
    if err:
        return err
    try:
        positions = eng.positions(args.get("underlying"))
    except Exception as e:
        return f"Couldn't fetch positions: {e}"
    if not positions:
        scope = f" in {tsym.normalize_underlying(args['underlying'])}" \
            if args.get("underlying") else ""
        return f"No open positions{scope}."
    lines, total, unmarked = [], 0.0, 0
    for p in positions[:12]:
        mark = eng.mark(p)
        u = p.unrealized(mark)
        side = "long" if p.quantity > 0 else "short"
        upnl = f"unrealized {_fmt(u)}" if u is not None else "no mark"
        if u is None:
            unmarked += 1
        else:
            total += u
        lines.append(f"{side} {abs(p.quantity):g} {p.symbol} "
                     f"(avg {_fmt(p.average_open_price)}, {upnl})")
    summary = f"Total unrealized {_fmt(total)}"
    if unmarked:
        summary += f" ({unmarked} position(s) unmarked)"
    return ". ".join(lines) + f". {summary}."


@tool({
    "name": "get_pnl",
    "description": (
        "Profit and loss report: realized P&L over a period (from "
        "transaction history, net of fees) plus current unrealized, "
        "optionally for one underlying. Period can be a named one or "
        "explicit dates. Use for 'how much did I make this week?', 'P&L "
        "on SPX for June', 'what's my realized today?'."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "period": {"type": "string",
                       "enum": ["today", "yesterday", "this_week",
                                "last_week", "this_month", "last_month",
                                "this_year"],
                       "description": "Named period (preferred)."},
            "start_date": {"type": "string", "description": "ISO date."},
            "end_date": {"type": "string", "description": "ISO date."},
            "underlying": {"type": "string"},
        },
    },
})
def get_pnl(ctx, args):
    eng, err = _engine()
    if err:
        return err
    start, end = _period_range(args)
    try:
        rep = eng.pnl_report(start, end, underlying=args.get("underlying"))
    except Exception as e:
        return f"Couldn't build the P&L report: {e}"
    scope = f" on {tsym.normalize_underlying(args['underlying'])}" \
        if args.get("underlying") else ""
    parts = [f"From {rep.start} to {rep.end}{scope}: realized "
             f"{_fmt(rep.realized)} net of {_fmt(rep.fees)} in fees "
             f"across {len(rep.events)} closed trades"]
    if rep.by_underlying and not args.get("underlying"):
        tops = sorted(rep.by_underlying.items(), key=lambda kv: -abs(kv[1]))[:4]
        parts.append("by underlying: " +
                     ", ".join(f"{u} {_fmt(v)}" for u, v in tops))
    if rep.unrealized is not None:
        parts.append(f"current unrealized {_fmt(rep.unrealized)}")
    if rep.note:
        parts.append(rep.note.strip())
    return ". ".join(parts) + "."


def _period_range(args):
    today = date.today()
    period = args.get("period")
    if not period:
        return (args.get("start_date") or today.isoformat(),
                args.get("end_date") or today.isoformat())
    monday = today - timedelta(days=today.weekday())
    first = today.replace(day=1)
    if period == "today":
        return today.isoformat(), today.isoformat()
    if period == "yesterday":
        d = today - timedelta(days=1)
        return d.isoformat(), d.isoformat()
    if period == "this_week":
        return monday.isoformat(), today.isoformat()
    if period == "last_week":
        return ((monday - timedelta(days=7)).isoformat(),
                (monday - timedelta(days=1)).isoformat())
    if period == "this_month":
        return first.isoformat(), today.isoformat()
    if period == "last_month":
        last_end = first - timedelta(days=1)
        return last_end.replace(day=1).isoformat(), last_end.isoformat()
    if period == "this_year":
        return today.replace(month=1, day=1).isoformat(), today.isoformat()
    return today.isoformat(), today.isoformat()


@tool({
    "name": "clear_ticket",
    "description": (
        "Discard the current order ticket and start fresh. Use for "
        "'scrap that', 'clear the ticket', 'start over'. Does not touch "
        "submitted orders."
    ),
    "input_schema": {"type": "object", "properties": {}},
})
def clear_ticket(ctx, args):
    tk = _load_ticket()
    if tk is None:
        return "There was no ticket to clear."
    tticket.clear()
    return f"Cleared the ticket ({tk.describe()})"
