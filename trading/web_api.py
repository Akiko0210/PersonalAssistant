"""JSON adapters for the dashboard's /api/trading/* routes.

dashboard.py imports this module lazily inside its route handlers, so the
dashboard keeps its instant, stdlib-only start until a Trading page is
actually opened. Every function takes plain dicts and returns
JSON-serializable dicts; anything that can fail returns {"error": ...}
with a human-readable message instead of raising.

The browser gets real-time quotes itself: /api/trading/quote-token hands
it the DXLink url+token and the page opens the websocket directly (the
same pattern Tasty-Web uses), so the dashboard process never streams.
"""

from datetime import date, timedelta

from trading import config as tcfg
from trading import strategies as tstrat
from trading import symbols as tsym
from trading import ticket as tticket
from trading.orders import read_log


def _engine():
    from trading.engine import get_engine
    return get_engine()


def _err(e):
    return {"error": str(e)}


def _chain_dict(chain):
    return {
        "underlying": chain.underlying,
        "underlying_streamer": chain.underlying_streamer,
        "is_futures": chain.is_futures,
        "expirations": chain.expirations,
        "strikes": chain.strikes,
        "options": {f"{exp}|{strike:g}|{cp}": {"occ": o.occ, "streamer": o.streamer}
                    for (exp, strike, cp), o in chain.options.items()},
    }


def _ticket_dict(tk, chain=None):
    d = tk.to_dict()
    d["description"] = tk.describe()
    d["fingerprint"] = tk.fingerprint()
    if chain is not None:
        d["problems"] = tk.validate(chain)
        for leg, info in zip(d["legs"], (
                chain.option(l.expiration, l.strike, l.cp[0]) for l in tk.legs)):
            leg["occ"] = info.occ if info else None
            leg["streamer"] = info.streamer if info else None
    return d


def status():
    problem = tcfg.credentials_problem()
    base = {"env": "live" if tcfg.is_live() else "sandbox",
            "configured": problem is None,
            "problem": problem,
            "strategies": tstrat.STRATEGIES}
    if problem:
        return base
    eng, err = _engine()
    if err:
        return {**base, "configured": False, "problem": err}
    base.update(eng.status())
    tk = tticket.load()
    base["ticket"] = tk.to_dict() if tk else None
    return base


def quote_token():
    eng, err = _engine()
    if err:
        return _err(err)
    try:
        tok = eng.client.quote_token()
        return {"token": tok.get("token"), "url": tok.get("dxlink-url")}
    except Exception as e:
        return _err(f"No quote token ({e}). Sandbox accounts have no "
                    "real-time data entitlement.")


def chain(symbol):
    eng, err = _engine()
    if err:
        return _err(err)
    try:
        return _chain_dict(eng.chain(symbol))
    except Exception as e:
        return _err(e)


def ticket_get():
    eng, err = _engine()
    if err:
        return _err(err)
    tk = tticket.load()
    if not tk:
        return {"ticket": None}
    try:
        ch = eng.chain(tk.underlying)
    except Exception:
        return {"ticket": _ticket_dict(tk)}
    return {"ticket": _ticket_dict(tk, ch)}


def ticket_post(body):
    """op=build: {strategy, symbol, expiration?, size?} from templates.
    op=set: full ticket fields {underlying, strategy?, legs, tif,
    limit_price, price_effect}. op=clear: discard."""
    eng, err = _engine()
    if err:
        return _err(err)
    op = body.get("op")
    if op == "clear":
        tticket.clear()
        return {"ticket": None}
    try:
        if op == "build":
            strategy = tstrat.match_strategy(body.get("strategy", ""))
            if not strategy:
                return _err(f"Unknown strategy: {body.get('strategy')}")
            symbol = tsym.normalize_underlying(body.get("symbol", ""))
            ch = eng.chain(symbol)
            q = eng.underlying_quote(symbol)
            price = q.mid if q and q.mid is not None else (q.last if q else None)
            legs = tstrat.build_legs(strategy, ch, price,
                                     expiration=body.get("expiration"),
                                     size=max(1, int(body.get("size") or 1)))
            if not legs:
                return _err(f"Couldn't build {strategy} on {symbol}.")
            tk = tticket.Ticket(underlying=symbol, strategy=strategy, legs=legs)
        elif op == "set":
            from trading.models import Leg
            tk = tticket.Ticket(
                underlying=tsym.normalize_underlying(body.get("underlying", "")),
                strategy=body.get("strategy") or None,
                legs=[Leg(strike=float(l["strike"]), cp=l["cp"],
                          expiration=l["expiration"], side=l["side"],
                          size=max(1, int(l.get("size") or 1)),
                          open_close=l.get("open_close", "Open"))
                      for l in body.get("legs", [])],
                tif=body.get("tif", "Day"),
                limit_price=(float(body["limit_price"])
                             if body.get("limit_price") not in (None, "") else None),
                price_effect=body.get("price_effect") or None)
            ch = eng.chain(tk.underlying)
        else:
            return _err(f"Unknown op: {op}")
    except Exception as e:
        return _err(e)
    tk.touch()
    tticket.save(tk)
    return {"ticket": _ticket_dict(tk, ch)}


def dry_run():
    eng, err = _engine()
    if err:
        return _err(err)
    tk = tticket.load()
    if not tk:
        return _err("No ticket to review.")
    try:
        ch = eng.chain(tk.underlying)
    except Exception as e:
        return _err(e)
    if tk.limit_price is None:
        net = tk.net_mid(ch, eng.leg_quote_fn(ch))
        if net is None:
            return _err("No limit price set and no live quotes for mid "
                        "pricing — set a price.")
        tk.apply_net_price(net)
        tticket.save(tk)
    r = eng.orders.review(tk, ch)
    return {"ok": r.ok, "errors": r.errors, "warnings": r.warnings,
            "buying_power_change": r.buying_power_change,
            "new_buying_power": r.new_buying_power,
            "margin_requirement": r.margin_requirement,
            "total_fees": r.total_fees,
            "fingerprint": tk.fingerprint(),
            "ticket": _ticket_dict(tk, ch)}


def submit(body):
    eng, err = _engine()
    if err:
        return _err(err)
    if not body.get("confirm"):
        return _err("Submit requires confirm:true from the confirmation "
                    "dialog.")
    tk = tticket.load()
    if not tk:
        return _err("No ticket to submit.")
    if body.get("fingerprint") and body["fingerprint"] != tk.fingerprint():
        return _err("The ticket changed since this dialog was opened — "
                    "review again.")
    try:
        ch = eng.chain(tk.underlying)
    except Exception as e:
        return _err(e)
    order, reason = eng.orders.submit(tk, ch, confirmed=True)
    if reason:
        return _err(reason)
    return {"order": {"id": order.id, "status": order.status,
                      "underlying": order.underlying}}


def cancel(body):
    eng, err = _engine()
    if err:
        return _err(err)
    order_id = str(body.get("order_id") or "").strip()
    if not order_id:
        return _err("No order id.")
    order, reason = eng.orders.cancel(order_id)
    if reason:
        return _err(reason)
    return {"order": {"id": order.id, "status": order.status}}


def orders_list(start_date=None):
    eng, err = _engine()
    if err:
        return _err(err)
    try:
        # Not today: a still-working order placed yesterday must not drop off
        # the page (tcfg.ORDERS_LOOKBACK_DAYS).
        orders = eng.orders.list_orders(
            start_date=start_date or tcfg.default_orders_start())
    except Exception as e:
        return _err(e)
    return {"orders": [{
        "id": o.id, "status": o.status, "underlying": o.underlying,
        "price": o.price, "price_effect": o.price_effect,
        "tif": o.time_in_force, "legs": o.legs, "created_at": o.created_at,
        "working": o.is_working} for o in orders],
        "log": read_log(50)}


def positions():
    eng, err = _engine()
    if err:
        return _err(err)
    try:
        pos = eng.positions()
    except Exception as e:
        return _err(e)
    rows, total, unmarked = [], 0.0, 0
    for p in pos:
        mark = eng.mark(p, wait_s=0.8)
        u = p.unrealized(mark)
        if u is None:
            unmarked += 1
        else:
            total += u
        rows.append({"symbol": p.symbol, "underlying": p.underlying,
                     "type": p.instrument_type, "quantity": p.quantity,
                     "avg_open": p.average_open_price,
                     "multiplier": p.multiplier, "mark": mark,
                     "streamer": p.streamer_symbol,
                     "unrealized": round(u, 2) if u is not None else None})
    return {"positions": rows, "total_unrealized": round(total, 2),
            "unmarked": unmarked}


def pnl(start, end, underlying=None):
    eng, err = _engine()
    if err:
        return _err(err)
    try:
        rep = eng.pnl_report(start or date.today().isoformat(),
                             end or date.today().isoformat(),
                             underlying=underlying or None)
    except Exception as e:
        return _err(e)
    return rep.to_dict()
