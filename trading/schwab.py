"""Charles Schwab: the broker variant behind Tom (see trading/main.py).

Same module surface as tastytrade.py, built on the Schwab Trader API
(developer.schwab.com, "Trader API - Individual", read 2026-10-15):

- Auth is three-legged OAuth2. The user signs in on Schwab's site and lands
  on the app's callback URL with a `code` in the address bar; exchange_code
  turns it into tokens. The access token lasts 30 minutes and is refreshed
  here silently; the refresh token lasts 7 days and then only a new sign-in
  helps — so every call that would fail on it raises BrokerError(code="auth")
  with the sign-in hint, and auth_state() warns a day ahead.
- Accounts are addressed by an encrypted hash (/accounts/accountNumbers).
- Quotes are REST snapshots (/quotes), not a stream: Schwab allows one
  streamer connection per user, which the agent and the dashboard would
  fight over, and /quotes already carries bid/ask/mark, IV and the greeks
  for options. _Quotes caches them for SCHWAB_QUOTE_TTL_S.
- Order entry covers EQUITY and OPTION only — no futures here, by Schwab's
  design; fetch_chain says so for "/ES".
- /previewOrder is a real dry run, so review_order keeps its meaning.

Field names on the Schwab side were taken from the portal's OpenAPI specs
and docs; the normalizers are small and isolated so the first live calls
can correct them in one place.
"""

import base64
import threading
import time
from datetime import date, datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

import requests

from lib.atomic_io import read_json, write_json_atomic
from trading import config as tcfg
from trading import symbols as tsym
from trading.models import (BrokerError, ChainData, DryRunResult, OptionInfo,
                            PlacedOrder, Position, Quote)

KEY = "schwab"
LABEL = "Charles Schwab"

SIGN_IN_HINT = ("Schwab needs you to sign in again — open the dashboard's "
                "Trading page and press Sign in to Schwab.")

_INDEXES = ("SPX", "RUT", "NDX", "VIX", "XSP", "DJX")


def credentials_problem():
    return tcfg.schwab_credentials_problem()


def env_label():
    return "live"   # Schwab has no sandbox; every token is a production one


def status(client):
    s = {"env": "live", "base_url": tcfg.SCHWAB_API, "account": None}
    s.update(client.auth_state())
    if s["signed_in"]:
        try:
            s["account"] = client.account_number()
        except BrokerError as e:
            s["problem"] = str(e)
    return s


def quote_symbol(underlying):
    """Schwab spells indexes with a leading '$' ('$SPX')."""
    return f"${underlying}" if underlying in _INDEXES else underlying


def _plain(symbol):
    return (symbol or "").lstrip("$")


def _iso(d):
    """Schwab's timestamp format for query params (UTC, milliseconds)."""
    if isinstance(d, str):
        d = datetime.fromisoformat(d)
    if isinstance(d, date) and not isinstance(d, datetime):
        d = datetime(d.year, d.month, d.day)
    return d.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z") \
        if d.tzinfo else d.strftime("%Y-%m-%dT%H:%M:%S.000Z")


# --- client -----------------------------------------------------------------

class Client:
    def __init__(self, client_id=None, client_secret=None, callback_url=None,
                 account_id=None, session=None, tokens_path=None, clock=time.time):
        self._client_id = client_id or tcfg.schwab_client_id()
        self._client_secret = client_secret or tcfg.schwab_client_secret()
        self.callback_url = callback_url or tcfg.schwab_callback_url()
        self._account_id = account_id or tcfg.schwab_account_id()
        self._http = session or requests.Session()
        self._http.headers.update({"Accept": "application/json"})
        self._tokens_path = tokens_path or tcfg.tokens_path(KEY)
        self._clock = clock
        self._tokens = read_json(self._tokens_path, {}, expect=dict)
        self._lock = threading.Lock()
        self._hash = None
        self._number = None

    # --- sign-in -----------------------------------------------------------

    def authorize_url(self):
        return (f"{tcfg.SCHWAB_API}/v1/oauth/authorize?client_id={self._client_id}"
                f"&redirect_uri={self.callback_url}")

    def exchange_code(self, code_or_url):
        """Finish a sign-in: `code_or_url` is the landing URL Schwab redirected
        to (a 404 page whose address bar holds the code) or the bare code."""
        text = (code_or_url or "").strip()
        if "://" in text:
            code = parse_qs(urlparse(text).query).get("code", [""])[0]
        else:
            code = text
        if not code:
            raise BrokerError("No code found — paste the whole address Schwab "
                              "sent you to after signing in.")
        self._save_tokens(self._token_post({
            "grant_type": "authorization_code", "code": code,
            "redirect_uri": self.callback_url}))

    def _token_post(self, form):
        basic = base64.b64encode(
            f"{self._client_id}:{self._client_secret}".encode()).decode()
        resp = self._http.post(
            f"{tcfg.SCHWAB_API}/v1/oauth/token", data=form,
            headers={"Authorization": f"Basic {basic}",
                     "Content-Type": "application/x-www-form-urlencoded"},
            timeout=15)
        if resp.status_code != 200:
            raise BrokerError(f"Schwab token request failed ({resp.status_code}). "
                              + SIGN_IN_HINT, status=resp.status_code, code="auth")
        return resp.json()

    def _save_tokens(self, body):
        now = self._clock()
        # A refresh reply carries the same refresh token; its 7-day clock
        # started at the sign-in, so keep that expiry rather than restart it.
        old_refresh = self._tokens.get("refresh_token")
        refresh_expires = (self._tokens.get("refresh_expires_at")
                           if body.get("refresh_token") == old_refresh and old_refresh
                           else now + tcfg.SCHWAB_REFRESH_TOKEN_LIFE_S)
        self._tokens = {
            "access_token": body["access_token"],
            "access_expires_at": now + float(body.get("expires_in", 1800)),
            "refresh_token": body.get("refresh_token") or old_refresh,
            "refresh_expires_at": refresh_expires,
        }
        tcfg.ensure_dirs(KEY)
        write_json_atomic(self._tokens_path, self._tokens)

    def auth_state(self):
        """{signed_in, hours_left, warning, problem} for status reports."""
        t = self._tokens
        if not t.get("refresh_token"):
            return {"signed_in": False, "hours_left": 0, "warning": None,
                    "problem": "Not signed in to Schwab yet. " + SIGN_IN_HINT}
        left = t.get("refresh_expires_at", 0) - self._clock()
        if left <= 0:
            when = datetime.fromtimestamp(t.get("refresh_expires_at", 0)).strftime("%b %d")
            return {"signed_in": False, "hours_left": 0, "warning": None,
                    "problem": f"Schwab's sign-in expired on {when}. " + SIGN_IN_HINT}
        hours = left / 3600
        warn = (f"Schwab's sign-in expires in {hours:.0f} hours — sign in again "
                "from the dashboard's Trading page before then."
                if left < tcfg.SCHWAB_REFRESH_WARN_S else None)
        return {"signed_in": True, "hours_left": round(hours, 1),
                "warning": warn, "problem": None}

    def _token(self):
        with self._lock:
            t = self._tokens
            if t.get("access_token") and self._clock() + 60 < t.get("access_expires_at", 0):
                return t["access_token"]
            state = self.auth_state()
            if not state["signed_in"]:
                raise BrokerError(state["problem"], code="auth")
            self._save_tokens(self._token_post({
                "grant_type": "refresh_token",
                "refresh_token": t["refresh_token"]}))
            return self._tokens["access_token"]

    # --- plumbing ----------------------------------------------------------

    def _request(self, method, path, params=None, json_body=None):
        resp = self._http.request(
            method, f"{tcfg.SCHWAB_API}{path}", params=params, json=json_body,
            headers={"Authorization": f"Bearer {self._token()}"}, timeout=30)
        if resp.status_code == 401:
            self._tokens["access_expires_at"] = 0   # refresh and retry once
            resp = self._http.request(
                method, f"{tcfg.SCHWAB_API}{path}", params=params, json=json_body,
                headers={"Authorization": f"Bearer {self._token()}"}, timeout=30)
        try:
            payload = resp.json() if resp.content else {}
        except ValueError:
            payload = {}
        if resp.status_code >= 400:
            errors = payload.get("errors") if isinstance(payload, dict) else None
            if isinstance(errors, dict):     # /quotes: {"errors": {"invalidSymbols": [...]}}
                errors = [{"message": f"{k}: {v}"} for k, v in errors.items()]
            elif not isinstance(errors, list):
                msg = payload.get("message") if isinstance(payload, dict) else None
                errors = [{"message": msg}] if msg else []
            raise BrokerError(f"{method} {path} failed ({resp.status_code})",
                              status=resp.status_code, errors=errors)
        if resp.status_code == 201 and resp.headers.get("Location"):
            # POST /orders answers with the new order's URL and no body.
            return {"orderId": resp.headers["Location"].rstrip("/").rsplit("/", 1)[-1]}
        return payload

    # --- account -----------------------------------------------------------

    def _resolve_account(self):
        if self._hash:
            return
        rows = self._request("GET", "/trader/v1/accounts/accountNumbers")
        rows = rows if isinstance(rows, list) else []
        if self._account_id:
            rows = [r for r in rows if str(r.get("accountNumber")) == str(self._account_id)]
        if len(rows) != 1:
            nums = ", ".join("…" + str(r.get("accountNumber"))[-4:] for r in rows) or "none"
            raise BrokerError(
                f"Schwab shared {len(rows)} account(s) ({nums}); set "
                "SCHWAB_ACCOUNT_ID in .env to the one to trade.")
        self._hash, self._number = rows[0]["hashValue"], str(rows[0]["accountNumber"])

    def account_hash(self):
        self._resolve_account()
        return self._hash

    def account_number(self):
        self._resolve_account()
        return self._number

    @property
    def account_id(self):
        return self._number or self._account_id

    def _acct(self, tail=""):
        return f"/trader/v1/accounts/{self.account_hash()}{tail}"

    def positions(self):
        data = self._request("GET", self._acct(), params={"fields": "positions"})
        return (data.get("securitiesAccount") or {}).get("positions", [])

    def transactions_all(self, start, end):
        """Every TRADE in [start, end] (ISO dates; at most one year)."""
        return self._request("GET", self._acct("/transactions"), params={
            "startDate": _iso(start), "endDate": _iso(end), "types": "TRADE"})

    # --- market data -------------------------------------------------------

    def quotes(self, symbols):
        return self._request("GET", "/marketdata/v1/quotes", params={
            "symbols": ",".join(symbols), "fields": "quote"})

    def option_chain(self, symbol):
        return self._request("GET", "/marketdata/v1/chains", params={
            "symbol": symbol, "contractType": "ALL", "strategy": "SINGLE"})

    # --- orders ------------------------------------------------------------

    def dry_run_order(self, payload):
        return self._request("POST", self._acct("/previewOrder"), json_body=payload)

    def place_order(self, payload):
        placed = self._request("POST", self._acct("/orders"), json_body=payload)
        order_id = placed.get("orderId")
        try:
            return self.get_order(order_id)
        except BrokerError:       # placed, just not readable yet
            return {"orderId": order_id, "status": "ACCEPTED"}

    def get_order(self, order_id):
        return self._request("GET", self._acct(f"/orders/{order_id}"))

    def list_orders(self, start_date=None, end_date=None):
        start = start_date or tcfg.default_orders_start()
        end = datetime.now(timezone.utc) + timedelta(days=1)
        return self._request("GET", self._acct("/orders"), params={
            "fromEnteredTime": _iso(start), "toEnteredTime": _iso(end)})

    def cancel_order(self, order_id):
        self._request("DELETE", self._acct(f"/orders/{order_id}"))
        try:
            return self.get_order(order_id)
        except BrokerError:
            return {"orderId": order_id, "status": "CANCELED"}


# --- quotes -----------------------------------------------------------------

def _f(v, default=None):
    try:
        x = float(v)
        return default if x != x else x
    except (TypeError, ValueError):
        return default


def _quote_from(symbol, q):
    """Quote from one /quotes entry's "quote" dict. Schwab's volatility is a
    percentage (14.2); Quote.iv is a fraction like dxFeed's (0.142)."""
    iv = _f(q.get("volatility"))
    return Quote(symbol=symbol, bid=_f(q.get("bidPrice")), ask=_f(q.get("askPrice")),
                 last=_f(q.get("lastPrice")) if _f(q.get("lastPrice")) is not None
                 else _f(q.get("mark")),
                 updated_at=time.time(),
                 iv=iv / 100 if iv is not None else None,
                 delta=_f(q.get("delta")), gamma=_f(q.get("gamma")),
                 theta=_f(q.get("theta")), vega=_f(q.get("vega")))


class _Quotes:
    """REST snapshots behind the QuoteService surface the engine uses.
    start() is always true (nothing to connect), subscribe() only notes the
    symbol, get() fetches when the cached quote is older than the TTL, and
    prefetch() prices a whole ticket in one /quotes call."""

    def __init__(self, client):
        self._client = client
        self._quotes = {}
        self._lock = threading.Lock()
        self._last_error = None

    def start(self):
        return True

    def stop(self):
        pass

    def subscribe(self, symbols, trade=False, greeks=False):
        pass

    @property
    def last_error(self):
        return self._last_error

    def _stale(self, symbols):
        now = time.time()
        with self._lock:
            return [s for s in symbols if s and
                    now - getattr(self._quotes.get(s), "updated_at", 0) > tcfg.SCHWAB_QUOTE_TTL_S]

    def prefetch(self, symbols):
        stale = list(dict.fromkeys(self._stale(symbols)))
        if not stale:
            return
        try:
            data = self._client.quotes(stale)
        except Exception as e:
            self._last_error = str(e)
            return
        with self._lock:
            for sym, entry in (data or {}).items():
                q = entry.get("quote") if isinstance(entry, dict) else None
                if q:
                    self._quotes[sym] = _quote_from(sym, q)

    def refresh(self, pairs):
        self.prefetch([key for _, key in pairs])

    def get(self, symbol, wait_s=0.0, greeks=False):
        self.prefetch([symbol])
        with self._lock:
            return self._quotes.get(symbol)


def make_quotes(client):
    return _Quotes(client)


# --- chains -----------------------------------------------------------------

# Same-date duplicates (SPX AM monthly vs SPXW PM weekly) resolve by
# expiration type, as on tastytrade: Standard > Quarterly > Month-end > Weekly.
_TYPE_PRIORITY = {"S": 0, "Q": 1, "M": 2, "W": 3}


def _normalize_chain(underlying, data):
    chain = ChainData(underlying=underlying,
                      underlying_streamer=quote_symbol(underlying),
                      expirations=[], strikes={}, options={},
                      fetched_at=time.time())
    best = {}   # (date, strike, cp) -> (priority, occ)
    for cp, key in (("C", "callExpDateMap"), ("P", "putExpDateMap")):
        for exp_key, by_strike in (data.get(key) or {}).items():
            exp = exp_key.split(":")[0]
            for strike_text, contracts in (by_strike or {}).items():
                for c in contracts or []:
                    sp = _f(c.get("strikePrice"), _f(strike_text))
                    occ = c.get("symbol")
                    if sp is None or not occ:
                        continue
                    prio = _TYPE_PRIORITY.get(c.get("expirationType"), 9)
                    k = (exp, sp, cp)
                    if k not in best or prio < best[k][0]:
                        best[k] = (prio, occ)
    for (exp, sp, cp), (_, occ) in best.items():
        if exp not in chain.strikes:
            chain.strikes[exp] = []
        if sp not in chain.strikes[exp]:
            chain.strikes[exp].append(sp)
        chain.options[(exp, sp, cp)] = OptionInfo(occ=occ, streamer=occ)
    chain.expirations = sorted(chain.strikes)
    for exp in chain.expirations:
        chain.strikes[exp].sort()
    return chain


def fetch_chain(client, underlying):
    if tsym.is_futures(underlying):
        raise ValueError(f"Schwab's API doesn't trade futures or their options, "
                         f"so {underlying} isn't available here — that's "
                         "tastytrade's, with Bonny.")
    return _normalize_chain(underlying, client.option_chain(quote_symbol(underlying)))


def underlying_api_symbol(chain):
    return chain.underlying_streamer


# --- orders -----------------------------------------------------------------

_INSTRUCTION = {"Buy to Open": "BUY_TO_OPEN", "Sell to Open": "SELL_TO_OPEN",
                "Buy to Close": "BUY_TO_CLOSE", "Sell to Close": "SELL_TO_CLOSE"}
_DURATION = {"Day": "DAY", "GTC": "GOOD_TILL_CANCEL"}


def order_payload(tk, chain):
    """The POST /orders and /previewOrder body (the portal's "Place Order
    Samples": LIMIT for one leg, NET_CREDIT/NET_DEBIT for a spread)."""
    legs = [{"instruction": _INSTRUCTION[leg.action], "quantity": leg.size,
             "instrument": {"symbol": occ, "assetType": "OPTION"}}
            for occ, leg in tk.order_legs(chain)]
    if len(legs) == 1:
        order_type = "LIMIT"
    else:
        order_type = "NET_CREDIT" if tk.price_effect == "Credit" else "NET_DEBIT"
    return {"orderType": order_type, "session": "NORMAL",
            "price": f"{tk.limit_price:.2f}", "duration": _DURATION[tk.tif],
            "orderStrategyType": "SINGLE", "orderLegCollection": legs}


def _messages(items):
    return [i.get("message") or i.get("activityMessage") or str(i)
            for i in items or [] if i]


def parse_dry_run(data):
    """DryRunResult from /previewOrder. Schwab reports a projected balance
    rather than a change, so buying_power_change stays None and the tool
    speaks new_buying_power instead."""
    v = data.get("orderValidationResult") or {}
    rejects = _messages(v.get("rejects"))
    bal = (data.get("orderStrategy") or {}).get("orderBalance") or {}
    fees = 0.0
    for group in ("commission", "fee"):
        for leg in ((data.get("commissionAndFee") or {}).get(group) or {}).get(
                f"{group}Legs", []) or []:
            for val in leg.get(f"{group}Values", []) or []:
                fees += _f(val.get("value"), 0.0)
    return DryRunResult(
        ok=not rejects, errors=rejects,
        warnings=_messages(v.get("warns")) + _messages(v.get("reviews"))
        + _messages(v.get("alerts")),
        new_buying_power=_f(bal.get("projectedBuyingPower")),
        total_fees=round(fees, 2) if data.get("commissionAndFee") else None,
        raw=data)


# Schwab's upper-case statuses in PlacedOrder's vocabulary (TERMINAL set).
_STATUS = {"FILLED": "Filled", "CANCELED": "Cancelled", "REJECTED": "Rejected",
           "EXPIRED": "Expired", "REPLACED": "Removed"}


def parse_order(o):
    status = str(o.get("status") or "Unknown")
    legs, underlying = [], ""
    for l in o.get("orderLegCollection") or []:
        inst = l.get("instrument") or {}
        underlying = underlying or _plain(inst.get("underlyingSymbol"))
        legs.append({"symbol": inst.get("symbol"),
                     "action": l.get("instruction", "").replace("_", " ").title(),
                     "quantity": _f(l.get("quantity")), "remaining": _f(o.get("remainingQuantity"))})
    otype = o.get("orderType", "")
    if otype in ("NET_CREDIT", "NET_DEBIT"):
        effect = "Credit" if otype == "NET_CREDIT" else "Debit"
    else:
        effect = "Credit" if legs and legs[0]["action"].startswith("Sell") else "Debit"
    if not underlying and legs:
        underlying = (tsym.parse_occ(legs[0]["symbol"] or "") or {}).get("root", "")
    return PlacedOrder(
        id=str(o.get("orderId", "")),
        status=_STATUS.get(status, status.replace("_", " ").title()),
        underlying=underlying,
        price=_f(o.get("price")), price_effect=effect,
        time_in_force={"DAY": "Day", "GOOD_TILL_CANCEL": "GTC"}.get(o.get("duration"), o.get("duration")),
        legs=legs, created_at=o.get("enteredTime"), raw=o)


def parse_order_list(data):
    return [parse_order(o) for o in (data if isinstance(data, list) else [])]


# --- positions & transactions -----------------------------------------------

def parse_positions(items):
    out = []
    for p in items or []:
        inst = p.get("instrument") or {}
        qty = _f(p.get("longQuantity"), 0.0) - _f(p.get("shortQuantity"), 0.0)
        if qty == 0:
            continue
        asset = inst.get("assetType", "")
        mult = 100.0 if asset == "OPTION" else 1.0
        mv = _f(p.get("marketValue"))
        out.append(Position(
            symbol=inst.get("symbol", ""),
            underlying=_plain(inst.get("underlyingSymbol")) or inst.get("symbol", ""),
            instrument_type="Equity Option" if asset == "OPTION" else asset.title(),
            quantity=qty, average_open_price=_f(p.get("averagePrice"), 0.0),
            multiplier=mult,
            # Schwab marks positions itself; keep it as the fallback mark.
            close_price=mv / (qty * mult) if mv is not None and qty else None,
            streamer_symbol=inst.get("symbol") or None,
            realized_today=_f(p.get("currentDayProfitLoss"), 0.0), raw=p))
    return out


def _normalize_transaction(tx):
    """One Schwab TRADE in pnl.py's native (tastytrade) shape, or None."""
    items = tx.get("transferItems") or []
    traded = [i for i in items if (i.get("instrument") or {}).get("assetType")
              not in (None, "CURRENCY") and not i.get("feeType")]
    if not traded:
        return None
    it = traded[0]
    inst = it.get("instrument") or {}
    amount = _f(it.get("amount"), 0.0)
    effect = (it.get("positionEffect") or "").upper()
    action = "Buy" if amount > 0 else "Sell"
    if effect == "OPENING":
        action += " to Open"
    elif effect == "CLOSING":
        action += " to Close"
    net = _f(tx.get("netAmount"), 0.0)
    gross = _f(it.get("cost"))
    if gross is None:
        mult = 100.0 if inst.get("assetType") == "OPTION" else 1.0
        gross = -abs(amount) * _f(it.get("price"), 0.0) * mult * (1 if amount > 0 else -1)
    return {"transaction-type": "Trade", "transaction-sub-type": "Trade",
            "symbol": inst.get("symbol", ""),
            "underlying-symbol": _plain(inst.get("underlyingSymbol")) or inst.get("symbol", ""),
            "action": action, "quantity": abs(amount),
            "executed-at": tx.get("time") or tx.get("tradeDate") or "",
            "value": abs(gross), "value-effect": "Debit" if gross < 0 else "Credit",
            "net-value": abs(net), "net-value-effect": "Debit" if net < 0 else "Credit"}


def transactions(client, start, end, underlying=None):
    end = end or date.today().isoformat()
    floor = (date.fromisoformat(end) - timedelta(days=tcfg.SCHWAB_TRANSACTIONS_MAX_DAYS))
    if not start or date.fromisoformat(start) < floor:
        start = floor.isoformat()
    out = []
    for tx in client.transactions_all(start, end) or []:
        n = _normalize_transaction(tx)
        if n and (not underlying or n["underlying-symbol"] == underlying):
            out.append(n)
    return out
