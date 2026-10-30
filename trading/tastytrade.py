"""tastytrade: the broker variant behind Bonny (see trading/main.py).

Everything shaped by the tastytrade Open API lives here — the OAuth2
refresh-token client, chain/order/position/transaction parsing, the order
payload — plus the DXLink quote stream it wires in from quotes.py. The rest
of the package (ticket, orders gate, engine, P&L math) sees only the
broker-neutral dataclasses in models.py and this module's uniform surface:

    KEY, LABEL, credentials_problem(), Client(), make_quotes(client),
    fetch_chain(client, underlying, ...), order_payload(ticket, chain),
    parse_dry_run(data), parse_order(data), parse_positions(items),
    parse_balance(raw), transactions(client, account, start, end, underlying),
    env_label(), status(client), chainless_quote_symbol(underlying)

and a Client whose account calls take the account number first
(positions(account), balances(account), dry_run_order(account, payload), ...),
plus account_numbers() and default_account.

Port of Tasty-Web's api/*.ts (see TRADING_RESEARCH.md §1-2). Auth: the
never-expiring personal-grant refresh token mints 15-minute access tokens at
/oauth/token; the token is cached and renewed 60 s before expiry. Every
request carries the mandatory User-Agent.
"""

import threading
import time

import requests

from trading import config as tcfg
from trading import symbols as tsym
from trading.models import (Balance, BrokerError, ChainData, DryRunResult,
                            OptionInfo, PlacedOrder, Position)
from trading.quotes import QuoteService

KEY = "tastytrade"
LABEL = "tastytrade"

TastyError = BrokerError  # the name the API research and older tests use


def _extract_errors(payload):
    """tastytrade nests rejection reasons under error.errors[]."""
    err = (payload or {}).get("error") or {}
    found = err.get("errors") or ([err] if err else [])
    return [e for e in found if isinstance(e, dict)]


def credentials_problem():
    return tcfg.credentials_problem()


def env_label():
    return "live" if tcfg.is_live() else "sandbox"


def status(client):
    s = {"env": env_label(), "base_url": client.base_url, "accounts": []}
    try:
        s["accounts"] = client.account_numbers()
    except BrokerError as e:
        s["problem"] = str(e)
    return s


class Client:
    def __init__(self, base_url=None, client_secret=None, refresh_token=None,
                 account_id=None, session=None):
        self.base_url = (base_url or tcfg.base_url()).rstrip("/")
        self._client_secret = client_secret or tcfg.client_secret()
        self._refresh_token = refresh_token or tcfg.refresh_token()
        self.account_id = account_id or tcfg.account_id()
        self._http = session or requests.Session()
        self._http.headers.update({"User-Agent": tcfg.USER_AGENT,
                                   "Accept": "application/json"})
        self._access_token = None
        self._token_expiry = 0.0
        self._token_lock = threading.Lock()
        self._numbers = None

    # --- auth ------------------------------------------------------------

    def _token(self):
        with self._token_lock:
            if self._access_token and time.time() + 60 < self._token_expiry:
                return self._access_token
            resp = self._http.post(
                f"{self.base_url}/oauth/token",
                json={"grant_type": "refresh_token",
                      "refresh_token": self._refresh_token,
                      "client_secret": self._client_secret},
                timeout=15)
            if resp.status_code != 200:
                raise BrokerError(
                    f"OAuth token request failed ({resp.status_code}). "
                    "Check the refresh token / client secret, and that "
                    f"TASTY_ENV matches the credentials ({self.base_url}).",
                    status=resp.status_code)
            body = resp.json()
            self._access_token = body["access_token"]
            self._token_expiry = time.time() + float(body.get("expires_in", 900))
            return self._access_token

    # --- plumbing --------------------------------------------------------

    def _request(self, method, path, params=None, json_body=None):
        resp = self._http.request(
            method, f"{self.base_url}{path}", params=params, json=json_body,
            headers={"Authorization": f"Bearer {self._token()}"}, timeout=30)
        if resp.status_code in (401,):
            # Token might have been revoked mid-flight; retry once fresh.
            self._token_expiry = 0.0
            resp = self._http.request(
                method, f"{self.base_url}{path}", params=params, json=json_body,
                headers={"Authorization": f"Bearer {self._token()}"}, timeout=30)
        try:
            payload = resp.json() if resp.content else {}
        except ValueError:
            payload = {}
        if resp.status_code >= 400:
            errors = _extract_errors(payload)
            raise BrokerError(
                f"{method} {path} failed ({resp.status_code})",
                status=resp.status_code, errors=errors)
        return payload.get("data", payload)

    def _get(self, path, **params):
        clean = {k: v for k, v in params.items() if v is not None}
        return self._request("GET", path, params=clean or None)

    # --- market data -----------------------------------------------------

    def quote_token(self):
        """{token, dxlink-url}; the token is valid 24 h."""
        return self._get("/api-quote-tokens")

    def market_data_by_type(self, **groups):
        """Snapshot quotes. Keyword args map to the endpoint's comma-lists:
        index='SPX', future='/ESU6', equity_option='SPX   ...', etc.
        Funded live accounts only — sandbox raises BrokerError."""
        params = {k.replace("_", "-"): ",".join(v) if isinstance(v, (list, tuple)) else v
                  for k, v in groups.items() if v}
        return self._request("GET", "/market-data/by-type", params=params)

    # --- instruments & chains -------------------------------------------

    def nested_option_chain(self, underlying):
        return self._get(f"/option-chains/{underlying}/nested")

    def nested_futures_option_chain(self, code):
        return self._get(f"/futures-option-chains/{code}/nested")

    def futures(self, code):
        return self._get("/instruments/futures", **{"product-code[]": code})

    # --- account ---------------------------------------------------------

    def account_numbers(self):
        """Every account on the login (/customers/me/accounts). An account
        set in .env stays usable if the listing is refused (sandbox)."""
        if self._numbers is None:
            try:
                data = self._get("/customers/me/accounts")
                found = [str(i["account"]["account-number"])
                         for i in data.get("items", [])
                         if (i.get("account") or {}).get("account-number")]
            except BrokerError:
                found = []
            if not found and self.account_id:
                found = [str(self.account_id)]
            if not found:
                raise BrokerError("tastytrade listed no accounts for this login.")
            self._numbers = found
        return list(self._numbers)

    @property
    def default_account(self):
        return self.account_id

    def _acct(self, account, tail):
        if not account:
            raise BrokerError("No tastytrade account chosen.")
        return f"/accounts/{account}{tail}"

    def balances(self, account):
        return self._get(self._acct(account, "/balances"))

    def positions(self, account):
        data = self._get(self._acct(account, "/positions"))
        return data.get("items", [])

    def transactions_page(self, account, page_offset=0, **filters):
        """One page of /transactions (250 max per page). Filters use the
        API's dasherized names: types=[...], underlying_symbol=...,
        start_date=..., end_date=..."""
        params = {"per-page": 250, "page-offset": page_offset}
        for k, v in filters.items():
            if v is None:
                continue
            params[k.replace("_", "-") + ("[]" if isinstance(v, (list, tuple)) else "")] = v
        return self._request("GET", self._acct(account, "/transactions"), params=params)

    def transactions_all(self, account, max_pages=40, **filters):
        """Every transaction matching the filters, following pagination.
        max_pages bounds a runaway fetch (40 pages = 10k transactions)."""
        items, offset = [], 0
        for _ in range(max_pages):
            page = self.transactions_page(account, page_offset=offset, **filters)
            batch = page.get("items", []) if isinstance(page, dict) else []
            items.extend(batch)
            if len(batch) < 250:
                break
            offset += 1
        return items

    # --- orders ----------------------------------------------------------

    def dry_run_order(self, account, payload):
        return self._request("POST", self._acct(account, "/orders/dry-run"), json_body=payload)

    def place_order(self, account, payload):
        return self._request("POST", self._acct(account, "/orders"), json_body=payload)

    def get_order(self, account, order_id):
        return self._get(self._acct(account, f"/orders/{order_id}"))

    def list_orders(self, account, start_date=None, end_date=None, status=None):
        return self._get(self._acct(account, "/orders"),
                         **{"start-date": start_date, "end-date": end_date,
                            "status[]": status})

    def cancel_order(self, account, order_id):
        return self._request("DELETE", self._acct(account, f"/orders/{order_id}"))


TastyClient = Client


# --- quotes ----------------------------------------------------------------

_INDEXES = ("SPX", "RUT", "NDX", "VIX", "XSP")


def _snapshot_group(api_symbol):
    """Which /market-data/by-type list an API symbol belongs in."""
    if api_symbol.startswith("./"):
        return "future-option"
    if api_symbol.startswith("/"):
        return "future"
    if tsym.parse_occ(api_symbol):
        return "equity-option"
    return "index" if api_symbol in _INDEXES else "equity"


class _Quotes(QuoteService):
    """The DXLink stream plus the REST snapshot fallback the engine reaches
    for when a symbol has no tick (sandbox has no entitlement)."""

    def refresh(self, pairs):
        """pairs = [(api_symbol, quote_key)]; errors are swallowed — the
        caller copes with a None quote."""
        for api_symbol, key in pairs:
            try:
                data = self._client.market_data_by_type(
                    **{_snapshot_group(api_symbol).replace("-", "_"): api_symbol})
                items = data.get("items", []) if isinstance(data, dict) else []
                self.snapshot_merge(items, {api_symbol: key})
            except Exception:
                pass


def make_quotes(client):
    return _Quotes(client)


# --- chains ----------------------------------------------------------------

# Where the same expiration date appears under two roots (SPX AM-settled
# monthly vs SPXW PM weekly), the duplicate is resolved by expiration-type
# priority — Regular > Quarterly > End-Of-Month > Weekly — the same rule
# Tasty-Web applies.
_TYPE_PRIORITY = {"Regular": 0, "Quarterly": 1, "End-of-Month": 2,
                  "End-Of-Month": 2, "Weekly": 3}


def _num(x):
    """API strike prices arrive as strings like '6000.0'."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _add_strikes(chain, exp, strike_rows):
    strikes = chain.strikes.setdefault(exp, [])
    for row in strike_rows:
        sp = _num(row.get("strike-price"))
        if sp is None:
            continue
        if sp not in strikes:
            strikes.append(sp)
        if row.get("call"):
            chain.options[(exp, sp, "C")] = OptionInfo(
                occ=row["call"], streamer=row.get("call-streamer-symbol", ""))
        if row.get("put"):
            chain.options[(exp, sp, "P")] = OptionInfo(
                occ=row["put"], streamer=row.get("put-streamer-symbol", ""))


def _normalize_equity(underlying, data):
    chain = ChainData(
        underlying=underlying,
        underlying_streamer=tcfg.UNDERLYING_QUOTE_SYMBOL.get(underlying, underlying),
        expirations=[], strikes={}, options={}, fetched_at=time.time())
    # Prefer the Standard chain; collect expirations across items, resolving
    # same-date duplicates by expiration-type priority.
    best_type = {}  # exp-date -> (priority, expiration-dict)
    for item in data.get("items", []):
        if item.get("option-chain-type") not in (None, "Standard"):
            continue
        for exp in item.get("expirations", []):
            date = exp.get("expiration-date")
            if not date:
                continue
            prio = _TYPE_PRIORITY.get(exp.get("expiration-type"), 9)
            if date not in best_type or prio < best_type[date][0]:
                best_type[date] = (prio, exp)
    for date in sorted(best_type):
        chain.expirations.append(date)
        _add_strikes(chain, date, best_type[date][1].get("strikes", []))
        chain.strikes[date].sort()
    return chain


def _normalize_futures(underlying, data, futures_items):
    chain = ChainData(
        underlying=underlying,
        underlying_streamer=tcfg.UNDERLYING_QUOTE_SYMBOL.get(underlying, underlying),
        expirations=[], strikes={}, options={}, is_futures=True,
        fetched_at=time.time())
    # Front month: the active contract's streamer symbol beats the
    # continuous-symbol default when we have it.
    for fut in futures_items or []:
        if fut.get("active-month") and fut.get("streamer-symbol"):
            chain.underlying_streamer = fut["streamer-symbol"]
            break
    by_date = {}
    for oc in data.get("option-chains", []):
        for exp in oc.get("expirations", []):
            date = exp.get("expiration-date")
            if not date:
                continue
            prio = _TYPE_PRIORITY.get(exp.get("expiration-type"), 9)
            if date not in by_date or prio < by_date[date][0]:
                by_date[date] = (prio, exp)
    for date in sorted(by_date):
        exp = by_date[date][1]
        chain.expirations.append(date)
        chain.expiration_contract[date] = exp.get("underlying-symbol", "")
        _add_strikes(chain, date, exp.get("strikes", []))
        chain.strikes[date].sort()
    return chain


def fetch_chain(client, underlying, start=None, end=None, every_strike=False):
    """ChainData for an underlying ('SPX', 'AAPL', '/ES'); the cache is
    chains.py's. tastytrade's nested chain is always whole, so the window
    chains.extend_chain asks Schwab for (#32) has nothing to add here."""
    if tsym.is_futures(underlying):
        code = tsym.product_code(underlying)
        data = client.nested_futures_option_chain(code)
        futs = data.get("futures") or []
        if not futs:
            try:
                futs = client.futures(code).get("items", [])
            except Exception:
                futs = []  # front-month streamer stays the continuous default
        return _normalize_futures(underlying, data, futs)
    return _normalize_equity(underlying, client.nested_option_chain(underlying))


def underlying_api_symbol(chain):
    """What the REST snapshot wants for the underlying: the front contract
    for futures, the ticker otherwise."""
    if chain.is_futures:
        return chain.expiration_contract.get(chain.expirations[0]) or chain.underlying
    return chain.underlying


def chainless_quote_symbol(underlying):
    """None: every underlying here, futures included, quotes via its chain."""
    return None


# --- orders ----------------------------------------------------------------

def order_payload(tk, chain):
    """The POST /orders body (port of buildOrderPayload). Raises ValueError
    if the ticket isn't orderable."""
    legs = []
    for occ, leg in tk.order_legs(chain):
        legs.append({
            "instrument-type": "Future Option" if occ.startswith("./")
                               else "Equity Option",
            "symbol": occ,
            "quantity": leg.size,
            "action": leg.action,
        })
    return {
        "time-in-force": tk.tif,
        "order-type": "Limit",
        "price": f"{tk.limit_price:.2f}",
        "price-effect": tk.price_effect,
        "source": "voice-agent",
        "legs": legs,
    }


def _money(d, key):
    """Signed float from tastytrade's value/effect pairs
    (Credit = +, Debit = -, None = missing)."""
    try:
        v = float(d.get(key))
    except (TypeError, ValueError):
        return None
    return -v if d.get(f"{key}-effect") == "Debit" else v


def parse_dry_run(data):
    bp = data.get("buying-power-effect") or {}
    fees = data.get("fee-calculation") or {}
    fee_total = _money(fees, "total-fees")
    return DryRunResult(
        ok=True,
        warnings=[w.get("message", str(w)) for w in data.get("warnings", [])],
        buying_power_change=_money(bp, "change-in-buying-power"),
        new_buying_power=_money(bp, "new-buying-power"),
        margin_requirement=_money(bp, "isolated-order-margin-requirement"),
        # Fees are always a debit; keep the magnitude (spoken as "fees
        # about 3.40", not "-3.40").
        total_fees=abs(fee_total) if fee_total is not None else None,
        raw=data)


def parse_order(o):
    o = o.get("order", o)
    return PlacedOrder(
        id=str(o.get("id", "")),
        status=o.get("status", "Unknown"),
        underlying=o.get("underlying-symbol", ""),
        price=float(o["price"]) if o.get("price") else None,
        price_effect=o.get("price-effect"),
        time_in_force=o.get("time-in-force"),
        legs=[{"symbol": l.get("symbol"), "action": l.get("action"),
               "quantity": l.get("quantity"),
               "remaining": l.get("remaining-quantity")}
              for l in o.get("legs", [])],
        created_at=o.get("received-at") or o.get("created-at"),
        raw=o)


def parse_order_list(data):
    items = data.get("items", []) if isinstance(data, dict) else data
    return [parse_order(o) for o in items]


# --- positions & transactions ----------------------------------------------

def _f(x, default=0.0):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def parse_positions(items):
    out = []
    for p in items:
        qty = _f(p.get("quantity"))
        if p.get("quantity-direction") == "Short":
            qty = -qty
        if qty == 0:
            continue
        out.append(Position(
            symbol=p.get("symbol", ""),
            underlying=p.get("underlying-symbol", p.get("symbol", "")),
            instrument_type=p.get("instrument-type", ""),
            quantity=qty,
            average_open_price=_f(p.get("average-open-price")),
            multiplier=_f(p.get("multiplier"), 1.0) or 1.0,
            close_price=_f(p.get("close-price"), None),
            streamer_symbol=p.get("streamer-symbol") or None,
            realized_today=_f(p.get("realized-today")),
            raw=p))
    return out


def parse_balance(raw):
    return Balance(net_liq=_f(raw.get("net-liquidating-value"), None),
                   cash=_f(raw.get("cash-balance"), None),
                   buying_power=_f(raw.get("derivative-buying-power"),
                                   _f(raw.get("equity-buying-power"), None)),
                   raw=raw)


def transactions(client, account, start, end, underlying=None):
    """Transactions in pnl.py's (tastytrade-native) shape."""
    filters = {"start_date": start, "end_date": end,
               "types": ["Trade", "Receive Deliver", "Money Movement"]}
    if underlying:
        filters["underlying_symbol"] = underlying
    return client.transactions_all(account, **filters)
