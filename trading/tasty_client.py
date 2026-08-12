"""Authenticated tastytrade REST client (OAuth2 refresh-token flow).

A thin, explicit wrapper: one method per endpoint we use, all returning
the parsed `data` payload. Port of Tasty-Web's api/*.ts server actions
(see TRADING_RESEARCH.md §1-2). Auth: the never-expiring personal-grant
refresh token mints 15-minute access tokens at /oauth/token; the token is
cached and renewed 60 s before expiry. Every request carries the
mandatory User-Agent.

Network errors and API rejections surface as TastyError with the API's
own error messages attached — callers turn them into spoken/JSON output.
"""

import threading
import time

import requests

from trading import config as tcfg


class TastyError(Exception):
    def __init__(self, message, status=None, errors=None):
        super().__init__(message)
        self.status = status
        self.errors = errors or []  # [{code, message}, ...] from the API

    def details(self):
        msgs = [e.get("message") or e.get("code", "") for e in self.errors]
        return "; ".join(m for m in msgs if m) or str(self)


def _extract_errors(payload):
    """tastytrade nests rejection reasons under error.errors[]."""
    err = (payload or {}).get("error") or {}
    found = err.get("errors") or ([err] if err else [])
    return [e for e in found if isinstance(e, dict)]


class TastyClient:
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
                raise TastyError(
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
            raise TastyError(
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
        Funded live accounts only — sandbox raises TastyError."""
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

    def _acct(self, tail):
        if not self.account_id:
            raise TastyError("No account id configured (TASTY_ACCOUNT_ID).")
        return f"/accounts/{self.account_id}{tail}"

    def balances(self):
        return self._get(self._acct("/balances"))

    def positions(self):
        data = self._get(self._acct("/positions"))
        return data.get("items", [])

    def transactions_page(self, page_offset=0, **filters):
        """One page of /transactions (250 max per page). Filters use the
        API's dasherized names: types=[...], underlying_symbol=...,
        start_date=..., end_date=..."""
        params = {"per-page": 250, "page-offset": page_offset}
        for k, v in filters.items():
            if v is None:
                continue
            params[k.replace("_", "-") + ("[]" if isinstance(v, (list, tuple)) else "")] = v
        return self._request("GET", self._acct("/transactions"), params=params)

    def transactions_all(self, max_pages=40, **filters):
        """Every transaction matching the filters, following pagination.
        max_pages bounds a runaway fetch (40 pages = 10k transactions)."""
        items, offset = [], 0
        for _ in range(max_pages):
            page = self.transactions_page(page_offset=offset, **filters)
            batch = page.get("items", []) if isinstance(page, dict) else []
            items.extend(batch)
            if len(batch) < 250:
                break
            offset += 1
        return items

    # --- orders ----------------------------------------------------------

    def dry_run_order(self, payload):
        return self._request("POST", self._acct("/orders/dry-run"), json_body=payload)

    def place_order(self, payload):
        return self._request("POST", self._acct("/orders"), json_body=payload)

    def get_order(self, order_id):
        return self._get(self._acct(f"/orders/{order_id}"))

    def list_orders(self, start_date=None, end_date=None, status=None):
        return self._get(self._acct("/orders"),
                         **{"start-date": start_date, "end-date": end_date,
                            "status[]": status})

    def cancel_order(self, order_id):
        return self._request("DELETE", self._acct(f"/orders/{order_id}"))

    def replace_order(self, order_id, payload):
        return self._request("PUT", self._acct(f"/orders/{order_id}"), json_body=payload)
