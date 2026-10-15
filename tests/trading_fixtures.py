"""Shared fixtures for the trading test suite: a synthetic SPX-like chain
and a synthetic /ES futures chain, built directly as ChainData (the tests
for API-shape parsing build raw payloads separately in test_trading_chains).
"""

from trading import symbols as tsym
from trading.models import ChainData, OptionInfo, Quote

SPX_EXPIRATIONS = ["2026-07-24", "2026-07-27", "2026-07-31",
                   "2026-08-07", "2026-08-21"]
SPX_STRIKES = [6300 + 5 * i for i in range(21)]        # 6300..6400, 5 wide


def spx_chain():
    chain = ChainData(underlying="SPX", underlying_streamer="SPX",
                      expirations=list(SPX_EXPIRATIONS), strikes={},
                      options={}, fetched_at=0.0)
    for exp in SPX_EXPIRATIONS:
        chain.strikes[exp] = [float(s) for s in SPX_STRIKES]
        for s in SPX_STRIKES:
            for cp in "CP":
                occ = tsym.build_occ("SPXW", exp, cp, s)
                chain.options[(exp, float(s), cp)] = OptionInfo(
                    occ=occ,
                    streamer=f".SPXW{exp[2:].replace('-', '')}{cp}{s}")
    return chain


ES_EXPIRATIONS = ["2026-08-21", "2026-09-18"]
ES_STRIKES = [6300 + 25 * i for i in range(9)]         # 25-point spacing


def es_chain():
    chain = ChainData(underlying="/ES", underlying_streamer="/ES:XCME",
                      expirations=list(ES_EXPIRATIONS), strikes={},
                      options={}, is_futures=True, fetched_at=0.0)
    for exp in ES_EXPIRATIONS:
        chain.strikes[exp] = [float(s) for s in ES_STRIKES]
        chain.expiration_contract[exp] = "/ESU6"
        y, m, d = exp.split("-")
        for s in ES_STRIKES:
            for cp in "CP":
                occ = f"./ESU6 EW3U6 {y[2:]}{m}{d}{cp}{s}"
                chain.options[(exp, float(s), cp)] = OptionInfo(
                    occ=occ, streamer=f"./EW3U26{cp}{s}:XCME")
    return chain


def quotes_for(chain, bid=10.0, ask=10.4):
    """A get_quote function giving every option the same bid/ask."""
    streamers = {o.streamer for o in chain.options.values()}

    def get_quote(streamer):
        if streamer in streamers:
            return Quote(symbol=streamer, bid=bid, ask=ask, updated_at=1.0)
        return None
    return get_quote


# --- HTTP fake for the broker clients ---------------------------------------

class FakeResponse:
    def __init__(self, status=200, body=None, headers=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {}
        self.content = b"x" if body is not None else b""

    def json(self):
        return self._body


class FakeHTTP:
    """A requests.Session stand-in: `script` maps "METHOD path-prefix" to a
    FakeResponse (or a list consumed in order); every call is recorded in
    `calls` as (method, url, kwargs)."""

    def __init__(self, script=None):
        self.script = dict(script or {})
        self.calls = []
        self.headers = {}

    def _answer(self, method, url):
        for key, resp in self.script.items():
            m, _, prefix = key.partition(" ")
            if m == method and prefix in url:
                if isinstance(resp, list):
                    return resp.pop(0) if len(resp) > 1 else resp[0]
                return resp
        return FakeResponse(404, {"message": f"unscripted {method} {url}"})

    def request(self, method, url, **kw):
        self.calls.append((method, url, kw))
        return self._answer(method, url)

    def post(self, url, **kw):
        return self.request("POST", url, **kw)


# --- a stand-in engine for persona->broker routing tests --------------------

class FakeEngine:
    """Just enough TradingEngine for the read tools: canned positions and a
    P&L report, keyed by broker so a test can tell which account answered."""

    def __init__(self, key, realized=0.0):
        self.key = key
        self.realized = realized
        self.calls = []

    def status(self):
        return {"env": "live", "account": f"{self.key}-acct", "stream": "ok"}

    def positions(self, underlying=None):
        self.calls.append(("positions", underlying))
        return []

    def pnl_report(self, start, end, underlying=None):
        from trading.models import PnLReport
        self.calls.append(("pnl", start, end, underlying))
        return PnLReport(start=start or "", end=end or "", realized=self.realized)
