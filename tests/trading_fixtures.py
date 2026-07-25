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
