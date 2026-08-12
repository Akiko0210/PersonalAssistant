"""Symbol handling: OCC option symbols, futures symbology, spoken forms.

Order symbols and streamer symbols are different alphabets (see
TRADING_RESEARCH.md §2). Chains return both, so nothing here *invents*
a streamer symbol — this module only parses/builds order symbols (needed
to read positions and transactions) and normalizes what the user says
("spx", "s&p", "e-mini") into an underlying ticker.
"""

import re

from trading import config as tcfg

# "SPXW  260227C06990000": 6-char padded root, yymmdd, C/P, strike*1000.
_OCC_RE = re.compile(
    r"^(?P<root>[A-Z0-9]{1,6})\s*(?P<date>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")

# "./ESU6 EW3U6 250919C6000": future-option — future contract, option
# product code, yymmdd, C/P, unpadded strike.
_FUT_OPT_RE = re.compile(
    r"^\.(?P<future>/[A-Z0-9]+)\s+(?P<code>[A-Z0-9]+)\s+"
    r"(?P<date>\d{6})(?P<cp>[CP])(?P<strike>\d+(?:\.\d+)?)$")


def parse_occ(symbol):
    """Parse an order symbol (equity OCC or future-option) into
    {root, expiration ("20yy-mm-dd"), cp ("C"/"P"), strike (float)} or
    None if it isn't an option symbol."""
    s = symbol.strip()
    m = _FUT_OPT_RE.match(s)
    if m:
        root, strike = m.group("future"), float(m.group("strike"))
    else:
        m = _OCC_RE.match(s)
        if not m:
            return None
        root, strike = m.group("root"), int(m.group("strike")) / 1000.0
    d = m.group("date")
    return {"root": root,
            "expiration": f"20{d[0:2]}-{d[2:4]}-{d[4:6]}",
            "cp": m.group("cp"),
            "strike": strike}


def build_occ(root, expiration, cp, strike):
    """Build an equity/index OCC symbol: 6-char padded root + yymmdd +
    C/P + 8-digit strike*1000. expiration is ISO 'yyyy-mm-dd'."""
    y, m, d = expiration.split("-")
    return f"{root:<6}{y[2:]}{m}{d}{cp[0].upper()}{int(round(float(strike) * 1000)):08d}"


def is_futures(symbol):
    return symbol.startswith("/")


def product_code(symbol):
    """CME product code for a futures underlying ('/ES' -> 'ES')."""
    return tcfg.FUTURES_PRODUCT_CODE.get(symbol, symbol.lstrip("/"))


def multiplier(underlying, instrument_type=""):
    """Dollar multiplier for one point of one contract/share."""
    if underlying in tcfg.FUTURES_MULTIPLIER:
        return tcfg.FUTURES_MULTIPLIER[underlying]
    if "Option" in instrument_type:
        return 100.0
    return 1.0


# Spoken/typed forms -> canonical underlying. Kept modest: the model
# already normalizes most speech; this catches the trading-specific ones.
_SPOKEN = {
    "spx": "SPX", "s&p": "SPX", "s and p": "SPX", "the index": "SPX",
    "spxw": "SPX",
    "es": "/ES", "/es": "/ES", "e-mini": "/ES", "e mini": "/ES",
    "s&p futures": "/ES", "es futures": "/ES",
    "mes": "/MES", "/mes": "/MES", "micro": "/MES", "micros": "/MES",
    "micro e-mini": "/MES", "micro futures": "/MES",
    "rut": "RUT", "russell": "RUT",
}


def normalize_underlying(text):
    """'spx', '/es', 'e-mini', 'AAPL' -> canonical underlying symbol."""
    t = (text or "").strip()
    if not t:
        return ""
    hit = _SPOKEN.get(t.lower())
    if hit:
        return hit
    if t.startswith("/"):
        return "/" + t[1:].upper()
    return t.upper()
