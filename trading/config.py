"""Trading configuration: credentials, environment gate, paths, constants.

Credentials come from environment variables (loaded from the project's
`.env`). The same OAuth credentials the Tasty-Web project uses work here —
either copy the keys into this project's `.env` or point `TASTY_ENV_FILE`
at Tasty-Web's `.env` and they are read from there as a fallback.

Recognised keys (Tasty-Web's names accepted as aliases):

    TASTY_ENV             "sandbox" (default) or "live"  — the safety gate
    TASTY_CLIENT_SECRET   or CLIENT_SECRET
    TASTY_REFRESH_TOKEN   or REFRESH_TOKEN
    TASTY_ACCOUNT_ID      or TASTYWORKS_ACCOUNT_ID
    TASTY_BASE_URL        optional explicit base URL override
    TASTY_ENV_FILE        optional path to another .env to fall back on

The environment gate is deliberate: unset or anything other than exactly
"live" means the sandbox base URL (api.cert.tastyworks.com). Real-money
trading must be opted into by writing TASTY_ENV=live.
"""

import os
from pathlib import Path

import config as app_config

# .env loading is idempotent and cheap; do it here too so the dashboard's
# trading routes work without voice_agent.py having run first.
try:
    from dotenv import load_dotenv
    load_dotenv(app_config.BASE_DIR / ".env")
except ImportError:  # env vars set another way still work
    pass

SANDBOX_BASE_URL = "https://api.cert.tastyworks.com"
LIVE_BASE_URL = "https://api.tastyworks.com"

# Mandatory on every request (including /oauth/token) — without it the API
# answers 401 with an nginx HTML page. See TRADING_RESEARCH.md §2.
USER_AGENT = "voice-agent-trading/1.0"

# --- Files ---------------------------------------------------------------

TRADING_DIR = app_config.DATA_DIR / "trading"
TICKET_PATH = TRADING_DIR / "ticket.json"          # the shared working ticket
ORDERS_LOG_PATH = TRADING_DIR / "orders_log.json"  # audit trail (append-only)

# --- Futures constants (ported from Tasty-Web lib/constants.ts) ----------

FUTURES_PRODUCT_CODE = {"/ES": "ES", "/MES": "MES"}
FUTURES_MULTIPLIER = {"/ES": 50.0, "/MES": 5.0}
# dxFeed streamer symbols for futures underlyings (equities/indices match
# their ticker; instrument streamer symbols always come from the API).
UNDERLYING_QUOTE_SYMBOL = {"/ES": "/ES:XCME", "/MES": "/MES:XCME"}

# How long a normalized option chain stays cached (strikes/expirations do
# not change intraday; new expirations appear overnight).
CHAIN_CACHE_TTL_S = 15 * 60

# Realized-P&L transaction fetch reaches back this far before the requested
# period start so opens that precede the period are matched (see pnl.py).
PNL_OPEN_LOOKBACK_DAYS = 370


def _fallback_env(key):
    """Read one key from TASTY_ENV_FILE (a foreign .env, e.g. Tasty-Web's)
    without letting it pollute os.environ."""
    path = os.environ.get("TASTY_ENV_FILE", "").strip()
    if not path or not Path(path).is_file():
        return None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if k.strip() == key:
            return v.strip().strip("'\"") or None
    return None


def _env(*names):
    """First non-empty value among aliases, checking the process env first,
    then the TASTY_ENV_FILE fallback."""
    for n in names:
        v = os.environ.get(n, "").strip()
        if v:
            return v
    for n in names:
        v = _fallback_env(n)
        if v:
            return v
    return None


def is_live():
    return (_env("TASTY_ENV") or "").lower() == "live"


def base_url():
    explicit = _env("TASTY_BASE_URL")
    if is_live():
        return explicit or LIVE_BASE_URL
    # In sandbox mode an explicit URL is honoured only if it *is* a cert
    # URL — a leftover production TASTY_BASE_URL must not silently defeat
    # the safety gate.
    if explicit and "cert" in explicit:
        return explicit
    return SANDBOX_BASE_URL


def client_secret():
    return _env("TASTY_CLIENT_SECRET", "CLIENT_SECRET")


def refresh_token():
    return _env("TASTY_REFRESH_TOKEN", "REFRESH_TOKEN")


def account_id():
    return _env("TASTY_ACCOUNT_ID", "TASTYWORKS_ACCOUNT_ID")


def credentials_problem():
    """None when fully configured, else a human-readable description of
    what is missing — used by tools and routes for friendly errors."""
    missing = [label for label, val in [
        ("client secret (TASTY_CLIENT_SECRET)", client_secret()),
        ("refresh token (TASTY_REFRESH_TOKEN)", refresh_token()),
        ("account id (TASTY_ACCOUNT_ID)", account_id()),
    ] if not val]
    if not missing:
        return None
    return ("Trading is not configured: missing " + ", ".join(missing) +
            ". Add the keys to .env, or set TASTY_ENV_FILE to a .env that "
            "has them (for example the Tasty-Web project's).")


def ensure_dirs():
    TRADING_DIR.mkdir(parents=True, exist_ok=True)
