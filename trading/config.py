"""Trading configuration: credentials, environment gate, paths, constants.

Credentials come from environment variables (loaded from the project's
`.env`). The same OAuth credentials the Tasty-Web project uses work here —
either copy the keys into this project's `.env` or point `TASTY_ENV_FILE`
at Tasty-Web's `.env` and they are read from there as a fallback.

Recognised keys (Tasty-Web's names accepted as aliases):

    TASTY_ENV             "sandbox" (default) or "live"  — the safety gate
    TASTY_CLIENT_SECRET   or CLIENT_SECRET
    TASTY_REFRESH_TOKEN   or REFRESH_TOKEN
    TASTY_ACCOUNT_ID      or TASTYWORKS_ACCOUNT_ID (optional: the account to start on)
    TASTY_BASE_URL        optional explicit base URL override
    TASTY_ENV_FILE        optional path to another .env to fall back on

    SCHWAB_CLIENT_ID      the Dev Portal app's App Key
    SCHWAB_CLIENT_SECRET  its Secret
    SCHWAB_CALLBACK_URL   the app's registered callback (default https://127.0.0.1)
    SCHWAB_ACCOUNT_ID     optional: the account to start on (the user can switch by voice)

The environment gate is deliberate: unset or anything other than exactly
"live" means the sandbox base URL (api.cert.tastyworks.com). Real-money
trading must be opted into by writing TASTY_ENV=live. Schwab has no sandbox:
its tokens are only ever production ones (see trading/schwab.py).
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

# --- Schwab Trader API (developer.schwab.com, read 2026-10-15) -------------

SCHWAB_API = "https://api.schwabapi.com"
# Refresh tokens die 7 days after the browser sign-in that minted them, and
# only a new sign-in revives them; access tokens last 30 minutes. The agent
# tracks both so "sign in again" is said before an order fails on it.
SCHWAB_REFRESH_TOKEN_LIFE_S = 7 * 24 * 3600
SCHWAB_REFRESH_WARN_S = 24 * 3600   # status starts warning this long before expiry
SCHWAB_DEFAULT_CALLBACK = "https://127.0.0.1"
# Schwab has no quote stream here (REST only, see schwab.py): a quote this
# fresh is reused rather than re-fetched, so pricing a 4-leg ticket is one
# /quotes call, not four, and two tools in one turn don't double it.
SCHWAB_QUOTE_TTL_S = 5.0
# /transactions takes at most one year per call; the P&L lookback (370 days
# below) is clamped to it rather than paged.
SCHWAB_TRANSACTIONS_MAX_DAYS = 365

# --- Files ---------------------------------------------------------------

TRADING_DIR = app_config.DATA_DIR / "trading"


# Each broker keeps its own working ticket, order audit log and (Schwab) token
# file under data/trading/<broker>/: a ticket built on one broker can never be
# submitted to the other, because the other's engine never sees the file.
def broker_dir(broker):
    return TRADING_DIR / broker


def ticket_path(broker):
    return broker_dir(broker) / "ticket.json"


def orders_log_path(broker):
    return broker_dir(broker) / "orders_log.json"


def tokens_path(broker):
    return broker_dir(broker) / "tokens.json"

# --- Trade log (thinkorswim/Schwab fills; see trading/trade_log.py) --------

TRADE_LOG_PATH = TRADING_DIR / "trade_log.db"
# Where "import my latest statement" looks when no path is given: the newest
# *AccountStatement*.csv here. Voice can't dictate a Windows path. Statements
# dropped on the dashboard's Trading page are kept here once imported. Under
# data/ (gitignored) by default — a personal folder path doesn't belong in the
# repo; point TRADE_STATEMENTS_DIR at one in .env instead.
STATEMENTS_DIR = Path(os.getenv("TRADE_STATEMENTS_DIR")
                      or TRADING_DIR / "statements")
# An alert is stamped with its email's send time, seconds after the fill. A
# statement whose last row is that fill must still supersede the alert's row,
# or the fill counts twice (alert # and statement REF # are different series).
STATEMENT_EMAIL_GRACE_MIN = 10
# An API sync supersedes the alert-logged rows of its span. An alert is
# stamped with the fill's own time (they matched the API to the second on
# 2026-10-15), so a minute past the last synced fill is enough; a longer
# grace would delete alerts for fills the API hasn't listed yet.
SCHWAB_SYNC_ALERT_GRACE_S = 60
# With nothing authoritative in the log yet, a sync reaches back this far
# (Schwab's one-year cap on /transactions). After that it resumes a day
# before the newest synced or imported row — the overlap is free, since a
# re-synced order replaces itself.
SCHWAB_SYNC_FIRST_DAYS = 365
# Position grouping tries every subset of the open trades touching a new trade
# (2^n). The statement's worst tangle is 6; past this the trades stay grouped
# as one unresolved position rather than stall the turn.
POSITION_SEARCH_MAX_TRADES = 16
# One tool result must fit the model's context with room to reason: ~130
# chars a trade line, so 200 lines is ~7k tokens. Over it, the tool says so.
TRADE_QUERY_MAX_ROWS = 200

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

# How far back "recent orders" reaches when no start date is given. The broker's
# order list is date-ranged, and this defaulted to *today* — so a working order
# silently vanished from both the dashboard and "any working orders?" the moment
# the date rolled over, while it was still live at the broker (two working SPX
# orders placed 2026-08-06 were invisible on 08-09). Working orders rest for
# days, so the default window has to outlive a weekend by a wide margin.
ORDERS_LOOKBACK_DAYS = 30


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
    ] if not val]   # TASTY_ACCOUNT_ID is optional: accounts are listed from the login
    if not missing:
        return None
    return ("tastytrade is not configured: missing " + ", ".join(missing) +
            ". Add the keys to .env, or set TASTY_ENV_FILE to a .env that "
            "has them (for example the Tasty-Web project's).")


def schwab_client_id():
    return _env("SCHWAB_CLIENT_ID")


def schwab_client_secret():
    return _env("SCHWAB_CLIENT_SECRET")


def schwab_callback_url():
    return _env("SCHWAB_CALLBACK_URL") or SCHWAB_DEFAULT_CALLBACK


def schwab_account_id():
    return _env("SCHWAB_ACCOUNT_ID")


def schwab_credentials_problem():
    missing = [label for label, val in [
        ("app key (SCHWAB_CLIENT_ID)", schwab_client_id()),
        ("secret (SCHWAB_CLIENT_SECRET)", schwab_client_secret()),
    ] if not val]
    if not missing:
        return None
    return ("Schwab is not configured: missing " + ", ".join(missing) +
            ". Add the keys from the Schwab Developer Portal app to .env.")


def default_orders_start():
    """ISO date the order list reaches back to when the caller gives none.
    Shared by the voice tool and the dashboard so both show the same set."""
    from datetime import date, timedelta
    return (date.today() - timedelta(days=ORDERS_LOOKBACK_DAYS)).isoformat()


def ensure_dirs(broker):
    """Create data/trading/<broker>/. tastytrade's ticket and audit log lived
    one level up before brokers were split (2026-10): the audit trail must
    survive the move, so the old files are carried in once, never copied."""
    d = broker_dir(broker)
    d.mkdir(parents=True, exist_ok=True)
    if broker == "tastytrade":
        for name in ("ticket.json", "orders_log.json"):
            old, new = TRADING_DIR / name, d / name
            if old.is_file() and not new.exists():
                old.replace(new)
