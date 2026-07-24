# Voice Trading — Architecture & Implementation Plan

Adds real trading capability to the voice agent: multi-leg option strategies,
real-time bid/ask, order review/submit/cancel, and realized/unrealized P&L —
by voice, and through a Trading page in the web dashboard.

Companion doc: **TRADING_RESEARCH.md** (broker/API research this plan is based
on). Everything here follows the project's design principle: self-contained
modules with small APIs, pure logic testable without a network or an API key.

---

## 1. Decisions (and why)

### 1.1 Broker: tastytrade, via a native Python client

**Decision: port the proven logic from the Tasty-Web project into a Python
`trading/` package — do not proxy through the Next.js app, and do not depend
on its server being up.**

Why not "connect to the other agent through its API": Tasty-Web has **no
external trading API**. All of its order logic runs as Next.js Server Actions
compiled into the browser bundle — there is no HTTP route an outside process
can call to place an order (its three `app/api/*` routes are watchlist,
symbol-search, and an Anthropic proxy). Exposing trading as REST would mean
adding and securing new order-placement routes to Tasty-Web *and* keeping its
dev server running whenever the voice agent trades, and the Python side would
still need its own DXLink stream for spoken real-time quotes. Two processes,
two failure modes, no code saved.

Why this isn't really "from scratch" either: the Python client is a **port of
a battle-tested reference implementation**. We reuse, unchanged:

- the OAuth2 refresh-token flow (`api/token.ts`) — and the **same credentials**
  from Tasty-Web's `.env` (`TASTY_BASE_URL`, `REFRESH_TOKEN`, `CLIENT_SECRET`,
  `TASTYWORKS_ACCOUNT_ID`), so there is zero new broker setup. (Research
  confirmed this is the only auth that still exists: legacy session tokens
  were decommissioned; a personal OAuth grant's refresh token never expires;
  access tokens last 15 minutes; a `User-Agent` header is mandatory on every
  request.);
- the exact order payload shape (`lib/utils.ts buildOrderPayload`);
- the 19 strategy leg templates (`buildStrategyLegs`) and their constraints
  (`MULTI_EXPIRY_STRATEGIES`, `STRIKE_LOCKED_GROUPS`);
- the DXLink websocket handshake (`option-chain-data/socket.js` — a clean,
  standalone reference);
- chain normalization + OCC/streamer symbol handling
  (`api/getOptionChains.ts`, `getFuturesChain.ts`, `parseOccSymbol`) — with
  one improvement: we fetch the **nested** chain endpoints, which return OCC
  *and* streamer symbols per strike directly, instead of the flat chain +
  hand-built maps Tasty-Web post-processes;
- futures constants (`/ES`→ES, ×$50; `/MES`→MES, ×$5; dxFeed suffix `:XCME`).

What we build new (Tasty-Web doesn't have it): realized P&L from transaction
history, and the period/position P&L reporting.

### 1.2 thinkorswim: data later maybe — execution no

The Schwab Trader API (thinkorswim's only sanctioned programmatic route)
**cannot place futures or futures-options orders at all** — its order entry is
limited to `EQUITY` and `OPTION` asset types (verified against the official
docs; see TRADING_RESEARCH.md). /ES automation through thinkorswim is simply
not possible today. It also forces a manual browser re-login every 7 days
(refresh-token hard expiry) and has no sandbox.

**Decision: all execution goes through tastytrade.** The `trading/` package
defines broker-agnostic types so a `SchwabBroker` (SPX-only trading, or
quotes) can be added later as one new module — but it is out of scope now.

### 1.3 New dependencies (2)

- `requests` — REST calls. (stdlib `urllib` was considered; not worth the
  clunkiness for a trading client that must handle JSON errors well.)
- `websocket-client` — DXLink streaming. Chosen over asyncio `websockets`
  because the agent is a threaded, blocking app; the streamer runs in a
  daemon thread like `media_control` does.

The dashboard keeps its instant-start guarantee: trading modules are imported
**lazily inside the trading route handlers**, never at dashboard start, so
`dashboard.py` still launches with stdlib only until a trading page is opened.

### 1.4 Safety model (real money is involved)

- **Environment gate**: `TASTY_ENV=sandbox|live` in `.env`. Default —
  including when unset — is **sandbox** (`api.cert.tastyworks.com`). Live
  trading must be opted into explicitly.
- **Review-before-submit, enforced by state**: `submit` refuses unless a
  dry-run **review** of the *current* ticket happened first. The review
  stamps a hash of the ticket (legs+price+tif+qty); any edit invalidates it.
  This holds for voice and web alike — it lives in the order manager, not UI.
- **Voice confirmation**: the submit tool takes `confirmed: true`, documented
  for the model as "set only after the user has heard the review and said
  yes". Two distinct turns: "review it" → spoken summary with cost, buying
  power effect, fees, warnings → "submit it".
- **Cancel is always one step** — the safe direction never needs ceremony.
- **Audit trail**: every dry-run, submit, cancel, and API error appends to
  `data/trading/orders_log.json` (atomic), shown in dashboard + session log.
- **No status polling**: the docs threaten API suspension for polling
  `/orders/live`. We fetch a specific order by id on demand (after submit,
  or when asked "did it fill?") and refresh dashboard order lists only on
  user action. The account streamer (push order updates) is a later
  addition if on-demand checks feel laggy.

---

## 2. Module map (all new files, one edit to an existing one)

```
trading/
  __init__.py       package doc; no imports with side effects
  config.py         env keys, base URLs, paths, futures constants; loads
                    Voice AI .env, optional TASTYWEB_ENV_PATH fallback so the
                    Tasty-Web credentials can be shared without copying
  models.py         dataclasses: Leg, Ticket, Quote, ChainData, DryRunResult,
                    PlacedOrder, Position, Transaction, PnLReport
  symbols.py        OCC build/parse, streamer-symbol conversion, futures
                    product codes/multipliers, spoken-form normalization
  tasty_client.py   TastyClient: OAuth refresh-token auth (cached, refreshed
                    60s before expiry), all REST endpoints, typed errors
  chains.py         chain fetch + normalization → expirations,
                    strikes-by-expiration, symbol map (occ+streamer); equity
                    (flat) and futures (nested) variants; TTL cache
  strategies.py     the 19 strategy templates; ATM/wing-step strike math;
                    strategy identification for display
  ticket.py         the working order ticket: build from strategy, edit legs,
                    validate every leg against the live chain (strike exists
                    for that expiration, expiration tradable, size ≥ 1),
                    net-price math, payload build; persisted atomically to
                    data/trading/ticket.json (shared voice ↔ web)
  quotes.py         QuoteService: DXLink streamer thread (handshake ported
                    from socket.js), subscribe/unsubscribe, latest-quote
                    cache, REST snapshot fallback, reconnect + token refresh
  orders.py         OrderManager: dry-run/review (hash-stamped), submit,
                    list, cancel, replace; order log
  pnl.py            unrealized P&L from positions (mark × multiplier × qty −
                    cost basis) — positions carry no live mark (close-price
                    is yesterday's), so marks come from quotes.py (DXLink
                    mid, REST snapshot fallback); realized P&L from
                    /transactions over a date range, grouped by underlying;
                    period + symbol filters; realized-today off the position
                    record as the same-day shortcut
tools/
  trading_tools.py  voice tools (§3) — the only integration point with the
                    agent, registered like every other tool module
dashboard.py        + trading routes (lazy imports); dashboard/ + Trading tab
tests/
  test_trading_symbols.py  test_trading_strategies.py  test_trading_ticket.py
  test_trading_pnl.py      test_trading_orders.py      (+ fixtures, no network)
```

The **only existing file edited** for the agent is `tools/__init__.py`
(one import line, the established extension mechanism), plus the dashboard
files for the UI. Nothing in the audio/LLM/notes path is touched.

Process model: the voice agent and the dashboard are separate processes, each
with its own `TastyClient` (tastytrade allows concurrent sessions). Shared
state (draft ticket, order log) goes through `data/trading/*.json` via
`atomic_io`, the same pattern the rest of the app uses.

---

## 3. Voice tools

| Tool | Does |
|---|---|
| `get_quote` | Real-time bid/ask/last for SPX, /ES, any ticker or current ticket legs |
| `build_strategy` | Start a ticket: strategy + symbol (+ optional expiration/strikes); legs auto-placed from live chain ATM, spoken back |
| `adjust_leg` | Change a leg's strike/expiration/size/side/type — revalidated against the chain, strike-lock groups respected |
| `set_order_terms` | Limit price (or "mid"), quantity, time-in-force |
| `review_order` | Dry-run: speaks net debit/credit, buying-power effect, fees, warnings; stamps the review hash |
| `submit_order` | Places the reviewed ticket; requires user's verbal yes (`confirmed: true`); refuses if ticket changed since review |
| `list_orders` | Working/filled orders today (or date range) |
| `cancel_order` | Cancel by spoken position ("the SPX iron condor", "last order") or id |
| `get_positions` | Open positions w/ unrealized P&L, optionally filtered by underlying |
| `get_pnl` | Realized + unrealized for a period ("this week", "June") and/or symbol |

Outputs are spoken-friendly strings (rounded, no tables), same contract as
every other tool. Tasty-Web's browser voice-agent tool set (`lib/voice/
tools.ts`) is the model for phrasing and the two-step submit flow.

---

## 4. Dashboard Trading page

New tab in the existing stdlib dashboard (`dashboard/` + routes in
`dashboard.py`), same visual language as the other tabs:

- **Watch & quotes** — live bid/ask/last tiles. The browser opens the DXLink
  websocket *directly* (the handshake is ~60 lines of JS, ported from
  socket.js; token fetched via `GET /api/trading/quote-token`) — true
  real-time in the UI with no server-side streaming.
- **Strategy builder** — strategy picker (all 19), leg rows (strike dropdowns
  from the real chain, expiration picker, B/S, C/P, qty), live per-leg and
  net mid pricing, validation errors inline. Mirrors `data/trading/
  ticket.json` so a ticket started by voice appears here and vice versa.
- **Review & submit** — dry-run panel (buying power, fees, warnings) →
  explicit confirm click → submit; working orders table with one-click
  cancel; environment badge (SANDBOX / LIVE) always visible.
- **Positions & P&L** — open positions with unrealized P&L; realized P&L
  over a selectable period, per-underlying breakdown.

Routes (all localhost-only, like the rest): `GET /api/trading/status`,
`quote-token`, `chain`, `ticket` (GET/POST), `dry-run`, `submit`, `cancel`,
`orders`, `positions`, `pnl`. Submit requires `{confirm: true}` and the
review hash, same enforcement as voice.

---

## 5. Implementation order

1. `symbols.py` + `models.py` + `config.py` — pure foundations, tested first.
2. `tasty_client.py` (auth + REST) and `chains.py` — verified against sandbox.
3. `strategies.py` + `ticket.py` — pure logic over chain fixtures.
4. `quotes.py` (DXLink thread) — then real-time bid/ask exists everywhere.
5. `orders.py` (dry-run → submit → cancel) — sandbox end-to-end.
6. `pnl.py` — transactions math over fixtures.
7. `tools/trading_tools.py` — the voice surface.
8. Dashboard routes + Trading tab.
9. Full test pass: new suites + the existing suite untouched and green.

Rollback story: everything lives behind one import line and new files —
reverting the branch removes the capability cleanly.

---

## 6. Out of scope (deliberately)

- Schwab/thinkorswim execution (no futures support; weekly manual re-auth —
  revisit if Schwab ships futures order entry).
- Market/stop orders — Limit only, like Tasty-Web. (Voice + market orders on
  futures is an easy way to get hurt; a limit at mid is the safer default.)
- Order *replace* by voice (cancel + rebuild is clearer spoken); replace is
  in the client for the dashboard.
- Portfolio greeks / risk graphs — later.
