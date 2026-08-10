# Voice Trading — Research Findings

Research conducted 2026-07-24 across three tracks: the Tasty-Web codebase
(`C:\Home\Proj\Tasty-Web`), the tastytrade Open API docs
(developer.tastytrade.com), and the Schwab Trader API (thinkorswim's only
programmatic route). Conclusions feed **TRADING_PLAN.md**.

---

## 1. The Tasty-Web project (what exists, what's reusable)

Next.js 16 / React 19 app with a working tastytrade integration and an
in-browser voice agent (Web Speech API + Anthropic proxy). Highlights:

- **Auth**: OAuth2 refresh-token grant (`POST {base}/oauth/token` with
  `grant_type=refresh_token`, `refresh_token`, `client_secret`), access token
  cached and refreshed ~60 s before expiry (`api/token.ts`). Credentials in
  `.env`: `TASTY_BASE_URL`, `REFRESH_TOKEN`, `CLIENT_SECRET`,
  `TASTYWORKS_ACCOUNT_ID`.
- **Endpoints used**: balances, `api-quote-tokens`, `option-chains/{sym}`
  (flat, post-processed), `instruments/futures?product-code=`,
  `futures-option-chains/{code}/nested`, orders (`dry-run`, POST, GET,
  DELETE, PUT), positions, `instruments/equities` (symbol search).
- **Streaming**: DXLink websocket from the browser
  (`contexts/AppContext.tsx`); standalone Node reference in
  `option-chain-data/socket.js` — SETUP → AUTH → CHANNEL_REQUEST(FEED) →
  FEED_SETUP (COMPACT; Quote: bid/ask, Trade: last) → FEED_SUBSCRIPTION;
  KEEPALIVE echo; re-auth on UNAUTHORIZED; 3 s reconnect.
- **Orders**: Limit only, up to 4 legs. Payload built in
  `lib/utils.ts buildOrderPayload`: `time-in-force`, `order-type: Limit`,
  `price` (abs, string), `price-effect: Credit|Debit` (sign of net),
  `legs[]` of `{instrument-type: "Equity Option"|"Future Option"` (by `./`
  OCC prefix)`, symbol: <occ>, quantity, action: "Buy to Open"|...}`.
  Dry-run is always called before submit.
- **19 strategy templates** (`buildStrategyLegs`): singles, verticals,
  straddles/strangles, calendars/diagonals, condors/butterflies/double
  diagonal — placed off the ATM strike with a ~5-point "wing step" derived
  from real strike spacing. Constraints: `MULTI_EXPIRY_STRATEGIES` (only
  calendars/diagonals/double-diagonal allow per-leg expirations),
  `STRIKE_LOCKED_GROUPS` (straddle legs, butterfly body, calendar legs share
  a strike).
- **Symbols**: OCC parse/build duplicated in `fetchOrders.ts`/
  `fetchPositions.ts`; futures constants: `/ES`→product `ES`, ×$50;
  `/MES`→`MES`, ×$5; dxFeed underlying overrides `/ES:XCME`, `/MES:XCME`.
- **P&L**: unrealized only, client-side: `mark × multiplier × qty × sign −
  cost basis`. **No transaction history / realized P&L** implemented.
- **No external trading API**: order logic runs as Server Actions invoked by
  the browser. The only HTTP routes are watchlist (no auth), symbol search,
  and an Anthropic proxy. An outside process cannot place orders through
  this app without new routes being added to it.
- `docs/es_trading_integration.md` (~920 lines) is a useful ES-futures +
  API reference but predates the code: it describes the now-dead
  session-token auth and slightly different chain paths. Trust the code and
  the current official docs over it.

## 2. tastytrade Open API (verified July 2026)

- **Auth: OAuth2 only.** Legacy `POST /sessions` + remember tokens were
  decommissioned (announced for 2025-12-01; release note 20260211 confirms
  full removal). Personal use: create an OAuth app under
  my.tastytrade.com → My Profile → API → OAuth Applications, then "Create
  Grant" for your own account → **refresh token that never expires**.
  `POST /oauth/token` mints an access token valid **15 minutes**.
  Tasty-Web's credentials are exactly this flow — reusable as-is.
- **Environments**: prod `api.tastyworks.com`; sandbox
  `api.cert.tastyworks.com` (separate sandbox user; resets every 24 h;
  quotes 15-min delayed; simulated fills: limit < $3 fills instantly,
  > $3 stays Live; no net-liq history or real-time data).
- **Chains**: `/option-chains/{sym}/nested` → expirations → strikes, each
  with `call`/`put` OCC symbols **and** `call-streamer-symbol`/
  `put-streamer-symbol` — no post-processing needed (Tasty-Web used the
  flat form + hand-built maps; nested is the better port target).
  Futures: `/futures-option-chains/{product_code}/nested` (product code
  `ES`, not `/ESU6`) returns `futures[]` + option chains.
  `/instruments/futures?product-code[]=ES` for contract months
  (URL-encode the `/` in symbols).
- **Symbology**: equity/index OCC = 6-char padded root + yymmdd + C/P +
  8-digit strike×1000 (`SPXW  220520C04025000`; root `SPX` = AM-settled
  monthlies, `SPXW` = weeklies/0DTE). Futures `/ESU6`; future options
  `./ESZ2 E1AZ2 221205P3720` (weekly product codes E1A…EW4 from
  `/instruments/future-option-products`). **Never construct
  streamer-symbols by hand** — read them off chain/instrument responses
  (e.g. `.SPY230731C393`, `/ESU23:XCME`, `./E3AN23C5600:XCME`).
- **Real-time quotes**: `GET /api-quote-tokens` (token valid 24 h, requires
  an opened customer account) → DXLink websocket
  `wss://tasty-openapi-ws.dxfeed.com/realtime`. Handshake as implemented in
  socket.js; use COMPACT format (FULL will be turned off); KEEPALIVE every
  ~30 s. Event types on this token: Profile, Quote, Trade, Summary, Greeks,
  TimeAndSale, Candle. **REST snapshot**: `GET /market-data/by-type?
  index=SPX&future=/ESU6&equity-option=...` (funded accounts only).
- **Orders**: `POST /accounts/{acct}/orders`; dry-run at `/orders/dry-run`
  returns `warnings[]`, `buying-power-effect` (change/current/new BP,
  margin requirement, `is-spread`, `impact`, `effect`) and
  `fee-calculation` (incl. `proprietary-index-option-fees` — matters for
  SPX; `total-fees`). Rejections: HTTP 422 `preflight_check_failure`.
  Cancel: DELETE `/orders/{id}` → `Cancel Requested`. Replace: PUT (price/
  type/TIF only, legs immutable). Statuses: Received, Routed, In Flight,
  Live, Cancel/Replace Requested, Contingent; terminal: Filled, Cancelled,
  Expired, Rejected, Removed, Partially Removed. Option orders max 4 legs;
  buys are Debit, sells Credit. Optional `source` tag on orders.
  **Do not poll `/orders/live`** (explicit in docs; can get access
  suspended) — order-status pushes come from the account streamer
  (`wss://streamer.tastyworks.com`, connect + heartbeat, full order
  objects on every transition). On-demand GET of a specific order id after
  submit is fine.
- **Positions & P&L**: positions carry `average-open-price`, `quantity`
  (positive) + `quantity-direction`, `multiplier`, `realized-today`,
  `realized-day-gain`, and `close-price` = **yesterday's close — there is
  no live mark field**; the docs' own formula marks positions with
  streaming quotes: unrealized = `(mark − avg-open) × qty × multiplier`
  (sign-flipped for shorts). Balances: `net-liquidating-value`,
  buying-power fields, margin requirements. Net-liq history:
  `/accounts/{acct}/net-liq/history?time-back=1m` (prod only).
- **Transactions** (realized P&L source): `GET /accounts/{acct}/
  transactions` with `types[]=Trade` (+ `Receive Deliver` for assignment/
  exercise/expiration), `underlying-symbol`, `start-date`/`end-date`;
  paginated 250/page. Fields: `transaction-type`, `-sub-type`, `action`,
  `quantity`, `price`, `value(+effect)`, `net-value(+effect)` (net of
  fees), `commission`, `executed-at`. Realized P&L = signed sum of
  `net-value` (Credit +, Debit −) with open/close matching per symbol;
  futures cash settlement appears as `Money Movement`/`Futures Settlement`
  and `Mark to Market` sub-types.
- **Hard requirements**: `User-Agent: <product>/<version>` on **every**
  request incl. OAuth (else a confusing nginx 401); dasherized JSON keys;
  `data`/`items` envelope; no published rate limits — "reasonable use",
  429 on abuse, streamers over polling. Market data is licensed for
  personal use on your own account — don't rebroadcast.
- **Python SDK option**: `tastytrade` (PyPI, tastyware org, v13.2.1,
  2026-07-24, Python ≥ 3.11, async/pydantic) covers sessions, DXLink,
  orders, transactions; linked from the official SDK page but community-
  maintained. We port by hand instead (see plan §1.3 rationale): the app is
  threaded-sync, the needed surface is small, we already have a verified
  reference implementation, and a hand-rolled client keeps the dashboard's
  lazy-import + no-network unit tests clean. The SDK remains the fallback
  if the API drifts faster than expected.

## 3. Schwab Trader API (thinkorswim)

thinkorswim is a front-end over a Schwab brokerage account; the **Schwab
Trader API** (developer.schwab.com, free, individual program with manual
app approval measured in days-to-weeks) is the only sanctioned automation
path. thinkScript cannot place orders; automating the ToS client violates
ToS.

- **Fatal for this project: no futures order entry.** Official docs state
  order entry is available only for `EQUITY` and `OPTION` asset types —
  **/ES and futures options cannot be traded**, only their L1 quotes
  streamed (`LEVELONE_FUTURES`). Corroborated by QuantConnect and Lumibot
  integrations. Rumors of future futures support have no timeline.
- SPX works: `$SPX` symbology, full chains
  (`/marketdata/v1/chains?symbol=$SPX`), multi-leg NET_DEBIT/NET_CREDIT
  orders via `orderLegCollection` (OCC-style padded symbols), cancel/
  replace, transactions.
- **Auth pain**: OAuth2 with 7-day hard-expiry refresh tokens — a manual
  browser re-login every week, no compliant workaround. No sandbox/paper
  environment at all. One streamer connection per user.
- Python: `schwab-py` (mature, order templates, streamer) or `schwabdev`.
- **Comparison**: tastytrade — instant self-serve access, sandbox,
  never-expiring personal refresh token, futures + futures-options
  trading, DXLink data. Schwab — approval queue, weekly re-auth, no
  sandbox, no futures execution.

**Conclusion** (drives plan §1.2): execute everything on tastytrade;
keep broker interfaces clean so a Schwab module (SPX-only execution or
quotes) can be added later if ever worth the weekly re-auth.
