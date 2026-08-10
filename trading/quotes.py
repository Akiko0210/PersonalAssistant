"""Real-time quotes over the DXLink websocket, with REST snapshot fallback.

A QuoteService runs one daemon thread speaking the DXLink protocol
(handshake ported from Tasty-Web's option-chain-data/socket.js, checked
against the official streaming guide): SETUP -> AUTH (api-quote-token,
fetched fresh on every connect; it lasts 24 h) -> CHANNEL_REQUEST(FEED)
-> FEED_SETUP (COMPACT: Quote bid/ask, Trade last) -> FEED_SUBSCRIPTION.
KEEPALIVEs are echoed and sent every ~30 s. Any disconnect or auth loss
reconnects after 3 s and resubscribes everything.

The latest tick per streamer symbol sits in an in-memory cache; get()
optionally waits briefly for a first tick after subscribing. Sandbox
accounts have no real-time entitlement — callers fall back to
snapshot_merge() (REST /market-data/by-type) or cope with None.
"""

import json
import threading
import time

from trading.models import Quote

RECONNECT_DELAY_S = 3
KEEPALIVE_EVERY_S = 30
FEED_CHANNEL = 1

QUOTE_FIELDS = ["eventSymbol", "bidPrice", "askPrice"]
TRADE_FIELDS = ["eventSymbol", "price"]


def _f(v):
    """DXLink numeric field -> float or None (handles 'NaN', null)."""
    try:
        x = float(v)
        return None if x != x else x
    except (TypeError, ValueError):
        return None


class QuoteService:
    def __init__(self, client):
        self._client = client                    # TastyClient (token fetch)
        self._quotes = {}                        # streamer symbol -> Quote
        self._subs = set()                       # Quote subscriptions
        self._trade_subs = set()                 # Trade subscriptions (last)
        self._lock = threading.Lock()
        self._ws = None
        self._ws_lock = threading.Lock()
        self._thread = None
        self._stopping = False
        self._feed_ready = threading.Event()
        self._last_error = None

    # --- public ----------------------------------------------------------

    def start(self):
        """Idempotent; returns False (with .last_error set) if the quote
        token could not be fetched — e.g. sandbox without entitlement."""
        if self._thread and self._thread.is_alive():
            return True
        try:
            self._client.quote_token()  # probe now for a friendly error
        except Exception as e:
            self._last_error = str(e)
            return False
        self._stopping = False
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="dxlink-quotes")
        self._thread.start()
        return True

    def subscribe(self, symbols, trade=False):
        """Subscribe streamer symbols for Quote (and optionally Trade)
        events. Safe before the feed is up — everything (re)subscribes on
        CHANNEL_OPENED."""
        symbols = [s for s in symbols if s]
        with self._lock:
            new_q = [s for s in symbols if s not in self._subs]
            self._subs.update(new_q)
            new_t = []
            if trade:
                new_t = [s for s in symbols if s not in self._trade_subs]
                self._trade_subs.update(new_t)
        add = ([{"type": "Quote", "symbol": s} for s in new_q] +
               [{"type": "Trade", "symbol": s} for s in new_t])
        if add and self._feed_ready.is_set():
            self._send({"type": "FEED_SUBSCRIPTION", "channel": FEED_CHANNEL,
                        "add": add})

    def get(self, symbol, wait_s=0.0):
        """Latest Quote for a streamer symbol, or None. wait_s > 0 waits
        for a first tick (after a fresh subscribe)."""
        deadline = time.time() + wait_s
        while True:
            with self._lock:
                q = self._quotes.get(symbol)
            if q and (q.bid is not None or q.last is not None):
                return q
            if time.time() >= deadline:
                return q
            time.sleep(0.05)

    def snapshot_merge(self, items, api_to_streamer):
        """Merge REST /market-data/by-type items into the cache.
        api_to_streamer maps the API symbol to the cache key."""
        now = time.time()
        with self._lock:
            for it in items or []:
                key = api_to_streamer.get(it.get("symbol"))
                if not key:
                    continue
                q = self._quotes.get(key) or Quote(symbol=key)
                # None-guards, not `or`: 0.0 is a real price (a deep-OTM option
                # with no buyer bids 0.00), and the old `or`-merge read it as
                # "missing" — keeping a stale non-zero bid standing, and on a
                # fresh Quote dropping the zero entirely. The streaming path
                # (_feed_data) already assigns directly.
                bid = _f(it.get("bid"))
                if bid is not None:
                    q.bid = bid
                ask = _f(it.get("ask"))
                if ask is not None:
                    q.ask = ask
                last = _f(it.get("last"))
                if last is None:
                    last = _f(it.get("mark"))  # mark fallback, as before
                if last is not None:
                    q.last = last
                q.updated_at = now
                self._quotes[key] = q

    @property
    def last_error(self):
        return self._last_error

    def stop(self):
        self._stopping = True
        with self._ws_lock:
            if self._ws:
                try:
                    self._ws.close()
                except Exception:
                    pass

    # --- websocket internals ---------------------------------------------

    def _send(self, msg):
        with self._ws_lock:
            if not self._ws:
                return
            try:
                self._ws.send(json.dumps(msg))
            except Exception as e:
                self._last_error = str(e)

    def _run(self):
        import websocket  # lazy: keep the dependency out of dashboard start
        while not self._stopping:
            self._feed_ready.clear()
            try:
                tok = self._client.quote_token()
                url, token = tok["dxlink-url"], tok["token"]
            except Exception as e:
                self._last_error = f"quote token: {e}"
                time.sleep(RECONNECT_DELAY_S)
                continue

            def on_open(ws):
                self._send({"type": "SETUP", "channel": 0, "version": "0.1",
                            "keepaliveTimeout": 60, "acceptKeepaliveTimeout": 60})

            def on_message(ws, raw):
                self._handle(json.loads(raw), token)

            def on_error(ws, err):
                self._last_error = str(err)

            ws = websocket.WebSocketApp(
                url, on_open=on_open, on_message=on_message, on_error=on_error)
            with self._ws_lock:
                self._ws = ws
            stop_beat = threading.Event()
            threading.Thread(target=self._heartbeat, args=(stop_beat,),
                             daemon=True, name="dxlink-keepalive").start()
            try:
                ws.run_forever()  # returns on close/error
            except Exception as e:
                self._last_error = str(e)
            finally:
                stop_beat.set()
                self._feed_ready.clear()
                with self._ws_lock:
                    self._ws = None
            if not self._stopping:
                time.sleep(RECONNECT_DELAY_S)

    def _heartbeat(self, stop):
        while not stop.wait(KEEPALIVE_EVERY_S):
            self._send({"type": "KEEPALIVE", "channel": 0})

    def _handle(self, msg, token):
        t = msg.get("type")
        if t == "SETUP":
            self._send({"type": "AUTH", "channel": 0, "token": token})
        elif t == "AUTH_STATE":
            if msg.get("state") == "AUTHORIZED":
                self._send({"type": "CHANNEL_REQUEST", "channel": FEED_CHANNEL,
                            "service": "FEED",
                            "parameters": {"contract": "AUTO"}})
            elif self._feed_ready.is_set():
                # Lost auth mid-stream (token aged out) — force reconnect;
                # _run fetches a fresh token.
                with self._ws_lock:
                    if self._ws:
                        self._ws.close()
        elif t == "CHANNEL_OPENED" and msg.get("channel") == FEED_CHANNEL:
            self._send({"type": "FEED_SETUP", "channel": FEED_CHANNEL,
                        "acceptAggregationPeriod": 0.1,
                        "acceptDataFormat": "COMPACT",
                        "acceptEventFields": {"Quote": QUOTE_FIELDS,
                                              "Trade": TRADE_FIELDS}})
            with self._lock:
                add = ([{"type": "Quote", "symbol": s} for s in self._subs] +
                       [{"type": "Trade", "symbol": s} for s in self._trade_subs])
            self._feed_ready.set()
            if add:
                self._send({"type": "FEED_SUBSCRIPTION",
                            "channel": FEED_CHANNEL, "reset": True, "add": add})
        elif t == "FEED_DATA":
            self._feed_data(msg.get("data") or [])
        elif t == "KEEPALIVE":
            self._send({"type": "KEEPALIVE", "channel": 0})

    def _feed_data(self, data):
        """COMPACT format: ["Quote", [flat values...]] pairs."""
        now = time.time()
        for i in range(0, len(data) - 1, 2):
            etype, values = data[i], data[i + 1]
            stride = 3 if etype == "Quote" else 2 if etype == "Trade" else 0
            if not stride or not isinstance(values, list):
                continue
            with self._lock:
                for j in range(0, len(values) - stride + 1, stride):
                    sym = values[j]
                    q = self._quotes.get(sym) or Quote(symbol=sym)
                    if etype == "Quote":
                        q.bid = _f(values[j + 1])
                        q.ask = _f(values[j + 2])
                    else:
                        q.last = _f(values[j + 1])
                    q.updated_at = now
                    self._quotes[sym] = q
