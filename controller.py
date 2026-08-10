"""The agent's control endpoint — a tiny HTTP server inside the agent process.

This is how the dashboard reaches a *live* agent: the dashboard proxies button
presses here (see dashboard.py), and the handler calls straight into the
methods that own the state. The routing is generic — GET /status and
POST /<action> dispatched through the service's `actions` dict — so adding a
control never touches this file: add a method in controller_service.py and
register it there.

Localhost only, and the port (config.CONTROL_PORT) is a soft dependency: if it
can't be bound the agent logs a warning and runs without dashboard control — a
mute button must never stop the voice agent from starting.

Threading: handler threads call the same agent methods the headset-button
thread already calls (they were built for cross-thread use); no extra locking.
"""

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import config as cfg

log = logging.getLogger("controller")


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    # Windows quirk: with the inherited allow_reuse_address=True a second bind
    # on the same port SUCCEEDS, so a port conflict would go undetected instead
    # of failing soft. Single-instance is the lock's job; this just keeps the
    # error honest.
    allow_reuse_address = False


class _Handler(BaseHTTPRequestHandler):
    server_version = "VoiceAgentController/1.0"

    def log_message(self, fmt, *args):  # the dashboard polls every 2 s
        pass

    def _send(self, status, body):
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path != "/status":
            return self._send(404, {"error": "not found"})
        self._send(200, self.server.service.status())

    def do_POST(self):
        action = self.server.service.actions.get(self.path.lstrip("/"))
        if action is None:
            return self._send(404, {"error": "no such action"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            self._send(200, action(payload))
        except ValueError as e:  # bad JSON, or the service refused the payload
            self._send(400, {"ok": False, "error": str(e)})
        except Exception as e:  # noqa: BLE001 - a bad request must not kill the thread
            log.exception("action %s failed", self.path)
            self._send(500, {"ok": False, "error": str(e)})


class ControlServer:
    """Serve `service` on 127.0.0.1:port from a daemon thread."""

    def __init__(self, service, port=None):
        self._service = service
        self._port = cfg.CONTROL_PORT if port is None else port
        self._server = None
        self._thread = None

    @property
    def port(self):
        return self._server.server_address[1] if self._server else None

    def start(self):
        try:
            self._server = _Server(("127.0.0.1", self._port), _Handler)
        except OSError as e:
            log.warning("could not bind control port %s (%s) — the agent runs "
                        "without dashboard control", self._port, e)
            return self
        self._server.service = self._service
        # Tight poll so stop() (which waits out one poll) returns promptly.
        self._thread = threading.Thread(
            target=lambda: self._server.serve_forever(poll_interval=0.1),
            daemon=True, name="controller")
        self._thread.start()
        log.info("control endpoint on 127.0.0.1:%s", self.port)
        return self

    def stop(self):
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=1.0)
        self._server = None
