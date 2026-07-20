"""Local web dashboard for the voice agent.

A visual companion to the running (or resting) agent: browse notes and
folders, read transcripts, inspect the live conversation history, long-term
memory staging, the knowledge base, Discord captures, and session logs — and
adjust the tunable config values (endpointing, settle window, barge-in,
models, ...) from a form instead of editing config.py.

Design constraints:
- Zero new dependencies: stdlib http.server only. Deliberately does NOT import
  notes.py / chromadb — everything is read straight from the JSON/markdown
  files on disk, so the dashboard starts instantly and can run alongside the
  agent without loading a second embedding model. Semantic search stays a
  voice feature; the dashboard's search is a plain substring scan.
- Read-mostly: the only thing it writes is data/config_overrides.json (via
  atomic_io, same as every other state file). Config changes are picked up by
  config.py at the agent's next start — the dashboard never reaches into a
  live process.
- Localhost only: binds 127.0.0.1; this is a private control panel, not a web
  service.

Run:  python dashboard.py [--port 8765] [--no-browser]
"""

import argparse
import json
import re
import threading
import webbrowser
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import categories
import config as cfg
from atomic_io import write_json_atomic
from single_instance import AlreadyRunning, SingleInstance

STATIC_DIR = cfg.BASE_DIR / "dashboard"
DEFAULT_PORT = 8765

NOTE_ID_RE = re.compile(r"^note_[\w.-]+$")
LOG_NAME_RE = re.compile(r"^session_[\w.-]+\.log$")

# --- Tunables metadata --------------------------------------------------------
# The UI is generated from this table: one entry per adjustable config value,
# grouped the way config.py groups them. `type` drives both the control shown
# and server-side validation; min/max are hard bounds enforced on save.
WHISPER_CHOICES = ["tiny.en", "base.en", "small.en", "medium.en", "large-v3"]

TUNABLES = [
    # -- Turn taking ----------------------------------------------------------
    dict(key="CONVO_ENDPOINT_MS", group="Turn taking", label="Conversation endpoint",
         type="int", min=200, max=3000, step=50, unit="ms",
         help="Trailing silence that ends an utterance in conversation mode. Lower = snappier turn-taking, higher = tolerates longer pauses."),
    dict(key="NOTE_ENDPOINT_MS", group="Turn taking", label="Notetaking endpoint",
         type="int", min=200, max=5000, step=50, unit="ms",
         help="Trailing silence that ends an utterance while recording a note."),
    dict(key="CONTINUATION_SETTLE_MS", group="Turn taking", label="Continuation settle",
         type="int", min=0, max=3000, step=50, unit="ms",
         help="After an utterance ends, wait this long before calling the model in case you were only pausing mid-thought. Adds latency but prevents a billed call per pause."),
    dict(key="CONTINUATION_GRACE_MS", group="Turn taking", label="Continuation grace",
         type="int", min=0, max=2000, step=50, unit="ms",
         help="One-time extension when speech is just starting as the settle window expires."),
    dict(key="MAX_CONTINUATION_ROUNDS", group="Turn taking", label="Max continuation rounds",
         type="int", min=1, max=20, step=1, unit="",
         help="Hard cap on settle-window restarts per turn, so background speech can't hold the turn hostage."),
    # -- Voice detection ------------------------------------------------------
    dict(key="VAD_AGGRESSIVENESS", group="Voice detection", label="VAD aggressiveness",
         type="int", min=0, max=3, step=1, unit="",
         help="0 (lenient) to 3 (strictest about calling noise non-speech)."),
    dict(key="SPEECH_PAD_MS", group="Voice detection", label="Speech pre-roll",
         type="int", min=0, max=1000, step=50, unit="ms",
         help="Audio kept from before detected speech so opening syllables aren't clipped."),
    dict(key="TRIGGER_RATIO", group="Voice detection", label="Trigger ratio",
         type="float", min=0.1, max=1.0, step=0.05, unit="",
         help="Fraction of the padded window that must be voiced to start capture."),
    dict(key="MAX_UTTERANCE_S", group="Voice detection", label="Max utterance",
         type="int", min=5, max=120, step=5, unit="s",
         help="Safety cap on a single captured utterance."),
    # -- Barge-in -------------------------------------------------------------
    dict(key="BARGE_IN", group="Barge-in", label="Barge-in enabled",
         type="bool",
         help="Stop speaking when the user starts talking. Best with headphones — on open speakers the mic can hear the agent and self-interrupt."),
    dict(key="BARGE_IN_MS", group="Barge-in", label="Qualifying audio",
         type="int", min=50, max=2000, step=50, unit="ms",
         help="Voiced audio that must accumulate to count as an interruption."),
    dict(key="BARGE_IN_DECAY", group="Barge-in", label="Counter decay",
         type="float", min=0.0, max=1.0, step=0.05, unit="",
         help="How much a non-qualifying frame decays the counter (0 = hard reset, 1 = symmetric). Lower it if the agent over-triggers."),
    dict(key="BARGE_IN_ENERGY", group="Barge-in", label="Energy floor",
         type="int", min=0, max=5000, step=50, unit="RMS",
         help="Absolute loudness (int16 RMS) the user's voice must exceed to count."),
    dict(key="BARGE_IN_ENERGY_RATIO", group="Barge-in", label="Echo ratio",
         type="float", min=1.0, max=10.0, step=0.5, unit="×",
         help="…and must exceed this multiple of the measured echo baseline."),
    dict(key="BARGE_IN_CALIB_MS", group="Barge-in", label="Echo calibration",
         type="int", min=100, max=2000, step=50, unit="ms",
         help="Initial playback window used to measure the echo baseline."),
    # -- Backchannel ----------------------------------------------------------
    dict(key="BACKCHANNEL_WORDS", group="Backchannel", label="Filler words",
         type="words",
         help="If a barge-in transcribes to nothing but these words (\"yeah\", \"uh-huh\"), the reply resumes instead of stopping. Real commands like \"stop\" must stay absent."),
    dict(key="BACKCHANNEL_MAX_WORDS", group="Backchannel", label="Max filler length",
         type="int", min=1, max=10, step=1, unit="words",
         help="An utterance longer than this is a real turn, not a filler."),
    # -- Models & tokens ------------------------------------------------------
    dict(key="CONVO_MODEL", group="Models & tokens", label="Conversation model",
         type="choice", choices=[
             dict(value=mid, label=cfg.CONVO_MODEL_LABELS.get(mid, mid))
             for mid in cfg.CONVO_MODELS.values()],
         help="Default model for spoken back-and-forth. Voice switching (set_conversation_model) still works and resets to this on restart."),
    dict(key="SUMMARY_MODEL", group="Models & tokens", label="Summary model",
         type="text",
         help="Model used for note summaries (quality matters more than latency here)."),
    dict(key="CONVO_MAX_TOKENS", group="Models & tokens", label="Reply budget",
         type="int", min=256, max=16384, step=256, unit="tokens",
         help="Must cover tool calls too — a saved note travels inside the reply. Billed as used, so a roomy cap costs nothing on short replies."),
    dict(key="SUMMARY_MAX_TOKENS", group="Models & tokens", label="Summary budget",
         type="int", min=256, max=8192, step=256, unit="tokens",
         help="Token budget for one note summary."),
    dict(key="CONVO_MAX_TOOL_ROUNDS", group="Models & tokens", label="Max tool rounds",
         type="int", min=1, max=50, step=1, unit="",
         help="Safety cap on model→tool→model rounds in one turn."),
    # -- Speech engines -------------------------------------------------------
    dict(key="WHISPER_MODEL", group="Speech engines", label="Whisper model",
         type="choice", choices=[dict(value=m, label=m) for m in WHISPER_CHOICES],
         help="base.en is faster, medium.en more accurate; small.en is the balanced default."),
    dict(key="TTS_RATE", group="Speech engines", label="Speaking rate",
         type="int", min=80, max=400, step=5, unit="wpm",
         help="Text-to-speech words per minute."),
    dict(key="TTS_VOICE", group="Speech engines", label="TTS voice",
         type="text", nullable=True,
         help="SAPI voice id substring; leave empty for the system default."),
    # -- Memory & search ------------------------------------------------------
    dict(key="HISTORY_MAX_MESSAGES", group="Memory & search", label="History window",
         type="int", min=4, max=200, step=2, unit="msgs",
         help="Messages kept when persisting/restoring conversation history."),
    dict(key="SEARCH_RESULTS", group="Memory & search", label="Note search results",
         type="int", min=1, max=20, step=1, unit="",
         help="Results per search_notes call."),
    dict(key="KB_SEARCH_RESULTS", group="Memory & search", label="Knowledge results",
         type="int", min=1, max=20, step=1, unit="",
         help="Chunks returned per search_knowledge call."),
    dict(key="MEMORY_SEARCH_RESULTS", group="Memory & search", label="Memory results",
         type="int", min=1, max=10, step=1, unit="",
         help="Summaries returned per search_past_conversations call."),
    # -- Headset button -------------------------------------------------------
    dict(key="MEDIA_KEEPALIVE", group="Headset button", label="Media keepalive",
         type="bool",
         help="Silent audio stream that keeps Bluetooth buttons routed here and the dongle awake. Costs some headset battery."),
    dict(key="MEDIA_CLICK_DEDUPE_S", group="Headset button", label="Click dedupe",
         type="float", min=0.05, max=1.0, step=0.05, unit="s",
         help="A press arriving on both listener channels within this window counts once."),
]

TUNABLES_BY_KEY = {t["key"]: t for t in TUNABLES}


def validate_override(meta, value):
    """Validate + coerce one override value against its TUNABLES entry.
    Returns (ok, coerced_or_error)."""
    t = meta["type"]
    try:
        if t == "bool":
            if not isinstance(value, bool):
                return False, "expected true/false"
            return True, value
        if t == "int":
            value = int(value)
        elif t == "float":
            value = float(value)
        elif t == "choice":
            value = str(value)
            if value not in [c["value"] for c in meta["choices"]]:
                return False, "not one of the allowed choices"
            return True, value
        elif t == "text":
            if value is None or value == "":
                if meta.get("nullable"):
                    return True, None
                return False, "may not be empty"
            return True, str(value).strip()
        elif t == "words":
            if not isinstance(value, list):
                return False, "expected a list of words"
            words = sorted({str(w).strip().lower() for w in value if str(w).strip()})
            if not words:
                return False, "word list may not be empty"
            return True, words
        else:
            return False, f"unknown type {t}"
    except (TypeError, ValueError):
        return False, f"expected a number"
    lo, hi = meta.get("min"), meta.get("max")
    if lo is not None and value < lo:
        return False, f"below minimum {lo}"
    if hi is not None and value > hi:
        return False, f"above maximum {hi}"
    return True, value


def validate_payload(payload):
    """Validate a {name: value_or_null} dict from the config form. Returns
    (overrides, errors): overrides is the cleaned dict ready to persist (null
    values dropped = reset to default), errors is {name: message}."""
    overrides, errors = {}, {}
    if not isinstance(payload, dict):
        return overrides, {"_": "expected an object of overrides"}
    for name, value in payload.items():
        meta = TUNABLES_BY_KEY.get(name)
        if meta is None:
            errors[name] = "not an adjustable setting"
            continue
        if value is None:
            continue  # reset to default — simply omit from the file
        ok, out = validate_override(meta, value)
        if ok:
            overrides[name] = out
        else:
            errors[name] = out
    return overrides, errors


# --- Data readers (all straight off disk, fresh per request) ------------------

def _read_json(path, fallback):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return fallback


def parse_frontmatter(text):
    # Mirror of notes.parse_frontmatter — duplicated so the dashboard never
    # imports notes.py (which drags in chromadb + the embedding stack).
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n?", text, re.DOTALL)
    if not m:
        return {}, text
    fields = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            fields[key.strip()] = value.strip()
    return fields, text[m.end():]


def agent_running():
    """True if the voice agent currently holds the single-instance lock."""
    try:
        with SingleInstance(cfg.LOCK_PATH):
            return False
    except AlreadyRunning:
        return True
    except OSError:
        return False


def folder_registry():
    categories.load_categories()  # pick up folders created by voice since we started
    return categories.NOTE_CATEGORIES


def note_index():
    return _read_json(cfg.INDEX_PATH, {})


def api_overview():
    index = note_index()
    folders = folder_registry()
    counts = {}
    for info in index.values():
        slug = info.get("category") or categories.DEFAULT_CATEGORY
        counts[slug] = counts.get(slug, 0) + 1
    history = _read_json(cfg.HISTORY_PATH, [])
    pending = _read_json(cfg.MEMORY_PENDING_PATH, [])
    manifest = _read_json(cfg.KNOWLEDGE_MANIFEST, {})
    logs = sorted(cfg.LOG_DIR.glob("session_*.log")) if cfg.LOG_DIR.exists() else []

    # Notes per day over the last 30 days, for the activity chart.
    by_day = {}
    for info in index.values():
        day = (info.get("date") or "")[:10]
        if day:
            by_day[day] = by_day.get(day, 0) + 1
    days = sorted(by_day)[-30:]

    return {
        "agent_running": agent_running(),
        "total_notes": len(index),
        "folder_counts": [
            {"slug": slug, "display": meta["display"], "count": counts.get(slug, 0)}
            for slug, meta in folders.items()
        ],
        "unknown_folders": [
            {"slug": slug, "count": n} for slug, n in counts.items()
            if slug not in folders
        ],
        "history_messages": len(history),
        "memory_pending": len(pending),
        "knowledge_docs": len(manifest),
        "log_files": len(logs),
        "convo_model": cfg.convo_model_label(cfg.CONVO_MODEL),
        "summary_model": cfg.SUMMARY_MODEL,
        "whisper_model": cfg.WHISPER_MODEL,
        "overrides_active": len(_read_json(cfg.OVERRIDES_PATH, {})),
        "activity": [{"day": d, "count": by_day[d]} for d in days],
        "recent_notes": [
            {"id": nid, **info}
            for nid, info in sorted(note_index().items(), reverse=True)[:8]
        ],
    }


def api_notes(folder=None):
    index = note_index()
    folders = folder_registry()
    items = sorted(index.items(), reverse=True)
    if folder:
        items = [(nid, info) for nid, info in items
                 if (info.get("category") or categories.DEFAULT_CATEGORY) == folder]
    return {
        "folders": [
            {"slug": slug, "display": meta["display"],
             "description": meta.get("description", "")}
            for slug, meta in folders.items()
        ],
        "notes": [{"id": nid, **info} for nid, info in items],
    }


def api_note(note_id):
    index = note_index()
    info = index.get(note_id)
    if not info or not NOTE_ID_RE.match(note_id):
        return None
    folders = folder_registry()
    slug = info.get("category") or categories.DEFAULT_CATEGORY
    folder = folders.get(slug, {}).get("folder", slug)
    base = cfg.DATA_DIR / folder
    summary, transcript = "", ""
    spath = base / f"{note_id}.md"
    if spath.exists():
        _, summary = parse_frontmatter(spath.read_text(encoding="utf-8"))
    tpath = base / f"{note_id}.transcript.md"
    if tpath.exists():
        transcript = tpath.read_text(encoding="utf-8")
    else:
        ppath = cfg.PENDING_DIR / f"{note_id}.md"
        if ppath.exists():
            transcript = ppath.read_text(encoding="utf-8")
    return {"id": note_id, **info,
            "folder_display": folders.get(slug, {}).get("display", slug),
            "summary": summary, "transcript": transcript}


def api_search(query):
    """Plain substring search over titles and summary bodies. Not semantic —
    Chroma stays out of the dashboard by design (see module docstring)."""
    q = query.strip().lower()
    if not q:
        return {"results": []}
    index = note_index()
    folders = folder_registry()
    results = []
    for nid, info in sorted(index.items(), reverse=True):
        slug = info.get("category") or categories.DEFAULT_CATEGORY
        folder = folders.get(slug, {}).get("folder", slug)
        title = info.get("title", "")
        snippet = ""
        hit = q in title.lower()
        path = cfg.DATA_DIR / folder / f"{nid}.md"
        if path.exists():
            try:
                _, body = parse_frontmatter(path.read_text(encoding="utf-8"))
                pos = body.lower().find(q)
                if pos >= 0:
                    hit = True
                    start = max(0, pos - 60)
                    snippet = ("…" if start else "") + " ".join(
                        body[start:pos + 160].split())
            except OSError:
                pass
        if hit:
            results.append({"id": nid, **info, "snippet": snippet})
        if len(results) >= 30:
            break
    return {"results": results}


def api_config():
    overrides = _read_json(cfg.OVERRIDES_PATH, {})
    out = []
    for meta in TUNABLES:
        key = meta["key"]
        current = getattr(cfg, key)
        default = cfg.CONFIG_DEFAULTS.get(key, current)
        if isinstance(current, frozenset):
            current = sorted(current)
        if isinstance(default, frozenset):
            default = sorted(default)
        out.append({**meta, "default": default, "value": current,
                    "override": overrides.get(key)})
    return {"tunables": out, "overrides_path": str(cfg.OVERRIDES_PATH),
            "agent_running": agent_running()}


def save_config(payload):
    overrides, errors = validate_payload(payload)
    if errors:
        return {"ok": False, "errors": errors}
    write_json_atomic(cfg.OVERRIDES_PATH, overrides)
    # Apply to this process too so the dashboard's own displays (overview
    # model label etc.) reflect the change immediately.
    cfg.apply_overrides(overrides)
    for key, default in cfg.CONFIG_DEFAULTS.items():
        if key not in overrides:
            setattr(cfg, key, default)
    return {"ok": True, "saved": sorted(overrides)}


def api_history(limit=200):
    messages = _read_json(cfg.HISTORY_PATH, [])
    return {"messages": messages[-limit:], "total": len(messages)}


def api_memory():
    return {"pending": _read_json(cfg.MEMORY_PENDING_PATH, []),
            "min_messages": cfg.MEMORY_MIN_MESSAGES}


def api_knowledge():
    manifest = _read_json(cfg.KNOWLEDGE_MANIFEST, {})
    docs = [{"hash": h[:12], **info} for h, info in manifest.items()]
    docs.sort(key=lambda d: d.get("ingested", ""), reverse=True)
    return {"docs": docs, "dir": str(cfg.KNOWLEDGE_DIR)}


def _tail_lines(path, n):
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return f.readlines()[-n:]
    except OSError:
        return []


def api_discord():
    trades = [ln.rstrip() for ln in _tail_lines(cfg.DISCORD_TRADES_PATH, 50) if ln.strip()]
    log = [ln.rstrip() for ln in _tail_lines(cfg.DISCORD_LOG_PATH, 120)]
    return {"available": cfg.DISCORD_DIR.exists(),
            "trades": trades, "log": log,
            "dir": str(cfg.DISCORD_DIR)}


def api_logs():
    files = []
    if cfg.LOG_DIR.exists():
        for p in sorted(cfg.LOG_DIR.glob("session_*.log"), reverse=True):
            st = p.stat()
            files.append({"name": p.name, "size": st.st_size,
                          "modified": datetime.fromtimestamp(st.st_mtime)
                          .isoformat(timespec="seconds")})
    return {"files": files}


def api_log(name, lines=300):
    if not LOG_NAME_RE.match(name):
        return None
    path = cfg.LOG_DIR / name
    if not path.exists():
        return None
    return {"name": name, "lines": [ln.rstrip("\n") for ln in _tail_lines(path, lines)]}


# --- HTTP plumbing ------------------------------------------------------------

MIME = {".html": "text/html; charset=utf-8",
        ".css": "text/css; charset=utf-8",
        ".js": "text/javascript; charset=utf-8",
        ".svg": "image/svg+xml"}


class Handler(BaseHTTPRequestHandler):
    server_version = "VoiceAgentDashboard/1.0"

    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    def _send(self, status, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False).encode("utf-8")
        elif isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _static(self, name):
        path = STATIC_DIR / name
        # STATIC_DIR files only — never serve arbitrary paths.
        if ".." in name or "/" in name or "\\" in name or not path.is_file():
            return self._send(404, {"error": "not found"})
        ctype = MIME.get(path.suffix, "application/octet-stream")
        self._send(200, path.read_bytes().decode("utf-8"), ctype)

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)

        def arg(name, default=None):
            return q.get(name, [default])[0]

        try:
            route = url.path
            if route in ("/", "/index.html"):
                return self._static("index.html")
            if route in ("/style.css", "/app.js"):
                return self._static(route.lstrip("/"))
            if route == "/api/overview":
                return self._send(200, api_overview())
            if route == "/api/notes":
                return self._send(200, api_notes(arg("folder")))
            if route == "/api/note":
                note = api_note(arg("id", ""))
                return self._send(200, note) if note else self._send(404, {"error": "no such note"})
            if route == "/api/search":
                return self._send(200, api_search(arg("q", "")))
            if route == "/api/config":
                return self._send(200, api_config())
            if route == "/api/history":
                return self._send(200, api_history(int(arg("limit", "200"))))
            if route == "/api/memory":
                return self._send(200, api_memory())
            if route == "/api/knowledge":
                return self._send(200, api_knowledge())
            if route == "/api/discord":
                return self._send(200, api_discord())
            if route == "/api/logs":
                return self._send(200, api_logs())
            if route == "/api/log":
                log = api_log(arg("name", ""), int(arg("lines", "300")))
                return self._send(200, log) if log else self._send(404, {"error": "no such log"})
            return self._send(404, {"error": "not found"})
        except Exception as e:  # one bad request must not kill the server
            return self._send(500, {"error": str(e)})

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/api/config":
            return self._send(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            result = save_config(payload)
            return self._send(200 if result["ok"] else 400, result)
        except (ValueError, TypeError) as e:
            return self._send(400, {"error": f"bad request: {e}"})
        except Exception as e:
            return self._send(500, {"error": str(e)})


def main():
    ap = argparse.ArgumentParser(description="Voice agent dashboard")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--no-browser", action="store_true",
                    help="don't open the dashboard in the default browser")
    args = ap.parse_args()

    cfg.ensure_dirs()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"Voice agent dashboard: {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, [url]).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
