"""Read-only access to the Discord Notifier's captured data.

A sibling project ("Discord Notifier") monitors Discord notifications and writes
them to two append-only files:

  * ``discord_log.md`` — every notification as a structured entry
    (sender, channel, capture time, body), entries separated by ``---``.
  * ``trades.txt``     — just the trade-ready lines (no timestamp/sender).

The data is tiny, so we simply re-read and parse the files on every call. That
keeps answers always fresh (the monitor may append at any time) and avoids any
indexing/caching machinery. All methods return human-readable strings so they
can be handed straight back to Claude as tool results.
"""

import logging
import re
from datetime import datetime

import config as cfg

log = logging.getLogger("discord")

# A log entry looks like:
#   ### Dan Sheridan (#channel-name, Text Channels)
#   **Time:** 2026-06-22 12:18:11
#
#   <body, possibly multi-line>
_HEADER_RE = re.compile(r"^###\s+(?P<sender>.+?)(?:\s+\((?P<channel>.+)\))?\s*$")
_TIME_RE = re.compile(r"^\*\*Time:\*\*\s*(?P<time>.+?)\s*$")
_TIME_FMT = "%Y-%m-%d %H:%M:%S"
# Discord wraps text in Unicode bidi formatting chars (isolates, marks, overrides);
# strip them so senders/bodies are clean for matching and text-to-speech.
_BIDI_RE = re.compile("[‎‏‪-‮⁦-⁩]")


def _clean(text: str) -> str:
    return _BIDI_RE.sub("", text)


class DiscordData:
    def __init__(self):
        self.log_path = cfg.DISCORD_LOG_PATH
        self.trades_path = cfg.DISCORD_TRADES_PATH

    # --- parsing -------------------------------------------------------------
    def _parse_log(self) -> list:
        """Return entries as dicts: {sender, channel, time(datetime|None), body}.

        Robust to malformed/partial blocks — anything we can't read is skipped.
        """
        if not self.log_path.exists():
            return []
        text = self.log_path.read_text(encoding="utf-8")
        entries = []
        for block in text.split("\n---\n"):
            block = _clean(block).strip()
            if not block or block.startswith("# Discord Notification Log"):
                continue
            sender = channel = None
            when = None
            body_lines = []
            for line in block.splitlines():
                if sender is None:
                    m = _HEADER_RE.match(line)
                    if m:
                        sender = m.group("sender").strip()
                        channel = (m.group("channel") or "").strip() or None
                        continue
                mt = _TIME_RE.match(line)
                if mt and when is None and not body_lines:
                    try:
                        when = datetime.strptime(mt.group("time"), _TIME_FMT)
                    except ValueError:
                        when = None
                    continue
                body_lines.append(line)
            if sender is None and not body_lines:
                continue
            body = "\n".join(body_lines).strip()
            entries.append(
                {"sender": sender or "Unknown", "channel": channel, "time": when, "body": body}
            )
        return entries

    @staticmethod
    def _one_line(text: str, limit: int = 300) -> str:
        flat = " ".join(text.split())
        return flat[: limit - 1] + "…" if len(flat) > limit else flat

    def _format(self, entry: dict) -> str:
        when = entry["time"].strftime("%b %d %H:%M") if entry["time"] else "unknown time"
        return f"{entry['sender']} ({when}): {self._one_line(entry['body'])}"

    # --- retrieval (used as Claude tools) ------------------------------------
    def recent_messages(self, n: int = 10, date: str = None) -> str:
        entries = self._parse_log()
        if not entries:
            return "No Discord notifications have been captured yet."

        if date:
            target = self._resolve_date(date)
            if target is None:
                return f"I couldn't understand the date '{date}'. Use 'today' or YYYY-MM-DD."
            entries = [e for e in entries if e["time"] and e["time"].date() == target]
            if not entries:
                return f"No Discord notifications found for {target.isoformat()}."

        # chronological order; keep the last n
        entries.sort(key=lambda e: e["time"] or datetime.min)
        chosen = entries[-n:]
        return "\n".join(self._format(e) for e in chosen)

    def search_messages(self, query: str, sender: str = None, n: int = 15) -> str:
        entries = self._parse_log()
        if not entries:
            return "No Discord notifications have been captured yet."

        q = (query or "").lower().strip()
        s = (sender or "").lower().strip()
        hits = []
        for e in entries:
            if s and s not in (e["sender"] or "").lower():
                continue
            if q:
                haystack = e["body"].lower()
                if not s:  # also match the sender when no explicit sender filter
                    haystack += " " + (e["sender"] or "").lower()
                if q not in haystack:
                    continue
            hits.append(e)

        if not hits:
            who = f" from {sender}" if sender else ""
            what = f" matching '{query}'" if query else ""
            return f"No Discord notifications found{what}{who}."

        hits.sort(key=lambda e: e["time"] or datetime.min, reverse=True)
        return "\n".join(self._format(e) for e in hits[:n])

    def recent_trades(self, n: int = 10) -> str:
        if not self.trades_path.exists():
            return "No trades have been captured yet."
        lines = [
            ln.strip()
            for ln in self.trades_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        if not lines:
            return "No trades have been captured yet."
        chosen = lines[-n:]
        return "\n".join(f"{i}. {ln}" for i, ln in enumerate(chosen, 1))

    # --- helpers -------------------------------------------------------------
    @staticmethod
    def _resolve_date(date: str):
        date = date.strip().lower()
        if date in ("today", "now"):
            return datetime.now().date()
        if date == "yesterday":
            from datetime import timedelta

            return (datetime.now() - timedelta(days=1)).date()
        try:
            return datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            return None
