"""Conversation-history persistence and repair.

Pure functions over the plain-dict message list the Anthropic API accepts.
The API enforces invariants a persisted history can silently break — every
tool_use answered by a tool_result in the next turn, strictly alternating
roles — and a history saved mid-tool-loop (e.g. a turn abandoned by the
barge-in path between the tool call and its result) violates them. Because
the file reloads on every launch, one bad save used to 400-brick the app;
`sanitize` makes histories self-heal instead. Kept free of I/O state and
model classes so it can be tested without hardware or an API key.
"""

import json
import logging
from datetime import datetime

from lib.atomic_io import read_json, write_json_atomic

log = logging.getLogger("history")


def now_iso():
    """One clock for every `ts` stamp: local time with offset, second precision.
    Shared by save() and the live append sites so a stamp compares cleanly
    against any other regardless of who wrote it."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def time_prefix(ts):
    """Render a stored `ts` as the spoken-time prefix the model sees on user
    messages, e.g. "(1:47pm 8/20/2026) ". Built by hand because strftime has no
    portable no-leading-zero directive. Empty or unparseable stamps render as
    "" — "time unknown" stays silent rather than inventing a moment."""
    if not ts:
        return ""
    try:
        dt = datetime.fromisoformat(ts)
    except ValueError:
        return ""
    hour = dt.hour % 12 or 12
    ampm = "am" if dt.hour < 12 else "pm"
    return f"({hour}:{dt.minute:02d}{ampm} {dt.month}/{dt.day}/{dt.year}) "


def sanitize(history):
    """Make a history safe to send. Drops any tool_use whose tool_result never
    arrived (and any tool_result with no matching tool_use), then merges
    adjacent same-role turns so roles keep alternating."""
    result_ids, use_ids = set(), set()
    for m in history:
        c = m.get("content")
        if isinstance(c, list):
            for b in c:
                if isinstance(b, dict):
                    if b.get("type") == "tool_result":
                        result_ids.add(b.get("tool_use_id"))
                    elif b.get("type") == "tool_use":
                        use_ids.add(b.get("id"))
    cleaned = []
    for m in history:
        c = m.get("content")
        if isinstance(c, list):
            blocks = []
            for b in c:
                if isinstance(b, dict):
                    if b.get("type") == "tool_use" and b.get("id") not in result_ids:
                        continue  # unanswered tool call — its result was lost
                    if b.get("type") == "tool_result" and b.get("tool_use_id") not in use_ids:
                        continue  # result with no surviving tool call
                blocks.append(b)
            if not blocks:
                continue  # message had only orphaned blocks — drop it whole
            m = {**m, "content": blocks}
        cleaned.append(m)
    # Dropping a message can leave two same-role turns adjacent, which the API
    # also rejects; fold them together so roles keep alternating.
    to_list = lambda c: [{"type": "text", "text": c}] if isinstance(c, str) else list(c)
    coalesced = []
    for m in cleaned:
        if coalesced and coalesced[-1]["role"] == m["role"]:
            prev = coalesced[-1]
            pc, mc = prev["content"], m["content"]
            if isinstance(pc, str) and isinstance(mc, str):
                prev["content"] = f"{pc}\n\n{mc}"
            else:
                prev["content"] = to_list(pc) + to_list(mc)
        else:
            content = m["content"] if isinstance(m["content"], str) else list(m["content"])
            coalesced.append({**m, "content": content})
    return coalesced


def trim(history, max_messages):
    """Cap the history and make sure it starts on a plain user message — a
    trim boundary must never orphan a tool_result from its tool_use, or the
    API rejects the conversation."""
    h = list(history[-max_messages:])
    while h and not (h[0].get("role") == "user" and isinstance(h[0].get("content"), str)):
        h.pop(0)
    return h


def load(path):
    """Read a saved history list from `path`. Returns [] when the file is
    missing or unreadable — a lost history must never block startup.

    Messages written before timestamps existed are marked with an empty `ts`
    rather than left bare: bare ones get stamped on the next save, which would
    backdate a whole restored backlog to the moment of the upgrade. An empty
    stamp means "time unknown" and the dashboard shows no time at all, which is
    the honest answer."""
    return [m if "ts" in m else {**m, "ts": ""}
            for m in read_json(path, [], expect=list,
                               warn=lambda e: log.warning(
                                   "could not load conversation history: %s", e))]


def save(path, history):
    """Persist `history` to `path`, sanitized so a turn abandoned mid-tool-loop
    can never write an orphaned tool_use that would 400 the next launch. Written
    atomically (temp + rename) so a power loss mid-save can't corrupt or empty
    the file — the previous good version survives intact.

    New messages are stamped with the local time here, at the one place every
    persisted message passes through, rather than at the five append sites (one
    of which would eventually be missed). The stamp is therefore the end of the
    turn, not the instant the words were said: a turn that runs a long tool loop
    labels its own question with the time the answer landed, seconds later. The
    `ts` key never reaches the API — `llm.cached` strips it."""
    now = now_iso()
    try:
        write_json_atomic(path, [m if "ts" in m else {**m, "ts": now}
                                 for m in sanitize(history)])
    except (OSError, TypeError) as e:
        log.warning("could not save conversation history: %s", e)
