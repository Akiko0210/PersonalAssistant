"""Per-persona working record: what each assistant is in the middle of.

Retrieval re-ranks the past every turn, and anything that loses the ranking
might as well not exist: Tom used the "red" account nickname in 8 of 8 turns
where the fact reached him and 0 of 16 where it didn't, and the campaign plan
lost to course transcripts on 9/30 (docs/analysis/2026-10-01-tracking-
failures.md). The working record is the opposite of retrieval: a small chart
the persona writes deliberately (tools/working_state_tools.py) and is shown
in full every turn (block), so an open campaign, a tracked pattern or a list
item never has to win anything to be seen.

Records have a lifecycle (open → closed) and a body holding the current state;
entries are the dated happenings under a record. Rows belong to one persona:
writes take the writer's key and nothing else, reads go through
agents.readable_owners — the same isolation rule as the private Chroma stores.

SQLite like the trade log, and the same connection-per-call shape: the memory
worker, a delegation thread and the foreground turn may all touch it.
"""

import logging
import sqlite3
from contextlib import contextmanager

import config as cfg
from brain import agents
from brain import history as hist

log = logging.getLogger("records")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id      INTEGER PRIMARY KEY,
    owner   TEXT NOT NULL,              -- persona key; only ctx.active_agent reaches it
    kind    TEXT NOT NULL DEFAULT '',   -- free label: campaign | pattern | item | ...
    title   TEXT NOT NULL,
    body    TEXT NOT NULL DEFAULT '',   -- the current state, <= WORKING_BODY_CHARS
    status  TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'closed')),
    created TEXT NOT NULL,              -- history.now_iso()
    updated TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS entries (
    id        INTEGER PRIMARY KEY,
    record_id INTEGER NOT NULL REFERENCES records ON DELETE CASCADE,
    ts        TEXT NOT NULL,
    text      TEXT NOT NULL,            -- <= WORKING_ENTRY_CHARS
    model     TEXT);                    -- the writer's live model
CREATE INDEX IF NOT EXISTS records_owner ON records (owner, status);
CREATE INDEX IF NOT EXISTS entries_record ON entries (record_id, ts);
"""

_PREAMBLE = (
    "Your working record — what you are in the middle of for the user, kept "
    "by you with update_record, add_entry and close_record. It is current and "
    "authoritative: where the Background or an older exchange disagrees with "
    "it, the record wins.\n")


@contextmanager
def _db():
    """One transaction (committed, or rolled back on error). The path is read
    per call so tests can repoint it."""
    path = cfg.WORKING_STATE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    try:
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys = ON")   # entries follow their record
        db.executescript(_SCHEMA)
        with db:
            yield db
    finally:
        db.close()


# --- writes: the writer's own records only ------------------------------------
def _checked(text, limit, what):
    """`text` stripped, or ValueError worded for the model to act on — an
    over-long body is condensed, not truncated, so no rule is silently cut."""
    text = (text or "").strip()
    if len(text) > limit:
        raise ValueError(
            f"That {what} is {len(text)} characters; the limit is {limit}. "
            "Condense it — keep the rules, levels and status, move history "
            "into entries — and try again.")
    return text


def _own(db, owner, rid):
    """The writer's record `rid` (open or closed), or ValueError."""
    row = db.execute("SELECT * FROM records WHERE id = ? AND owner = ?",
                     (rid, owner)).fetchone()
    if row is None:
        raise ValueError(f"You have no record #{rid}.")
    return row


def _wrote(owner, rid, action, title, delta):
    # The one line per write that scripts/audit_sessions.py counts.
    log.info("record %s #%d %s %r (%+d chars)", owner, rid, action, title, delta)


def create(owner, title, *, kind="", body=""):
    """A new open record; returns its id."""
    title = _checked(title, 200, "title")
    if not title:
        raise ValueError("A new record needs a title.")
    body = _checked(body, cfg.WORKING_BODY_CHARS, "body")
    now = hist.now_iso()
    with _db() as db:
        rid = db.execute(
            "INSERT INTO records (owner, kind, title, body, created, updated) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (owner, (kind or "").strip().lower(), title, body, now, now)).lastrowid
    _wrote(owner, rid, "created", title, len(body))
    return rid


def update(owner, rid, *, title=None, body=None, kind=None):
    """Replace any of title, body, kind on an open record."""
    with _db() as db:
        row = _own(db, owner, rid)
        if row["status"] != "open":
            raise ValueError(f"Record #{rid} is closed; start a new one.")
        new_title = _checked(title, 200, "title") if title else row["title"]
        new_body = (row["body"] if body is None
                    else _checked(body, cfg.WORKING_BODY_CHARS, "body"))
        new_kind = row["kind"] if kind is None else kind.strip().lower()
        db.execute("UPDATE records SET title = ?, body = ?, kind = ?, updated = ? "
                   "WHERE id = ?", (new_title, new_body, new_kind,
                                    hist.now_iso(), rid))
    _wrote(owner, rid, "updated", new_title, len(new_body) - len(row["body"]))


def add_entry(owner, rid, text, *, model=None):
    """A dated line under an open record."""
    text = _checked(text, cfg.WORKING_ENTRY_CHARS, "entry")
    if not text:
        raise ValueError("The entry is empty.")
    now = hist.now_iso()
    with _db() as db:
        row = _own(db, owner, rid)
        if row["status"] != "open":
            raise ValueError(f"Record #{rid} is closed; start a new one.")
        db.execute("INSERT INTO entries (record_id, ts, text, model) "
                   "VALUES (?, ?, ?, ?)", (rid, now, text, model))
        db.execute("UPDATE records SET updated = ? WHERE id = ?", (now, rid))
    _wrote(owner, rid, "entry", row["title"], len(text))


def close(owner, rid, outcome, *, model=None):
    """Close a record with its outcome — the closing entry is the point of
    closing, so it is required."""
    outcome = _checked(outcome, cfg.WORKING_ENTRY_CHARS, "outcome")
    if not outcome:
        raise ValueError("Say how it ended: closing needs an outcome.")
    now = hist.now_iso()
    with _db() as db:
        row = _own(db, owner, rid)
        if row["status"] != "open":
            raise ValueError(f"Record #{rid} is already closed.")
        db.execute("INSERT INTO entries (record_id, ts, text, model) "
                   "VALUES (?, ?, ?, ?)", (rid, now, "Closed: " + outcome, model))
        db.execute("UPDATE records SET status = 'closed', updated = ? "
                   "WHERE id = ?", (now, rid))
    _wrote(owner, rid, "closed", row["title"], len(outcome))


# --- reads ----------------------------------------------------------------------
def title(caller, rid):
    """Record `rid`'s title if `caller` may read it, else None. The trade
    log's positions name their campaign by id; this is the permission check,
    so that store never needs this one's tables."""
    owners = agents.readable_owners(caller)
    if not owners:
        return None
    with _db() as db:
        found = _records(db, owners, "AND id = ?", (rid,))
    return found[0]["title"] if found else None


def _records(db, owners, where="", args=()):
    marks = ",".join("?" * len(owners))
    return [dict(r) for r in db.execute(
        f"SELECT * FROM records WHERE owner IN ({marks}) {where}",
        (*owners, *args))]


def _entries(db, ids):
    out = {i: [] for i in ids}
    if ids:
        marks = ",".join("?" * len(ids))
        for e in db.execute(f"SELECT * FROM entries WHERE record_id IN ({marks}) "
                            "ORDER BY ts, id", tuple(ids)):
            out[e["record_id"]].append(dict(e))
    return out


def _title_line(r):
    kind = f" [{r['kind']}]" if r["kind"] else ""
    closed = " (closed)" if r["status"] == "closed" else ""
    return (f"#{r['id']}{kind} {r['title']}{closed} — opened "
            f"{hist.time_label(r['created'])}, updated {hist.time_label(r['updated'])}")


def _header(r):
    return _title_line(r) + (f"\n{r['body']}" if r["body"] else "")


def _entry_line(e):
    return f"  - {hist.time_label(e['ts'])}: {e['text']}"


def render(caller, rid):
    """Record `rid` with its full history, if `caller` may read it."""
    owners = agents.readable_owners(caller)
    if not owners:
        return None
    with _db() as db:
        recs = _records(db, owners, "AND id = ?", (rid,))
        if not recs:
            return None
        entries = _entries(db, [rid])[rid]
    return "\n".join([_header(recs[0])] + [_entry_line(e) for e in entries])


def listing(caller, closed=10):
    """Open records plus the last `closed` closed ones, one line each."""
    owners = agents.readable_owners(caller)
    if not owners:
        return []
    with _db() as db:
        rows = _records(db, owners, "AND status = 'open' ORDER BY id")
        rows += _records(db, owners, "AND status = 'closed' "
                         "ORDER BY updated DESC LIMIT ?", (closed,))
    return [_title_line(r) for r in rows]


def block(owner):
    """The always-shown block for `owner`: every open record, packed into
    WORKING_RECORD_CHARS by a fixed order — headers (most recently updated
    first, title-only once a full header no longer fits), then each record's
    last WORKING_RECENT_ENTRIES entries, then the newest of the rest. Rendered
    in creation order, entries oldest first, so the bytes only change when a
    record does and the block stays prompt-cached. "" when nothing is open."""
    with _db() as db:
        recs = _records(db, (owner,), "AND status = 'open' ORDER BY id")
        if not recs:
            return ""
        entries = _entries(db, [r["id"] for r in recs])
    left = cfg.WORKING_RECORD_CHARS - len(_PREAMBLE)
    full, shown = set(), set()
    # Stamps have one-second resolution, so ties fall back to the row id —
    # "newest" must not be arbitrary when a turn writes twice in a second.
    by_update = sorted(recs, key=lambda r: (r["updated"], r["id"]), reverse=True)
    for r in by_update:
        h = _header(r)
        if len(h) < left:
            full.add(r["id"])
            left -= len(h) + 1
        else:  # a title is always shown: knowing a record exists is the point
            left -= len(_title_line(r)) + 1
    def take(e):
        nonlocal left
        cost = len(_entry_line(e)) + 1
        if cost > left:
            return False
        shown.add(e["id"])
        left -= cost
        return True
    for r in by_update:
        for e in reversed(entries[r["id"]][-cfg.WORKING_RECENT_ENTRIES:]):
            take(e)
    rest = sorted((e for es in entries.values() for e in es
                   if e["id"] not in shown),
                  key=lambda e: (e["ts"], e["id"]), reverse=True)
    for e in rest:
        if not take(e):
            break
    lines = [_PREAMBLE]
    for r in recs:
        lines.append(_header(r) if r["id"] in full else
                     _title_line(r) + f" (details: read_record {r['id']})")
        lines += [_entry_line(e) for e in entries[r["id"]] if e["id"] in shown]
    total = sum(len(es) for es in entries.values())
    hidden = total - len(shown)
    if hidden:
        lines.append(f"({hidden} older entries not shown — read_record(id) "
                     "for a record's full history.)")
    if hidden or len(full) < len(recs):
        log.info("working record for %s truncated: %d/%d entries, %d titles only",
                 owner, len(shown), total, len(recs) - len(full))
    return "\n".join(lines)
