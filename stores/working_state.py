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
from collections import namedtuple
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
-- The record a persona is focused on (#63). Here rather than in session state
-- because "continue a little bit" the next morning has to land on the same
-- project, and in the same store as the record so closing one can end its
-- focus in one transaction: a closed project can never still be in focus.
CREATE TABLE IF NOT EXISTS focus (
    owner     TEXT PRIMARY KEY,
    record_id INTEGER NOT NULL REFERENCES records ON DELETE CASCADE);
"""

_PREAMBLE = (
    "Your working record — what you are in the middle of for the user, kept "
    "by you with update_record, add_entry and close_record. It is current and "
    "authoritative: where the Background or an older exchange disagrees with "
    "it, the record wins.\n")

_FOCUS_LINE = (
    "In focus: #{id} {title}. The user asked to work on this project, so its "
    "conversations and facts come first in your Background; the rest of your "
    "memory is still there. Work on it (rules, decisions, changes) goes in "
    "#{id}, never a new record. Before telling the user something was not "
    "done or not saved, call recall and say what you checked. Stay in focus "
    "until the user says otherwise.\n")


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
        db.execute("DELETE FROM focus WHERE record_id = ?", (rid,))
    _wrote(owner, rid, "closed", row["title"], len(outcome))


def focus_on(owner, rid):
    """Put the writer's open record `rid` in focus, replacing any other."""
    with _db() as db:
        row = _own(db, owner, rid)
        if row["status"] != "open":
            raise ValueError(f"Record #{rid} is closed, so it can't be focused. "
                             "Its memory is still searchable.")
        db.execute("INSERT OR REPLACE INTO focus (owner, record_id) VALUES (?, ?)",
                   (owner, rid))
    log.info("focus %s -> #%d %r", owner, rid, row["title"])


def leave_focus(owner):
    """End `owner`'s focus; returns the record that was in focus, or None."""
    was = focused(owner)
    if was:
        with _db() as db:
            db.execute("DELETE FROM focus WHERE owner = ?", (owner,))
        log.info("focus %s left #%d %r", owner, was["id"], was["title"])
    return was


def focused(owner):
    """The record `owner` is focused on, as a dict, or None."""
    with _db() as db:
        row = db.execute("SELECT r.* FROM focus f JOIN records r ON r.id = f.record_id "
                         "WHERE f.owner = ?", (owner,)).fetchone()
    return dict(row) if row else None


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


Packed = namedtuple("Packed", "recs entries focus_line full shown")


def _short_line(r):
    return _title_line(r) + f" (details: read_record {r['id']})"


def _pack(owner):
    """Which parts of `owner`'s open records fit WORKING_RECORD_CHARS, or
    None when nothing is open. `full` holds the ids shown with their body,
    `shown` the entry ids shown. The order things claim the budget:

    1. Every record's title. Knowing a record exists is the point.
    2. The record in focus, whole: its body, then its journal newest first.
       This block is what focus means for the persona's own notes.
    3. Every record's newest WORKING_RECENT_ENTRIES entries, before any body.
       An entry is where a step gets recorded as done; a body is the plan
       as first written. With bodies first, 10.7k of bodies filled Tom's
       8k on 2026-11-12, so none of his 32 entries was shown. One of them
       said his upside-calendar rule was already spent, and he recited it
       as pending.
    4. Bodies, most recently updated first.
    5. The rest of the journals, newest first."""
    in_focus = focused(owner)
    with _db() as db:
        recs = _records(db, (owner,), "AND status = 'open' ORDER BY id")
        if not recs:
            return None
        entries = _entries(db, [r["id"] for r in recs])
    focus_line = (_FOCUS_LINE.format(id=in_focus["id"], title=in_focus["title"])
                  if in_focus else "")
    focus_id = in_focus["id"] if in_focus else None
    left = cfg.WORKING_RECORD_CHARS - len(_PREAMBLE) - len(focus_line)
    left -= sum(len(_short_line(r)) + 1 for r in recs)
    full, shown = set(), set()

    def take(e):
        nonlocal left
        cost = len(_entry_line(e)) + 1
        if cost > left:
            return False
        shown.add(e["id"])
        left -= cost
        return True

    def widen(r):  # title line -> full header, if the difference fits
        nonlocal left
        extra = len(_header(r)) - len(_short_line(r))
        if extra <= left:
            full.add(r["id"])
            left -= extra
    # Stamps have one-second resolution, so ties fall back to the row id —
    # "newest" must not be arbitrary when a turn writes twice in a second.
    by_update = sorted(recs, key=lambda r: (r["id"] == focus_id, r["updated"], r["id"]),
                       reverse=True)
    if focus_id is not None:
        widen(by_update[0])
        for e in reversed(entries[focus_id]):
            if not take(e):
                break
    for r in by_update:
        for e in reversed(entries[r["id"]][-cfg.WORKING_RECENT_ENTRIES:]):
            take(e)
    for r in by_update:
        widen(r)
    rest = sorted((e for es in entries.values() for e in es
                   if e["id"] not in shown),
                  key=lambda e: (e["ts"], e["id"]), reverse=True)
    for e in rest:
        if not take(e):
            break
    return Packed(recs, entries, focus_line, full, shown)


def block(owner):
    """The always-shown block for `owner`: every open record, packed into
    WORKING_RECORD_CHARS by _pack's order. Rendered in creation order,
    entries oldest first, so the bytes only change when a record does and
    the block stays prompt-cached. "" when nothing is open."""
    p = _pack(owner)
    if p is None:
        return ""
    lines = [_PREAMBLE + p.focus_line]
    for r in p.recs:
        lines.append(_header(r) if r["id"] in p.full else _short_line(r))
        lines += [_entry_line(e) for e in p.entries[r["id"]] if e["id"] in p.shown]
    total = sum(len(es) for es in p.entries.values())
    hidden = total - len(p.shown)
    if hidden:
        lines.append(f"({hidden} older entries not shown — read_record(id) "
                     "for a record's full history.)")
    if hidden or len(p.full) < len(p.recs):
        log.info("working record for %s truncated: %d/%d entries; titles only: %s; "
                 "entries hidden in: %s", owner, len(p.shown), total,
                 " ".join(f"#{r['id']}" for r in p.recs if r["id"] not in p.full) or "none",
                 " ".join(f"#{i}" for i, es in p.entries.items()
                          if any(e["id"] not in p.shown for e in es)) or "none")
    return "\n".join(lines)


def open_titles(owner):
    """{id: title} of `owner`'s open records."""
    with _db() as db:
        return {r["id"]: r["title"] for r in
                _records(db, (owner,), "AND status = 'open' ORDER BY id")}


def unshown(owner, rid, budget):
    """The parts of open record `rid` that block() leaves out, as text for
    the turn's Background: its body when only the title fit, then the
    entries that didn't fit, newest first while `budget` lasts (shown
    oldest first). "" when the block already shows all of it."""
    p = _pack(owner)
    rec = next((r for r in p.recs if r["id"] == rid), None) if p else None
    if rec is None:
        return ""
    body = rec["body"] if rid not in p.full and rec["body"] else ""
    left = budget - len(body)
    hidden = [e for e in p.entries[rid] if e["id"] not in p.shown]
    picked = []
    for e in reversed(hidden):
        line = _entry_line(e)
        if len(line) + 1 > left:
            break
        picked.insert(0, line)
        left -= len(line) + 1
    if not body and not picked:
        return ""
    dropped = len(hidden) - len(picked)
    return "\n".join(
        [f"#{rid} {rec['title']} — what the record block above leaves out:"]
        + ([body] if body else [])
        + ([f"  ({dropped} older entries: read_record {rid})"] if dropped else [])
        + picked)
