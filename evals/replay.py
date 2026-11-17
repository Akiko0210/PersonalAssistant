"""Point-in-time replay: put a cloned state back to how it stood at one
moment, so a real incident can be asked again exactly as Tom met it.

A case with "as_of" (the moment the user asked) is cut back before Tom
starts. Everything written at or after that moment is removed, and what
was overwritten since is restored where a record of it survives:

- working records: those created later are gone, later entries are gone,
  a record closed later is open again, and a body or title edited later is
  rebuilt from the session logs (snapshot.record_writes). The store
  overwrites bodies in place, so the logs are their only history. A body
  with no logged version from before the moment is kept as it is now, and
  the trace lists it under "unreconstructed".
- the exchange index: Tom's exchanges from the moment on are deleted.
- facts: those learned from later exchanges are deleted. A fact that a
  later one replaced is current again.
- the trade log: fills from the moment on are deleted, with positions
  created since.
- the conversation thread: rebuilt from the exchanges just before the
  moment, so the verbatim tail is what Tom actually had in front of him.
- the clock: now_iso, recency scoring and get_current_time read the moment.

What can't be known is listed, not guessed:
- Fills imported from a statement after the moment (the log keeps no
  import time) are kept if they happened before it.
- Facts carry the time of the exchange they came from. They were rebuilt in
  order on 10/22, so they approximate what was known then.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path

OWNER = "tom"
THREAD_EXCHANGES = 20   # messages the live thread held: HISTORY_MAX_MESSAGES is 40


def _epoch(iso):
    return datetime.fromisoformat(iso).timestamp()


def _local(iso):
    """A stored stamp as a comparable aware datetime (logs are naive local)."""
    d = datetime.fromisoformat(iso.replace(",", "."))
    return d if d.tzinfo else d.astimezone()


def cut_records(db_path, as_of, writes_path):
    """Working records as they stood at `as_of`. Returns the ids whose
    body or title changed later with no earlier logged version."""
    t = _local(as_of)
    writes = []
    if Path(writes_path).exists():
        writes = [json.loads(line) for line in Path(writes_path).read_text(
            encoding="utf-8").splitlines() if line.strip()]
    con = sqlite3.connect(db_path)
    unknown = []
    with con:
        rows = con.execute("SELECT id, title, body, created, updated, status "
                           "FROM records WHERE owner = ?", (OWNER,)).fetchall()
        for rid, title, body, created, updated, status in rows:
            if _local(created) >= t:
                con.execute("DELETE FROM entries WHERE record_id = ?", (rid,))
                con.execute("DELETE FROM records WHERE id = ?", (rid,))
                continue
            mine = [w for w in writes if w["owner"] == OWNER and w["id"] == rid
                    and w["tool"] == "update_record"]
            before = [w for w in mine if _local(w["ts"]) < t]
            after = [w for w in mine if _local(w["ts"]) >= t]
            new_body, new_title = body, title
            if any("body" in w["args"] for w in after):
                last = next((w for w in reversed(before) if "body" in w["args"]), None)
                if last:
                    new_body = last["args"]["body"]
                else:
                    unknown.append(rid)
            if any(w["args"].get("title") for w in after):
                last = next((w for w in reversed(before) if w["args"].get("title")), None)
                new_title = last["args"]["title"] if last else title
            # Entry stamps carry offsets; compare as datetimes, not text.
            for eid, ts in con.execute("SELECT id, ts FROM entries WHERE record_id = ?",
                                       (rid,)).fetchall():
                if _local(ts) >= t:
                    con.execute("DELETE FROM entries WHERE id = ?", (eid,))
            last_ts = con.execute("SELECT max(ts) FROM entries WHERE record_id = ?",
                                  (rid,)).fetchone()[0]
            stamps = [created] + ([last_ts] if last_ts else []) + [w["ts"] for w in before]
            stamp = (updated if _local(updated) < t else
                     _local(max(stamps, key=_local)).isoformat(timespec="seconds"))
            # Closed later = open then (its "Closed:" entry went with the later entries).
            reopen = status == "closed" and _local(updated) >= t
            con.execute("UPDATE records SET title = ?, body = ?, updated = ?, status = ? "
                        "WHERE id = ?", (new_title, new_body, stamp,
                                         "open" if reopen else status, rid))
        con.execute("DELETE FROM focus")
    con.close()
    return unknown


def cut_trades(db_path, as_of):
    """The trade log without fills from `as_of` on."""
    if not Path(db_path).exists():
        return
    t = _local(as_of).replace(tzinfo=None).isoformat(timespec="seconds")
    con = sqlite3.connect(db_path)
    with con:
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("DELETE FROM legs WHERE (account, ref) IN "
                    "(SELECT account, ref FROM trades WHERE executed_at >= ?)", (t,))
        con.execute("DELETE FROM trades WHERE executed_at >= ?", (t,))
        con.execute("DELETE FROM positions WHERE created >= ?", (t,))
        con.execute("UPDATE trades SET position_id = NULL WHERE position_id NOT IN "
                    "(SELECT id FROM positions)")
    con.close()


def open_legs(db_path, as_of):
    """The option legs open at `as_of` per the cut trade log, as a sentence
    a get_positions fixture can return. A reconstruction, and labelled as
    one: the live account at that moment is not recoverable."""
    day = _local(as_of).date().isoformat()
    con = sqlite3.connect(db_path)
    rows = con.execute(
        "SELECT t.account, l.expiration, l.strike, l.option_type, sum(l.quantity) "
        "FROM legs l JOIN trades t USING (account, ref) "
        "GROUP BY t.account, l.expiration, l.strike, l.option_type "
        "HAVING sum(l.quantity) != 0 AND l.expiration >= ? "
        "ORDER BY t.account, l.expiration, l.strike", (day,)).fetchall()
    con.close()
    if not rows:
        return "No open option positions."
    out = {}
    for acct, exp, strike, kind, qty in rows:
        side = "long" if qty > 0 else "short"
        out.setdefault(acct[-7:-4], []).append(
            f"{side} {abs(qty)} SPX {exp} {strike:g} {'call' if kind == 'C' else 'put'}"
            f"{'s' if abs(qty) > 1 else ''}")
    return " ".join(f"Account …{a}: " + "; ".join(legs) + "." for a, legs in out.items())


def cut_memory(as_of):
    """Exchanges and facts from `as_of` on, through the app's own stores
    (the worker has imported them inside the sandbox)."""
    from stores import chroma_store
    t = _epoch(as_of)
    convo = chroma_store.collection(f"conversations_{OWNER}")
    gone = convo.get(where={"epoch": {"$gte": t}}, include=[])["ids"]
    if gone:
        convo.delete(ids=gone)
    facts = chroma_store.collection(f"facts_{OWNER}")
    later = facts.get(where={"epoch": {"$gte": t}}, include=[])["ids"]
    if later:
        facts.delete(ids=later)
    # A fact replaced from the moment on is the current one again.
    replaced = facts.get(where={"superseded_by": {"$ne": ""}}, include=["metadatas"])
    revive = [i for i, m in zip(replaced["ids"], replaced["metadatas"])
              if m.get("superseded_by") in set(later) or m.get("superseded_by") in set(gone)
              or (m.get("superseded_epoch") or 0) >= t]
    if revive:
        facts.update(ids=revive, metadatas=[{"superseded_by": ""} for _ in revive])
    return {"exchanges_removed": len(gone), "facts_removed": len(later),
            "facts_current_again": len(revive)}


def rebuild_thread(as_of, history_path):
    """Tom's thread as it stood: the last exchanges before `as_of`, as
    user/assistant messages with their stamps. Tool calls in them are
    gone: the index keeps an exchange's words, not its tool blocks."""
    from stores import chroma_store
    convo = chroma_store.collection(f"conversations_{OWNER}")
    got = convo.get(include=["documents", "metadatas"])
    rows = sorted((m.get("epoch") or 0, m.get("ts", ""), d)
                  for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])
                  if not i.endswith("_a") and d and d.startswith("user: "))
    msgs = []
    for _, ts, doc in rows[-THREAD_EXCHANGES:]:
        user, *said = doc[len("user: "):].split("\nassistant: ")
        if msgs and msgs[-1]["role"] == "user":     # an exchange with no reply
            msgs[-1]["content"] += "\n" + user.strip()
        else:
            msgs.append({"role": "user", "content": user.strip(), "ts": ts})
        if said:
            msgs.append({"role": "assistant",
                         "content": [{"type": "text", "text": "\n".join(said).strip()}]})
    Path(history_path).write_text(json.dumps(msgs, indent=1), encoding="utf-8")
    return len(msgs)


def freeze_clock(as_of):
    """Every clock Tom's turn reads says `as_of`."""
    import time as real_time
    from brain import context, history
    from tools import time_tools
    moment = _local(as_of)

    class _Time:
        def __getattr__(self, name):
            return getattr(real_time, name)

        @staticmethod
        def time():
            return moment.timestamp()

    class _DateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return moment if tz is None else moment.astimezone(tz)

    history.now_iso = lambda: moment.isoformat(timespec="seconds")
    context.time = _Time()
    time_tools.datetime = _DateTime


def apply(case, cfg, run_root):
    """Cut the clone back to the case's `as_of`; returns what was done, for
    the trace. Run inside the sandbox, before Tom starts."""
    as_of = case["as_of"]
    report = {"as_of": as_of}
    report["unreconstructed_bodies"] = cut_records(
        cfg.WORKING_STATE_PATH, as_of, Path(run_root) / "logs" / "record_writes.jsonl")
    from trading import config as tcfg
    cut_trades(tcfg.TRADE_LOG_PATH, as_of)
    report.update(cut_memory(as_of))
    report["thread_messages"] = rebuild_thread(as_of, cfg.history_path(OWNER))
    report["positions_at_moment"] = (open_legs(tcfg.TRADE_LOG_PATH, as_of)
                                     if Path(tcfg.TRADE_LOG_PATH).exists() else "")
    freeze_clock(as_of)
    return report
