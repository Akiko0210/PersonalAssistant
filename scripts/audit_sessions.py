"""Session-log audit: the baseline metrics behind
docs/analysis/2026-10-01-tracking-failures.md, recomputed the same way on
any window of logs so a before/after comparison measures one thing.

    python -m scripts.audit_sessions logs/session_2026-09-2*.log --to "2026-10-01 14:57:39"   # the baseline
    python -m scripts.audit_sessions logs/session_*.log --from 2026-10-07

Read-only: parses log lines, touches no state. The correction regex is the
baseline's — change it and the numbers stop being comparable."""

import argparse
import re
from collections import Counter

TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d{3} (\S+)\s+(\w+)\s+(.*)$")

# User wording that corrects or pushes back on the agent. A heuristic floor,
# frozen at the 2026-10-01 baseline.
CORRECTION = re.compile(
    r"\b(no[,.]? (you|that|we|i)|wrong|you said|i (just )?told you|already "
    r"(went|told|said|did|have|saved|logged)|again\b|forgot|forget|you('re| are) "
    r"(drifting|confused|not|supposed)|didn'?t (you|we|i tell)|not what i|"
    r"dumb|stupid|why (can'?t|don'?t|didn'?t|are|do) you|supposed to|"
    r"you missed|missed|quit |stop (telling|saying|repeating)|i don'?t know what you|"
    r"doesn'?t make sense|that'?s not|you keep|keep (giving|telling|saying)|"
    r"we (talked|discussed|decided|agreed)|remember (that|when|what|the)|"
    r"do you remember|last (time|night|week)|yesterday|earlier)\b", re.I)


def parse(paths, start=None, end=None):
    """One dict per user turn, plus per-day event counts."""
    turns, events, persona, cur = [], Counter(), None, None
    for path in paths:
        for raw in open(path, encoding="utf-8", errors="replace"):
            m = TS.match(raw.rstrip("\n"))
            if not m:
                continue
            ts, logger, level, msg = m.groups()
            day = ts[:10]
            # Bounds compare at their own precision: a day, or a timestamp —
            # the baseline ends mid-log at 2026-10-01 14:57:39.
            if (start and ts[:len(start)] < start) or (end and ts[:len(end)] > end):
                continue
            if logger == "llm" and msg.startswith("active agent ->"):
                persona = msg.split("->")[1].split()[0]
            elif logger == "agent" and msg.startswith("you: "):
                cur = {"day": day, "persona": persona, "user": msg[5:],
                       "tools": 0, "rewrite_failed": False, "pinned": None}
                turns.append(cur)
            elif cur and logger == "agent" and msg.startswith("you (continued): "):
                cur["user"] += " " + msg[17:]
            elif cur and logger == "llm" and msg.startswith("tool_use "):
                cur["tools"] += 1
            elif cur and logger == "query" and "query understanding failed" in msg:
                cur["rewrite_failed"] = True
                events["rewrite_failed"] += 1
            elif cur and logger == "context" and msg.startswith("context pull"):
                pm = re.search(r"\((\d+) pinned\)", msg)
                cur["pinned"] = int(pm.group(1)) if pm else None
            elif "your note is ready to file" in msg:
                events["note_interruption"] += 1
            elif "saved conversation note" in msg:
                events["note_saved"] += 1
            elif "fact extraction failed" in msg or "facts extraction failed" in msg:
                events["extraction_failed"] += 1
            elif "conversation turn failed" in msg:
                events["turn_failed"] += 1
            elif "out of balance" in msg or "Insufficient Balance" in msg:
                events["balance_error"] += 1
            elif logger == "records" and (rm := re.match(r"record (\w+) #\d+ (\w+)", msg)):
                # stores/working_state.py: one line per write
                events[("record_write", rm.group(1), rm.group(2))] += 1
            elif logger == "records" and msg.startswith("working record for"):
                events["record_truncated"] += 1
    for t in turns:
        t["correction"] = bool(CORRECTION.search(t["user"]))
    return turns, events


def report(turns, events):
    days = sorted({t["day"] for t in turns})
    print(f"window: {days[0]} .. {days[-1]}  ({len(days)} days, {len(turns)} user turns)\n")
    print(f"{'persona':8} {'turns':>6} {'corrections':>12} {'rate':>6} {'no-tool':>8} {'no-tool %':>9}")
    for p in ("tom", "linda", "alice", "bob"):
        ts = [t for t in turns if t["persona"] == p]
        if not ts:
            continue
        c = [t for t in ts if t["correction"]]
        nt = sum(1 for t in c if not t["tools"])
        print(f"{p:8} {len(ts):6} {len(c):12} {len(c)/len(ts):6.1%} {nt:8} "
              f"{(nt/len(c) if c else 0):9.0%}")
    pulled = [t for t in turns if t["pinned"] is not None]
    ok = [t for t in pulled if not t["rewrite_failed"]]
    bad = [t for t in pulled if t["rewrite_failed"]]
    share = lambda xs: (sum(1 for t in xs if t["pinned"]) / len(xs)) if xs else 0
    print(f"\nrewrite failures:            {events['rewrite_failed']}")
    print(f"pinned facts | rewrite ok:   {share(ok):.0%} of {len(ok)} turns")
    print(f"pinned facts | rewrite fail: {share(bad):.0%} of {len(bad)} turns")
    print(f"fact extraction failures:    {events['extraction_failed']}")
    print(f"turns failed outright:       {events['turn_failed']}")
    print(f"provider balance errors:     {events['balance_error']}")
    print(f"note-save interruptions:     {events['note_interruption']} "
          f"({events['note_interruption']/len(days):.1f}/day)")
    print(f"notes saved via dialogue:    {events['note_saved']}")
    # Working record (from 2026-10-06): does each persona actually keep it?
    print(f"\n{'record writes':14} {'created':>8} {'updated':>8} {'entry':>6} "
          f"{'closed':>7} {'per day':>8}")
    for p in ("tom", "linda", "alice"):
        n = {a: events[("record_write", p, a)]
             for a in ("created", "updated", "entry", "closed")}
        print(f"{p:14} {n['created']:8} {n['updated']:8} {n['entry']:6} "
              f"{n['closed']:7} {sum(n.values())/len(days):8.1f}")
    print(f"record blocks truncated:     {events['record_truncated']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--from", dest="start", help="first day or moment, 'YYYY-MM-DD[ HH:MM:SS]'")
    ap.add_argument("--to", dest="end", help="last day or moment, 'YYYY-MM-DD[ HH:MM:SS]'")
    a = ap.parse_args()
    report(*parse(a.logs, a.start, a.end))
