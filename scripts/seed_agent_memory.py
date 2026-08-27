"""One-shot: seed each persona's conversation archive from the session logs.

The per-agent memory split starts every persona with an empty archive, but the
recent session logs (logs/session_*.log) hold full transcripts — and every
turn in them is attributable: each boot starts on DEFAULT_AGENT, and every
switch is logged ("=== talking to X" / "active agent -> x"). This script
replays that attribution, groups the turns into persona-per-day chunks,
summarises each with the same CONSOLIDATE_PROMPT the live consolidator uses,
and upserts into conversations_<key>. Ids are deterministic
(conv_<date>_<key>_seed), so re-running overwrites rather than duplicates.

ONLY THE LAST `--days` DAYS OF LOGS ARE READ (default 7). Personas did not
exist before 2026-07-20 — the first switch line in these logs names "Cobe",
a persona since renamed and no longer in the alias map. In an older log there
is nothing to attribute turns to, so every one of them would be filed under
DEFAULT_AGENT and Alice would "remember" months of conversations that were
never hers. A first run against all 24 logs produced exactly that: 26 Alice
records reaching back to June. The window is the guard; seeded records that
fall outside it are pruned on each run, so narrowing the window cleans up
after a wider one.

Cost: one model call per persona-day chunk with enough lines. Run with the
agent OFF:

    python -m scripts.seed_agent_memory [--days N] [--dry-run]
"""

import argparse
import re
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from dotenv import load_dotenv
    load_dotenv()  # ANTHROPIC_API_KEY, same as voice_agent's startup
except ImportError:
    pass

import config as cfg  # noqa: E402
from brain import agents  # noqa: E402
from brain.memory import CONSOLIDATE_PROMPT  # noqa: E402
from lib.single_instance.main import AlreadyRunning, SingleInstance  # noqa: E402

# One log line: "2026-08-10 09:22:11,286 agent    INFO    <message>".
_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2}) \d{2}:\d{2}:\d{2},\d+ "
                   r"\S+\s+\S+\s+(.*)$")
_SWITCH_BANNER = re.compile(r"^=== talking to (\w+) ")
_SWITCH_LOG = re.compile(r"^active agent -> (\w+)")
# "you:" variants that are real user turns ("you (folder choice)" is not).
_USER = re.compile(r"^you(?: \(typed\)| \(continued\))?: (.*)$")
_AGENT = re.compile(r"^agent: (.*)$")

_CHUNK_CHAR_BUDGET = 100_000  # ~25k tokens; newest lines win, like recall
_LOG_NAME = re.compile(r"session_(\d{4}-\d{2}-\d{2})\.log$")
DEFAULT_DAYS = 7  # see the module docstring: older logs predate personas


def log_date(path):
    """The date in a session log's filename, or None if it doesn't match."""
    m = _LOG_NAME.search(path.name)
    return date.fromisoformat(m.group(1)) if m else None


def select_logs(paths, days, today=None):
    """(kept, skipped) split of log paths by the `days`-day window. Kept in
    filename order; a path whose name doesn't parse is skipped, not guessed
    at."""
    cutoff = (today or date.today()) - timedelta(days=days)
    kept, skipped = [], []
    for path in paths:
        d = log_date(path)
        (kept if d and d >= cutoff else skipped).append(path)
    return kept, skipped, cutoff


def parse_logs(paths):
    """(date, persona, 'role: text') turns from the session logs, in order.

    Attribution state machine: a boot marker resets the persona to
    DEFAULT_AGENT (a fresh process always starts there); a switch line flips
    it. Lines without the timestamp prefix continue the previous message —
    the file logger writes multi-line replies that way."""
    turns = []
    for path in paths:
        persona = agents.DEFAULT_AGENT
        current = None  # last appended turn, open for continuation lines
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = _LINE.match(raw)
            if m is None:
                if current is not None and raw.strip():
                    current[2] = current[2] + " " + raw.strip()
                continue
            date, msg = m.group(1), m.group(2)
            current = None
            if msg.startswith("startup took"):
                persona = agents.DEFAULT_AGENT
                continue
            sw = _SWITCH_BANNER.match(msg) or _SWITCH_LOG.match(msg)
            if sw:
                persona = agents.resolve(sw.group(1)) or persona
                continue
            user = _USER.match(msg)
            if user:
                current = [date, persona, "user: " + user.group(1)]
                turns.append(current)
                continue
            reply = _AGENT.match(msg)
            if reply:
                current = [date, persona, "assistant: " + reply.group(1)]
                turns.append(current)
    return turns


def chunk_by_persona_day(turns):
    """{(persona, date): [line, ...]} preserving turn order."""
    chunks = {}
    for date, persona, line in turns:
        chunks.setdefault((persona, date), []).append(line)
    return chunks


def prune_outside_window(cutoff, dry_run=False):
    """Drop previously seeded records older than the cutoff. Touches ONLY
    records this script wrote (metadata seeded=True), so a summary the live
    agent consolidated can never be caught by it. Ids are fetched and
    filtered here rather than handed to Chroma as a range query — the dates
    are strings, and a where= comparison on those is not worth trusting with
    a delete."""
    from brain import agents as agent_registry
    from stores import chroma_store

    removed = 0
    for key in agent_registry.AGENTS:
        col = chroma_store.collection(cfg.agent_memory_collection(key))
        got = col.get(where={"seeded": True}, include=["metadatas"])
        stale = [i for i, m in zip(got["ids"], got["metadatas"])
                 if (m or {}).get("date", "") < cutoff.isoformat()]
        if not stale:
            continue
        print(f"  {'would prune' if dry_run else 'pruned'} {len(stale)} "
              f"stale seeded record(s) from {key}", flush=True)
        if not dry_run:
            col.delete(ids=stale)
        removed += len(stale)
    return removed


def summarise_and_store(chunks, dry_run=False):
    import anthropic
    from stores import chroma_store

    client = None if dry_run else anthropic.Anthropic()
    seeded = 0
    # `day`, not `date`: the module-level datetime.date import must stay
    # reachable from inside this function.
    for (persona, day), lines in sorted(chunks.items(), key=lambda i: i[0][1]):
        if len(lines) < cfg.MEMORY_MIN_MESSAGES:
            print(f"  skip {persona} {day}: only {len(lines)} line(s)", flush=True)
            continue
        if dry_run:
            print(f"  would seed {persona} {day}: {len(lines)} line(s)", flush=True)
            continue
        transcript = f"[{day}]\n" + "\n".join(lines)
        # Newest-last budget, mirroring recall_staged: a marathon day must not
        # blow the request; the tail of the day is the part worth keeping.
        transcript = transcript[-_CHUNK_CHAR_BUDGET:]
        resp = client.messages.create(
            model=cfg.CONVO_MODEL,
            max_tokens=cfg.MEMORY_MAX_TOKENS,
            thinking={"type": "disabled"},
            messages=[{"role": "user",
                       "content": CONSOLIDATE_PROMPT + transcript}],
        )
        summary = "".join(b.text for b in resp.content
                          if b.type == "text").strip()
        if not summary:
            print(f"  {persona} {day}: model returned nothing; skipped",
                  flush=True)
            continue
        col = chroma_store.collection(cfg.agent_memory_collection(persona))
        col.upsert(ids=[f"conv_{day}_{persona}_seed"],
                   documents=[summary],
                   metadatas=[{"date": day, "messages": len(lines),
                               "seeded": True}])
        seeded += 1
        # flush: stdout is block-buffered when redirected, and a run that
        # dies mid-way otherwise leaves a log showing nothing happened.
        print(f"  seeded {persona} {day}: {len(lines)} line(s)", flush=True)
    return seeded


def main():
    parser = argparse.ArgumentParser(
        description="Seed per-agent conversation archives from session logs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report the chunks without model calls or writes")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"How many days back to read logs from "
                             f"(default {DEFAULT_DAYS}). Logs older than this "
                             f"predate personas, so their turns cannot be "
                             f"attributed — see the module docstring.")
    args = parser.parse_args()
    if args.days < 1:
        print("--days must be at least 1.")
        return 1

    paths = sorted(cfg.LOG_DIR.glob("session_*.log"))
    if not paths:
        print(f"No session logs found in {cfg.LOG_DIR}.")
        return 1
    paths, skipped, cutoff = select_logs(paths, args.days)
    print(f"Window: logs from {cutoff} onward ({args.days} day(s)) — "
          f"{len(paths)} log(s) in, {len(skipped)} older log(s) skipped.",
          flush=True)
    if not paths:
        print("No logs inside the window; nothing to seed.")
        return 1

    if not args.dry_run:
        try:
            lock = SingleInstance(cfg.LOCK_PATH).acquire()
        except AlreadyRunning:
            print("A voice agent (or ingest) is running — close it first: "
                  "seeding writes the same Chroma store.")
            return 1
    try:
        turns = parse_logs(paths)
        chunks = chunk_by_persona_day(turns)
        print(f"{len(turns)} attributed turn(s) across {len(paths)} log(s), "
              f"{len(chunks)} persona-day chunk(s):", flush=True)
        pruned = prune_outside_window(cutoff, dry_run=args.dry_run)
        seeded = summarise_and_store(chunks, dry_run=args.dry_run)
        if not args.dry_run:
            print(f"Done — {seeded} archive record(s) written, "
                  f"{pruned} stale record(s) pruned.")
    finally:
        if not args.dry_run:
            lock.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
