"""One-shot: seed each persona's exchange index from the session logs.

The session logs (logs/session_*.log) hold full transcripts, and every turn
in them is attributable: each boot starts on DEFAULT_AGENT, and every switch
is logged ("=== talking to X" / "active agent -> x"). This script replays that
attribution and indexes the turns as exchange records through the same
ConversationMemory.index_exchanges the live agent uses — no model call, one
record per user/assistant exchange, stamped with the log line's time. Ids are
deterministic, so re-running skips what is already indexed.

ONLY THE LAST `--days` DAYS OF LOGS ARE READ (default 7). Personas did not
exist before 2026-07-20 — the first switch line in these logs names "Cobe",
a persona since renamed and no longer in the alias map. In an older log there
is nothing to attribute turns to, so every one of them would be filed under
DEFAULT_AGENT and Alice would "remember" months of conversations that were
never hers. A first run against all 24 logs produced exactly that: 26 Alice
records reaching back to June. The window is the guard.

Run with the agent OFF (it writes the same Chroma store):

    python -m scripts.seed_agent_memory [--days N] [--dry-run]
"""

import argparse
import re
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config as cfg  # noqa: E402
from brain import agents  # noqa: E402
from lib.single_instance.main import AlreadyRunning, SingleInstance  # noqa: E402

# One log line: "2026-08-10 09:22:11,286 agent    INFO    <message>".
_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2}) (\d{2}:\d{2}:\d{2}),\d+ "
                   r"\S+\s+\S+\s+(.*)$")
_SWITCH_BANNER = re.compile(r"^=== talking to (\w+) ")
_SWITCH_LOG = re.compile(r"^active agent -> (\w+)")
# "you:" variants that are real user turns ("you (folder choice)" is not).
_USER = re.compile(r"^you(?: \(typed\)| \(continued\))?: (.*)$")
_AGENT = re.compile(r"^agent: (.*)$")

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
    """[date, persona, 'role: text', ts] turns from the session logs, in
    order — `ts` is the line's local time as ISO, the stamp the exchange
    record is dated by.

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
            date, clock, msg = m.group(1), m.group(2), m.group(3)
            ts = f"{date}T{clock}"
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
                current = [date, persona, "user: " + user.group(1), ts]
                turns.append(current)
                continue
            reply = _AGENT.match(msg)
            if reply:
                current = [date, persona, "assistant: " + reply.group(1), ts]
                turns.append(current)
    return turns


def chunk_by_persona_day(turns):
    """{(persona, date): [line, ...]} preserving turn order."""
    chunks = {}
    for date, persona, line, _ in turns:
        chunks.setdefault((persona, date), []).append(line)
    return chunks


def index_turns(turns, dry_run=False):
    """Index the attributed turns as exchange records, per persona, in log
    order. Returns how many new records were written (0 on a dry run)."""
    by_persona = {}
    for _, persona, line, ts in turns:
        role, _, text = line.partition(": ")
        by_persona.setdefault(persona, []).append(
            {"role": role, "content": text, "ts": ts})
    if dry_run:
        for persona, msgs in by_persona.items():
            print(f"  would index {persona}: {len(msgs)} turn(s)", flush=True)
        return 0
    from brain.memory import ConversationMemory  # heavy; only for a real run
    memory = ConversationMemory()
    written = 0
    for persona, msgs in by_persona.items():
        n = memory.index_exchanges(msgs, persona)
        written += n
        # flush: stdout is block-buffered when redirected, and a run that
        # dies mid-way otherwise leaves a log showing nothing happened.
        print(f"  {persona}: {n} new exchange(s) from {len(msgs)} turn(s)",
              flush=True)
    return written


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
        written = index_turns(turns, dry_run=args.dry_run)
        if not args.dry_run:
            print(f"Done — {written} new exchange record(s) written.")
    finally:
        if not args.dry_run:
            lock.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
