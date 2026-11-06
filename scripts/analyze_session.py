"""Post-session analysis: hand the last session in a daily log to the Claude
Code CLI and write what the user ran into — and what is and isn't working
about the agent — to logs/session_<date>_<HHMM>_analyzed.md.

    python -m scripts.analyze_session                                  # newest log
    python -m scripts.analyze_session logs/session_2026-10-15.log

run.bat / run.sh start this in the background after a plain launch exits.
Read-only on everything but its one output file; a failure (CLI missing, not
logged in, timed out) is written into that file rather than lost."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import config as cfg
from lib.atomic_io import write_text_atomic
from scripts.audit_sessions import TS, parse

PROMPT = """\
You are reviewing one session of the Voice AI agent in this repository. Read
CLAUDE.md first: the user has tunnel vision and runs daily life and options
trades by voice, so a wrong number, a lost note, or a silent failure costs far
more than slowness.

Below are deterministic metrics for the session, then the session's log with
transport chatter and the context DEBUG tables removed. The raw log is
{log} (session = lines {first}-{last}); Grep it when retrieval detail matters,
and Read/Grep the code to explain causes. Cite log times (HH:MM:SS) for every
claim and file:line for every cause you name.

Write a very detailed Markdown report, no preamble, with these sections:

## Session overview
Duration, personas and models used, provider failovers, what the user was
trying to get done, and whether they got it done.

## Issues the user faced
One subsection per issue, most harmful first. For each: when; what the user
said (quote); what the agent did; impact on this user; root cause with code
references (say plainly when it is a guess); severity (critical/high/medium/low).
Include friction the user worked around: repeats, corrections, interruptions,
mishearings, wrong persona, long silences, tools that returned nothing.

## What worked
Concrete things that went right, with times — keep these as specific as the issues.

## What didn't work, by subsystem
Speech-to-text and turn-taking; barge-in; TTS; headset button; LLM provider,
failover and cost; tool calls; memory, recall and retrieval; working record;
trading and broker data; notes; dashboard. Skip a subsystem the session never
touched, and say so in one line.

## Accuracy and trust
Every number, position, price, fill or date the agent stated: was it from a
tool result this turn, from memory, or guessed? Flag anything stale or unbacked.

## Recommendations
Ranked fixes, each tied to the issues above, with the file to change and the
rough size of the change.

=== Metrics ===
{metrics}

=== Session log ===
"""


# --- Session slicing -----------------------------------------------------------

def last_session(lines):
    """(first line number, lines) of the last agent session in a daily log.
    One file holds every launch that day; a session ends at the agent's
    "shut down" line. A tail with no "Ready." in it is a launch that never
    started (an --ingest pass, a refused second copy), so the session before it
    is the one that just ended; a crash leaves no "shut down", so its tail is
    the session."""
    def agent_says(line, prefix):
        m = TS.match(line.rstrip("\n"))
        return bool(m) and m.group(2) == "agent" and m.group(4).startswith(prefix)

    chunks, start = [], 0
    for i, line in enumerate(lines):
        if agent_says(line, "shut down"):
            chunks.append((start, i + 1))
            start = i + 1
    if start < len(lines):
        chunks.append((start, len(lines)))
    started = [c for c in chunks
               if any(agent_says(l, "Ready.") for l in lines[c[0]:c[1]])]
    a, b = (started or chunks or [(0, 0)])[-1]
    return a + 1, lines[a:b]


def condense(lines):
    """Drop DEBUG records and transport chatter, continuation lines included
    (a multi-line record's tail belongs to whatever its head was)."""
    out, keep = [], True
    for line in lines:
        m = TS.match(line.rstrip("\n"))
        if m:
            _, logger, level, _ = m.groups()
            keep = level in ("WARNING", "ERROR", "CRITICAL") or (
                level != "DEBUG" and logger not in cfg.LOG_ANALYSIS_NOISE)
        if keep:
            out.append(line)
    return "".join(out)


def metrics(path, session):
    """The audit script's counters over this session's time window, so the
    numbers are computed, not estimated by the model."""
    stamps = [m.group(1) for l in session if (m := TS.match(l.rstrip("\n")))]
    if not stamps:
        return stamps, "(no timestamped lines)"
    turns, events = parse([path], stamps[0], stamps[-1])
    flagged = [t["user"] for t in turns if t["correction"]]
    personas = sorted({t["persona"] for t in turns if t["persona"]})
    rows = [f"- window: {stamps[0]} .. {stamps[-1]}",
            f"- user turns: {len(turns)} (personas: {', '.join(personas) or 'n/a'})",
            f"- turns with no tool call: {sum(1 for t in turns if not t['tools'])}",
            f"- turns failed outright: {events['turn_failed']}",
            f"- provider balance errors: {events['balance_error']}",
            f"- query-rewrite failures: {events['rewrite_failed']}",
            f"- fact-extraction failures: {events['extraction_failed']}",
            f"- user turns matching the correction heuristic: {len(flagged)}"]
    rows += [f"  - \"{u}\"" for u in flagged]
    return stamps, "\n".join(rows)


# --- Claude CLI ------------------------------------------------------------------

def run_claude(stdin_text):
    """(ok, markdown or failure reason). The whole prompt goes through stdin:
    on Windows the npm install is a .cmd shim, and cmd.exe mangles long
    multi-line arguments."""
    exe = shutil.which(cfg.LOG_ANALYSIS_CLI)
    if not exe:
        return False, (f"`{cfg.LOG_ANALYSIS_CLI}` is not on PATH. Install Claude "
                       "Code (npm install -g @anthropic-ai/claude-code), run "
                       "`claude` once to log in, then rerun this script.")
    try:
        r = subprocess.run([exe, "-p", "--allowedTools", "Read,Grep,Glob"],
                           input=stdin_text, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", cwd=cfg.BASE_DIR,
                           timeout=cfg.LOG_ANALYSIS_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {cfg.LOG_ANALYSIS_TIMEOUT_S}s."
    if r.returncode != 0 or not r.stdout.strip():
        return False, (f"`claude` exited {r.returncode}.\n\n```\n"
                       f"{(r.stdout + r.stderr).strip()[-2000:]}\n```")
    return True, r.stdout.strip()


def analyze(path):
    """Write the _analyzed file for the last session in `path`; (its path, ok)."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines(True)
    first, session = last_session(lines)
    last = first + len(session) - 1
    stamps, numbers = metrics(path, session)
    hhmm = stamps[0][11:16].replace(":", "") if stamps else "0000"
    out = path.with_name(f"{path.stem}_{hhmm}_analyzed.md")
    ok, body = run_claude(PROMPT.format(log=path, first=first, last=last,
                                        metrics=numbers) + condense(session))
    head = f"# Session analysis — {path.name}, lines {first}-{last}\n\n{numbers}\n\n"
    write_text_atomic(out, head + (body if ok else f"## Analysis failed\n\n{body}")
                      + "\n")
    return out, ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log", nargs="?", help="session log (default: the newest)")
    a = ap.parse_args()
    logs = sorted(cfg.LOG_DIR.glob("session_*.log"), key=lambda p: p.stat().st_mtime)
    target = a.log or (logs[-1] if logs else None)
    if target is None:
        sys.exit(f"No session logs in {cfg.LOG_DIR}.")
    out, ok = analyze(target)
    print(("wrote " if ok else "analysis failed, see ") + str(out))
    sys.exit(0 if ok else 1)
