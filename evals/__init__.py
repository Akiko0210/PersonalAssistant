"""Tom's test environment (#66). It replays his real (or a synthetic) state
on any code version, sandboxed, with models on the user's Claude
subscription. The runbook is in docs/TEST_ENV.md.

    snapshot.py   capture / verify / clone a read-only state snapshot
    sandbox.py    the isolation checks a run must pass before Tom starts
    gateway.py    anthropic-SDK-shaped client over `claude -p` (subscription)
    mcp_proxy.py  hands the CLI's tool calls back to the worker
    worker.py     one case on one code version in one fresh clone
    runner.py     export code, clone, run, judge, compare
    cases/        synthetic cases (public). Cases about the user's real
                  data live in <testenv>/cases/ and never enter git.
"""

import os
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent
REPO_DIR = HARNESS_DIR.parent
# The judge reads Tom's answer against the case's rubric. It is the
# strongest model, because a lenient judge would hide exactly the confident
# wrong answers these tests exist to catch. Its verdict is a triage aid; the
# report shows every answer for a human read.
JUDGE_MODEL = "claude-opus-5-5"


def testenv_root() -> Path:
    """Private, outside git: snapshots, runs, traces, reports, real cases."""
    return Path(os.getenv("VOICE_AI_TESTENV") or Path.home() / ".voice-agent-testenv")
