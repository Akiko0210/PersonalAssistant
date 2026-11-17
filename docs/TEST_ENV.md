# Tom's test environment

Replays Tom's real state (or a synthetic one) on any code version, in a
sandbox, with models on the user's Claude subscription. A memory fix is
judged by running the same case on the old and the new code from the same
frozen state, not by how a live conversation went afterwards (#66).

Run everything with the interpreter that has the app's dependencies (the one
`.claude/launch.json` pins), from a checkout of this repo.

## Commands

| Step | Command |
| --- | --- |
| Capture Tom's live state (safe while the agent runs) | `python -m evals snapshot` → `~/.voice-agent-testenv/snapshots/real-YYYY-MM-DD` |
| Build a synthetic state from a fixture case | `python -m evals synth SYN-FIXTURE-overflow` |
| Check no master snapshot changed | `python -m evals verify` |
| Is the evidence in Tom's prompt? (no model calls) | `python -m evals run REAL-WR-01 --ref <ref> --no-model` |
| One case, end to end | `python -m evals run REAL-WR-01 --ref <ref>` |
| Baseline vs candidate | `python -m evals compare SYN-WR-01 REAL-WR-01 --baseline <ref> --candidate WORKTREE --repeats 3` |
| Answer with another Claude model | add `--tom-model claude-haiku-4-5` |

`--ref` is any git ref. `WORKTREE` means this checkout including uncommitted
edits. The report goes to `~/.voice-agent-testenv/reports/<time>/report.md`.
Every run's trace (`trace.json`) and exact prompts stay in its run folder.
Set `VOICE_AI_TESTENV` to move the root.

## What a run is

1. The code at `--ref` is exported into a fresh run folder, without
   `evals/`, so both versions are measured by this checkout's harness.
2. The case's snapshot is cloned beside it (`data/`, `knowledge/`). The app
   derives every path from the folder it runs in, so it reads and writes the
   clone and nothing else.
3. `evals/worker.py` starts in that folder and passes the isolation checks
   (`evals/sandbox.py`). Only then does it import the app, apply the case's
   setup, and run its turns through the real `Claude.converse`.
4. The runner has the judge grade the last answer against the case's ground
   truth (`evals.JUDGE_MODEL`, also on the subscription). It then verifies
   the master snapshot's hashes.

## Isolation (tests/test_eval_harness.py)

- A state folder that resolves outside the run folder (junction, symlink),
  a run folder inside production, or any configured `*_PATH`/`*_DIR` outside
  the run folder refuses the run before the app is imported.
- Credentials are removed from the environment. The export has no `.env`,
  and the snapshot has no token files.
- The worker can open loopback sockets only. Every tool not in
  `sandbox.LOCAL_TOOLS` is replaced: it returns a case fixture, or fails
  with "no fixture". Order, cancel, send and sync calls are recorded and
  rejected.
- Models go through `claude -p` with `ANTHROPIC_API_KEY` stripped. The
  gateway refuses to start unless `claude auth status` reports claude.ai
  (subscription) auth.

## Where it differs from production (recorded in every trace)

- **Model.** A persona set to DeepSeek runs on the Claude model its
  registry would use (`gateway.STAND_INS`). Tom is DeepSeek V4 Pro live and
  Sonnet 5.5 here. The rewrite and fact extraction run on their configured
  fallbacks. Pass rates are this model's, not DeepSeek's. Baseline and
  candidate always get the same model.
- **Wire shape.** The verbatim tail becomes a labelled transcript inside one
  prompt. Tools arrive over MCP (`mcp__tom__<name>`). The CLI chooses
  thinking and adds a short environment note.
- **Memory writes.** Fact extraction after a turn is off unless the case
  sets `"learn": true`. The exchange is still indexed.
- **Clock.** Real time.

## Cases

A case is JSON: `snapshot`, optional `setup` ops (`create_record`,
`add_entry`, `close_record`, `clear_thread`), `turns`, `evidence` (strings
that must reach Tom), `expected`, `pass_if`, `fail_if`, and optional
`fixtures` (tool name → canned text). Synthetic cases live in `evals/cases/`.
Cases about the user's real data live in `~/.voice-agent-testenv/cases/` and
never enter git (the repo is public).

Each run reports evidence by stage: **stored**, **in the prompt**,
**fetched by a tool**. A failure can then be told apart as absent from
memory, not retrieved, or retrieved but misread.
