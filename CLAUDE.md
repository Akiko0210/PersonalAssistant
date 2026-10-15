# Claude Code instructions — Voice AI

Read first: `docs/PROJECT.md` (architecture + the reasoning behind every
design decision), `README.md` (user-facing docs), `docs/MEDIA_CONTROL.md`
(headset button arcana). Tests: `python -m unittest discover tests`.

## Who the user is (consider it in every change)

The app is for a visually impaired user who isn't blind but has **tunnel vision**.
They run daily life and options trades through it, mostly by voice. Before
shipping anything, ask how it works for someone who can't scan a screen:

- **Voice is the primary interface.** Every feature must be usable by
  speaking and listening; a screen-only feature is unfinished. Spoken replies
  should be short, lead with the answer, and read numbers naturally.
- **Accuracy beats speed.** The user can't easily check a number against the
  screen, so a stale or guessed position, price, or fill does real harm (the
  2026-10-15 phantom short leg). Answer from current data, and say so plainly
  when the data might be stale.
- **Never hide state in visuals.** A colour, an icon, or a toast alone is
  invisible to this user; every state change also needs text next to its
  control, or a spoken line.
- **The dashboard rules below** (narrow column, 17px+ text, ≥46px targets)
  follow from this; apply the same thinking to anything new.

## Architecture in one paragraph

One process. `voice_agent.py` (root) owns the microphone, TTS, and all live
state — and serves the web dashboard from inside itself
(`web/server.py`'s `serve_embedded`, port `config.DASHBOARD_PORT`), so a
dashboard button is a direct method call on the live Agent. `dashboard.bat`
runs the same server standalone for browsing/config/ingest while the agent is
off; its control routes answer honestly ("agent not running" / "use the
agent's dashboard"). Code lives in topic packages: `speech/` (mic, STT, TTS,
barge-in), `media_control/` (headset buttons: portable gesture decoding in
`gestures.py`, per-OS channels in `windows.py`/`macos.py`/`linux.py`, wired
by `main.py`), `brain/` (llm/ — the engine in `main.py`, providers in
`anthropic.py`/`deepseek.py` — plus agents, history, memory, facts, query,
context), `stores/`
(notes, knowledge, categories), `lib/` (leaf utilities), `web/` (server +
static), `tools/` (the tool registry), and `trading/`. Entry points and
`config.py` stay at root.

**To add a live dashboard control**: one handler in `web/server.py`'s
`CONTROL_ACTIONS` (validate the payload; raise `ValueError` for a bad one →
400), one public method on `Agent`, one front-end button POSTing
`/api/control/<name>`. Do not touch the dispatch or routing — they are
generic on purpose, and they carry the one provenance log line. If you find
yourself editing them, stop and reconsider.

## Code-writing practice (user-dictated — follow it)

- **Lean first.** Build the minimum that satisfies the request. A feature's
  size should embarrass no one: a button should cost tens of lines, not
  hundreds.
- **Duplication is strictly limited.** No wrapper whose body is a delegation
  plus a log — log at the dispatch site. Before writing a function, look for
  the existing one (`atomic_io.read_json`, `frontmatter.parse_frontmatter`,
  `agents.registry_model`, `categories.slug_of`, `llm._tool_loop`,
  `tests/*_fixtures.py` all exist because their logic was once copied
  instead). The same ten lines twice is a bug waiting to diverge — the tool
  loop existed three times and only one copy handled truncation.
- **Seams over speculation.** Make the *next* feature cheap (a registry, an
  injected callable, a generic route) rather than building machinery for
  futures nobody asked for. Reusable structure yes; speculative robustness no.
- **Flag extras, don't ship them silently.** If robustness, edge-case
  handling, or infrastructure beyond the ask seems warranted, say so and give
  the cost — the user decides. Never let scope grow quietly.
- **Tests sized to the feature.** ~100-180 lines for a typical feature; test
  the seams and the honest-failure paths, not every permutation. Shared fakes
  live in `tests/agent_fixtures.py`, `tests/llm_fixtures.py`,
  `tests/trading_fixtures.py` — extend those, don't redefine them.
- **House comment style.** Comments explain *why* (often naming the failure
  they prevent), not what; section banners look like `# --- Title ---…`;
  constants live in `config.py` with a rationale comment.
- **Dashboard UI is for a tunnel-vision user.** Narrow single column (~560px),
  stacked layout, 17px+ text, big targets (≥46px), one obvious action per
  card, status text right next to the control that produced it. No wide
  tables.

## Issue → branch → PR workflow (user-dictated — follow it)

Every task that changes the repo is traceable: an issue says why, a PR says
what. This is standing authorization to open issues, push feature branches,
and open PRs on `origin` — merging stays with the user. Questions and
read-only investigations need no issue.

1. **Issue first**, before editing (`gh issue create`). Body: the problem or
   request, acceptance criteria, and what is out of scope. Labels come from
   `gh label list` — one type (`bug`, `enhancement`, `documentation`, …) plus
   an area if one fits; create a new label only when none fits, and say so.
2. **One branch per issue**, cut from `main`: `<type>/<issue#>-<slug>`
   (e.g. `feat/42-upload-button`). Work that turns up mid-task and isn't
   this issue gets its own issue — never fold it in silently.
3. **One PR per issue** (`gh pr create`), body opening with `Closes #<n>`:
   - **What changed and why** — per file/area, plain language.
   - **Before / after** — behaviour, output, or UI, side by side.
   - **Tests** — what was added, the exact command, and its result line;
     say plainly what was *not* tested.
   - **Visuals** where they help — a Mermaid diagram for flow/structure
     changes (GitHub renders it), before/after screenshots for dashboard
     changes. `gh` can't upload images; if one can't be attached, say so.

## Gotchas

- **Variant-file convention.** The app is cross-platform (Windows / macOS /
  Linux) and multi-provider (Anthropic / DeepSeek). Variant-dependent code
  lives in variant-named files inside its package —
  `windows.py`/`macos.py`/`linux.py` (or `posix.py` when macOS and Linux
  genuinely share one implementation), `anthropic.py`/`deepseek.py` — wired by
  that package's `main.py`; consumers import `<pkg>.main`. Common logic stays
  in `main.py`; a variant file is the only place its OS/provider library may
  be imported. Extend a variant file or add a new one — never scatter
  `sys.platform`/provider checks elsewhere. Current variant packages:
  `media_control/`, `speech/tts/`, `speech/sound/`, `lib/single_instance/`,
  `brain/llm/`, `trading/` (`schwab.py`/`tastytrade.py`, keyed by the
  persona's `brokers` in `brain/agents.py`).
- Several Pythons per machine, only one with the deps: `.claude/launch.json`
  pins it (Windows Store Python on the PC, `.venv` on the Mac) — keep the pin
  pointed at whichever interpreter has the deps on the machine you're on.
- `data/` is the user's own state and is gitignored: use `lib/atomic_io` for
  state files (atomic temp-and-rename, with a retry for Windows file-lock
  violations); avoid chatty disk writes.
- The single-instance lock (`lib/single_instance/`) means agent-side
  scripts and a running agent are mutually exclusive; probing the lock
  deletes the lock file when free (`web/server.py` memoises the probe).
- `web/server.py` must never import `stores/notes.py`/chromadb at module
  level — the standalone dashboard's instant start depends on it.
