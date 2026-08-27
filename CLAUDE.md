# Claude Code instructions — Voice AI

Read first: `docs/PROJECT.md` (architecture + the reasoning behind every
design decision), `README.md` (user-facing docs), `docs/MEDIA_CONTROL.md`
(headset button arcana). Tests: `python -m unittest discover tests`.

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
`anthropic.py`/`deepseek.py` — plus agents, history, memory), `stores/`
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
  `brain/llm/`.
- Several Pythons per machine, only one with the deps: `.claude/launch.json`
  pins it (Windows Store Python on the PC, `.venv` on the Mac) — keep the pin
  pointed at whichever interpreter has the deps on the machine you're on.
- `data/` is Dropbox-synced: use `lib/atomic_io` for state files (it retries
  Windows sharing violations); avoid chatty disk writes.
- The single-instance lock (`lib/single_instance/`) means agent-side
  scripts and a running agent are mutually exclusive; probing the lock
  deletes the lock file when free (`web/server.py` memoises the probe).
- `web/server.py` must never import `stores/notes.py`/chromadb at module
  level — the standalone dashboard's instant start depends on it.
