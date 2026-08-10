# Claude Code instructions — Voice AI

Read first: `PROJECT.md` (architecture + the reasoning behind every design
decision), `README.md` (user-facing docs), `MEDIA_CONTROL.md` (headset button
arcana). Tests: `python -m unittest discover tests`.

## Architecture in one paragraph

Two processes. `voice_agent.py` owns the microphone, TTS, and all live state;
`dashboard.py` is a separate stdlib-only web server that reads state files and
edits config. They meet in exactly one place: the agent hosts a localhost HTTP
control endpoint (`controller.py`, port `config.CONTROL_PORT`) whose actions
live in `controller_service.py`; the dashboard proxies `/api/control/*` to it.

**To add a live dashboard control**: add one method to `controller_service.py`
(validate the payload; raise `ValueError` for a bad one → the server answers
400), register it in `actions`, and add a front-end button that POSTs
`/api/control/<name>`. Do not touch `controller.py` or dashboard routing —
they are generic on purpose. If you find yourself editing them, stop and
reconsider.

## Code-writing practice (user-dictated — follow it)

- **Lean first.** Build the minimum that satisfies the request. A feature's
  size should embarrass no one: a button should cost tens of lines, not
  hundreds.
- **Seams over speculation.** Make the *next* feature cheap (a registry, an
  injected callable, a generic route) rather than building machinery for
  futures nobody asked for. Reusable structure yes; speculative robustness no.
- **Flag extras, don't ship them silently.** If robustness, edge-case
  handling, or infrastructure beyond the ask seems warranted, say so and give
  the cost — the user decides. Never let scope grow quietly.
- **Tests sized to the feature.** ~100-180 lines for a typical feature; test
  the seams and the honest-failure paths, not every permutation.
- **House comment style.** Comments explain *why* (often naming the failure
  they prevent), not what; section banners look like `# --- Title ---…`;
  constants live in `config.py` with a rationale comment.
- **Dashboard UI is for a tunnel-vision user.** Narrow single column (~560px),
  stacked layout, 17px+ text, big targets (≥46px), one obvious action per
  card, status text right next to the control that produced it. No wide
  tables.

## Gotchas

- Two Pythons on this machine: only the Windows Store Python has the deps;
  `.claude/launch.json` pins it — keep that pin.
- `data/` is Dropbox-synced: use `atomic_io.write_json_atomic` for state files
  (it retries Windows sharing violations); avoid chatty disk writes.
- The single-instance lock (`single_instance.py`) means agent-side scripts and
  a running agent are mutually exclusive; probing the lock deletes the lock
  file when free.
