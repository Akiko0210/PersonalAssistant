# Voice AI Agent — Session Handoff

_Prepared 2026-07-15. Structured so the next session can pick up cold. Open concerns are front and center._

---

## Context

Local Windows voice notetaking agent (Python). Two modes: **conversation** and **notetaking**. Local STT (faster-whisper), TTS (Windows SAPI), semantic search (Chroma + sentence-transformers); Claude for reasoning and note summaries. Working directory: `C:\Users\EAC\Dropbox\1eac notes\Voice AI` (inside Dropbox). This session did a large refactor plus several safety/cost fixes.

---

## Git state (IMPORTANT)

- **Currently on branch `refactor`. 12 commits ahead of `main`. NOT merged.**
- `main` has only the first fix (history-sanitize crash fix + media-control). The user's **running app runs from `main`**, so it does **not** yet have: the wasted-API-call fix, single-instance lock, atomic writes, the refactor, or the two new tools.
- **55 unit tests, all passing** — run `python -m unittest discover tests`. No hardware needed for tests.

---

## What shipped on `refactor`

1. **Crash fix (also on `main`):** an orphaned `tool_use` in `history.json` caused a `400` that bricked startup on every launch. `history.sanitize()` now runs on load/send/save; per-turn API errors are caught instead of killing the app.
2. **Refactor Phase 1–2:** extracted `history.py`, `barge_in.py`, `gestures.py`, `categories.py` (out of `config.py`); built a **tool registry** (`tools/` — adding a tool = one decorated function + one import). `llm.py` went 729 → ~340 lines.
3. **New tools:** `set_conversation_model` (switch Haiku / Sonnet / Opus by voice) and `describe_project` (answers questions about itself from `PROJECT.md`).
4. **Wasted-API-call fix ("settle before answering"):** the old code fired a speculative `converse()` at every mid-thought pause and discarded it — N pauses = N billed calls. Now it waits `CONTINUATION_SETTLE_MS` (600 ms) of silence, merges continuations, and calls the model **once**.
5. **Single-instance lock** (`single_instance.py`, Windows-only): a second launch is refused so two agents can't corrupt data or talk over each other.
6. **Atomic writes** (`atomic_io.py`): all state-file writes (`history.json`, `memory_pending.json`, `index.json`, `categories.json`, `manifest.json`, note `.md` files) now use temp + fsync + rename, so a power loss can't corrupt or empty them.

---

## OPEN CONCERNS / DECISIONS PENDING

1. **Merge `refactor` → `main`** — the biggest item. Until then the running app still burns credits on mid-thought pauses, has no single-instance guard, and does non-atomic writes. Merge-vs-cherry-pick not yet decided.
2. **`HISTORY_MAX_MESSAGES = 40` is small** for the user's long technical conversations (~10–20 exchanges once tool calls are counted). Open: raise to ~150–200. Trade-off: per-turn input cost. **Prompt caching** was offered to keep a big window cheap — not implemented.
3. **Mid-session memory blind spot:** consolidation runs **only at boot**, so content that ages out of the 40-message window during a session isn't searchable until restart. Offered fix: periodic consolidation — not done.
4. **Long-term memory is lossy** (summaries, not verbatim) and requires the model to *choose* to call `search_past_conversations`. Inherent to the design; the durable path is saving a note.
5. **Settle latency is tunable:** ~1.4 s of silence before a reply (`CONVO_ENDPOINT_MS` 800 + `CONTINUATION_SETTLE_MS` 600). Suggested trying 800 / 400 — not changed.
6. **Roadmap Phase 3–4** (in `PROJECT.md`): event-driven actor core (enables streaming replies, wake word) and swappable engine interfaces. Not started.

---

## Key gotchas for whoever continues

- **The Windows lock file MUST stay empty** — writing or truncating it after `msvcrt.locking` silently drops the lock. (Cost a debugging cycle; the lock is the mechanism, not the file contents.)
- **Atomic writes:** use `write_json_atomic` / `write_text_atomic` from `atomic_io.py` for any new state file. Intentional non-atomic exceptions: the live-transcript **append** (`notes.append_transcript`) and the WAV keepalive.
- **History invariant:** never persist an assistant `tool_use` without its `tool_result` — `history.sanitize()` guards this; don't bypass it. This is what once bricked startup.
- **Adding a tool:** one `@tool`-decorated function in `tools/<domain>_tools.py` + an import in `tools/__init__.py`. Handlers take `(ctx, args)` and return a string.
- **Data safety:** `data/`, `logs/`, `knowledge/`, `.env` are gitignored — switching branches never touches user data. All three memory layers (live window, long-term summaries, notes) live under `data/`.

---

## Recommended immediate next action

Decide on **merging `refactor` → `main`** (or cherry-picking the settle fix + single-instance lock + atomic writes), so the running app actually benefits. Then optionally bump `HISTORY_MAX_MESSAGES` and add prompt caching.