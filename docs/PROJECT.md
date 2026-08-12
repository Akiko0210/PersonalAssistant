# Voice AI Notetaking Agent — Project Description

A local, voice-driven notetaking assistant for Windows. You talk to it through a
headset; it answers aloud, records notes, files them into folders, and remembers
past conversations. Everything runs on-device except the Claude API calls (the
reasoning and the note summaries).

This document is the detailed technical description of the project — how it is
structured, how a turn flows through it, where data lives, and how to extend it.
It is also the knowledge source the agent uses to answer questions about itself
(via the `describe_project` tool), so it is kept accurate and current.

---

## 1. What it does

Two modes, switched by a headset button:

- **Conversation mode** (default): a spoken back-and-forth with Claude. Ask about
  saved notes, Discord trade alerts, ingested trading books, the time, or past
  conversations. Manage note folders by voice. Save part of a conversation as a
  note. Switch which Claude model answers.
- **Notetaking mode**: a silent recorder. It listens without speaking; on stop it
  writes a transcript, generates an AI summary, files it into a category folder
  (decided through a short spoken dialogue), and reads the summary back.

Local components: microphone capture + voice-activity detection (webrtcvad),
speech-to-text (faster-whisper), text-to-speech (Windows SAPI), and semantic
search (Chroma + sentence-transformers embeddings). Only the language model is
remote.

---

## 2. Architecture at a glance

The code is layered: hardware/IO services at the bottom, a thin orchestration
layer on top, and pure logic pulled out into standalone, testable modules.

```
                  voice_agent.py  (Agent: main loop + orchestration; root)
                   |         |            |             |            |
     +-------------+         |            |             |            +---------+
     |                       |            |             |                      |
  speech/                 speech/      buttons/      brain/llm.py         web/server.py
  audio.py (mic+VAD)      tts.py       gestures.py   (Claude)             (dashboard,
  barge_in.py (interrupt) sound.py     media_control  |     |              embedded)
                          (speak/cue)  (SMTC button)  |     |
                                            brain/history.py, memory.py   tools/ (registry)
                                            stores/ notes, knowledge,     |
                                                    discord_data,   ------+
                                                    categories

  config.py (root) — constants shared by everything;  lib/ — leaf utilities
```

Design principle: **each module is a self-contained service with a small API and
no knowledge of the others.** The tricky, bug-prone logic (history repair,
barge-in detection, click-gesture decoding) lives in pure modules that can be
unit-tested without a microphone, speakers, or an API key.

---

## 3. Module map

### Orchestration
- **`voice_agent.py`** — entry point and the `Agent` class. Owns the main loop,
  the two modes (`run_conversation_turn`, `run_notetaking`), the `say()` speech
  path (with barge-in orchestration), hotkey handling, and the
  listen-while-thinking logic. Also holds CLI subcommands: `--selftest`,
  `--miccheck`, `--ingest`, `--kb-list`, `--resync`.

### Audio IO
- **`speech/audio.py`** — `AudioEngine`: one always-on input stream pushing fixed-size
  frames onto a queue. `collect_utterance` pulls whole utterances using VAD so
  long silences cost almost nothing. `poll_speech` and `pushback` support
  barge-in (retaining the audio consumed while detecting an interruption).
- **`speech/stt.py`** — `Transcriber`: faster-whisper wrapper (`small.en` by default,
  `vad_filter=True` to reject hallucinated text from silence).
- **`speech/tts.py`** — `Speaker`: Windows SAPI backend (async speak + purge, which is
  what enables barge-in; plus pause/resume, which is what lets a mute click
  leave a reply intact) with a synchronous pyttsx3 fallback. `Announcer` is a
  deliberately separate second voice — one SpVoice serialises its utterances,
  so a notice that must be heard *over* a playing reply ("Muted.") needs its
  own voice object, in its own thread and COM apartment.
- **`speech/sound.py`** — `IdleSound`: loops a "thinking" WAV while the agent waits on
  the model. Idempotent, thread-safe, never raises (missing file = silence).

### Button / interruption logic (extracted, pure-ish, tested)
- **`speech/barge_in.py`** — `BargeInDetector`: decides when the user's voice should
  interrupt playback. Calibrates an echo baseline from the first part of
  playback, then requires frames that are both voiced (VAD) and louder than that
  baseline; a *leaky* counter tolerates brief mid-word dropouts. Retains the
  consumed frames so the user's opening words aren't lost on interruption.
- **`buttons/gestures.py`** — `ClickGestureDecoder`: turns raw button presses into
  single/double/triple gestures. Dedupes presses that arrive on both listener
  channels, counts clicks within a window, and fires the gesture on a timer.
  Thread-safe.
- **`buttons/media_control.py`** — `MediaButtonListener`: a Windows System Media Transport
  Controls (SMTC) session so Bluetooth-native headset buttons (AVRCP, which never
  appear as key events) are received, plus the silent keepalive stream. See
  `MEDIA_CONTROL.md` for the hardware reasoning.

### Language model
- **`brain/llm.py`** — `Claude`: the conversation loop (`converse`, with the tool-call
  loop), the folder-choice dialogue (`choose_folder_via_dialogue`), note
  summarisation (`summarize`), and memory consolidation. Holds a `ToolContext`
  and reads the active conversation model from it each call.
- **`brain/history.py`** — pure functions over the message list: `sanitize` (drop any
  tool_use whose tool_result never arrived, and merge adjacent same-role turns),
  `trim` (rolling window that starts on a clean user message), `load`/`save`.
  This is what makes a conversation persisted mid-tool-loop safe to reload — see
  §6.
- **`brain/memory.py`** — `ConversationMemory`: long-term memory. Stages messages that
  fall off the live window, then at boot consolidates the staged text into one
  dense record embedded in a Chroma `conversations` collection; `search` backs
  the `search_past_conversations` tool.

### Stores
- **`stores/notes.py`** — `NoteStore`: note storage, retrieval, semantic search, folder
  management (create/rename/delete/move), and a `resync` repair pass.
- **`stores/knowledge.py`** — `KnowledgeStore`: ingests reference PDFs/text/video into a
  Chroma `knowledge` collection (idempotent, content-hashed via `manifest.json`)
  and searches it. Video/audio is transcribed by Whisper first and chunked with
  timestamps, so hits cite a moment rather than a page; because that costs ~20 min
  per hour of recording, the agent's boot scan passes `include_media=False` and
  only reports new media. `run.bat` covers the gap by running `--ingest` as a
  separate process *before* launching the agent, so a plain double-click still
  absorbs everything — with console progress, and with the single-instance lock
  released before the agent asks for it. Encrypted PDFs need `cryptography`
  (in requirements.txt); without it pypdf fails with "AES algorithm" and the
  scan reports the file as failed.
- **`stores/discord_data.py`** — `DiscordData`: read-only view over a sibling "Discord
  Notifier" project's captured messages and trade alerts.
- **`stores/categories.py`** — the note-folder registry: seed folders, the
  voice-created/renamed overlay (persisted to `data/categories.json`),
  `category_dir`, and the add/rename/delete API. This is runtime-mutable *state*,
  deliberately separate from `config.py`.

### Configuration
- **`config.py`** — pure constants: paths, audio/VAD parameters, barge-in
  thresholds, model ids, the system prompt, and `ensure_dirs()`. At import it
  overlays `data/config_overrides.json` (whitelisted tunables only) so values
  adjusted in the dashboard reach the agent on its next start.

### Dashboard
- **`web/server.py`** + **`web/static/`** — the web dashboard
  (http://127.0.0.1:8765). Served two ways from the same module: **embedded in
  the agent process** (`serve_embedded`, started by `Agent.run()`, fails soft
  on a taken port), or **standalone** (`python -m web.server` /
  `dashboard.bat`) for browsing/config/ingest while the agent is off. Browse
  notes/folders/transcripts, inspect the live conversation history, memory
  staging, the knowledge base, Discord captures, and session logs — and edit
  the tunable config values from a form. Stdlib-only and read-mostly: it writes
  `data/config_overrides.json` (atomically) and accepts knowledge uploads. Its
  search is a plain substring scan — semantic search stays a voice feature.
  - The **Knowledge page is laid out for a user with tunnel vision**: a single
    ~560px column, numbered steps, 17px+ text, full-width 54px buttons, and
    stacked rows instead of a multi-column table (horizontal scanning is the
    pattern to avoid). A `@media (max-width: 620px)` rule tightens padding so
    the same layout survives heavy browser zoom. Keep new UI here to that shape.
  - **Knowledge upload/ingest** is the one heavyweight path. Files dropped on
    the Knowledge page stream to `knowledge/` via a `.part` rename, and
    "Ingest" runs a real ingest on a background thread. Pending (not-yet-
    ingested) files can be deleted from the page; ingested ones cannot, since
    their chunks live in Chroma — `remove_pending` refuses anything the manifest
    claims. That thread imports
    `KnowledgeStore` *inside itself*, so a dashboard that never ingests still
    starts instantly and loads no embedding model, and it holds the agent's
    single-instance lock for the duration — embedding writes the same Chroma
    store the agent has open, so an ingest and a live agent are mutually
    exclusive by design. The UI disables the button and says so while the agent
    runs; the server re-checks the lock regardless.
  - **Live controls** (the sidebar mute button, and the Conversation
    page's chat composer) are the one
    place the dashboard touches a *running* agent — and embedded, that touch
    is a direct method call: `/api/control/<name>` dispatches through the
    `CONTROL_ACTIONS` table onto `server.agent` (`agent.set_mute`,
    `agent.queue_typed_message` — public methods, called from HTTP handler
    threads exactly as the headset-button thread calls them). Standalone,
    `server.agent` is None and the routes answer honestly: "agent not
    running", or "the agent is running — use its dashboard" when the lock
    says one is alive in its own process. **To add a control**: one handler
    in `CONTROL_ACTIONS` (validate the payload, raise `ValueError` for a bad
    one → 400), one public Agent method, one button POSTing
    `/api/control/<name>`; the dispatch — which carries the one provenance
    log line — never changes. Typed messages ride the agent's existing
    interjection wake-up: queued, answered aloud at the next utterance
    boundary, never truncating speech, and working while muted.

### Tools package
- **`tools/`** — the tool registry (§5). `__init__.py` holds the `@tool`
  decorator, the `ToolContext` dataclass, `api_tools()` (schemas for the API
  call), and `dispatch()`. Each `*_tools.py` module registers handlers for one
  domain.

### Trading package
- **`trading/`** — real trading on the tastytrade API: multi-leg option
  strategy building (19 named strategies, validated against the live option
  chain), real-time bid/ask over a DXLink websocket, order dry-run → confirmed
  submit → cancel, and realized/unrealized P&L from transaction history and
  live marks. Self-contained: the agent reaches it only through
  `tools/trading_tools.py`, the dashboard through lazily imported
  `/api/trading/*` routes, and an unconfigured machine degrades to a friendly
  "not configured" message. Credentials come from `.env` (`TASTY_*` keys, or
  `TASTY_ENV_FILE` pointing at the Tasty-Web project's `.env`); orders go to
  the sandbox unless `TASTY_ENV=live` is set explicitly. Submitting requires a
  dry-run review of the exact current ticket (fingerprint-checked) plus an
  explicit spoken confirmation — the gate lives in `trading/orders.py`, shared
  by voice and web. Design + API research: **TRADING_PLAN.md**,
  **TRADING_RESEARCH.md**.

### Tests & scripts
- **`tests/`** — `unittest` suite over the pure logic (history, barge-in,
  gestures, summary parsing, model + project tools). Run:
  `python -m unittest discover tests`.
- **`scripts/`** — manual hardware probes (`test_pynput.py`, `test_rawinput*.py`)
  used while developing button handling; not part of the test suite.

---

## 4. How a conversation turn flows

1. `run_conversation_turn` calls `audio.collect_utterance` — blocks (cheaply)
   until a spoken utterance is captured, using VAD + endpointing.
2. `stt.transcribe` turns the audio into text.
3. `_converse_with_followups` **settles before calling**: after the utterance
   ends it listens for `CONTINUATION_SETTLE_MS` (`_await_continuation`); if the
   user resumes, the continuation is captured, merged, and the window restarts.
   Only once they've finished does it call `llm.converse` — **once**, with the
   complete utterance. This is what keeps a multi-part question (spoken with
   pauses) to a single billed model call. (An earlier design fired a speculative
   `converse` at each pause and discarded the reply when the user kept talking;
   that billed a full call per pause — §11.)
4. `converse` sanitizes + trims history, appends the user message, then loops:
   call the model → if it returned `tool_use`, dispatch each tool via the
   registry and feed results back → repeat until the model returns plain text.
   The "thinking" cue loops the whole time.
5. `say()` speaks the reply. While speaking, every mic frame is fed to a
   `BargeInDetector`; if it fires, TTS stops, the captured speech is pushed back
   for the next turn, and (optionally) the unsaid tail is remembered for a
   "continue" command.
6. `_after_reply` drains, in fixed order, everything the reply's tool calls
   deferred: background delegations start (`ask_agent`), a prepared note runs
   the folder-choice dialogue and is filed (`save_conversation_note`), and a
   persona hand-off is performed (`switch_agent`, answering any forwarded
   question in the new voice). A note prepared *inside* a forwarded turn is
   drained right there — deferred work must never outlive the turn that
   created it (that leak once ate a "switch me back" command).
7. History is saved to `data/history.json` after the turn.

### Personas and background delegation

Alice (general), Bob (notes/memory), and Tom (trading) each keep their OWN
conversation with the user (`brain/agents.py`, one history file per persona)
and their own private memory. Switching personas swaps the whole thread
(`Claude.switch_to`): what one persona was told, another cannot see, and
context crosses over only as an `ask_agent` summary — ask Tom what you told
Alice and he says he doesn't have access, then offers to ask her. Isolation
is structural (which files and Chroma collections get read), never a prompt
rule; the registry's `reads` grants are the one deliberate exception (a
future reviewer persona may be granted read access to a specialist's private
stores). Knowledge splits the same way: the shared `knowledge` collection
plus a private `knowledge_<key>` per persona, chosen per ingest.

A request can move between personas two ways, and the difference is the core
of the design:

- **`switch_agent` — the user moves.** They talk to the other persona from now
  on. Announced aloud with the persona's model ("Bob here, running on Haiku
  4.5") because the model is the one part of a switch you can't hear.
- **`ask_agent` — a task moves.** The other persona runs it in the background
  (`Claude.run_delegated_task`: an isolated mini tool-loop on its own thread,
  its own message list and `ToolContext` scoped to the delegate — so a
  delegated Alice searches ALICE's memory, which is what makes "ask Alice
  what we discussed" work), while the user keeps talking to the persona they
  were with. The result is
  queued as a spoken **interjection**, delivered in the worker persona's own
  voice at the next utterance boundary: after the current reply finishes, or
  immediately if the agent is idle (a `wake` event ends the listening wait,
  but never mid-utterance — the user is never cut off, and never spoken
  over). The delivery announcement is deliberately not interruptible: it is
  the one moment the background work has to reach the user. If the task
  prepared a note, the interjection runs the normal folder-choice dialogue —
  so it is the *worker* persona's voice asking "which folder?". What happened
  is folded back into the active persona's history via `record_tool_event`,
  and a note still waiting for its folder question at quit is rescued into
  its suggested folder rather than lost.

---

## 5. The tool registry

Tools are how the model reaches the app's data and actions. Each tool is a
decorated handler co-located with its JSON schema:

```python
@tool({"name": "...", "description": "...", "input_schema": {...}})
def my_tool(ctx, args):
    return "spoken-friendly result string"
```

- **`ctx`** is the shared `ToolContext` (the stores: `store`, `discord`, `kb`,
  `memory`; plus mutable session state: `pending_note`, `pending_switch`,
  `pending_delegations`, `convo_model`, `active_agent`, `focus`). Retrieval
  tools pass `ctx.active_agent` as the caller and the stores resolve what it
  may read (`agents.readable_owners`) — there is no parameter through which a
  tool can name another persona's collection.
- **`args`** is the raw input dict from the model.
- The return string becomes the `tool_result` fed back to the model.

`api_tools()` produces the `tools=` list for the API call (optionally excluding
some, e.g. during the folder dialogue). `dispatch()` runs one call and turns any
exception into a `"Tool error: ..."` string so a tool bug can never kill the
conversation loop.

**Adding a capability is one function in one file** under `tools/` — no edits to a
central list or dispatch chain.

### Current tools by domain
- **notes** (`note_tools.py`): `search_notes`, `list_recent_notes`, `read_note`,
  `list_folders`, `create_folder`, `rename_folder`, `delete_folder`, `move_note`,
  `count_notes`, `save_conversation_note`.
- **discord** (`discord_tools.py`): `get_recent_discord_messages`,
  `search_discord_messages`, `get_recent_trades`.
- **time** (`time_tools.py`): `get_current_time`.
- **memory** (`memory_tools.py`): `search_past_conversations` — the caller's
  OWN staging, archive, and saved live window (plus the pre-isolation shared
  archive, labelled); never another persona's.
- **knowledge** (`knowledge_tools.py`): `search_knowledge` — common plus the
  caller's private collection(s), merged by distance.
- **focus** (`focus_tools.py`, Tom only): `set_focus`, `clear_focus`,
  `get_focus` — narrow retrieval to a strategy/underlying until cleared.
  A hard metadata filter on private collections, soft (fall back to
  unfiltered) on common; the active focus is folded into the system prompt so
  the model knows its retrieval is narrowed. Session state, never persisted.
- **model** (`model_tools.py`): `get_current_model` — a live read of which
  model and provider are answering, and as which persona. It resolves the same
  expression `converse()` routes the API call with, so it cannot disagree with
  reality. It exists because the model answers "which model are you?" from
  conversation history, which carries model choices that have since changed.
  The system prompt has always stated the truth, and the model overrode it
  from history three times in one session — "I'm Opus" while on Haiku,
  "DeepSeek V4 Flash" while on Opus 5 (session_2026-07-31.log 11:32 / 11:48 /
  12:23; histories were still shared across personas then, but a persona's
  own stale turns reproduce the failure fine).
  `config.model_identity_block` therefore makes it a hard rule: identity
  questions are answered by this tool or not at all. The tool result lands at
  the *end* of the context, where it outweighs old turns, and leaves an
  auditable `tool_use` line in the log.
- **model** (`model_tools.py`): `set_conversation_model` — switch the
  conversation model between Haiku 4.5, Sonnet 5, Opus 5, and (when
  `DEEPSEEK_API_KEY` is set) DeepSeek V4 Flash / Pro by voice. DeepSeek runs
  through its Anthropic-compatible endpoint via `Claude.client_for`, so every
  model shares one code path; `config.model_provider` is the routing rule.
  A mid-session choice is remembered **per persona** (`Claude._model_overrides`)
  and follows that persona everywhere: `Claude.model_for(key)` is the one
  resolver behind `converse()`, background delegation, and the switch
  announcement ("Bob here, running on Haiku 4.5" — spoken because the model is
  the one part of a switch you can't hear). Delegation deliberately shares it:
  when it ran on the registry default instead, a delegated Alice answered
  "Haiku" seconds before the switch announcement said "DeepSeek V4 Flash"
  (session_2026-08-10.log 18:22).
- **project** (`project_tools.py`): `describe_project` — returns this document so
  the agent can answer questions about its own design.
- **agents** (`agent_tools.py`): `switch_agent` — hand the user over to another
  persona (deferred via `pending_switch`, so the goodbye finishes in the old
  voice); `ask_agent` — delegate a task to another persona in the background
  (deferred via `pending_delegations`; result spoken as an interjection — §4).
- **trading** (`trading_tools.py`): `trading_status`, `get_quote`,
  `list_expirations`, `build_strategy`, `adjust_leg`, `set_order_terms`,
  `review_order`, `submit_order` (requires review + explicit confirmation),
  `list_orders`, `cancel_order`, `get_positions`, `get_pnl`, `clear_ticket` —
  real trading via the tastytrade API (see the trading package above).

---

## 6. Conversation history & the invariant that once bricked the app

Each persona's live conversation is persisted to `data/history_<key>.json`
after every turn and restored on the next boot (the pre-isolation shared
`history.json` is parked as `.bak` by a one-time migration). The Anthropic API enforces invariants a saved history
can silently violate: every `tool_use` must be answered by a `tool_result` in the
next turn, and roles must alternate. A turn abandoned mid-tool-loop (e.g. by the
barge-in / listen-while-thinking path) could persist an assistant `tool_use` with
no matching `tool_result`. Because the file reloads on every launch, one such bad
save caused a `400 Bad Request` on startup **every time** — the app was bricked,
not merely crashed once.

`history.sanitize()` fixes this: it drops any `tool_use` without a result (and any
orphaned `tool_result`), then merges adjacent same-role turns so alternation
holds. It runs on load, before every send, and before every save, so a history
can never be persisted — or replayed — in a shape the API rejects. `converse`
also catches per-turn API errors so one bad request logs and continues instead of
taking down the whole session.

---

## 7. Memory: three layers, each per persona

1. **Live window** — each persona's recent turns in `data/history_<key>.json`
   (`HISTORY_MAX_MESSAGES` each).
2. **Long-term memory** — text that ages out of a window is staged to
   `data/memory_pending.json` tagged with its persona, then consolidated at
   boot (one summary call per persona with enough material) into that
   persona's Chroma collection `conversations_<key>`.
   `search_past_conversations` reads the caller's own staging, archive, and
   saved live window — plus the pre-isolation `conversations` collection,
   whose mixed history is readable by all and labelled as legacy.
   `scripts/seed_agent_memory.py` backfills the per-persona archives from the
   session logs (every turn is attributable: boots start on the default
   agent, switches are logged).
3. **Notes** — deliberate, saved artifacts (recorded sessions or
   conversation-derived), filed in category folders and semantically
   searchable. Notes and the common `knowledge` collection are the SHARED
   write paths: information meant for every persona belongs there, not in a
   private thread.

---

## 8. Where data lives

```
data/<Folder>/       notes: <id>.md (summary + frontmatter) + <id>.transcript.md
data/pending/        transient live transcript while recording
data/chroma/         Chroma index: notes, knowledge, conversations, plus
                     per-agent knowledge_<key> / conversations_<key>
data/index.json      ordered record of every note (title, date, category)
data/categories.json voice-created/renamed folders overlaid on the seed defaults
data/history_<key>.json  each persona's live window (sanitized on every save)
data/memory_pending.json  staged text awaiting consolidation, tagged by persona
knowledge/           reference PDFs/text/video you ingest + manifest.json
logs/                dated session logs
```

Everything under `data/`, `logs/`, `knowledge/`, and `.env` is gitignored — the
user's content and keys never enter version control.

Only one agent may run against this directory at a time. At startup
`lib/single_instance.py` takes a Windows `msvcrt` lock on `data/agent.lock`; a second
launch finds it held and exits with a spoken notice, so two instances can't race
on `history.json` / the Chroma index (which would corrupt them) or talk over each
other. The OS drops the lock when the process exits — including on a crash — so
no stale lock is ever left behind.

Every state file the app rewrites — `history.json`, `memory_pending.json`,
`index.json`, `categories.json`, the knowledge `manifest.json`, and each note's
`.md` summary — is written **atomically** via `atomic_io`
(`write_text_atomic` / the `write_json_atomic` convenience wrapper): temp file,
fsync, then `os.replace` (an atomic same-volume rename). A power loss mid-save
therefore leaves either the complete old file or the complete new one, never a
torn or empty file. The two intentional exceptions are the live-transcript
*append* (incremental durability by design) and the WAV keepalive (not user
data). Chroma is SQLite, already crash-safe by design.

---

## 9. Headset button & barge-in

One physical play/pause button drives everything, listened for on two channels at
once (keyboard hook for wired/dongle headsets; SMTC for Bluetooth-native ones).
`ClickGestureDecoder` dedupes across channels and resolves gestures: **1 click =
mute, 2 = toggle notetaking, 3 = quit**. Every accepted press immediately
*pauses* output (the "hush" path) because a state-tracking dongle swallows the
next press if playback continues through it — but pausing, not purging, because
at press time the gesture hasn't resolved yet. `_hold_for_gesture` then waits for
it: **mute deafens the microphone at once and lets the reply finish** (muting
stops listening, not talking), while notetaking and quit end the reply for good.
Mute is also the one command that never abandons a turn — a click while the
model is still thinking leaves the answer to arrive and be spoken; only
notetaking and quit abort (the `silence` event marks which is which). The
acknowledgement rides a second `Announcer` voice so it overlaps the reply
instead of queueing behind it. A press with no gesture behind it resumes after
`GESTURE_VERDICT_TIMEOUT_S`, and a backend that can't pause falls back to
stopping. Voice barge-in is separate:
while the agent speaks, start talking and `BargeInDetector` stops it and captures
your words. See `MEDIA_CONTROL.md` for the full hardware story.

---

## 10. Extending the project

### Where each kind of addition goes

Each row is meant to be the ONLY place you touch. If a change starts spreading
past its row, that is the signal to stop and look for the seam you're missing
rather than to keep editing — see the duplication rule in `CLAUDE.md`.

| To add… | Edit | Notes |
| --- | --- | --- |
| **A voice tool** | one decorated function under `tools/`, plus its module import at the bottom of `tools/__init__.py` | the `@tool` schema IS the content; the body should be a call into a store (§5) |
| **A dashboard control** (button that acts on the live agent) | one handler in `CONTROL_ACTIONS` (`web/server.py`), one public `Agent` method, one front-end button POSTing `/api/control/<name>` | never the dispatch or routing — they are generic and carry the one provenance log line |
| **A persona** | one entry in `brain/agents.py`'s `AGENTS` (`name, role, persona, aliases, model, tools, tts_voice, tts_rate`) | aliases must stay globally unique or `match_address` becomes ambiguous |
| **A note folder** | created by voice at runtime; or seed one in `stores/categories.py` | |
| **A tunable** | the constant in `config.py` (with its rationale comment), its type in `OVERRIDABLE`, and a `TUNABLES` row in `web/server.py` | the row generates the Config-page control AND its server-side validation |
| **A dashboard page** | a `views.<name>` function in `web/static/app.js`, a nav link in `index.html`, and its read-only API route | 12 views exist; copy the nearest one's shape |
| **A different speech/embedding engine** | `speech/stt.py`, `speech/tts.py`, or the store's `_ensure_chroma` | orchestration never names a backend |
| **A shared test fake** | `tests/agent_fixtures.py`, `tests/llm_fixtures.py`, `tests/trading_fixtures.py` | extend these; a fourth copy of a fake is the bug this convention exists to stop |

### Limits, and the number that triggers each

None of these are bugs — they are deliberate bounds. They are written down
because code that *assumes* a limit away breaks quietly: the dashboard's chat
box first shipped watching the history message count grow, which a full
history never does.

| Limit | Now | What happens past it |
| --- | --- | --- |
| Conversation history | **40 messages** (`HISTORY_MAX_MESSAGES`, adjustable 4–200 on the Config page) | oldest turns fall off the window and are staged into long-term memory (§7) — the count stops growing, it does not grow forever |
| Dashboard note search | every note file opened per query (**291 notes** when this was written — check the Overview page for today's count) | linear; fine at hundreds, slow in the low thousands. Deliberately a substring scan — semantic search stays a voice feature (Chroma), so this path never loads the embedding model |
| Turns | **one at a time** | the loop is blocking. A typed message waits for an utterance boundary; during note-taking it waits for the note to end. Background delegations (`ask_agent`) are the exception — they run on threads and speak their result at the next gap |
| Tool rounds per turn | **15** conversation, **8** delegated | the loop bails with a spoken "I got stuck repeating tool calls" rather than billing forever |
| Knowledge ingest | mutually exclusive with a running agent | both write the same Chroma store; the single-instance lock enforces it, so the dashboard's Ingest button is dead while the agent runs |
| Agent instances | **one** | the lock (`lib/single_instance.py`). A second would double the Whisper + embedding memory and race on history and the index |
| Typed message | **4000 chars** (`MAX_MESSAGE_CHARS`) | rejected with a 400; a message is a question, not a document |
| One utterance | **30 s** (`MAX_UTTERANCE_S`) | capture is cut and transcribed as-is |

### Tuning

Audio thresholds, models, endpointing, barge-in sensitivity, and the history
window are constants in `config.py`, adjustable visually on the dashboard's
Config page — edits persist to `data/config_overrides.json` and apply at the
agent's next start.

### Roadmap (not yet built)
- **Event-driven core** — replace the blocking loop with actors (Ears, Mouth,
  Brain, Buttons) coordinated by an explicit state machine, so features like
  streaming replies, a wake word, or a second IO surface become declarative
  transitions instead of interleaved polling.
- **Engine interfaces** — formal `SpeechToText` / `TextToSpeech` / `ChatModel`
  protocols so backends (streaming Whisper, neural TTS, streaming Claude) are
  swappable without touching orchestration.

---

## 11. One reply per turn (and why the model isn't called until you finish)

Speech recognition ends an utterance after `CONVO_ENDPOINT_MS` of trailing
silence — but a natural pause mid-thought can be longer than that, so the app
must tolerate you continuing after it thinks you stopped.

The **original** approach fired `converse()` speculatively the instant the
utterance endpointed, in a background thread, while watching the mic. If you
kept talking, it waited for that in-flight call to finish, threw the reply away
(`discard_last_turn`), and re-asked with the merged text. Because Anthropic's
non-streaming `messages.create` generates (and bills) the whole response
server-side, **every mid-thought pause cost a full, discarded model call** — and
more if the speculative call fanned out into a tool loop. On a pricier model
(the `set_conversation_model` tool can switch to Opus) that adds up fast.

The **current** approach settles first: after the utterance ends,
`_await_continuation` listens for `CONTINUATION_SETTLE_MS`; if you resume, the
continuation is captured, merged, and the window restarts; only once you've
truly finished is `converse()` called — exactly once, with the complete
utterance. You hear the same single reply you always did (the speculative
intermediate replies were never spoken anyway), but nothing is generated or
billed until the turn is complete. The cost is a small, tunable latency
(`CONTINUATION_SETTLE_MS`) before each reply. The behaviour is covered by
`tests/test_continuation.py` (one `converse` per turn, continuations merged,
coughs and hotkeys cost nothing).
