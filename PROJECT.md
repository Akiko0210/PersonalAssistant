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
                         voice_agent.py  (Agent: main loop + orchestration)
                          |      |      |        |          |
        +-----------------+      |      |        |          +----------------+
        |                        |      |        |                           |
   audio.py                   tts.py  sound.py  gestures.py            llm.py (Claude)
   (mic + VAD)                (speak) (cue)     (button decode)         |    |     |
        |                        |                                      |    |     |
   barge_in.py              media_control.py                      history.py |  tools/  (registry)
   (interrupt logic)        (SMTC button + keepalive)            (persist)   |    |
                                                                     memory.py  notes.py
                                                                  (long-term)  knowledge.py
                                                                               discord_data.py
                                                                               categories.py

   config.py — constants shared by everything
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
- **`audio.py`** — `AudioEngine`: one always-on input stream pushing fixed-size
  frames onto a queue. `collect_utterance` pulls whole utterances using VAD so
  long silences cost almost nothing. `poll_speech` and `pushback` support
  barge-in (retaining the audio consumed while detecting an interruption).
- **`stt.py`** — `Transcriber`: faster-whisper wrapper (`small.en` by default,
  `vad_filter=True` to reject hallucinated text from silence).
- **`tts.py`** — `Speaker`: Windows SAPI backend (async speak + purge, which is
  what enables barge-in) with a synchronous pyttsx3 fallback.
- **`sound.py`** — `IdleSound`: loops a "thinking" WAV while the agent waits on
  the model. Idempotent, thread-safe, never raises (missing file = silence).

### Button / interruption logic (extracted, pure-ish, tested)
- **`barge_in.py`** — `BargeInDetector`: decides when the user's voice should
  interrupt playback. Calibrates an echo baseline from the first part of
  playback, then requires frames that are both voiced (VAD) and louder than that
  baseline; a *leaky* counter tolerates brief mid-word dropouts. Retains the
  consumed frames so the user's opening words aren't lost on interruption.
- **`gestures.py`** — `ClickGestureDecoder`: turns raw button presses into
  single/double/triple gestures. Dedupes presses that arrive on both listener
  channels, counts clicks within a window, and fires the gesture on a timer.
  Thread-safe.
- **`media_control.py`** — `MediaButtonListener`: a Windows System Media Transport
  Controls (SMTC) session so Bluetooth-native headset buttons (AVRCP, which never
  appear as key events) are received, plus the silent keepalive stream. See
  `MEDIA_CONTROL.md` for the hardware reasoning.

### Language model
- **`llm.py`** — `Claude`: the conversation loop (`converse`, with the tool-call
  loop), the folder-choice dialogue (`choose_folder_via_dialogue`), note
  summarisation (`summarize`), and memory consolidation. Holds a `ToolContext`
  and reads the active conversation model from it each call.
- **`history.py`** — pure functions over the message list: `sanitize` (drop any
  tool_use whose tool_result never arrived, and merge adjacent same-role turns),
  `trim` (rolling window that starts on a clean user message), `load`/`save`.
  This is what makes a conversation persisted mid-tool-loop safe to reload — see
  §6.
- **`memory.py`** — `ConversationMemory`: long-term memory. Stages messages that
  fall off the live window, then at boot consolidates the staged text into one
  dense record embedded in a Chroma `conversations` collection; `search` backs
  the `search_past_conversations` tool.

### Stores
- **`notes.py`** — `NoteStore`: note storage, retrieval, semantic search, folder
  management (create/rename/delete/move), and a `resync` repair pass.
- **`knowledge.py`** — `KnowledgeStore`: ingests reference PDFs/text into a Chroma
  `knowledge` collection (idempotent, content-hashed via `manifest.json`) and
  searches it.
- **`discord_data.py`** — `DiscordData`: read-only view over a sibling "Discord
  Notifier" project's captured messages and trade alerts.
- **`categories.py`** — the note-folder registry: seed folders, the
  voice-created/renamed overlay (persisted to `data/categories.json`),
  `category_dir`, and the add/rename/delete API. This is runtime-mutable *state*,
  deliberately separate from `config.py`.

### Configuration
- **`config.py`** — pure constants: paths, audio/VAD parameters, barge-in
  thresholds, model ids, the system prompt, and `ensure_dirs()`.

### Tools package
- **`tools/`** — the tool registry (§5). `__init__.py` holds the `@tool`
  decorator, the `ToolContext` dataclass, `api_tools()` (schemas for the API
  call), and `dispatch()`. Each `*_tools.py` module registers handlers for one
  domain.

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
6. History is saved to `data/history.json` after the turn.

If the reply's tool calls included `save_conversation_note`, the agent then runs
the folder-choice dialogue and files the note.

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
  `memory`; plus mutable session state: `pending_note`, `convo_model`).
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
- **memory** (`memory_tools.py`): `search_past_conversations`.
- **knowledge** (`knowledge_tools.py`): `search_knowledge`.
- **model** (`model_tools.py`): `set_conversation_model` — switch the
  conversation model between Haiku 4.5, Sonnet 5, and Opus 4.8 by voice.
- **project** (`project_tools.py`): `describe_project` — returns this document so
  the agent can answer questions about its own design.

---

## 6. Conversation history & the invariant that once bricked the app

The live conversation is persisted to `data/history.json` after every turn and
restored on the next boot. The Anthropic API enforces invariants a saved history
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

## 7. Memory: three layers

1. **Live window** — recent turns in `data/history.json` (`HISTORY_MAX_MESSAGES`).
2. **Long-term memory** — text that ages out of the window is staged to
   `data/memory_pending.json`, then consolidated at boot into dense summaries
   embedded in the Chroma `conversations` collection. `search_past_conversations`
   retrieves them ("what did we talk about last week?").
3. **Notes** — deliberate, saved artifacts (recorded sessions or
   conversation-derived), filed in category folders and semantically searchable.

---

## 8. Where data lives

```
data/<Folder>/       notes: <id>.md (summary + frontmatter) + <id>.transcript.md
data/pending/        transient live transcript while recording
data/chroma/         Chroma index: notes, knowledge, conversations collections
data/index.json      ordered record of every note (title, date, category)
data/categories.json voice-created/renamed folders overlaid on the seed defaults
data/history.json    live conversation window (sanitized on every save)
data/memory_pending.json  staged text awaiting consolidation
knowledge/           reference PDFs/text you ingest + manifest.json
logs/                dated session logs
```

Everything under `data/`, `logs/`, `knowledge/`, and `.env` is gitignored — the
user's content and keys never enter version control.

---

## 9. Headset button & barge-in

One physical play/pause button drives everything, listened for on two channels at
once (keyboard hook for wired/dongle headsets; SMTC for Bluetooth-native ones).
`ClickGestureDecoder` dedupes across channels and resolves gestures: **1 click =
mute, 2 = toggle notetaking, 3 = quit**. Every accepted press immediately
silences output (the "hush" path) because a state-tracking dongle swallows the
next press if playback continues through it. Voice barge-in is separate: while the
agent speaks, start talking and `BargeInDetector` stops it and captures your
words. See `MEDIA_CONTROL.md` for the full hardware story.

---

## 10. Extending the project

- **New tool / capability** — add one decorated function under `tools/` and import
  the module in `tools/__init__.py`. Nothing else to wire.
- **New note folder** — created by voice at runtime, or seed one in
  `categories.py`.
- **Swap an engine** — STT/TTS/embedding choices are isolated behind `stt.py` /
  `tts.py` / the stores; a new backend is a drop-in.
- **Tuning** — audio thresholds, models, endpointing, and barge-in sensitivity
  are all constants in `config.py`.

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
