# Hands-Free Voice Notes Agent

A regular foreground, voice-first personal assistant for Windows. It does two jobs:

1. **Note-taking** — On command it records you for a long, mostly-silent session (hours of
   wall-clock time, but typically only minutes of real speech). Silence is never processed.
   On stop it produces a transcript and a summary, both saved locally.
2. **Conversational assistant** — A smooth, interruptible voice Q&A mode that can answer
   questions about your notes and serve general voice-assistant purposes.

The whole thing is **one agent** whose LLM (Claude) calls note-taking and retrieval as
**tools**. A master **mute** control lets you stop it listening entirely.

See [`voice-notes-agent-build-plan.md`](voice-notes-agent-build-plan.md) for the full design
spec. This implementation follows that plan section-by-section.

---

## Architecture at a glance

One Python process hosts two mutually-exclusive subsystems plus a shared local store:

```
 Mic ──▶ Audio Router + Silero VAD ──▶ State Machine (MUTED / LISTENING / CAPTURING)
                                          │                     │
                       CAPTURE SUBSYS (local)          CONVERSATION SUBSYS (Pipecat)
                       VAD-gated recorder              cloud STT → Claude(tools) → cloud TTS
                       → local Whisper                 barge-in, turn-taking
                          │                                  │ tool calls
                          ▼                          start/stop/summarize/search
                  local disk (audio/text)  ◀─────────────────┘
                          │
                  ChromaDB vector index (RAG)
```

**Key rule:** the conversation and capture subsystems are *never active at once*. Entering
capture mode suspends the cloud loop; note audio is handled entirely locally and never leaves
the device.

## Module map

| Module | Responsibility | Spec refs |
|--------|----------------|-----------|
| `config.py` | Load + validate `config.yaml` into typed dataclasses | §14 |
| `state.py` | Three-state machine + guards + transition events | §4 |
| `paths.py` | project-local `.voice-notes-agent` storage layout | §9 |
| `audio/vad.py` | Silero VAD wrapper + segmenter (pre-roll, hangover, min-len) | §5.2 |
| `audio/router.py` | Shared mic input stream; hard mute releases the device | §5.7 |
| `capture/recorder.py` | VAD-gated recorder, incremental crash-safe writes | §5.2 |
| `capture/transcriber.py` | Background faster-whisper worker, anti-hallucination | §5.3 |
| `summarize/summarizer.py` | Transcript → Claude → spoken + full summary | §5.4 |
| `storage/store.py` | Dated per-session files | §5.5, §9 |
| `storage/rag.py` | ChromaDB index + `search_notes` | §5.5 |
| `agent/tools.py` | LLM function specs + implementations | §8 |
| `agent/conversation.py` | Pipecat real-time pipeline, barge-in | §5.6 |
| `feedback/earcons.py` | Earcon vocabulary (tone synthesis) | §10 |
| `feedback/voice.py` | Earcon + spoken-confirmation manager | §5.9, §10 |
| `controls/hotkeys.py` | Global hotkeys + headset media button | §5.8, §11 |
| `app.py` | Wires everything together; owns the state machine | §3 |

## Setup

Target platform is Windows (single user). A wired headset is strongly recommended — see
risk R-1 in the build plan (Bluetooth drops to narrowband when the mic opens).

```bash
run.bat                           # creates .venv and .voice-notes-agent\config.yaml
```

Set provider API keys as environment variables (never commit them):

```
ANTHROPIC_API_KEY   # Claude — summary + conversation LLM (required)
DEEPGRAM_API_KEY    # cloud streaming STT + TTS for Q&A (required for conversation)
```

First run downloads the Silero VAD and faster-whisper models (a few hundred MB) to the
local cache. Note transcription is CPU-only by design (§C2).

## Run

```bash
run.bat                              # starts MUTED (privacy by default, §C7)
```

**Controls.** The agent starts muted. Three control surfaces drive it, in order of
reliability:

1. **Terminal keys** (always works) — with the `run.bat` window focused:
   - `m` — toggle **Listening / Muted**
   - `n` — start / stop note capture
   - `s` — speak current status
   - `q` — quit
2. **Global hotkeys** (work app-wide when the `keyboard` hook is available): `Ctrl+Alt+M`
   (mute toggle), `Ctrl+Alt+N` (notes), `Ctrl+Alt+Space` (push-to-talk).
3. **Headset media button** — only works if the headset actually emits media keys
   (play/pause/next). Many **wired** headsets, including Apple EarPods, do **not** send
   media keys on Windows, so their inline button does nothing here — use the terminal
   keys or global hotkeys instead. Bluetooth headsets usually do emit media keys.

Then drive it by ear:

- Press `m` (or the mute hotkey) to un-mute → **"Listening"**
- Say *"take notes"* → **"Recording notes"** (capture begins, cloud loop suspended)
- Talk for as long as you like; only speech is recorded and transcribed
- Say *"stop notes"* (or press `n`) → **"Stopped. Summarizing."** → spoken summary
- Ask *"what did I note about the budget?"* → answered via RAG over your notes

Audio in/out uses your **Windows default** recording and playback devices. Set them in
Windows Sound settings before launch, or pin explicit PyAudio indices via
`conversation.input_device_index` / `output_device_index` in `config.yaml` (run
`python -m voice_notes_agent --list-pyaudio-devices` to see the indices).

Every state change and result is conveyed by audio (earcons + speech); the screen is never
required (§C5).

## Testing

```bash
pytest                  # unit tests for VAD segmenter, state machine, storage
```

The capture pipeline (`tests/test_segmenter.py`) is tested with synthetic audio so it runs
without a microphone or models.

## Status / phase coverage

This codebase implements the phased build plan in §12:

- **Phase 0–2** (capture core, transcription, summary, RAG): implemented and unit-tested.
- **Phase 3–4** (Pipecat conversation + tool wiring): implemented; requires Pipecat + cloud
  keys to run live.
- **Phase 5–7** (mute/controls, voice feedback, resilience): implemented; hotkeys
  and hard-mute device release are Windows-targeted.
