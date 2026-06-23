# Voice AI Notetaking Agent

A local, voice-driven assistant with two modes:

- **Conversation mode** (default) — talk to it like you talk to Claude. Ask about
  your saved notes ("what's my latest note?", "what did I say about the budget?").
- **Notetaking mode** — a silent recorder. It listens without speaking, handles long
  silences efficiently (an hour-long session with only a few minutes of speech does
  almost no transcription work), and on stop it saves a transcript + an AI summary to
  disk and reads the summary back to you.

Everything runs locally except Claude (the brains + summaries): transcription
(faster-whisper), speech (Windows SAPI), and semantic note search (Chroma +
sentence-transformers) are all on-device. No UI — just your voice and hotkeys —
but everything is logged to `logs/`.

## Setup

1. Install Python dependencies:

   ```sh
   pip install -r requirements.txt
   ```

   > On Windows this uses `webrtcvad-wheels` (a precompiled build of `webrtcvad`)
   > so you don't need Visual C++ Build Tools. It imports as `webrtcvad` either way.

2. Provide your Claude API key. Easiest: copy `.env.example` to `.env` and put your
   key in it — it's loaded automatically on startup and is gitignored so it never
   gets committed:

   ```sh
   copy .env.example .env       # then edit .env and set ANTHROPIC_API_KEY
   ```

   Alternatively, set it as an environment variable instead of using `.env`:

   ```powershell
   $env:ANTHROPIC_API_KEY = "sk-ant-..."   # current PowerShell session only
   ```

3. First run downloads model weights once: faster-whisper `small.en` (~0.5 GB) and
   the embedding model `all-MiniLM-L6-v2` (~90 MB).

## Run

```sh
python voice_agent.py            # start the agent (conversation mode)
python voice_agent.py --selftest # check mic, STT, TTS, Claude, and note search
```

## Controls

All controls work globally (even when another window is focused).

### Headset button (play/pause)

| Action                       | Gesture               |
| ---------------------------- | --------------------- |
| Toggle mute (stop listening) | Single click          |
| Toggle notetaking            | Double click          |
| Quit                         | Triple click          |

The headset button listens for `media_play_pause` events. Multi-click
detection uses a 450 ms window — clicks within that window count together.

### Barge-in (interrupt the agent)

While the agent is speaking a reply, just start talking — it will stop and
listen. Say "continue", "go on", or "keep going" to resume where it left off.

## Where things are saved

```
data/transcripts/  raw transcripts (written live during a session)
data/summaries/    AI summaries with title/date frontmatter
data/chroma/       semantic search index
logs/              dated session logs of everything that happened
```

## Tuning

All settings live in `config.py`:

- `WHISPER_MODEL` — `base.en` (faster) ↔ `small.en` (default) ↔ `medium.en` (more accurate).
- `CONVO_MODEL` / `SUMMARY_MODEL` — Claude models (conversation defaults to Haiku for
  low latency; summaries use Sonnet for quality).
- `CONVO_ENDPOINT_MS` / `NOTE_ENDPOINT_MS` — how much trailing silence ends an utterance.
- `VAD_AGGRESSIVENESS` — 0–3; raise it if background noise is being picked up as speech.
- `TTS_RATE` / `TTS_VOICE` — speech speed and voice selection.
