# Voice AI Notetaking Agent

A local, voice-driven assistant with two modes:

- **Conversation mode** (default) — talk to it like you talk to Claude. Ask about
  your saved notes ("what's my latest note?", "what did I say about the budget?"),
  switch which Claude model answers ("switch to Opus"), or ask how the agent
  itself works ("how does barge-in work?").
- **Notetaking mode** — a silent recorder. It listens without speaking, handles long
  silences efficiently (an hour-long session with only a few minutes of speech does
  almost no transcription work), and on stop it saves a transcript + an AI summary to
  disk and reads the summary back to you.

Everything runs locally except Claude (the brains + summaries): transcription
(faster-whisper), speech (Windows SAPI), and semantic note search (Chroma +
sentence-transformers) are all on-device. No UI — just your voice and hotkeys —
but everything is logged to `logs/`.

For a detailed technical walkthrough — module map, data flows, the tool
registry, and extension points — see **[PROJECT.md](PROJECT.md)**. (You can also
just ask the agent: "tell me about this project.")

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

Button presses are listened for on two channels at once (see
`MEDIA_CONTROL.md`): a keyboard hook (how wired headsets and USB wireless
dongles deliver presses) and a Windows media session (SMTC — how
Bluetooth-native headsets like AirPods deliver them; those never appear as key
events). A press arriving on both channels is counted once. Multi-click
detection uses a 450 ms window — clicks within that window count together.
Headsets that decode multi-press in firmware (e.g. AirPods) send Next/Previous
instead; those map to the same double/triple actions. A silent keepalive
stream runs continuously (`MEDIA_KEEPALIVE` in `config.py`) so the headset's
audio link never spins up from silence — wireless dongles drop presses during
those first seconds — and every click briefly pauses it so state-tracking
dongles stay in sync (see `MEDIA_CONTROL.md`).

### Barge-in (interrupt the agent)

While the agent is speaking a reply, just start talking — it will stop and
listen. Say "continue", "go on", or "keep going" to resume where it left off.

### Pausing mid-sentence

If you pause long enough that the agent starts thinking, just keep talking:
the mic stays live while it waits on the model, so the rest of your sentence is
captured, the premature reply is silently discarded, and the model is re-asked
with your complete sentence.

## Conversation memory

The conversation is saved to `data/history.json` after every turn and restored
on the next start, so the agent remembers your last conversation across
restarts. The live window keeps the most recent exchanges
(`HISTORY_MAX_MESSAGES` in `config.py`).

Older conversation isn't lost when it ages out of that window: its text is
staged to `data/memory_pending.json`, and at boot the agent consolidates the
staged text — one quick model call summarises it into a dense memory record
embedded in a persistent `conversations` collection in Chroma. Ask "what did we
talk about last week?" or "didn't we discuss X before?" and the agent searches
those archived summaries (`search_past_conversations`). Consolidation only runs
when enough has accumulated, and if it fails (e.g. offline) the staged text is
kept and retried next boot.

You can also turn part of a conversation into a note without switching to
note-taking mode: ask something ("what did we talk about trading?"), then say
"save that as a note". The agent writes the note from the conversation and runs
the usual folder dialogue to ask where to file it.

## Where things are saved

Notes are sorted into category folders. Each finished note lives in its category
folder as two files: the AI summary (`<id>.md`, with title/date/category
frontmatter) and the raw transcript (`<id>.transcript.md`).

```
data/Trading/        notes filed under "Trading"  (<id>.md + <id>.transcript.md)
data/TherapyBooks/   notes filed under "Therapy book"
data/General/        everything else
data/pending/        transient: live transcript while a session is recording
data/chroma/         semantic search index (note + knowledge collections)
data/index.json      ordered record of every note (title, date, category)
knowledge/           reference PDFs/text you ingest + manifest.json (see below)
logs/                dated session logs of everything that happened
```

When a note-taking session ends, the agent suggests the best-fitting category and
talks it through with you — you can just agree, name a different folder, or ask
questions first ("what folders do I have?", "how many notes are in General?")
before deciding. It files the note only once you commit. Queries ("what's my last
note", "what did I think about X") search across **all** categories by default,
and can be scoped to one folder by naming it ("what's the latest note in my
General folder", "search my Trading notes for spreads").

The built-in categories are defined in `categories.py` under `NOTE_CATEGORIES` —
each entry has a folder name and a description of what belongs there. You can also
manage folders **by voice** in conversation mode:

- **Create** — "create a folder called Recipes".
- **Rename** — "rename Ideas to Brainstorms". Existing notes stay filed under it
  (the slug is preserved) and the folder is moved on disk.
- **Delete** — "delete the Recipes folder". Notes are never lost: they move to
  General by default, or to a folder you name ("delete Recipes and move its notes
  to Ideas"). The General folder can't be deleted.
- **Move a note** — "move my last note to Ideas", "put the grocery note in
  Recipes". The agent looks the note up, then moves its files.

Voice-created and renamed folders are persisted to `data/categories.json` and
overlaid on the defaults at startup.

## Trading knowledge base (PDF)

You can give the agent reference material to draw on — e.g. a trading book — so it
can answer questions from it without you pasting anything into the conversation.

1. Drop one or more `.pdf`, `.txt`, or `.md` files into the `knowledge/` folder
   (at the project root, next to `run.bat`).
2. Run `run.bat --ingest` (or just start the agent — it auto-scans on boot).

Each file is chunked and embedded **once** into a persistent `knowledge` collection
in `data/chroma`. Ingestion is idempotent: files are identified by content hash and
recorded in `knowledge/manifest.json`, so re-scanning an unchanged folder is
near-instant and never re-embeds. On boot the agent runs the same scan
automatically — nothing new means no delay; a genuinely new book is embedded once
before the agent starts listening.

After that, ask trading questions in conversation ("what does my trading book say
about iron condors?"). The agent uses the `search_knowledge` tool on demand and
cites the source (and page, for PDFs). `run.bat --kb-list` shows what's been
ingested. The
content stays local and, like the rest of `data/`, is gitignored.

## Switching the Claude model by voice

Conversation defaults to **Haiku 4.5** for low latency. Ask for a different model
mid-conversation and it switches from that reply onward:

- "switch to Opus" / "use the smartest model" → **Opus 4.8** (most capable, slowest)
- "use Sonnet" → **Sonnet 5** (stronger reasoning, a little slower)
- "go back to the fast one" → **Haiku 4.5**

The choice lasts for the session and resets to the fast default on restart (so you
never get silently left on an expensive model). Note summaries always use
`SUMMARY_MODEL` regardless. The models are defined in `config.py` under
`CONVO_MODELS`.

## Asking the agent about itself

The agent can answer questions about its own design — "how does barge-in work?",
"where are my notes stored?", "what tools do you have?", "how do I switch models?".
It reads [PROJECT.md](PROJECT.md) (via the `describe_project` tool) and answers
from it, so its self-knowledge stays in sync with the documentation.

## Project layout

```
voice_agent.py   entry point + Agent orchestration (main loop, modes, say/barge-in)
audio.py stt.py tts.py sound.py     mic/VAD, transcription, speech, thinking cue
barge_in.py gestures.py media_control.py   interrupt logic, button decode, SMTC
llm.py history.py memory.py         Claude loop, history repair, long-term memory
notes.py knowledge.py discord_data.py categories.py   stores + folder registry
config.py        shared constants (paths, audio params, models, system prompt)
tools/           tool registry — one file per domain (notes, discord, model, ...)
tests/           unittest suite over the pure logic (no hardware needed)
scripts/         manual hardware probes used while developing button handling
```

Adding a capability is one decorated function under `tools/` — see
[PROJECT.md](PROJECT.md) §5. Run the tests with:

```sh
python -m unittest discover tests
```

## Tuning

All settings live in `config.py`:

- `WHISPER_MODEL` — `base.en` (faster) ↔ `small.en` (default) ↔ `medium.en` (more accurate).
- `CONVO_MODELS` / `CONVO_MODEL` / `SUMMARY_MODEL` — Claude models. Conversation
  defaults to Haiku for low latency and can be switched by voice (see above);
  summaries use Sonnet for quality.
- `CONVO_ENDPOINT_MS` / `NOTE_ENDPOINT_MS` — how much trailing silence ends an utterance.
- `VAD_AGGRESSIVENESS` — 0–3; raise it if background noise is being picked up as speech.
- `TTS_RATE` / `TTS_VOICE` — speech speed and voice selection.
