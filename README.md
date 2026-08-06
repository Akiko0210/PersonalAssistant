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

Only one agent can run at a time: a second `python voice_agent.py` detects the
first (via a lock on `data/agent.lock`) and exits immediately with a spoken
notice, so two instances can't talk over each other or corrupt your history and
note index. The lock is released automatically when the agent stops — even on a
crash — so there's nothing to clean up.

## Controls

All controls work globally (even when another window is focused).

### Headset button (play/pause)

| Action                       | Gesture               |
| ---------------------------- | --------------------- |
| Toggle mute (stop listening) | Single click          |
| Toggle notetaking            | Double click          |
| Quit                         | Triple click          |

**Mute stops the microphone, not the agent.** Click it while a reply is
playing and the reply finishes — you just stop being heard. The confirmation
("Muted." / "Listening.") is spoken *over* the reply on a second, quieter
voice, so it lands the moment you press the button instead of waiting for the
reply to end; on a machine with more than one voice installed it deliberately
uses a different one, so you can tell the two apart. There's a brief hitch
where the click landed: playback pauses the instant the button goes down,
because the wireless dongle only passes on the *next* click of a double/triple
if the host really stops (see `MEDIA_CONTROL.md`), and the reply picks up again
as soon as the click turns out to be a single.

Muting while the agent is still *thinking* doesn't cancel anything either —
the question you already asked is answered and spoken as usual; you just
aren't being listened to while it happens. Double-click (notetaking) and
triple-click (quit) do end a reply immediately, and to cut one short without
changing anything, just start talking (barge-in, below).

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

You don't have to say everything in one breath. When your utterance ends, the
agent waits a short **settle window** (`CONTINUATION_SETTLE_MS` in `config.py`,
default 600 ms) — listening, not yet answering. If you keep talking within it,
the continuation is captured and merged and the window restarts; only once
you've truly finished (the window passes in silence) does it call the model,
**once**, with your complete question. So a multi-part question with pauses
costs exactly one reply, not one per pause. The trade-off is a little latency
before each reply (`CONVO_ENDPOINT_MS + CONTINUATION_SETTLE_MS` of silence):
lower the settle window to answer sooner, raise it if your natural pauses get
cut off mid-thought.

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

If you're talking to a persona that doesn't own notes (say, Tom mid-trading
discussion), the save is **delegated**: Tom hands the task to Bob, who works in
the background while your conversation continues uninterrupted. When the note
is ready, Bob speaks up in his own voice at the next pause — "Bob here — your
note is ready to file" — and asks which folder. You never leave the
conversation you were in.

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
knowledge/           reference PDFs/text/video you ingest + manifest.json (see below)
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

## Trading knowledge base (PDFs and course videos)

You can give the agent reference material to draw on — a trading book, or a
recorded course — so it can answer questions from it without you pasting anything
into the conversation.

Two ways in — the dashboard, or the folder plus a command:

**From the dashboard** (easiest): open the Knowledge page, which walks through
three numbered steps in a single narrow column:

1. **Add files** — press **Upload files** (or drag them onto the box below it).
2. **Waiting to add** — everything uploaded but not yet in the knowledge base.
   Each file has a **Remove** button for undoing a wrong upload; it asks once
   before deleting, since there's no undo. Press **Ingest** to process the list.
3. **In the knowledge base** — what the agent can already search.

A file that's already been ingested has no Remove button, and the server refuses
to delete it even if asked directly: its chunks live in Chroma, so deleting the
file alone would leave the agent citing a source that no longer exists. Ask the
agent to *forget* it instead.

**From the folder:**
1. Drop files into the `knowledge/` folder (at the project root, next to
   `run.bat`). Documents: `.pdf`, `.txt`, `.md`. Video/audio: `.mp4`, `.mkv`,
   `.mov`, `.webm`, `.m4a`, `.mp3`, `.wav`.
2. Just launch with `run.bat`. A plain launch ingests everything new — video
   included — before starting the agent, printing progress as it goes. Launches
   with nothing new cost a couple of seconds.

`run.bat --ingest` still does the ingest alone. Note that the pre-launch pass is
skipped whenever you pass arguments, so `run.bat --selftest` and friends start
immediately.

Each file is chunked and embedded **once** into a persistent `knowledge` collection
in `data/chroma`. Ingestion is idempotent: files are identified by content hash and
recorded in `knowledge/manifest.json`, so re-scanning an unchanged folder is
near-instant and never re-embeds.

**Video and audio are transcribed on the way in** by the same local Whisper model
the agent uses for dictation (decoding is handled by PyAV, bundled with
faster-whisper — no ffmpeg install needed). Each chunk keeps the timestamp it was
spoken at, so a citation points you at the moment to rewatch.

Transcription is slow — on this machine `small.en` runs about 3.3× faster than
real time, so budget **roughly 20 minutes per hour of recording** (more with a
bigger `KB_MEDIA_MODEL`). You pay it exactly once per file, and progress is
printed every few seconds so a long run never looks like a hang.

That is far too long to hold the agent's own startup open, so **the agent process
never transcribes**: its boot scan takes documents only and just names any video
still waiting. `run.bat` does the media pass first, as a separate `--ingest`
process, then launches the agent. Same end result — everything searchable by the
time the agent is listening — but the wait is visible and attributable rather
than a half-started agent that looks wedged.

**An ingest and a running agent are mutually exclusive.** Embedding writes the
same Chroma index the agent has open, and two writers corrupt it — so the
dashboard's ingest takes the agent's single-instance lock. If the agent is
running, close it, ingest, then start it again. (The same lock is why `--ingest`
refuses to run beside a live agent.) While an ingest is in flight, leave the
dashboard running; closing it cancels the job. Cancelling is safe — the manifest
is written per file and chunk ids derive from the file hash, so a re-run
overwrites rather than duplicates.

Two knobs in `config.py`: `KB_MEDIA_MODEL` (default `small.en`; `medium.en` is
noticeably better on jargon and, as a one-time cost, usually worth it) and
`KB_MEDIA_EXTS`. `KB_MEDIA_MODEL` is also settable from the dashboard.

After that, ask trading questions in conversation ("what does my course say about
iron condors?"). The agent uses the `search_knowledge` tool on demand and cites the
source — a page for books, a timestamp like `14:32` for video. `run.bat --kb-list`
shows what's been ingested, with page count or running time. The content stays
local and, like the rest of `data/`, is gitignored.

One limitation worth knowing: only the spoken audio is captured. Whatever is drawn
on a chart or slide is lost, so an instructor saying "as you can see here" ingests
as exactly that.

## Switching the model by voice

Conversation defaults to **Haiku 4.5** for low latency. Ask for a different model
mid-conversation and it switches from that reply onward:

- "switch to Opus" / "use the smartest model" → **Opus 5** (most capable, slowest)
- "use Sonnet" → **Sonnet 5** (stronger reasoning, a little slower)
- "go back to the fast one" → **Haiku 4.5**
- "use the cheap one" / "switch to DeepSeek" → **DeepSeek V4 Flash** (external, by far the cheapest)
- "use DeepSeek pro" → **DeepSeek V4 Pro** (DeepSeek's strongest)

The DeepSeek options talk to DeepSeek's Anthropic-compatible API through the same
client code; they require `DEEPSEEK_API_KEY` in `.env` (see `.env.example`) and
are refused out loud when it's missing. Your conversation history goes to
DeepSeek's servers while one is active.

The choice lasts for the session and resets to the fast default on restart (so you
never get silently left on an expensive model). Note summaries always use
`SUMMARY_MODEL` regardless. The models are defined in `config.py` under
`CONVO_MODELS`.

**Each persona remembers its own model.** Leave Tom on DeepSeek V4 Pro, switch to
Bob, and Bob answers on his own model; switch back and Tom is still on DeepSeek.
The switch announcement says which, since the model is the one thing you can't
hear:

> "Bob here, running on Haiku 4.5."
> "Tom here, running on DeepSeek V4 Pro."

**Asking "what model are you on?" reads the real setting.** The personas share
one conversation history, so it fills up with model talk that no longer applies
— a switch Tom made, a choice from an hour ago. Left to answer from that, the
agent guesses, and it has guessed wrong (claiming Opus while on Haiku, and
DeepSeek while on Opus). Every persona now has a `get_current_model` tool and a
hard rule to call it before saying anything about models, so the answer is a
live read rather than a recollection — and it shows up as a `tool_use` line in
`logs/` if you want to check.

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
