"""Central configuration for the voice notetaking agent.

Everything tunable lives here so behaviour can be changed without touching the
logic modules.
"""

import json
from pathlib import Path

# --- Paths -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"
LOG_DIR = BASE_DIR / "logs"
INDEX_PATH = DATA_DIR / "index.json"
# Single-instance lock: a second launch takes a lock on this file (Windows
# msvcrt; see single_instance.py) and exits if it's already held, so two agents
# can't talk over each other or race on history.json and the Chroma index
# (which would corrupt them).
LOCK_PATH = DATA_DIR / "agent.lock"
# Detailed project description — the knowledge source the describe_project tool
# reads so the agent can answer questions about its own design.
PROJECT_DOC_PATH = BASE_DIR / "PROJECT.md"
# User-created / renamed folders are persisted here and overlaid on the built-in
# NOTE_CATEGORIES defaults below at startup (see load_categories).
CATEGORIES_PATH = DATA_DIR / "categories.json"
# Dashboard edits to the agent personas (agents.py registry) are persisted here
# and overlaid on the built-in defaults at startup (see agents.load_agents) —
# the same pattern as categories.json above.
AGENTS_PATH = DATA_DIR / "agents.json"
# Which persona is currently active, written by llm.Claude.switch_to on every
# switch so the dashboard can show who the user is talking to. Read-only
# telemetry — nothing in the agent reads it back.
AGENT_STATE_PATH = DATA_DIR / "agent_state.json"
# The agent's control endpoint (controller.py): a localhost-only HTTP port the
# dashboard proxies live controls to (mute, send a typed message). One below
# the dashboard's own default of 8765. Soft dependency — if the port can't be
# bound the agent logs a warning and runs without dashboard control.
CONTROL_PORT = 8766
# Live transcripts are appended here while recording, then moved into the chosen
# category folder when the note is saved (the category isn't known until the end).
PENDING_DIR = DATA_DIR / "pending"
# Reference material (trading books/PDFs/text) the user drops in to build a
# searchable knowledge base. Kept at the project root (not under data/) so it's
# easy to find and manage. Ingested once into a persistent Chroma collection —
# which lives in data/chroma — and queried on demand via the search_knowledge
# tool, never pasted into the conversation.
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
KNOWLEDGE_MANIFEST = KNOWLEDGE_DIR / "manifest.json"  # {sha256: {source,title,...}}
# Conversation memory: the chat history is saved here after every turn and
# restored (trimmed) on the next boot, so the agent remembers the last
# conversation across restarts.
HISTORY_PATH = DATA_DIR / "history.json"
HISTORY_MAX_MESSAGES = 40   # messages kept when persisting/restoring history
# Long-term memory: messages that fall off the window above are not lost — their
# text is staged here, then consolidated (summarised by the model and embedded
# into a persistent Chroma collection) so older conversations stay searchable
# via the search_past_conversations tool.
MEMORY_PENDING_PATH = DATA_DIR / "memory_pending.json"
# Legacy flat locations — only referenced by the one-time migration in notes.py.
SUMMARY_DIR = DATA_DIR / "summaries"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"

# --- Note categories -----------------------------------------------------------
# The category registry (seed folders, voice-created overlays, category_dir and
# the add/rename/delete API) lives in categories.py — it is runtime-mutable
# state, not configuration. Only its persistence path stays here (defined above
# as CATEGORIES_PATH); the data/categories.json format is unchanged.

# Discord Notifier (sibling project) — read-only access to its captured data.
DISCORD_DIR = BASE_DIR.parent / "Discord Notifier"
DISCORD_LOG_PATH = DISCORD_DIR / "discord_log.md"
DISCORD_TRADES_PATH = DISCORD_DIR / "trades.txt"

# --- Audio capture -----------------------------------------------------------
SAMPLE_RATE = 16000           # Hz; webrtcvad supports 8/16/32/48 kHz
FRAME_MS = 30                 # 10/20/30 ms are the only valid VAD frame sizes
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000   # 480 samples per frame
VAD_AGGRESSIVENESS = 3        # 0 (lenient) .. 3 (strictest about calling noise non-speech)

# Endpointing: how much trailing silence marks the end of an utterance.
CONVO_ENDPOINT_MS = 800       # snappier turn-taking in conversation
NOTE_ENDPOINT_MS = 1000       # a beat longer while notetaking
SPEECH_PAD_MS = 300           # pre-roll kept before detected speech
TRIGGER_RATIO = 0.6           # fraction of padded window that must be voiced
MAX_UTTERANCE_S = 30          # safety cap on a single captured utterance

# --- Speech-to-text (local, faster-whisper) ----------------------------------
WHISPER_MODEL = "small.en"    # base.en is faster, medium.en more accurate
WHISPER_DEVICE = "cpu"        # set to "cuda" if you have a supported GPU
WHISPER_COMPUTE = "int8"      # int8 is a good CPU default; float16 for GPU

# --- Embeddings / semantic search --------------------------------------------
EMBED_MODEL = "all-MiniLM-L6-v2"
SEARCH_RESULTS = 5

# --- Knowledge base (ingested PDFs, text, and video/audio) --------------------
KNOWLEDGE_COLLECTION = "knowledge"   # Chroma collection, separate from "notes"
KB_CHUNK_CHARS = 1000                # target characters per embedded chunk
KB_CHUNK_OVERLAP = 150               # characters shared between adjacent chunks
KB_SEARCH_RESULTS = 5                # chunks returned per search_knowledge call
# Recorded lecture material, transcribed by Whisper on the way in. Decoding is
# handled by PyAV (bundled with faster-whisper), so no ffmpeg install is needed.
KB_MEDIA_EXTS = (".mp4", ".m4a", ".mp3", ".mkv", ".mov", ".wav", ".webm")
# Transcription is a one-time cost per file, so it can afford a bigger model than
# live dictation: medium.en is noticeably better on jargon and worth it here.
KB_MEDIA_MODEL = "small.en"

# --- Long-term conversation memory --------------------------------------------
MEMORY_COLLECTION = "conversations"  # Chroma collection of archived summaries
MEMORY_MIN_MESSAGES = 6              # consolidate only once this many lines staged
MEMORY_MAX_TOKENS = 700              # budget for one consolidation summary
MEMORY_SEARCH_RESULTS = 3            # summaries returned per search

# --- Text-to-speech (local, Windows SAPI via pyttsx3) ------------------------
TTS_RATE = 175                # words per minute
TTS_VOICE = None              # None = system default; or a SAPI voice id substring

# Spoken notices ("Muted.", "Listening.") are said on a SECOND voice so they can
# be heard *over* a reply that is still playing — one SAPI voice queues its
# utterances, so an overlapping notice needs its own. Kept quieter and, where
# the machine has more than one voice installed, deliberately a different one
# from the reply's, so two simultaneous voices stay tellable apart.
ANNOUNCE_VOLUME = 80          # 0-100, against the reply's 100
ANNOUNCE_RATE = 185           # words per minute; brisk — these are two words

# --- "Thinking" audio cue ----------------------------------------------------
# Looped whenever the agent is busy with the model and there's nothing to hear —
# answering you (including tool calls), summarising a note, or deciding its
# folder — as a "still working" cue. WAV only (played via the built-in winsound,
# which loops and stops cleanly). Reuses the summarising clip for now; point it
# at a dedicated idle file, or set to None to disable. Missing file = silent, no
# error.
IDLE_SOUND = BASE_DIR / "assets" / "summarizing.wav"

# --- Headset button -------------------------------------------------------------
# The button is listened to on two channels at once: the keyboard hook (how wired
# headsets and USB wireless dongles deliver presses, as media-key events) and an
# SMTC media session (how Bluetooth-native headsets like AirPods deliver them —
# their presses never appear as key events). A press that shows up on both
# channels within MEDIA_CLICK_DEDUPE_S counts once.
MEDIA_CLICK_DEDUPE_S = 0.15
# MEDIA_KEEPALIVE loops a silent audio stream continuously. It serves two
# purposes: (1) it makes our media session the active one, so Bluetooth AVRCP
# buttons route to us; (2) it means the headset's audio stream NEVER starts
# from silence — the Yealink dongle drops button presses during the first
# seconds after a stream spins up, so a reply beginning to play used to open
# an uninterruptible window. Every accepted click briefly pauses the keepalive
# (see duck() in media_control.py) so the dongle sees its "pause" honoured and
# never desyncs. Costs some headset battery (the radio link stays active).
MEDIA_KEEPALIVE = True
# A click landing mid-reply pauses playback immediately (the dongle only
# transmits the gesture's next click if the host really stops), and the reply
# then waits here for the multi-click window to say what the gesture was:
# mute resumes it, note-taking/quit end it. Long enough to cover a triple
# click (two ~250 ms gaps plus the 450 ms window) with room to spare; if no
# verdict arrives by then the reply resumes rather than hang mid-word.
GESTURE_VERDICT_TIMEOUT_S = 2.0

# --- Settle before answering ---------------------------------------------------
# After an utterance endpoints, the agent waits this long — listening, not yet
# calling the model — in case the user was only pausing mid-thought. If they
# resume within the window, the continuation is captured and merged, and the
# window restarts; only once it elapses in silence is the model called, ONCE,
# with the complete utterance. This is what prevents a wasted (billed) model
# call per mid-thought pause. It adds this much latency to the start of each
# reply, so it trades a little responsiveness for cost: lower it to answer
# sooner, raise it if your natural pauses are being cut off mid-thought. The
# effective silence before a reply is CONVO_ENDPOINT_MS + this.
CONTINUATION_SETTLE_MS = 600
# If speech is just starting as the settle window expires — some voiced frames
# consumed but not yet enough to trigger — the window is extended once by this
# much, so an onset straddling the boundary becomes a real continuation instead
# of having its opening frames silently dropped.
CONTINUATION_GRACE_MS = 400
# Hard cap on continuation rounds per turn. Continuous background speech (a TV,
# a second person) can otherwise re-trigger the settle window forever, holding
# the turn hostage and merging the noise into the user's question.
MAX_CONTINUATION_ROUNDS = 6

# --- Barge-in (interrupt the agent by speaking while it talks) ---------------
# Works best with headphones. On open speakers the mic can hear the agent's own
# voice and self-interrupt; raise BARGE_IN_MS or set BARGE_IN = False if so.
BARGE_IN = True               # stop speaking when the user starts talking
BARGE_IN_MS = 250             # qualifying audio (ms) that must accumulate to count as an interruption
# Qualifying frames add to the counter; non-qualifying frames *decay* it by this
# fraction of a frame rather than resetting it to zero. A leaky counter tolerates
# the brief VAD/energy dropouts that happen mid-word, so real speech reliably
# accumulates instead of being wiped by a single quiet frame. 0 = hard reset
# (old behaviour); 1.0 = symmetric decay. Lower it if the agent over-triggers.
BARGE_IN_DECAY = 0.4
# Loudness gate: a frame only counts toward an interruption if it is BOTH speech
# (per VAD) AND louder than the agent's own echo. The threshold is the larger of
# an absolute floor and a multiple of the measured echo level, so it adapts to
# your speaker volume. Lower these if real speech isn't interrupting; raise them
# if the agent still interrupts itself. Check the logs: each reply logs the
# loudest speech RMS it saw vs the threshold, so you can tune from real numbers.
BARGE_IN_ENERGY = 200        # absolute RMS floor (int16) the user's voice must exceed
BARGE_IN_ENERGY_RATIO = 3.0   # ...and must exceed this multiple of the echo baseline
BARGE_IN_CALIB_MS = 350       # initial window used to measure the echo baseline

# Backchannel tolerance: listeners naturally say "yeah" / "uh-huh" / "okay"
# WHILE someone talks, meaning "I'm listening", not "stop". The detector can't
# tell a filler from a real interruption (it only sees voiced energy), so the
# distinction is made AFTER the barge-in fires: the captured speech is
# transcribed, and when it is nothing but words from this set, the agent
# resumes the reply where it left off instead of abandoning it. Real commands
# ("stop", "wait", "no") are deliberately absent. Whisper's spellings of the
# common grunts (mm-hmm, uh-huh, aha) are tokenized on non-letters, so the set
# holds the individual tokens.
BACKCHANNEL_WORDS = frozenset({
    "yeah", "yes", "yep", "yup", "ok", "okay", "sure", "right", "alright",
    "uh", "huh", "mm", "mhm", "hmm", "hm", "aha", "ah", "oh", "wow",
    "cool", "nice", "true", "gotcha", "got", "it", "i", "see", "interesting",
})
BACKCHANNEL_MAX_WORDS = 3     # "oh okay yeah" is a filler; a longer utterance is a turn

# --- Claude ------------------------------------------------------------------
# Conversation models the user can switch between by voice (see the
# set_conversation_model tool). Spoken name -> API id; the active choice lives
# on the running agent (ToolContext.convo_model), so it resets to the default
# on restart rather than silently leaving you on an expensive model.
CONVO_MODELS = {
    "haiku": "claude-haiku-4-5",   # fastest, lowest latency — the default
    "sonnet": "claude-sonnet-5",   # stronger reasoning, a bit slower
    "opus": "claude-opus-5",       # most capable, slowest and priciest
    # DeepSeek, served through their Anthropic-compatible endpoint (see
    # DEEPSEEK_BASE_URL below) — same Messages API, same tool_use/tool_result
    # blocks, so the whole tool loop and history machinery work unchanged.
    # Cheapest by far; needs DEEPSEEK_API_KEY in .env or the switch is refused.
    "deepseek": "deepseek-v4-flash",      # cheap + fast external option
    "deepseek pro": "deepseek-v4-pro",    # DeepSeek's strongest
}
CONVO_MODEL_LABELS = {
    "claude-haiku-4-5": "Haiku 4.5",
    "claude-sonnet-5": "Sonnet 5",
    "claude-opus-5": "Opus 5",
    "deepseek-v4-flash": "DeepSeek V4 Flash",
    "deepseek-v4-pro": "DeepSeek V4 Pro",
}
CONVO_MODEL = CONVO_MODELS["haiku"]   # low latency for back-and-forth (default)
SUMMARY_MODEL = "claude-sonnet-5"     # higher quality for note summaries

# DeepSeek's Anthropic-compatible endpoint: speaks the Messages API (client-side
# tool use fully supported) so it works through the same anthropic SDK client,
# just with this base_url and its own key. Known differences, all harmless here:
# cache_control is ignored (DeepSeek caches automatically server-side, with its
# own cache-hit pricing), thinking budget_tokens is ignored, and images are
# unsupported (this app never sends any).
DEEPSEEK_BASE_URL = "https://api.deepseek.com/anthropic"


def model_provider(model_id: str) -> str:
    """Which API serves a model id: 'anthropic' or 'deepseek'. The single
    routing rule for client selection — keep any new provider logic here."""
    return "deepseek" if (model_id or "").startswith("deepseek") else "anthropic"


# Spoken names for the providers, for the get_current_model readout ("served
# by Anthropic") — the raw keys above are routing identifiers, not speech.
PROVIDER_LABELS = {"anthropic": "Anthropic", "deepseek": "DeepSeek"}


def provider_label(model_id: str) -> str:
    """Friendly spoken name of the API serving a model id."""
    return PROVIDER_LABELS.get(model_provider(model_id), "an unknown provider")

# --- thinking ----------------------------------------------------------------
# Effort level for models that support it. Conversation runs low: turns are short
# spoken answers plus tool calls, not deep reasoning, and effort is the main lever
# on both latency and token spend. Summaries run a step higher — a note is
# written once and kept, so quality matters more there than the seconds or cents.
CONVO_EFFORT = "low"
SUMMARY_EFFORT = "medium"
# Models that accept adaptive thinking and the effort parameter. Haiku 4.5
# accepts neither — it has no adaptive mode, and sending `effort` is rejected
# outright — so it is deliberately absent and falls through to thinking-off,
# which suits the low-latency default anyway. DeepSeek models are absent for
# the same reason: adaptive thinking and output_config are Anthropic-specific,
# and thinking-off keeps spoken turns snappy there too.
#
# Everything else must NOT run with thinking disabled: on Opus 5 a
# thinking-disabled turn occasionally writes a tool call into its visible text
# instead of emitting a real tool_use block. The call never runs, no error is
# raised, and the turn looks successful — the same silent-failure shape as the
# truncated-tool-call bug behind CONVO_MAX_TOKENS above, and fatal to an agent
# whose ground rule is "if you did not call a tool, nothing happened".
ADAPTIVE_THINKING_MODELS = frozenset({
    "claude-sonnet-5", "claude-opus-5", "claude-fable-5",
    "claude-opus-4-8", "claude-opus-4-7", "claude-sonnet-4-6",
})
# Reply budget must cover TOOL CALLS too, not just spoken sentences: a
# save_conversation_note carrying a full written-up document is one big JSON
# blob inside the reply. At 1024 a long note was truncated mid-JSON
# (stop_reason max_tokens), never dispatched, and the turn died silently —
# the model kept announcing "saving now" while unable to ever finish
# (session_2026-07-19.log, 19:53-20:05). Tokens are billed as used, so a
# roomy cap costs nothing on ordinary short replies.
CONVO_MAX_TOKENS = 4096
SUMMARY_MAX_TOKENS = 2000
# Safety cap on model->tool->model rounds within one turn. Real turns use a
# handful; a model stuck re-calling tools without ever answering would
# otherwise bill unbounded API calls with no way to stop it by voice.
CONVO_MAX_TOOL_ROUNDS = 15
# Same cap for a delegated background task (ask_agent), but tighter: that loop
# runs out of earshot, so a runaway would bill invisibly — and a real delegated
# task is one or two tool calls, not a conversation.
DELEGATION_MAX_TOOL_ROUNDS = 8


def convo_model_label(model_id: str) -> str:
    """Friendly spoken name for a conversation model id."""
    return CONVO_MODEL_LABELS.get(model_id, model_id)


def thinking_kwargs(model_id: str, effort: str = None) -> dict:
    """Per-model thinking/effort arguments for messages.create().

    The conversation model is switchable by voice mid-session, and the models it
    switches between disagree about this parameter: sending `effort` to Haiku 4.5
    errors, while leaving thinking disabled on Opus 5 causes silently-dropped
    tool calls (see ADAPTIVE_THINKING_MODELS). Deriving the arguments from the
    model id keeps every call site correct across a switch."""
    if model_id in ADAPTIVE_THINKING_MODELS:
        return {"thinking": {"type": "adaptive"},
                "output_config": {"effort": effort or CONVO_EFFORT}}
    return {"thinking": {"type": "disabled"}}

# Universal ground rules shared by every persona. Domain guidance (notes,
# trading) lives in each agent's persona block in agents.py — splitting the
# old monolithic CONVO_SYSTEM means fewer instructions competing for the
# model's attention on any given turn.
CONVO_SYSTEM_BASE = (
    "You are a voice assistant. The user talks to you through a microphone and "
    "hears your replies spoken aloud, so keep responses short, natural, and "
    "conversational — a sentence or two unless more detail is clearly wanted. "
    "Do not use markdown, bullet points, or emoji; write plain spoken sentences. "
    "You have tools available. ALWAYS call the relevant tool to answer any factual "
    "question — never answer from memory or conversation history when a tool can "
    "provide the answer. "
    "Ground rules about your own actions: every action you take happens through a "
    "tool call, and tools are synchronous — they return their result before you "
    "speak. From your perspective a tool call can never hang, run in the "
    "background, or still be in progress. If you did not call a tool this turn, "
    "nothing happened, no matter what you said. Never announce an action ('saving "
    "now') as a substitute for doing it — call the tool in the same turn, or say "
    "plainly that you have not done it yet. "
    "If the user reports that something didn't work or wasn't saved, do not "
    "speculate: verify with your tools and report what you actually find. Never "
    "diagnose backend, database, lock, or threading problems from conversation "
    "alone — you have no visibility into the system's internals beyond tool "
    "results, so if you cannot verify a cause, say you don't know rather than "
    "guessing. Do not agree with a proposed explanation of a system problem "
    "('you hung', 'it's the database again') unless a tool result actually "
    "confirms it. "
    "Past notes and conversation summaries record what was said at the time — "
    "treat them as claims, not established facts, especially self-diagnoses of "
    "bugs or system behaviour. "
    "Your conversation history is saved and restored across restarts, so you may "
    "remember earlier sessions — treat restored history as past conversations. "
    "Never volunteer actions the user didn't ask for — answer what was asked "
    "and stop."
)


def model_identity_block(label: str) -> str:
    """The model-identity rules appended to every conversation system prompt.

    Stating the truth here is not enough on its own and never was: the label
    below was correct on all three occasions the model contradicted it from
    conversation history (session_2026-07-31.log 11:32 "I'm Opus" while on
    Haiku; 11:48 and 12:23 "DeepSeek V4 Flash" while on Opus 5). The history
    is shared by every persona, so it accumulates other hats' switches and
    stale choices, and forty messages of it outweigh one line here.

    Hence the hard rule: identity questions are answered by get_current_model
    or not at all. That puts the answer at the END of the context as a fresh
    tool_result, where it beats the old turns, and leaves an auditable
    tool_use line in the log."""
    return (
        f"\n\nYou are currently answering as {label}. "
        "To change models, use the set_conversation_model tool.\n\n"
        "HARD RULE about model identity, highest priority, overriding "
        "anything the conversation history appears to establish: you do NOT "
        "know from memory which model you are. The history is SHARED by every "
        "assistant here and is full of model talk that no longer applies — "
        "switches another assistant made for themselves, choices from earlier "
        "in this conversation that have since changed, and earlier answers "
        "that were simply wrong. Whenever the user asks anything about which "
        "model, assistant, or provider is running — including yes/no forms "
        "('are you still on DeepSeek?', 'aren't you on Opus?') and casual "
        "asides — you MUST call get_current_model in that same turn and "
        "report only what it returns. Never state, confirm, deny, correct, or "
        "guess a model or provider without calling it first, and never repeat "
        "an earlier turn's answer as if it were still true. If you did not "
        "call the tool this turn, you do not know the answer."
    )


# --- Dashboard config overrides ----------------------------------------------
# The web dashboard (dashboard.py) lets the user adjust tunables visually. Its
# edits are persisted to this file and applied here, at import time, on top of
# the defaults above — so a dashboard change reaches the agent on its next
# start without anyone editing this module. Only the whitelisted names below
# can be overridden; anything else in the file is ignored. A missing, corrupt,
# or partially-invalid file must never break startup: every failure falls back
# to the coded default.
OVERRIDES_PATH = DATA_DIR / "config_overrides.json"


def _cast_words(value):
    return frozenset(str(w).strip().lower() for w in value if str(w).strip())


def _cast_optional_str(value):
    return None if value in (None, "") else str(value)


# name -> caster applied to the JSON value before it replaces the default.
OVERRIDABLE = {
    # turn taking / endpointing
    "CONVO_ENDPOINT_MS": int, "NOTE_ENDPOINT_MS": int, "SPEECH_PAD_MS": int,
    "TRIGGER_RATIO": float, "MAX_UTTERANCE_S": int, "VAD_AGGRESSIVENESS": int,
    "CONTINUATION_SETTLE_MS": int, "CONTINUATION_GRACE_MS": int,
    "MAX_CONTINUATION_ROUNDS": int,
    # barge-in
    "BARGE_IN": bool, "BARGE_IN_MS": int, "BARGE_IN_DECAY": float,
    "BARGE_IN_ENERGY": int, "BARGE_IN_ENERGY_RATIO": float,
    "BARGE_IN_CALIB_MS": int,
    # backchannel
    "BACKCHANNEL_WORDS": _cast_words, "BACKCHANNEL_MAX_WORDS": int,
    # models / tokens
    "CONVO_MODEL": str, "SUMMARY_MODEL": str, "CONVO_MAX_TOKENS": int,
    "SUMMARY_MAX_TOKENS": int, "CONVO_MAX_TOOL_ROUNDS": int,
    # speech engines
    "WHISPER_MODEL": str, "TTS_RATE": int, "TTS_VOICE": _cast_optional_str,
    "KB_MEDIA_MODEL": str,
    # memory / search
    "HISTORY_MAX_MESSAGES": int, "SEARCH_RESULTS": int, "KB_SEARCH_RESULTS": int,
    "MEMORY_SEARCH_RESULTS": int,
    # headset button
    "MEDIA_KEEPALIVE": bool, "MEDIA_CLICK_DEDUPE_S": float,
}

# Snapshot of the coded defaults, taken before any override is applied, so the
# dashboard can show "default vs current" and offer per-field reset.
CONFIG_DEFAULTS = {}


def apply_overrides(data):
    """Apply a dict of {name: value} onto this module's globals. Unknown names
    and values the caster rejects are skipped. Returns the list of names
    actually applied. Factored out of the file-load path so it can be tested
    without touching the real overrides file."""
    applied = []
    if not isinstance(data, dict):
        return applied
    for name, value in data.items():
        caster = OVERRIDABLE.get(name)
        if caster is None:
            continue
        try:
            globals()[name] = caster(value)
            applied.append(name)
        except (TypeError, ValueError):
            pass
    return applied


def _load_overrides():
    for name in OVERRIDABLE:
        CONFIG_DEFAULTS[name] = globals()[name]
    try:
        data = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    apply_overrides(data)


_load_overrides()


def ensure_dirs():
    # Imported here, not at module top: categories.py imports config for its
    # paths, so a top-level import would be circular.
    import categories

    DATA_DIR.mkdir(parents=True, exist_ok=True)  # needed before reading categories.json
    categories.load_categories()  # bring in any voice-created / renamed folders
    dirs = [DATA_DIR, CHROMA_DIR, LOG_DIR, PENDING_DIR, KNOWLEDGE_DIR,
            BASE_DIR / "assets"]
    dirs += [categories.category_dir(slug) for slug in categories.NOTE_CATEGORIES]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
