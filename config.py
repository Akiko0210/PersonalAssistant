"""Central configuration for the voice notetaking agent.

Everything tunable lives here so behaviour can be changed without touching the
logic modules.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"
LOG_DIR = BASE_DIR / "logs"
INDEX_PATH = DATA_DIR / "index.json"
# Live transcripts are appended here while recording, then moved into the chosen
# category folder when the note is saved (the category isn't known until the end).
PENDING_DIR = DATA_DIR / "pending"
# Legacy flat locations — only referenced by the one-time migration in notes.py.
SUMMARY_DIR = DATA_DIR / "summaries"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"

# --- Note categories ---------------------------------------------------------
# The single source of truth for note folders. Each note (summary + transcript)
# is filed under data/<folder>/. "description" tells the model what belongs in a
# category — both when auto-classifying a finished note and when resolving a
# folder name the user speaks. Add a category by adding an entry here.
NOTE_CATEGORIES = {
    "trading": {
        "display": "Trading",
        "folder": "Trading",
        "description": "Any option trading note.",
    },
    "therapy_book": {
        "display": "Therapy book",
        "folder": "TherapyBooks",
        "description": "Any physical therapy or neurology book notes.",
    },
    "to-do": {
        "display": "To-do",
        "folder": "To-do",
        "description": "My to-do notes."
    },
    "ideas": {
        "display": "Ideas",
        "folder": "Ideas",
        "description": "Thoughts I wanna get back to"
    },
    "reminders": {
        "display": "Reminders",
        "folder": "Reminders",
        "description": "Things that I wanna remember later."
    },
    "general": {
        "display": "General",
        "folder": "General",
        "description": "Anything that doesn't fit another category.",
    },
}
DEFAULT_CATEGORY = "general"


def category_dir(slug):
    """Absolute path to a category's folder; unknown slugs fall back to default."""
    slug = slug if slug in NOTE_CATEGORIES else DEFAULT_CATEGORY
    return DATA_DIR / NOTE_CATEGORIES[slug]["folder"]

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

# --- Text-to-speech (local, Windows SAPI via pyttsx3) ------------------------
TTS_RATE = 175                # words per minute
TTS_VOICE = None              # None = system default; or a SAPI voice id substring

# --- "Thinking" audio cue ----------------------------------------------------
# Looped whenever the agent is busy with the model and there's nothing to hear —
# answering you (including tool calls), summarising a note, or deciding its
# folder — as a "still working" cue. WAV only (played via the built-in winsound,
# which loops and stops cleanly). Reuses the summarising clip for now; point it
# at a dedicated idle file, or set to None to disable. Missing file = silent, no
# error.
IDLE_SOUND = BASE_DIR / "assets" / "summarizing.wav"

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

# --- Claude ------------------------------------------------------------------
CONVO_MODEL = "claude-haiku-4-5"   # low latency for back-and-forth
SUMMARY_MODEL = "claude-sonnet-4-6"   # higher quality for note summaries
CONVO_MAX_TOKENS = 1024
SUMMARY_MAX_TOKENS = 2000

CONVO_SYSTEM = (
    "You are a voice assistant. The user talks to you through a microphone and "
    "hears your replies spoken aloud, so keep responses short, natural, and "
    "conversational — a sentence or two unless more detail is clearly wanted. "
    "Do not use markdown, bullet points, or emoji; write plain spoken sentences. "
    "You have tools available. ALWAYS call the relevant tool to answer any factual "
    "question — never answer from memory or conversation history when a tool can "
    "provide the answer. This applies to notes, Discord notifications, trades, "
    "note counts, the current time, and anything else the tools cover. "
    "You can also answer questions about the user's captured Discord notifications "
    "and trade alerts using the Discord tools. Use get_recent_trades for the latest "
    "trade lines; for time-based questions like 'what trades came in today', use "
    "get_recent_discord_messages with the date, since the trade list itself has no "
    "timestamps. Read trade details aloud naturally rather than reciting symbols "
    "character by character."
)



def ensure_dirs():
    dirs = [DATA_DIR, CHROMA_DIR, LOG_DIR, PENDING_DIR, BASE_DIR / "assets"]
    dirs += [category_dir(slug) for slug in NOTE_CATEGORIES]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
