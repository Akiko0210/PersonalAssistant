"""Central configuration for the voice notetaking agent.

Everything tunable lives here so behaviour can be changed without touching the
logic modules.
"""

import json
import re
from pathlib import Path

# --- Paths -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"
LOG_DIR = BASE_DIR / "logs"
INDEX_PATH = DATA_DIR / "index.json"
# User-created / renamed folders are persisted here and overlaid on the built-in
# NOTE_CATEGORIES defaults below at startup (see load_categories).
CATEGORIES_PATH = DATA_DIR / "categories.json"
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

# --- Note categories ---------------------------------------------------------
# The built-in seed folders. Each note (summary + transcript) is filed under
# data/<folder>/. "description" tells the model what belongs in a category — both
# when auto-classifying a finished note and when resolving a folder name the user
# speaks. These are defaults: at startup load_categories() overlays any folders
# the user has created or renamed by voice (persisted in data/categories.json),
# and add_category()/rename_category() mutate this live dict and re-save.
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


# --- Runtime folder management (create / rename by voice) --------------------
def _slugify(name: str) -> str:
    """Stable dict key derived from a display name: lowercase, non-alphanumeric
    runs collapsed to single hyphens. Slugs never change once assigned so notes,
    the index, and Chroma metadata stay linked across renames."""
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")
    return slug or "folder"


def _folder_name(name: str) -> str:
    """Filesystem-friendly directory name from a display name (e.g. 'Meeting
    notes' -> 'MeetingNotes')."""
    folder = re.sub(r"[^0-9A-Za-z]+", "", (name or "").strip().title())
    return folder or "Folder"


def _unique(value: str, existing, sep: str = "") -> str:
    """Return `value`, or value-2/value2/... if it clashes with `existing`."""
    if value not in existing:
        return value
    i = 2
    while f"{value}{sep}{i}" in existing:
        i += 1
    return f"{value}{sep}{i}"


def load_categories():
    """Overlay user-persisted folders onto the built-in defaults. Idempotent, so
    it's safe to call on every ensure_dirs()."""
    if not CATEGORIES_PATH.exists():
        return
    try:
        data = json.loads(CATEGORIES_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if isinstance(data, dict):
        for slug, meta in data.items():
            if isinstance(meta, dict) and {"display", "folder"} <= meta.keys():
                NOTE_CATEGORIES[slug] = meta


def save_categories():
    CATEGORIES_PATH.write_text(
        json.dumps(NOTE_CATEGORIES, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def add_category(display: str, description: str = "") -> str:
    """Create a new folder from a spoken name. Returns its (new, unique) slug and
    creates the directory on disk."""
    display = (display or "").strip()
    slug = _unique(_slugify(display), set(NOTE_CATEGORIES), sep="-")
    folder = _unique(_folder_name(display),
                     {m["folder"] for m in NOTE_CATEGORIES.values()})
    NOTE_CATEGORIES[slug] = {
        "display": display,
        "folder": folder,
        "description": (description or "").strip() or f"Notes about {display}.",
    }
    category_dir(slug).mkdir(parents=True, exist_ok=True)
    save_categories()
    return slug


def rename_category(slug: str, new_display: str):
    """Rename an existing folder's display name (and its on-disk directory) while
    keeping the slug stable so saved notes stay linked."""
    meta = NOTE_CATEGORIES[slug]
    old_dir = category_dir(slug)
    meta["display"] = (new_display or "").strip()
    meta["folder"] = _unique(
        _folder_name(new_display),
        {m["folder"] for s, m in NOTE_CATEGORIES.items() if s != slug},
    )
    new_dir = category_dir(slug)
    if new_dir != old_dir:
        if old_dir.exists():
            old_dir.rename(new_dir)
        else:
            new_dir.mkdir(parents=True, exist_ok=True)
    save_categories()


def delete_category(slug: str):
    """Remove a folder from the registry and drop its (expected-empty) directory.
    Callers must relocate any notes first — this only removes the directory when
    it's empty, so stray files are left in place rather than destroyed."""
    old_dir = category_dir(slug)  # resolve before popping (category_dir needs the entry)
    NOTE_CATEGORIES.pop(slug, None)
    save_categories()
    try:
        if old_dir.exists() and not any(old_dir.iterdir()):
            old_dir.rmdir()
    except OSError:
        pass

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

# --- Knowledge base (ingested PDFs) ------------------------------------------
KNOWLEDGE_COLLECTION = "knowledge"   # Chroma collection, separate from "notes"
KB_CHUNK_CHARS = 1000                # target characters per embedded chunk
KB_CHUNK_OVERLAP = 150               # characters shared between adjacent chunks
KB_SEARCH_RESULTS = 5                # chunks returned per search_knowledge call

# --- Long-term conversation memory --------------------------------------------
MEMORY_COLLECTION = "conversations"  # Chroma collection of archived summaries
MEMORY_MIN_MESSAGES = 6              # consolidate only once this many lines staged
MEMORY_MAX_TOKENS = 700              # budget for one consolidation summary
MEMORY_SEARCH_RESULTS = 3            # summaries returned per search

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

# --- Listening while thinking --------------------------------------------------
# The model reply is fetched in a background thread while the mic stays live, so
# words spoken during transcription/thinking are never lost. If the user resumes
# talking before the reply is spoken (the agent endpointed too early), the stale
# reply is discarded and the model is re-asked with the completed sentence. The
# grace window lets speech that starts JUST as the reply arrives win the race
# instead of being talked over.
CONTINUATION_GRACE_MS = 400

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
    "You can manage the user's note folders too: create, rename, or delete a "
    "folder, and move a note from one folder to another, using the create_folder, "
    "rename_folder, delete_folder, and move_note tools. To move a note, first look "
    "up its id with search_notes or list_recent_notes, then call move_note. "
    "When a question is scoped to one folder (e.g. 'my latest note in General'), "
    "pass that folder to search_notes or list_recent_notes instead of filtering "
    "yourself. "
    "Your conversation history is saved and restored across restarts, so you may "
    "remember earlier sessions — treat restored history as past conversations. "
    "Conversations older than the current window are archived as searchable "
    "summaries: use search_past_conversations for 'what did we talk about last "
    "week' or anything you don't see in the current history. "
    "Only save a conversation as a note when the user explicitly asks you to "
    "('save that as a note', 'make a note of that'). Never suggest, offer, or "
    "prompt to save a note on your own — do not ask whether they want to save "
    "anything. When they do ask, call save_conversation_note with a clear title "
    "and well-formed markdown content drawn from the conversation, then reply with "
    "one short acknowledgement; the system handles asking which folder to file it "
    "in, so never ask about folders yourself. "
    "You have a trading knowledge base built from reference material the user "
    "ingested (books and PDFs). Use search_knowledge for questions about trading "
    "concepts, strategies, or definitions that such material would cover, and cite "
    "the source and page when it helps. "
    "You can also answer questions about the user's captured Discord notifications "
    "and trade alerts using the Discord tools. Use get_recent_trades for the latest "
    "trade lines; for time-based questions like 'what trades came in today', use "
    "get_recent_discord_messages with the date, since the trade list itself has no "
    "timestamps. Read trade details aloud naturally rather than reciting symbols "
    "character by character. "
    "HARD RULE, highest priority: never volunteer note actions. Do not offer, "
    "suggest, or ask about saving, updating, or filing notes — replies like 'would "
    "you like me to save that as a note?' or 'do you want me to update that note?' "
    "are forbidden, no matter what. Just acknowledge what the user said and stop. "
    "Note actions happen only when the user's own current message explicitly "
    "requests one. If earlier messages in this conversation show you offering to "
    "save or update notes, those were errors — never imitate them."
)



def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)  # needed before reading categories.json
    load_categories()  # bring in any voice-created / renamed folders
    dirs = [DATA_DIR, CHROMA_DIR, LOG_DIR, PENDING_DIR, KNOWLEDGE_DIR,
            BASE_DIR / "assets"]
    dirs += [category_dir(slug) for slug in NOTE_CATEGORIES]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
