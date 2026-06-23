"""Central configuration for the voice notetaking agent.

Everything tunable lives here so behaviour can be changed without touching the
logic modules.
"""

from pathlib import Path

# --- Paths -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"
SUMMARY_DIR = DATA_DIR / "summaries"
CHROMA_DIR = DATA_DIR / "chroma"
LOG_DIR = BASE_DIR / "logs"
INDEX_PATH = DATA_DIR / "index.json"

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

# --- Barge-in (interrupt the agent by speaking while it talks) ---------------
# Works best with headphones. On open speakers the mic can hear the agent's own
# voice and self-interrupt; raise BARGE_IN_MS or set BARGE_IN = False if so.
BARGE_IN = True               # stop speaking when the user starts talking
BARGE_IN_MS = 400             # consecutive qualifying audio (ms) that counts as an interruption
# Loudness gate: a frame only counts toward an interruption if it is BOTH speech
# (per VAD) AND louder than the agent's own echo. The threshold is the larger of
# an absolute floor and a multiple of the measured echo level, so it adapts to
# your speaker volume. Lower these if real speech isn't interrupting; raise them
# if the agent still interrupts itself.
BARGE_IN_ENERGY = 300        # absolute RMS floor (int16) the user's voice must exceed
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
    "When the user asks about their notes (their latest note, what they said about "
    "something, etc.), use the provided tools to look it up rather than guessing."
)



def ensure_dirs():
    for d in (DATA_DIR, TRANSCRIPT_DIR, SUMMARY_DIR, CHROMA_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
