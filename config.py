"""Central configuration — reads from environment variables or .env file."""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

BASE_DIR = Path(__file__).parent
DB_PATH   = BASE_DIR / "data" / "news.db"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR    = BASE_DIR / "logs"

# ── Whisper ──────────────────────────────────────────────────────────────────
# Use "small" on CPU server (good speed/accuracy balance)
# Use "medium" if Oracle Cloud 24GB RAM
WHISPER_MODEL        = os.getenv("WHISPER_MODEL", "small")
WHISPER_DEVICE       = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")

# ── Groq (free tier: 14 400 req/day, llama-3.3-70b) ─────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL   = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_ENABLED = bool(GROQ_API_KEY)

# ── Collector ─────────────────────────────────────────────────────────────────
CHUNK_DURATION     = int(os.getenv("CHUNK_DURATION", "30"))   # seconds per capture
MIN_WORDS_TO_SAVE  = 8    # ignore chunks with less than N words
MAX_WORKERS        = int(os.getenv("MAX_WORKERS", "3"))       # parallel channels

# ── SNRT Channels ─────────────────────────────────────────────────────────────
# Set URLs via environment variables on the server
CHANNELS = [
    {
        "id": "al_aoula",
        "name": "Al Aoula",
        "url": os.getenv("AL_AOULA_URL", ""),
        "language": "ar",
        "active": bool(os.getenv("AL_AOULA_URL")),
    },
    {
        "id": "arryadia",
        "name": "Arryadia",
        "url": os.getenv("ARRYADIA_URL", ""),
        "language": "ar",
        "active": bool(os.getenv("ARRYADIA_URL")),
    },
    {
        "id": "assadissa",
        "name": "Assadissa",
        "url": os.getenv("ASSADISSA_URL", ""),
        "language": "ar",
        "active": bool(os.getenv("ASSADISSA_URL")),
    },
    {
        "id": "arrabia",
        "name": "Arrabia",
        "url": os.getenv("ARRABIA_URL", ""),
        "language": "fr",
        "active": bool(os.getenv("ARRABIA_URL")),
    },
    {
        "id": "amazighie",
        "name": "Amazighie",
        "url": os.getenv("AMAZIGHIE_URL", ""),
        "language": "ar",
        "active": bool(os.getenv("AMAZIGHIE_URL")),
    },
]

# ── Dashboard ─────────────────────────────────────────────────────────────────
DASHBOARD_PORT            = int(os.getenv("DASHBOARD_PORT", "8501"))
DASHBOARD_REFRESH_SECONDS = int(os.getenv("DASHBOARD_REFRESH", "20"))
