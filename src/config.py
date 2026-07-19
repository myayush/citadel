import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# A single DATABASE_URL wins over the POSTGRES_* values so the same code runs
# against local docker and against Neon in Iteration 7 with no edits.
DATABASE_URL = os.getenv("DATABASE_URL", "")

DB_PARAMS = {
    "host": os.getenv("POSTGRES_HOST", "localhost"),
    "port": os.getenv("POSTGRES_PORT", "5432"),
    "user": os.getenv("POSTGRES_USER", "citadel"),
    "password": os.getenv("POSTGRES_PASSWORD", "citadel"),
    "dbname": os.getenv("POSTGRES_DB", "citadel"),
}

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
STRONG_MODEL = os.getenv("STRONG_MODEL", "")
CHEAP_MODEL = os.getenv("CHEAP_MODEL", "")

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMS = 384

RRF_K = 60
TOP_K = 5