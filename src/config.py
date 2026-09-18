import os
from dotenv import load_dotenv
from pathlib import Path

# Load variables from .env
# Walk up from this file to find the project root .env
_project_root = Path(__file__).resolve().parent.parent
load_dotenv(_project_root / ".env")

# --- Gemini (active LLM provider) ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MODEL_NAME = GEMINI_MODEL  # alias for backward compatibility

# --- OpenRouter (optional alternative) ---
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Embedding model
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ChromaDB settings
CHROMA_COLLECTION_NAME = "environmental_knowledge"
CHROMA_PERSIST_DIR = str(_project_root / "chroma_db")

# Knowledge base path
KNOWLEDGE_BASE_PATH = str(_project_root / "data" / "environmental_data" / "knowledge.json")

# Conversation settings
MAX_CONVERSATION_HISTORY = 20