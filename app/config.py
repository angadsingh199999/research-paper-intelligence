import os
from pathlib import Path
from dotenv import load_dotenv

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    load_dotenv(env_file)

DATA_DIR = PROJECT_ROOT / "data"
RAW_PAPERS_DIR = DATA_DIR / "raw_papers"
CHROMA_DIR = DATA_DIR / "chroma"

# Ensure critical directories exist
RAW_PAPERS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Model Settings
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Retrieval & Fusion Parameters
DEFAULT_RRF_K = int(os.getenv("RRF_K", 60))
DEFAULT_TOP_K = int(os.getenv("TOP_K", 8))
DEFAULT_CANDIDATE_K = int(os.getenv("CANDIDATE_K", 24))

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# If local HF cache exists, default offline flag to avoid repeated network DNS timeouts
hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
if hf_cache.exists():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
