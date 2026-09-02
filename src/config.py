import os
from pathlib import Path

from dotenv import load_dotenv


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
STORAGE_DIR = BASE_DIR / "storage"
CHROMA_DIR = STORAGE_DIR / "chroma_db"

# Name of the Chroma collection containing searchable child chunks
CHROMA_COLLECTION = "research_paper_children"

PARENTS_PATH = STORAGE_DIR / "parents.json"
CHAT_DB_PATH = STORAGE_DIR / "chat_history.db"


# Load private environment variables
load_dotenv(BASE_DIR / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
)


# Local Hugging Face models
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"



# Hierarchical chunking
PARENT_CHUNK_SIZE = 1500
PARENT_CHUNK_OVERLAP = 150

CHILD_CHUNK_SIZE = 350
CHILD_CHUNK_OVERLAP = 50


# Retrieval
DENSE_K = 10
BM25_K = 10
FINAL_CHILD_K = 10
MAX_PARENT_CONTEXTS = 5


# Ensure generated-storage directories exist
STORAGE_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)