import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Native (non-Docker) runs read settings from .env; variables already set in
# the environment (e.g. by docker-compose) take precedence.
load_dotenv(BASE_DIR / ".env", override=False)
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
FAISS_INDEX_PATH = PROCESSED_DIR / "faiss.index"
BM25_INDEX_PATH = PROCESSED_DIR / "bm25.pkl"

# Multilingual model: the knowledge base and queries are in Russian, and
# all-MiniLM-L6-v2 (English-only) would give poor semantic search quality here.
EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# ANTHROPIC_API_KEY itself is read directly by the Anthropic SDK client, not here.
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")

# "anthropic" (paid, Claude API) or "ollama" (free, local model, no API key needed)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

CHUNK_SIZE_WORDS = int(os.environ.get("CHUNK_SIZE_WORDS", "220"))
CHUNK_OVERLAP_WORDS = int(os.environ.get("CHUNK_OVERLAP_WORDS", "40"))

TOP_K_VECTOR = int(os.environ.get("TOP_K_VECTOR", "8"))
TOP_K_BM25 = int(os.environ.get("TOP_K_BM25", "8"))
TOP_K_FINAL = int(os.environ.get("TOP_K_FINAL", "5"))
RRF_K = int(os.environ.get("RRF_K", "60"))

# RRF fused scores are rank-based (~constant magnitude regardless of true
# relevance), so the "I don't know" gate instead checks the top result's raw
# vector cosine similarity, which is a real bounded relevance signal.
# Calibrated for paraphrase-multilingual-MiniLM-L12-v2 on the Russian corpus:
# on-topic questions score ~0.50-0.70, clearly off-topic ones ~0.05-0.10,
# finance-adjacent-but-uncovered ones ~0.45-0.47.
MIN_RELEVANCE_SCORE = float(os.environ.get("MIN_RELEVANCE_SCORE", "0.48"))
