import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Для запуска без Docker настройки читаются из .env; уже заданные переменные
# окружения (например, из docker-compose) имеют приоритет.
load_dotenv(BASE_DIR / ".env", override=False)

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
FAISS_INDEX_PATH = PROCESSED_DIR / "faiss.index"
BM25_INDEX_PATH = PROCESSED_DIR / "bm25.pkl"

# База знаний и вопросы на русском, поэтому модель мультиязычная.
EMBEDDING_MODEL_NAME = os.environ.get(
    "EMBEDDING_MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# "anthropic" — Claude API (ANTHROPIC_API_KEY читает сам SDK), "ollama" — локальная модель.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

CHUNK_SIZE_WORDS = int(os.environ.get("CHUNK_SIZE_WORDS", "220"))
CHUNK_OVERLAP_WORDS = int(os.environ.get("CHUNK_OVERLAP_WORDS", "40"))

TOP_K_VECTOR = int(os.environ.get("TOP_K_VECTOR", "8"))
TOP_K_BM25 = int(os.environ.get("TOP_K_BM25", "8"))
TOP_K_FINAL = int(os.environ.get("TOP_K_FINAL", "5"))
RRF_K = int(os.environ.get("RRF_K", "60"))

# Порог отказа «не знаю» по максимальной косинусной близости среди найденных
# фрагментов (RRF-скор основан на рангах и абсолютной релевантности не отражает).
# Значение подобрано вручную для paraphrase-multilingual-MiniLM-L12-v2 на этом
# корпусе; при смене модели эмбеддингов порог нужно калибровать заново.
MIN_RELEVANCE_SCORE = float(os.environ.get("MIN_RELEVANCE_SCORE", "0.48"))
