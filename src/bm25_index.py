import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from src.config import BM25_INDEX_PATH
from src.schemas import Chunk

# Includes Cyrillic (а-яё) alongside Latin/digits so Russian-language chunks
# and queries tokenize into real words instead of being dropped entirely.
_TOKEN_RE = re.compile(r"[a-zA-Zа-яёА-ЯЁ0-9']+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self) -> None:
        self.bm25: BM25Okapi | None = None
        self.chunk_ids: list[str] = []

    def build(self, chunks: list[Chunk]) -> None:
        corpus = [tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(corpus)
        self.chunk_ids = [c.chunk_id for c in chunks]

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if self.bm25 is None:
            raise RuntimeError("BM25 index not built/loaded")
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.chunk_ids, scores), key=lambda x: x[1], reverse=True)
        return [(cid, float(score)) for cid, score in ranked[:top_k] if score > 0]

    def save(self, path: Path = BM25_INDEX_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump({"bm25": self.bm25, "chunk_ids": self.chunk_ids}, f)

    def load(self, path: Path = BM25_INDEX_PATH) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.chunk_ids = data["chunk_ids"]
