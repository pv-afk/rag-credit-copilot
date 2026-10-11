import json
from pathlib import Path

import faiss
import numpy as np

from src.config import FAISS_INDEX_PATH
from src.schemas import Chunk


class VectorStore:
    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.chunk_ids: list[str] = []

    def build(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.chunk_ids = [c.chunk_id for c in chunks]

    def search(self, query_embedding: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if self.index is None:
            raise RuntimeError("Векторный индекс не построен и не загружен")
        scores, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
        results = []
        for idx, score in zip(indices[0], scores[0], strict=True):
            if idx == -1:
                continue
            results.append((self.chunk_ids[idx], float(score)))
        return results

    def save(self, index_path: Path = FAISS_INDEX_PATH) -> None:
        if self.index is None:
            raise RuntimeError("Индекс не построен: сохранять нечего")
        faiss.write_index(self.index, str(index_path))
        ids_path = index_path.with_suffix(".ids.json")
        ids_path.write_text(json.dumps(self.chunk_ids), encoding="utf-8")

    def load(self, index_path: Path = FAISS_INDEX_PATH) -> None:
        self.index = faiss.read_index(str(index_path))
        ids_path = index_path.with_suffix(".ids.json")
        self.chunk_ids = json.loads(ids_path.read_text(encoding="utf-8"))
