from src.bm25_index import BM25Index
from src.chunk_store import load_chunks
from src.config import RRF_K, TOP_K_BM25, TOP_K_FINAL, TOP_K_VECTOR
from src.embeddings import embed_query
from src.schemas import Chunk, RetrievedChunk
from src.vectorstore import VectorStore


def reciprocal_rank_fusion(
    vector_results: list[tuple[str, float]],
    bm25_results: list[tuple[str, float]],
    rrf_k: int = RRF_K,
) -> dict[str, dict[str, float]]:
    """Объединяет два ранжированных списка по RRF: score = Σ 1 / (k + rank + 1).

    Возвращает словарь chunk_id -> {vector_score, bm25_score, fused_score}; исходные
    скоры сохраняются (0.0, если фрагмент нашёл только один из методов).
    """
    fused: dict[str, dict[str, float]] = {}

    for rank, (chunk_id, score) in enumerate(vector_results):
        entry = fused.setdefault(chunk_id, {"vector_score": 0.0, "bm25_score": 0.0, "fused_score": 0.0})
        entry["vector_score"] = score
        entry["fused_score"] += 1.0 / (rrf_k + rank + 1)

    for rank, (chunk_id, score) in enumerate(bm25_results):
        entry = fused.setdefault(chunk_id, {"vector_score": 0.0, "bm25_score": 0.0, "fused_score": 0.0})
        entry["bm25_score"] = score
        entry["fused_score"] += 1.0 / (rrf_k + rank + 1)

    return fused


class HybridRetriever:
    def __init__(self) -> None:
        self.vector_store = VectorStore()
        self.bm25_index = BM25Index()
        self.chunks: dict[str, Chunk] = {}

    def load(self) -> None:
        self.vector_store.load()
        self.bm25_index.load()
        self.chunks = load_chunks()
        # Индексы строятся одним запуском build_index; расхождение означает частичную пересборку.
        if set(self.vector_store.chunk_ids) != set(self.chunks) or set(self.bm25_index.chunk_ids) != set(self.chunks):
            raise RuntimeError("Индексы в data/processed не согласованы. Пересоберите: python -m src.build_index")

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_FINAL,
        top_k_vector: int = TOP_K_VECTOR,
        top_k_bm25: int = TOP_K_BM25,
    ) -> list[RetrievedChunk]:
        query_embedding = embed_query(query)
        vector_results = self.vector_store.search(query_embedding, top_k_vector)
        bm25_results = self.bm25_index.search(query, top_k_bm25)

        fused = reciprocal_rank_fusion(vector_results, bm25_results)
        ranked_ids = sorted(fused.items(), key=lambda x: x[1]["fused_score"], reverse=True)

        return [RetrievedChunk(chunk=self.chunks[chunk_id], **scores) for chunk_id, scores in ranked_ids[:top_k]]
