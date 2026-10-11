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
    """Combines two ranked lists into fused RRF scores, keyed by chunk_id."""
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

        results: list[RetrievedChunk] = []
        for chunk_id, scores in ranked_ids[:top_k]:
            chunk = self.chunks.get(chunk_id)
            if chunk is None:
                continue
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    vector_score=scores["vector_score"],
                    bm25_score=scores["bm25_score"],
                    fused_score=scores["fused_score"],
                )
            )
        return results
