import pytest

from src.hybrid import HybridRetriever, reciprocal_rank_fusion


def test_rrf_boosts_documents_ranked_high_in_both_lists():
    vector_results = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    bm25_results = [("b", 5.0), ("a", 3.0), ("d", 2.0)]

    fused = reciprocal_rank_fusion(vector_results, bm25_results, rrf_k=60)

    # "a" and "b" appear in both lists near the top, so they should outrank
    # "c" and "d" which only appear in one list.
    assert fused["a"]["fused_score"] > fused["c"]["fused_score"]
    assert fused["b"]["fused_score"] > fused["d"]["fused_score"]


def test_rrf_preserves_individual_scores():
    vector_results = [("a", 0.42)]
    bm25_results = [("a", 3.14)]

    fused = reciprocal_rank_fusion(vector_results, bm25_results)

    assert fused["a"]["vector_score"] == 0.42
    assert fused["a"]["bm25_score"] == 3.14


def test_rrf_handles_disjoint_results():
    vector_results = [("a", 0.9)]
    bm25_results = [("b", 5.0)]

    fused = reciprocal_rank_fusion(vector_results, bm25_results)

    assert set(fused.keys()) == {"a", "b"}
    assert fused["a"]["bm25_score"] == 0.0
    assert fused["b"]["vector_score"] == 0.0


def test_rrf_empty_inputs():
    fused = reciprocal_rank_fusion([], [])
    assert fused == {}


def test_retriever_rejects_inconsistent_indexes(monkeypatch):
    retriever = HybridRetriever()
    vectors, bm25 = retriever.vector_store, retriever.bm25_index
    monkeypatch.setattr(vectors, "load", lambda: setattr(vectors, "chunk_ids", ["a", "b"]))
    monkeypatch.setattr(bm25, "load", lambda: setattr(bm25, "chunk_ids", ["a"]))  # BM25 собран по другим фрагментам
    monkeypatch.setattr("src.hybrid.load_chunks", lambda: {"a": object(), "b": object()})

    with pytest.raises(RuntimeError, match="не согласованы"):
        retriever.load()
