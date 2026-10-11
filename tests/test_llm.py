import json

import pytest

from src import llm
from src.schemas import Chunk, RetrievedChunk


def make_retrieved(vector_score: float) -> list[RetrievedChunk]:
    chunk = Chunk(
        chunk_id="doc1_0", doc_id="doc1", title="Basel III", source_path="doc1.pdf", chunk_index=0, text="CET1 4.5%"
    )
    return [RetrievedChunk(chunk=chunk, vector_score=vector_score, fused_score=0.03)]


class FakeOllamaResponse:
    def __init__(self, content: str) -> None:
        self._content = content

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return {"message": {"content": self._content}}


def test_low_relevance_short_circuits_without_calling_llm(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("LLM must not be called below MIN_RELEVANCE_SCORE")

    monkeypatch.setattr(llm, "_generate_answer_anthropic", fail)
    monkeypatch.setattr(llm, "_generate_answer_ollama", fail)

    result = llm.generate_answer("Off-topic question", make_retrieved(vector_score=0.05))

    assert result.is_answerable is False
    assert result.confidence == "insufficient_context"
    assert result.citations == []


def test_empty_retrieval_short_circuits():
    result = llm.generate_answer("Anything", [])
    assert result.is_answerable is False


def test_ollama_valid_json_is_parsed(monkeypatch):
    payload = {
        "answer": "4.5% RWA",
        "citations": [{"chunk_id": "doc1_0", "doc_title": "Basel III", "quote": "CET1 4.5%"}],
        "confidence": "high",
        "is_answerable": True,
    }
    monkeypatch.setattr(llm.requests, "post", lambda *a, **kw: FakeOllamaResponse(json.dumps(payload)))

    result = llm._generate_answer_ollama("CET1?", make_retrieved(vector_score=0.6))

    assert result.answer == "4.5% RWA"
    assert result.citations[0].chunk_id == "doc1_0"


@pytest.mark.parametrize("content", ["not json", json.dumps({"answer": "x", "confidence": "very_sure"})])
def test_ollama_invalid_output_raises_llm_output_error(monkeypatch, content):
    monkeypatch.setattr(llm.requests, "post", lambda *a, **kw: FakeOllamaResponse(content))

    with pytest.raises(llm.LLMOutputError):
        llm._generate_answer_ollama("CET1?", make_retrieved(vector_score=0.6))


def test_gate_uses_best_vector_score_not_first_fused_result(monkeypatch):
    # Top fused hit came from BM25 only (vector_score 0.0); a lower-ranked chunk is semantically relevant.
    bm25_only, relevant = make_retrieved(vector_score=0.0) + make_retrieved(vector_score=0.6)
    sentinel = object()
    monkeypatch.setattr(llm, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(llm, "_generate_answer_ollama", lambda q, r: sentinel)

    assert llm.generate_answer("CET1?", [bm25_only, relevant]) is sentinel
