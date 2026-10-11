import pytest
from fastapi.testclient import TestClient

from api.main import app, pipeline
from src.schemas import QueryResponse


def test_health_reports_index_state(monkeypatch):
    monkeypatch.setattr(pipeline, "_loaded", False)
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "index_loaded": False}


def test_query_returns_503_when_index_not_loaded(monkeypatch):
    monkeypatch.setattr(pipeline, "_loaded", False)
    client = TestClient(app)

    response = client.post("/query", json={"question": "What is Basel III?"})

    assert response.status_code == 503


def test_query_rejects_empty_question(monkeypatch):
    monkeypatch.setattr(pipeline, "_loaded", True)
    client = TestClient(app)

    response = client.post("/query", json={"question": "   "})

    assert response.status_code == 400


@pytest.mark.parametrize("top_k", [0, -2, 21])
def test_query_rejects_out_of_range_top_k(monkeypatch, top_k):
    monkeypatch.setattr(pipeline, "_loaded", True)
    client = TestClient(app)

    response = client.post("/query", json={"question": "CET1?", "top_k": top_k})

    assert response.status_code == 422


def test_query_returns_answer_when_loaded(monkeypatch):
    monkeypatch.setattr(pipeline, "_loaded", True)

    fake_response = QueryResponse(
        question="What is the minimum CET1 ratio?",
        answer="4.5% of RWA.",
        citations=[],
        confidence="high",
        is_answerable=True,
        retrieved_chunk_ids=["doc1_0"],
    )
    monkeypatch.setattr(pipeline, "answer", lambda question, top_k=None: fake_response)

    client = TestClient(app)
    response = client.post("/query", json={"question": "What is the minimum CET1 ratio?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "4.5% of RWA."
    assert body["is_answerable"] is True


def test_query_maps_invalid_llm_output_to_502(monkeypatch):
    from src.llm import LLMOutputError

    monkeypatch.setattr(pipeline, "_loaded", True)

    def raise_output_error(question, top_k=None):
        raise LLMOutputError("bad json")

    monkeypatch.setattr(pipeline, "answer", raise_output_error)

    client = TestClient(app)
    response = client.post("/query", json={"question": "What is the minimum CET1 ratio?"})

    assert response.status_code == 502
