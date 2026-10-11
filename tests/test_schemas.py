import pytest
from pydantic import ValidationError

from src.schemas import AnswerResponse, Citation


def test_answer_response_valid():
    response = AnswerResponse(
        answer="Minimum CET1 is 4.5% of RWA.",
        citations=[Citation(chunk_id="doc1_0", doc_title="Basel III", quote="minimum 4.5% of RWA")],
        confidence="high",
        is_answerable=True,
    )
    assert response.is_answerable is True
    assert response.citations[0].chunk_id == "doc1_0"


def test_answer_response_rejects_invalid_confidence():
    with pytest.raises(ValidationError):
        AnswerResponse(
            answer="...",
            citations=[],
            confidence="very_confident",  # not in the allowed literal set
            is_answerable=True,
        )


def test_answer_response_allows_empty_citations_when_not_answerable():
    response = AnswerResponse(
        answer="I don't know.",
        citations=[],
        confidence="insufficient_context",
        is_answerable=False,
    )
    assert response.citations == []
