import json

import anthropic
import requests
from pydantic import ValidationError

from src.config import ANTHROPIC_MODEL, LLM_PROVIDER, MIN_RELEVANCE_SCORE, OLLAMA_BASE_URL, OLLAMA_MODEL
from src.schemas import AnswerResponse, RetrievedChunk

SYSTEM_PROMPT = """Ты — RAG-ассистент, который помогает кредитным аналитикам отвечать на вопросы, используя \
фиксированную базу знаний из публичных документов (Базель III, МСФО 9, методология рейтингов, ковенанты, \
стресс-тестирование и т.д.).

Правила:
- Отвечай строго на основе предоставленных фрагментов контекста. Не используй знания вне контекста.
- Каждое фактическое утверждение в ответе должно подкрепляться минимум одной цитатой на chunk_id из контекста \
с короткой дословной цитатой в его поддержку.
- Если в контексте недостаточно информации для ответа на вопрос, установи is_answerable в false, \
confidence в "insufficient_context", оставь citations пустым и укажи в ответе, что база знаний не содержит \
информации по этому вопросу.
- Отвечай кратко и точно, на русском языке. Это профессиональный кредитный анализ, а не общий чат.
- Верни один JSON-объект, соответствующий требуемой схеме. Никакого текста вне JSON."""


class LLMOutputError(RuntimeError):
    """The LLM responded, but not with a valid AnswerResponse."""


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    parts = []
    for rc in chunks:
        parts.append(f"[chunk_id={rc.chunk.chunk_id}] ({rc.chunk.title})\n{rc.chunk.text}")
    return "\n\n---\n\n".join(parts)


def _build_user_message(question: str, retrieved: list[RetrievedChunk]) -> str:
    context_block = _build_context_block(retrieved)
    return (
        f"Фрагменты контекста:\n\n{context_block}\n\n---\n\nВопрос: {question}\n\n"
        "Отвечай строго на основе контекста выше, на русском языке. Указывай chunk_id для каждого утверждения. "
        "Если контекста недостаточно, скажи об этом явно."
    )


def _insufficient_context_response() -> AnswerResponse:
    return AnswerResponse(
        answer="Не знаю. База знаний не содержит информации, релевантной этому вопросу.",
        citations=[],
        confidence="insufficient_context",
        is_answerable=False,
    )


def _generate_answer_anthropic(question: str, retrieved: list[RetrievedChunk]) -> AnswerResponse:
    client = anthropic.Anthropic()
    response = client.messages.parse(
        model=ANTHROPIC_MODEL,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_message(question, retrieved)}],
        output_format=AnswerResponse,
    )
    if response.parsed_output is None:
        # e.g. stop_reason == "refusal" or "max_tokens": no schema-valid JSON came back.
        raise LLMOutputError(f"Model returned no structured output (stop_reason={response.stop_reason})")
    return response.parsed_output


def _generate_answer_ollama(question: str, retrieved: list[RetrievedChunk]) -> AnswerResponse:
    """Free, local generation via Ollama. No API key, no external calls."""
    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_message(question, retrieved)},
            ],
            "format": AnswerResponse.model_json_schema(),
            "stream": False,
            "options": {"temperature": 0.0},
        },
        timeout=180,
    )
    resp.raise_for_status()
    content = resp.json()["message"]["content"]
    try:
        return AnswerResponse.model_validate(json.loads(content))
    except (json.JSONDecodeError, ValidationError) as e:
        raise LLMOutputError(f"Ollama returned output that does not match AnswerResponse: {e}") from e


def top_vector_score(retrieved: list[RetrievedChunk]) -> float:
    # retrieved[0] is the top *fused* result, which may be a BM25-only hit with
    # vector_score == 0.0, so gate on the best cosine similarity instead.
    return max((rc.vector_score for rc in retrieved), default=0.0)


def generate_answer(question: str, retrieved: list[RetrievedChunk]) -> AnswerResponse:
    if top_vector_score(retrieved) < MIN_RELEVANCE_SCORE:
        return _insufficient_context_response()

    if LLM_PROVIDER == "ollama":
        return _generate_answer_ollama(question, retrieved)
    return _generate_answer_anthropic(question, retrieved)
