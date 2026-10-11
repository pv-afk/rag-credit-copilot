from contextlib import asynccontextmanager

import anthropic
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config import FAISS_INDEX_PATH
from src.embeddings import get_embedding_model
from src.llm import LLMOutputError
from src.pipeline import RAGPipeline
from src.schemas import QueryRequest, QueryResponse

pipeline = RAGPipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if FAISS_INDEX_PATH.exists():
        pipeline.load()
        get_embedding_model()  # загружаем модель заранее, иначе первый запрос ждёт её ~10 с
    yield


app = FastAPI(
    title="RAG Copilot для кредитного аналитика",
    description=(
        "Ответы на вопросы на основе публичной базы знаний по кредитному риску "
        "(Базель III, МСФО 9, методология рейтингов, ковенанты, стресс-тестирование и т.д.), "
        "с гибридным (BM25 + векторным) поиском и цитируемыми структурированными ответами."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "index_loaded": pipeline.is_loaded}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if not pipeline.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Индекс не загружен. Сначала выполните `python -m src.build_index`.",
        )
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Поле `question` не должно быть пустым.")
    try:
        return pipeline.answer(request.question, top_k=request.top_k)
    except anthropic.AuthenticationError as e:
        raise HTTPException(
            status_code=502,
            detail="Ошибка аутентификации Anthropic API. Проверьте ANTHROPIC_API_KEY.",
        ) from e
    except TypeError as e:
        # Без ключа SDK бросает TypeError («Could not resolve authentication method») ещё до запроса.
        if "authentication" in str(e).lower():
            raise HTTPException(
                status_code=502,
                detail="ANTHROPIC_API_KEY не задан. Экспортируйте его или добавьте в .env перед запросом.",
            ) from e
        raise
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"Ошибка Anthropic API: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise HTTPException(status_code=502, detail="Не удалось связаться с Anthropic API.") from e
    except requests.ConnectionError as e:
        raise HTTPException(
            status_code=502,
            detail="Не удалось связаться с Ollama. Она запущена? Попробуйте `ollama serve` "
            "или `brew services start ollama`.",
        ) from e
    except requests.Timeout as e:
        raise HTTPException(status_code=504, detail="Превышено время ожидания ответа от Ollama.") from e
    except requests.HTTPError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama вернула ошибку: {e}. Проверьте, что модель скачана (`ollama pull <model>`).",
        ) from e
    except LLMOutputError as e:
        raise HTTPException(status_code=502, detail=f"LLM вернула некорректный структурированный ответ: {e}") from e
