from contextlib import asynccontextmanager

import anthropic
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config import FAISS_INDEX_PATH
from src.llm import LLMOutputError
from src.pipeline import RAGPipeline
from src.schemas import QueryRequest, QueryResponse

pipeline = RAGPipeline()


@asynccontextmanager
async def lifespan(app: FastAPI):
    if FAISS_INDEX_PATH.exists():
        pipeline.load()
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
    return {"status": "ok", "index_loaded": pipeline._loaded}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    if not pipeline._loaded:
        raise HTTPException(
            status_code=503,
            detail="Индекс не загружен. Сначала выполните `python -m src.build_index`.",
        )
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Поле `question` не должно быть пустым.")
    try:
        return pipeline.answer(request.question, top_k=request.top_k)
    except anthropic.AuthenticationError:
        raise HTTPException(
            status_code=502,
            detail="Ошибка аутентификации Anthropic API. Проверьте ANTHROPIC_API_KEY.",
        )
    except TypeError as e:
        if "authentication" in str(e).lower():
            raise HTTPException(
                status_code=502,
                detail="ANTHROPIC_API_KEY не задан. Экспортируйте его или добавьте в .env перед запросом.",
            )
        raise
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"Ошибка Anthropic API: {e.message}")
    except anthropic.APIConnectionError:
        raise HTTPException(status_code=502, detail="Не удалось связаться с Anthropic API.")
    except requests.ConnectionError:
        raise HTTPException(
            status_code=502,
            detail="Не удалось связаться с Ollama. Она запущена? Попробуйте `ollama serve` "
            "или `brew services start ollama`.",
        )
    except requests.Timeout:
        raise HTTPException(status_code=504, detail="Превышено время ожидания ответа от Ollama.")
    except requests.HTTPError as e:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama вернула ошибку: {e}. Проверьте, что модель скачана (`ollama pull <model>`).",
        )
    except LLMOutputError as e:
        raise HTTPException(status_code=502, detail=f"LLM вернула некорректный структурированный ответ: {e}")
