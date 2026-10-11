# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
# Сначала ставим CPU-сборку torch: колесо с PyPI для linux/aarch64 тянет весь
# стек NVIDIA CUDA (3–4 ГБ), который в контейнере без GPU не нужен.
# sentence-transformers затем использует уже установленный torch.
# Увеличенные таймаут и число повторов — для больших колёс на медленной сети.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout=180 --retries=8 torch --index-url https://download.pytorch.org/whl/cpu
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout=180 --retries=8 -r requirements.txt

COPY src ./src
COPY api ./api
COPY data/raw ./data/raw
COPY streamlit_app.py .

# Индексы собираются при сборке образа, поэтому контейнер сразу готов отвечать.
RUN python -m src.build_index

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
