# syntax=docker/dockerfile:1
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install CPU-only torch first: the default PyPI wheel for linux/aarch64 pulls
# in the full NVIDIA CUDA stack (~3-4 GB of nvidia_* packages) that's useless
# here (no GPU in this container) and just slows/breaks the build. Once torch
# is satisfied, sentence-transformers in requirements.txt won't re-resolve it.
# Large wheels can also exceed pip's default 15s socket timeout on slow
# connections, so give installs more room and retries.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout=180 --retries=8 torch --index-url https://download.pytorch.org/whl/cpu
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --timeout=180 --retries=8 -r requirements.txt

COPY src ./src
COPY api ./api
COPY data/raw ./data/raw
COPY streamlit_app.py .

# Pre-build the retrieval indexes at image build time so the container
# starts ready to serve queries.
RUN python -m src.build_index

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
