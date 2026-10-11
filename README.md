# RAG Credit Copilot

A retrieval-augmented question-answering system for credit analysts, built end to end:
document ingestion → chunking → dense + sparse retrieval → rank fusion → grounded,
citation-backed structured generation → retrieval evaluation.

The knowledge base and the queries are in **Russian**, which drives several design choices
below (multilingual embeddings, a Cyrillic-aware BM25 tokenizer, a Russian system prompt).

> **About the data.** The 14 documents in `data/raw/` are **synthetic** educational texts
> on public credit-risk topics (Basel III, IFRS 9, PD/LGD/EAD, rating methodology,
> covenants, collateral, stress testing, early-warning indicators, restructuring,
> syndicated lending, sovereign and counterparty risk, KYC/AML). They were written for
> this project and contain no internal bank documents, client data, or proprietary code.
> `scripts/generate_docs.py` shows how they were rendered from `data/sources/*.md` into
> a mixed PDF/HTML corpus to exercise the ingestion layer.

---

## Project Overview

The goal is a small but complete RAG system where each stage is explicit, inspectable,
and testable, with no orchestration framework hiding the mechanics. Given a question, the
system:

1. retrieves candidate chunks with **FAISS** (cosine similarity over sentence embeddings)
   and **BM25** in parallel,
2. merges both rankings with **Reciprocal Rank Fusion (RRF)**,
3. **refuses to answer** ("I don't know") if no retrieved chunk is semantically close
   enough to the question, without calling the LLM,
4. otherwise asks an LLM to answer **only from the retrieved context**, returning a
   schema-validated JSON object with per-claim citations (`chunk_id` + verbatim quote)
   and a confidence label.

Generation runs on either the **Claude API** or a **local model via Ollama**, selected by
one environment variable. Retrieval runs fully locally and needs no API key.

## Architecture

```
data/raw/*.pdf, *.html
        │
        ▼
 ingestion.py ─────── pypdf / BeautifulSoup → Document(doc_id, title, text, source_path)
        │
        ▼
 chunking.py ──────── sliding window: 220 words, 40-word overlap
        │
        ├─────────────────────────────┐
        ▼                             ▼
 embeddings.py                    bm25_index.py
 multilingual MiniLM-L12 (384-d)  BM25Okapi, Cyrillic-aware tokenizer
        │                             │
        ▼                             │
 vectorstore.py                       │
 FAISS IndexFlatIP (L2-normalized     │
 vectors → cosine similarity)         │
        │   top-8                     │   top-8
        └──────────► hybrid.py ◄──────┘
                 Reciprocal Rank Fusion (k=60) → top-5
                         │
                         ▼
                      llm.py
        relevance gate: max cosine < MIN_RELEVANCE_SCORE → "I don't know"
        else: Claude (messages.parse + Pydantic schema)
              or Ollama (/api/chat with JSON-schema-constrained decoding)
                         │
                         ▼
                pipeline.py → QueryResponse
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
     api/main.py (FastAPI)    streamlit_app.py (UI, calls the API)
```

Build-time artifacts (`data/processed/`: `chunks.jsonl`, `faiss.index`, `faiss.ids.json`,
`bm25.pkl`) are produced by `python -m src.build_index` and are not committed.

## Features

**Ingestion**
- PDF text extraction (`pypdf`) and HTML parsing (`BeautifulSoup`, `<script>`/`<style>` stripped).
- Whitespace normalization; title taken from `<title>`/`<h1>` (HTML) or the first non-empty line (PDF).
- Stable document IDs (SHA-1 of the file name). Source paths are stored relative to the repo root.

**Chunking**
- Word-based sliding window (default 220 words, 40-word overlap) over paragraph-split text, preserving paragraph breaks.
- Each chunk carries `doc_id`, `title`, `source_path`, and `chunk_index`; IDs look like `<doc_id>_<n>`.

**Retrieval**
- Dense: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, normalized embeddings, exact search with `faiss.IndexFlatIP`.
- Sparse: `rank_bm25.BM25Okapi` with a regex tokenizer that keeps Latin and Cyrillic word tokens (no stemming or lemmatization).
- Fusion: RRF, `score(d) = Σ 1 / (k + rank(d) + 1)` over both lists, `k = 60`. RRF uses ranks only, so the incompatible scales of cosine similarity and BM25 scores don't need calibrating.

**Answerability gate**
- RRF scores are rank-based and carry no absolute relevance signal, so the gate uses the **best raw cosine similarity** among the retrieved chunks instead. Below `MIN_RELEVANCE_SCORE` (default `0.48`), the pipeline returns `is_answerable=false` without an LLM call.

**Generation**
- System prompt restricts answers to the provided context and requires a `chunk_id` citation with a short verbatim quote for each factual claim.
- Output schema (`AnswerResponse`): `answer`, `citations[]`, `confidence ∈ {high, medium, low, insufficient_context}`, `is_answerable`.
- Claude: `client.messages.parse(..., output_format=AnswerResponse)` returns a validated Pydantic object.
- Ollama: `format=<JSON schema>` (constrained decoding), `temperature=0`, then Pydantic validation.
- Malformed or missing structured output raises `LLMOutputError`, which the API maps to HTTP 502.

**Serving**
- FastAPI: `POST /query`, `GET /health`, OpenAPI docs at `/docs`. Typed error mapping: 503 (index not built), 400 (empty question), 502 (auth, provider, or invalid-output errors), 504 (Ollama timeout).
- Streamlit UI with a Top-K slider, a confidence label, a citation list, and retrieved chunk IDs for debugging.
- Docker Compose with two services (API + UI). Indexes are built into the image at build time.

## Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.11+ |
| Parsing | `pypdf`, `beautifulsoup4` |
| Embeddings | `sentence-transformers` (multilingual MiniLM-L12, 384-d), PyTorch CPU |
| Vector search | `faiss-cpu` (`IndexFlatIP`) |
| Lexical search | `rank-bm25` |
| LLM | Anthropic Claude API (`anthropic` SDK), or Ollama (default `llama3.1:8b`) |
| Schemas | Pydantic v2 |
| API / UI | FastAPI + Uvicorn, Streamlit |
| Testing | pytest, FastAPI `TestClient`; GitHub Actions CI |
| Packaging | Docker, Docker Compose |

## Installation

```bash
git clone https://github.com/pv-afk/rag-credit-copilot.git
cd rag-credit-copilot

python -m venv .venv
source .venv/bin/activate

# Optional but recommended on machines without a GPU: install CPU-only PyTorch first
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt     # runtime + test dependencies

cp .env.example .env                    # then edit as needed (see Configuration)
python -m src.build_index               # parse docs, embed, build FAISS + BM25 indexes
```

The first index build downloads the embedding model (~0.5 GB of weights) from the Hugging Face Hub.

### Docker

```bash
cp .env.example .env
docker compose up --build
# API:       http://localhost:8000/docs
# Streamlit: http://localhost:8501
```

With `LLM_PROVIDER=ollama`, the API container reaches the host's Ollama at
`host.docker.internal:11434`.

## Configuration

All settings are environment variables, read from `.env` for native runs (via
`python-dotenv`) and passed through by `docker-compose.yml` for container runs.
Variables already set in the environment take precedence over `.env`.

| Variable | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | `anthropic` (Claude API) or `ollama` (local) |
| `ANTHROPIC_API_KEY` | — | Required only for `LLM_PROVIDER=anthropic` |
| `ANTHROPIC_MODEL` | `claude-opus-4-8` | Claude model ID |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint (native runs) |
| `OLLAMA_MODEL` | `llama3.1:8b` | Local model name |
| `EMBEDDING_MODEL_NAME` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Embedding model |
| `CHUNK_SIZE_WORDS` / `CHUNK_OVERLAP_WORDS` | `220` / `40` | Chunking window |
| `TOP_K_VECTOR` / `TOP_K_BM25` | `8` / `8` | Candidates per retriever |
| `TOP_K_FINAL` | `5` | Chunks passed to the LLM after fusion |
| `RRF_K` | `60` | RRF smoothing constant |
| `MIN_RELEVANCE_SCORE` | `0.48` | Cosine threshold for the answerability gate |

Changing the chunking parameters or the embedding model requires re-running `python -m src.build_index`.

**Running fully local, with no API key:**

```bash
# macOS; see https://ollama.com/download for other platforms
brew install ollama && brew services start ollama
ollama pull llama3.1:8b
# in .env:  LLM_PROVIDER=ollama
```

## Usage

```bash
uvicorn api.main:app --reload        # API on :8000
streamlit run streamlit_app.py       # UI on :8501 (in a second terminal)
```

```bash
curl -s localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "Каков минимальный норматив CET1 по Базелю III с учётом буфера консервации капитала?"}'
```

Actual response from a local run (`LLM_PROVIDER=ollama`, `llama3.1:8b`):

```json
{
  "question": "Каков минимальный норматив CET1 по Базелю III с учётом буфера консервации капитала?",
  "answer": "7%",
  "citations": [
    {
      "chunk_id": "44dfa53be049_0",
      "doc_title": "Базель III: требования к достаточности капитала",
      "quote": "Это доводит эффективный минимальный норматив CET1 до 7,0% для банков, желающих избежать ограничений на распределение дивидендов."
    }
  ],
  "confidence": "high",
  "is_answerable": true,
  "retrieved_chunk_ids": ["44dfa53be049_0", "23bf22f06632_1", "bd2d7f43fa04_1", "44dfa53be049_2", "44dfa53be049_1"]
}
```

The request body also accepts an optional `top_k` to override `TOP_K_FINAL`.

## Evaluation

`eval/eval_set.json` holds 24 hand-written questions: 23 in-domain, each labeled with the
source document expected to contain the answer, and 1 out-of-domain question that should
trigger a refusal.

```bash
python -m eval.evaluate --k 5
```

The script measures **document-level Recall@k** of the hybrid retriever: a hit means at
least one of the top-k chunks comes from the expected document. It needs no LLM calls.
For the out-of-domain question, it reports whether the answerability gate would refuse.

Results on the current corpus (14 documents → 31 chunks), default settings:

| Metric | Value |
|---|---|
| Recall@1 | 19/23 = 82.6% |
| Recall@3 | 21/23 = 91.3% |
| Recall@5 | 23/23 = 100% |
| Out-of-domain refusal | 1/1 (top cosine 0.464 vs. threshold 0.48) |

**How to read these numbers.** The corpus is tiny, so k=5 already covers ~16% of all
chunks, and Recall@5 = 100% is a sanity check, not evidence of retrieval quality at scale.
The relevance threshold was calibrated by hand on this same corpus; there is no held-out
split. **Generation quality is not evaluated automatically**: no faithfulness, citation
precision, or answer-correctness metrics exist yet.

## Tests

```bash
pytest
```

23 offline tests (no network, no LLM, no embedding model download):

- `test_chunking.py`: window sizes, overlap, sequential unique IDs.
- `test_hybrid_retrieval.py`: RRF scoring as a pure function.
- `test_ingestion.py`: parses the bundled PDF/HTML corpus and checks for relative paths and unique IDs.
- `test_llm.py`: answerability gate (including BM25-only top hits), Ollama JSON parsing and invalid-output handling.
- `test_schemas.py`: `AnswerResponse` validation.
- `test_api.py`: `/health` and `/query` behavior with a mocked pipeline, including error mapping.

CI runs the suite on every push and pull request (`.github/workflows/tests.yml`).

## Project Structure

```
├── api/main.py              # FastAPI app: /query, /health, error mapping
├── src/
│   ├── config.py            # env-driven settings (+ .env loading)
│   ├── schemas.py           # Pydantic models: Document, Chunk, AnswerResponse, ...
│   ├── ingestion.py         # PDF/HTML → Document
│   ├── chunking.py          # sliding-window chunker
│   ├── chunk_store.py       # JSONL persistence of chunks
│   ├── embeddings.py        # sentence-transformers wrapper (lazy singleton)
│   ├── vectorstore.py       # FAISS index build/search/save/load
│   ├── bm25_index.py        # BM25 index + tokenizer
│   ├── hybrid.py            # RRF + HybridRetriever
│   ├── llm.py               # answerability gate, Claude / Ollama structured generation
│   ├── pipeline.py          # retrieval + generation → QueryResponse
│   └── build_index.py       # CLI: full offline indexing
├── streamlit_app.py         # UI (talks to the API over HTTP)
├── eval/                    # eval_set.json + Recall@k script
├── scripts/generate_docs.py # dev-only: data/sources/*.md → data/raw/*.pdf|html
├── data/sources/            # synthetic markdown sources
├── data/raw/                # ingestion input (generated PDF/HTML)
├── tests/
├── Dockerfile, docker-compose.yml
└── requirements.txt, requirements-dev.txt
```

## Limitations

- **Synthetic, small corpus.** 14 short documents, 31 chunks. The retrieval numbers above don't carry over to a realistic knowledge base.
- **Fragile answerability gate.** A single global cosine threshold separates on-topic queries (~0.50–0.70) from finance-adjacent off-topic queries (~0.46–0.50) by a thin margin. In a manual test, *"Какая рекомендуемая процентная ставка по 30-летней ипотеке?"* scored 0.502, passed the gate, and `llama3.1:8b` answered "20–70%" by misreading a Basel III mortgage risk weight as an interest rate. The prompt-level `is_answerable` instruction did not catch it with the 8B model. A cross-encoder reranker or an LLM-based answerability check would be the next step.
- **No generation evaluation.** Citations are requested and schema-validated, but nothing checks that quotes actually appear in the cited chunk or that the answer is faithful to them.
- **No reranking, query rewriting, or multi-hop retrieval.** Retrieval is a single round of BM25 + dense search + RRF.
- **Lexical search without morphology.** BM25 tokens are raw lowercase word forms. Russian inflection reduces lexical recall, and a lemmatizer or stemmer (e.g. `pymorphy3`, Snowball) would help.
- **Exact, in-memory indexes.** `IndexFlatIP` and pickled BM25 suit this corpus size. There is no incremental indexing, no ANN index, and no persistence beyond local files.
- **Local-model quality.** With `llama3.1:8b`, answers are terse, citations are sometimes generic, and each query took ~1 min on a laptop CPU in testing.
- **Unauthenticated demo API.** CORS is fully open and there is no auth or rate limiting. Not meant for exposure beyond localhost.
