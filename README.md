# Legal RAG

A retrieval-augmented question-answering system for **Bulgarian law**. Ask a natural-language question in Bulgarian or English; the system retrieves relevant articles from the statutes it has indexed and generates an LLM-grounded answer with citations.

Designed to be consumed both by humans (via HTTP) and by **agents over MCP** — answers are concise, structured, and use a stable citation format (`[Чл. X, <law_id>]`) so callers can branch on machine-parseable output.

## Scope

Currently indexes three statutes scraped from [lex.bg](https://lex.bg):

- **Кодекс на труда** (Labour Code)
- **Закон за защита на потребителите** (Consumer Protection Act)
- **Закон за задълженията и договорите** (Obligations and Contracts Act)

Adding more laws is a matter of running the scraper for them and dropping the resulting JSON into `documentation/`.

## How it works

```
Question
   │
   ▼
┌─────────────────────────────────────────────────────┐
│  Hybrid Retrieval                                   │
│  ┌──────────────┐    ┌────────────────────────┐     │
│  │ FAISS (k=25) │    │ BM25 (k=25)            │     │
│  │ BGE-M3       │    │ Bulgarian tokenizer    │     │
│  │ embeddings   │    │ (Cyrillic + suffixes)  │     │
│  └──────┬───────┘    └────────────┬───────────┘     │
│         └───────┬───────────────────┘               │
│                 ▼                                   │
│      Reciprocal Rank Fusion (k=60)                  │
│                 │                                   │
│                 ▼                                   │
│   Cross-encoder rerank (BGE-Reranker-v2-M3)         │
│   sigmoid → [0,1] → threshold filter → top 5        │
└─────────────────────────┬───────────────────────────┘
                          ▼
                  ┌───────────────┐
                  │   LLM         │
                  │ Claude        │  ← grounded prompt
                  │ (or Ollama)   │     with citations
                  └───────┬───────┘
                          ▼
                       Answer
```

1. **Ingest** — JSON articles are loaded and chunked at paragraph/word boundaries (`CHUNK_MAX_CHARS=600`, `CHUNK_OVERLAP_CHARS=80`). Law/chapter/article metadata is preserved per chunk.
2. **Index** — FAISS vector index (BGE-M3 embeddings) + BM25 sparse index with a Bulgarian-aware tokenizer (Cyrillic-lowercase + light suffix stripping). Both are cached to `backend/.cache/indices/`.
3. **Retrieve** — Top-25 from each index, fused via Reciprocal Rank Fusion, then reranked by a multilingual cross-encoder. Raw cross-encoder logits are sigmoid-normalized to `[0, 1]` for a model-independent threshold.
4. **Generate** — The top reranked chunks are stuffed into a strict, grounding-focused prompt and sent to Claude (or Ollama). Output uses inline `[Чл. X, <law_id>]` citations and emits an explicit `INSUFFICIENT_CONTEXT:` signal when retrieval comes up short.

## Repository layout

```
legal_rag/
├── README.md                  # this file
├── CLAUDE.md                  # project-specific instructions for Claude Code
├── documentation/             # scraped law JSON files (input data)
│   ├── kodeks_na_truda.json
│   ├── zakon_za_zashtita_na_potrebitelite.json
│   └── zakon_za_zadalzheniqta_i_dogovorite.json
├── client/                    # (frontend, separate)
└── backend/
    ├── main.py                # FastAPI app: /api/health, /api/query, /api/query/stream
    ├── config.py              # All config + env vars (single source of truth)
    ├── templates.py           # System prompt for the LLM
    ├── pyproject.toml         # Dependencies
    ├── .env.example           # Copy to .env and fill in
    ├── models/
    │   └── rag_pipeline.py    # Orchestrator: load → chunk → index → query
    ├── rag/
    │   ├── chunker.py         # Article-aware chunker
    │   ├── retrieval.py       # HybridRetriever: FAISS + BM25 → RRF → rerank
    │   ├── reranker.py        # CrossEncoder wrapper (auto MPS/CUDA/CPU, sigmoid-normalized)
    │   ├── tokenization.py    # Bulgarian BM25 tokenizer
    │   └── generation.py      # LLM answer generation (Claude or Ollama, streaming + non-streaming)
    ├── scripts/
    │   ├── scrape_laws.py     # One-shot scraper: lex.bg HTML → structured JSON
    │   └── clear_cache.py     # Wipe .cache/indices/ to force reindex
    └── tests/
```

## Setup

### Prerequisites

- **Python 3.11+**
- **~5 GB free disk** for the embedding + reranker models (downloaded on first run, cached under `~/.cache/huggingface/`)
- **Claude API access** (default) or a local **Ollama** install (fallback)

### Install

```bash
cd backend
uv sync
source .venv/bin/activate
```

`uv sync` creates `.venv/` and installs everything from `pyproject.toml` (including the `dev` extras) in one step.

### Configure

```bash
cp .env.example .env
```

Edit `.env`:

| Var                        | Default                       | Notes                                                     |
| -------------------------- | ----------------------------- | --------------------------------------------------------- |
| `USE_OLLAMA`               | `false`                       | Set `true` to use Ollama instead of Claude                |
| `CLAUDE_API_KEY`           | —                             | Required if `USE_OLLAMA=false`                            |
| `CLAUDE_MODEL`             | `anthropic--claude-4.5-haiku` | Any Claude model your endpoint accepts                    |
| `CLAUDE_URL`               | `http://localhost:6655`       | Anthropic-compatible endpoint                             |
| `OLLAMA_HOST`              | `http://localhost:11434`      | Used when `USE_OLLAMA=true`                               |
| `OLLAMA_MODEL`             | `llama3`                      | Pick a model that handles Bulgarian well                  |
| `EMBEDDING_MODEL`          | `BAAI/bge-m3`                 | Multilingual; downloads on first run (~2.3 GB)            |
| `RERANKER_MODEL`           | `BAAI/BGE-Reranker-v2-M3`     | Multilingual; downloads on first run (~2.3 GB)            |
| `CHUNK_MAX_CHARS`          | `600`                         |                                                           |
| `CHUNK_OVERLAP_CHARS`      | `80`                          |                                                           |
| `VECTOR_RETRIEVAL_K`       | `25`                          | FAISS top-k before fusion                                 |
| `BM25_RETRIEVAL_K`         | `25`                          | BM25 top-k before fusion                                  |
| `RERANKER_TOP_N`           | `5`                           | Final number of chunks sent to the LLM                    |
| `RERANKER_SCORE_THRESHOLD` | `0.4`                         | Sigmoid-normalized score in `[0, 1]`                      |
| `MIN_RETRIEVED_DOCS`       | `1`                           | Don't return empty even if all scores are below threshold |

### Get the data

If `documentation/` is empty:

```bash
cd backend
python scripts/scrape_laws.py
```

This pulls from lex.bg and writes JSON files to `../documentation/`.

## Run

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**First boot is slow** — the app downloads BGE-M3 + BGE-Reranker-v2-M3 (~4.5 GB combined), then builds the FAISS and BM25 indices from the source JSONs. Subsequent boots load everything from cache and start in seconds.

To pre-warm the model cache before launching:

```bash
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('BAAI/bge-m3'); snapshot_download('BAAI/bge-reranker-v2-m3')"
```

## API

### Health

```bash
curl http://localhost:8000/api/health
```

```json
{ "status": "ok", "message": "server is running", "pipeline_loaded": true }
```

### Query (non-streaming)

```bash
curl -X POST http://localhost:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "Колко дни е платеният годишен отпуск?", "top_n": 5}'
```

```json
{
  "question": "Колко дни е платеният годишен отпуск?",
  "answer": "Минималният размер на платения годишен отпуск е 20 работни дни [Чл. 155, Кодекс на труда]...",
  "sources": [
    { "rank": 1, "source": "kodeks_na_truda.json", "snippet": "...", "score": 0.92 },
    ...
  ]
}
```

### Query (streaming, SSE)

```bash
curl -X POST http://localhost:8000/api/query/stream \
  -H 'Content-Type: application/json' \
  -d '{"question": "Колко дни е платеният годишен отпуск"}'
```

Stream events:

1. `data: {"type":"metadata","sources":[...],"question":"..."}`
2. `data: {"type":"chunk","content":"..."}` (repeated)
3. `data: {"type":"done"}`

## Cache invalidation

Indices are cached to `backend/.cache/indices/` with a manifest fingerprinted on:

- `EMBEDDING_MODEL`, `RERANKER_MODEL`
- `CHUNK_MAX_CHARS`, `CHUNK_OVERLAP_CHARS`
- The mtimes of every file in `documentation/`

Changing any of these auto-invalidates the cache. **Code changes** to `chunker.py`, `tokenization.py`, etc. are NOT fingerprinted — wipe the cache manually after such changes:

```bash
python scripts/clear_cache.py
```

## Tests

```bash
cd backend
pytest
```

## Agentic use (MCP)

The system prompt in `backend/templates.py` is built for tool-use:

- **Strict grounding** — the model is told to use only the provided context.
- **Explicit out-of-scope signal** — when retrieval is insufficient, the model returns `INSUFFICIENT_CONTEXT: <reason>` so the calling agent can branch.
- **Stable citations** — `[Чл. X, <law_id>]` inline after every claim. Regex-parseable.
- **Language mirroring** — answers in the question's language; legal text quoted verbatim from the source.

To expose this over MCP, wrap `RAGPipeline.query` (or `query_with_contexts` for evaluation) as an MCP tool. The `query_with_contexts` method returns `{question, answer, contexts, sources}` — the shape expected by RAGAS / TruLens-style evaluation harnesses.
