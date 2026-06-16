# Legal RAG — Project Context

## What this is
Agentic RAG system that answers questions about Bulgarian law. Users ask natural-language questions in Bulgarian or English; the system retrieves relevant articles and generates an LLM answer with citations.

**Current scope:** 3 laws scraped from lex.bg:
- Кодекс на труда (Labour Code)
- Закон за защита на потребителите (Consumer Protection Act)
- Закон за задълженията и договорите (Obligations and Contracts Act)

## Stack
- **Backend:** FastAPI (Python 3.11+), `backend/` is the working directory
- **Embeddings:** `BAAI/bge-m3` via HuggingFace (multilingual, no prefix convention)
- **Vector store:** FAISS (in-memory, cached to disk under `.cache/indices/`)
- **Keyword search:** BM25 via LangChain community, with a Bulgarian-aware tokenizer (Cyrillic-lowercase + light suffix stripping)
- **Reranker:** `BAAI/BGE-Reranker-v2-M3` (sentence_transformers); raw logits sigmoid-normalized to 0..1 before thresholding
- **LLM:** Claude API (default) or Ollama (fallback, toggled by `USE_OLLAMA` in config)
- **Data:** JSON files in `documentation/` (sibling of `backend/`), produced by `scripts/scrape_laws.py`

## Key files
```
legal_rag/
├── documentation/          # Scraped law JSON files (input to pipeline)
├── backend/
│   ├── main.py             # FastAPI app: /api/health, /api/query[/stream], /api/agent[/stream]
│   ├── config.py           # All constants and env vars (single source of truth)
│   ├── templates.py        # System prompt for the baseline RAG generator
│   ├── prompt_rules.py     # Shared LANGUAGE_RULE / CITATION_RULE / ANSWER_SHAPE_RULE — used by both templates.py AND agent/prompts.py
│   ├── citations.py        # Single source of truth for parsing `[Чл. X, <law_id>]` markers (NFC, letter-suffix-aware). Imported by agent/runner.py and evaluation/metrics.py.
│   ├── models/
│   │   └── rag_pipeline.py # Pipeline orchestrator: load → chunk → index → query
│   ├── rag/
│   │   ├── chunker.py      # Splits articles into ≤CHUNK_MAX_CHARS chunks at paragraph/word boundaries with CHUNK_OVERLAP_CHARS overlap. Carries law/chapter/article in metadata only — the prefix is rendered later by Generator's doc_prompt, NOT embedded in page_content.
│   │   ├── retrieval.py    # HybridRetriever: FAISS + BM25 → RRF fusion → cross-encoder rerank
│   │   ├── reranker.py     # CrossEncoder wrapper (auto-selects MPS/CUDA/CPU); sigmoid-normalizes scores to 0..1
│   │   └── generation.py   # Generator: LangChain chain → LLM answer (non-streaming and streaming)
│   ├── agent/              # LangGraph ReAct agent over MCP (the "agentic" arm of the thesis)
│   │   ├── graph.py        # StateGraph: call_model → tools → tick (with hard cap + force_synthesize escape hatch)
│   │   ├── runner.py       # run_agent / stream_agent entrypoints; _SourceCollector dedup (by law_id+article); citation passthrough validation (preserved/dropped/invented)
│   │   ├── mcp_client.py   # Singleton MultiServerMCPClient (SSE → localhost:8001/sse). AGENT_ALLOWED_TOOLS gate.
│   │   └── prompts.py      # REACT_SYSTEM_PROMPT and FORCE_SYNTHESIS_PROMPT (composed from prompt_rules.py)
│   ├── evaluation/         # Head-to-head harness: baseline vs agent
│   │   ├── evaluator.py        # Baseline pipeline runner (rank-aware precision@k / recall@k)
│   │   ├── agent_evaluator.py  # Agent runner (set_recall@k, since agent source order isn't a ranking)
│   │   ├── metrics.py          # citation_recall/precision (None on no-citations), faithfulness, answer_*. Backward-compat aliases for tests.
│   │   ├── aggregation.py      # {n,mean,median,p95,min,max,stdev,sum} per metric; deltas() table
│   │   ├── dataset.json        # Single-hop questions
│   │   └── dataset_multihop.json # 15 multi-hop questions across multi-article-same-law / cross-law / sequential-dependency
│   ├── scripts/
│   │   ├── scrape_laws.py      # One-shot scraper: fetches lex.bg HTML → structured JSON articles
│   │   ├── clear_cache.py      # Wipes `.cache/indices/`; run after chunker/retrieval logic changes
│   │   └── run_head_to_head.py # Eval entrypoint: writes results/head_to_head_<ts>.json with full reproducibility envelope (git SHA, dataset SHA-256, all model/retrieval knobs)
│   └── tests/                  # 101 unit tests: test_agent.py, test_metrics.py, test_dataset.py, …
└── mcp/                    # Standalone FastMCP server: thin HTTP client over the backend
    └── src/mcp_server.py   # Tools: query_rag_tool, batch_query_tool, retrieve_documents, check_rag_health
```

## Data format
Each JSON in `documentation/` is a list of article objects:
```json
{
  "law_id": "Кодекс на труда",
  "chapter": "...",
  "section": "...",
  "article": "Чл. 1",
  "title": "...",
  "content": "full article text",
  "cross_references": ["Чл. 2", ...]
}
```

## Pipeline flow (baseline RAG)
1. `RAGPipeline.__init__`: loads JSON → `Document` objects → `Chunker` → FAISS + BM25 indices (cached)
2. Query: `HybridRetriever.search` (FAISS k=25 + BM25 k=25 → RRF → CrossEncoder top_n=5, threshold from `RERANKER_SCORE_THRESHOLD`)
3. Generate: `Generator.generate` / `generate_stream` via LangChain `create_stuff_documents_chain`

## Environment gotchas
- Python stdout is full-buffered when piped through `tee`. Always run scripts with `python -u …` for live progress; per-question prints already use `flush=True` but stack traces still buffer.

## Agent flow (LangGraph ReAct over MCP)
1. `run_agent(question)` builds initial state `{messages: [HumanMessage], tool_calls_used: 0}`.
2. Graph: `START → call_model → after_model → {tools | force_synthesize | END}; tools → tick → after_tick → {call_model | force_synthesize}`.
3. Hard cap: `AGENT_MAX_TOOL_CALLS=6`. On cap, `force_synthesize` fakes "budget exhausted" `ToolMessage`s for any pending tool_use blocks (Anthropic 400-proofing) and re-prompts the model with `FORCE_SYNTHESIS_PROMPT`.
4. Repeated-tool-call detector (`LOOP_DETECTOR_WINDOW=4`) short-circuits to force-synthesis when the same `(name, args)` pair appears twice within the window.
5. Output: `{question, answer, sources, trace, tool_calls_used, citation_validation}`. `citation_validation` is preserved/dropped/invented diff between sub-answer markers and final-answer markers.
6. Sources are deduplicated cross-call by `(law_id, article)` whenever EITHER is non-empty; otherwise by `(source_file, snippet[:80])`. Order is first-appearance, NOT a relevance ranking — hence the agent eval reports `set_recall@k` rather than `precision@k`.

## Evaluation
- `scripts/run_head_to_head.py` runs both pipelines on the same dataset and writes `results/head_to_head_<ts>.json`.
- Output JSON includes a `config` envelope (model, temperature, retrieval params, MCP URL, dataset path + SHA-256, git SHA, git dirty flag, timestamp) so any run is reproducible.
- Aggregates per metric: `{n, mean, median, p95, min, max, stdev, sum}` — `None` values are skipped (e.g. `citation_precision` when no citations were emitted, `faithfulness` when no contexts).
- Apples-to-apples comparison is on `recall@k` / `set_recall@k` and citation metrics. Baseline-only: `precision@k`. Agent-only: `set_recall@k`, `tool_calls_used`, `citations_preserved/dropped/invented`.
- Loads dataset once and slices in memory (no temp files). Run unbuffered for live progress: `python -u scripts/run_head_to_head.py --dataset evaluation/dataset_multihop.json`.

