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
- **Reranker:** `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (sentence_transformers); raw logits sigmoid-normalized to 0..1 before thresholding
- **LLM:** Claude API (default) or Ollama (fallback, toggled by `USE_OLLAMA` in config)
- **Data:** JSON files in `documentation/` (sibling of `backend/`), produced by `scripts/scrape_laws.py`

## Key files
```
legal_rag/
├── documentation/          # Scraped law JSON files (input to pipeline)
├── backend/
│   ├── main.py             # FastAPI app: /api/health, /api/query, /api/query/stream
│   ├── config.py           # All constants and env vars (single source of truth)
│   ├── templates.py        # System prompt — NOT YET IMPLEMENTED (empty string)
│   ├── models/
│   │   └── rag_pipeline.py # Pipeline orchestrator: load → chunk → index → query
│   ├── rag/
│   │   ├── chunker.py      # Splits articles into ≤CHUNK_MAX_CHARS chunks at paragraph/word boundaries with CHUNK_OVERLAP_CHARS overlap. Carries law/chapter/article in metadata only — the prefix is rendered later by Generator's doc_prompt, NOT embedded in page_content.
│   │   ├── retrieval.py    # HybridRetriever: FAISS + BM25 → RRF fusion → cross-encoder rerank
│   │   ├── reranker.py     # CrossEncoder wrapper (auto-selects MPS/CUDA/CPU); sigmoid-normalizes scores to 0..1
│   │   └── generation.py   # Generator: LangChain chain → LLM answer (non-streaming and streaming)
│   └── scripts/
│       ├── scrape_laws.py  # One-shot scraper: fetches lex.bg HTML → structured JSON articles
│       └── clear_cache.py  # Wipes `.cache/indices/`; run after chunker/retrieval logic changes
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

## Pipeline flow
1. `RAGPipeline.__init__`: loads JSON → `Document` objects → `Chunker` → FAISS + BM25 indices (cached)
2. Query: `HybridRetriever.search` (FAISS k=25 + BM25 k=25 → RRF → CrossEncoder top_n=5, threshold=0.2)
3. Generate: `Generator.generate` / `generate_stream` via LangChain `create_stuff_documents_chain`

## Known gaps (to implement)
- `templates.py` — `SYSTEM_TEXT_TEMPLATE` is an empty string; system prompt for the LLM is not written yet
- The system prompt should be domain-specific: instruct the LLM to answer questions about Bulgarian law using only the provided article context, cite articles by number, and respond in the same language as the question
