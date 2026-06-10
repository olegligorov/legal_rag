"""
Configuration file for RAG system
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# Base paths
BASE_DIR = Path(__file__).resolve().parent  # backend/
DATA_PATH = BASE_DIR.parent / "documentation"
CACHE_DIR = BASE_DIR / ".cache" / "indices"

# Models
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/BGE-Reranker-v2-M3")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))

# OLLAMA
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# Claude API Keys
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "anthropic--claude-4.5-haiku")
CLAUDE_URL = os.getenv("CLAUDE_URL", "http://localhost:6655")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

# Retrieval
VECTOR_RETRIEVAL_K = int(os.getenv("VECTOR_RETRIEVAL_K", "25"))
BM25_RETRIEVAL_K = int(os.getenv("BM25_RETRIEVAL_K", "25"))

# Chunking
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "600"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "80"))

# Reranker
# Score is sigmoid-normalized to [0, 1] in Reranker.rerank_with_scores; 0.5 is neutral.
RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", "5"))
RERANKER_SCORE_THRESHOLD = float(os.getenv("RERANKER_SCORE_THRESHOLD", "0.4"))
MIN_RETRIEVED_DOCS = int(os.getenv("MIN_RETRIEVED_DOCS", "1"))
