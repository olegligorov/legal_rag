"""
Configuration file for RAG system
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = os.path.join(BASE_DIR, "documentation")
CACHE_DIR = os.path.join(BASE_DIR, "backend", ".cache", "indices")

# Models
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_TEMPERATURE = 0.2

# OLLAMA
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = "llama3"
USE_OLLAMA = False

# Claude API Keys
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "anthropic--claude-4.5-haiku")
CLAUDE_URL = os.getenv("CLAUDE_URL", "http://localhost:6655")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

# Retrieval
VECTOR_RETRIEVAL_K = 25
BM25_RETRIEVAL_K = 25

# Reranker
RERANKER_TOP_N = 5
RERANKER_SCORE_THRESHOLD = 0.2
MIN_RETRIEVED_DOCS = 1

# LLM_MODEL = "llama3" TODO check this?
