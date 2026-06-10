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

# Retrieval
VECTOR_RETRIEVAL_K = 25
BM25_RETRIEVAL_K = 25

# Reranker
RERANKER_TOP_N = 5
RERANKER_SCORE_THRESHOLD = 0.2
MIN_RETRIEVED_DOCS = 1

# LLM_MODEL = "llama3" TODO check this?
