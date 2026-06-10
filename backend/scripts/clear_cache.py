#!/usr/bin/env python
"""
Utility script to clear cached indices.

Run this when:
- Source documents have been updated
- Chunking parameters have changed
- You want to force rebuild indices

Usage:
    python clear_cache.py
"""

import shutil
from pathlib import Path

from config import CACHE_DIR

if __name__ == "__main__":
    print("Clearing cached indices...")

    cache_dir = Path(CACHE_DIR)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Cache cleared successfully at {cache_dir}")
    else:
        print(f"No cache found at {cache_dir}")

    print("Next server startup will rebuild indices from scratch.")
