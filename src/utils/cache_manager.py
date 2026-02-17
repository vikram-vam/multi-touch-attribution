"""
Cache manager for pre-computed attribution results.
Uses diskcache to store parameter-grid computations for instant UI retrieval.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

import diskcache
import pandas as pd


def _find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "config").is_dir():
            return current
        current = current.parent
    return Path.cwd()


CACHE_DIR = _find_project_root() / ".cache" / "attribution"


class CacheManager:
    """
    Manages a diskcache-based store for pre-computed attribution results.

    Keys are parameter hashes; values are DataFrames serialized via Parquet bytes.
    """

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = diskcache.Cache(str(self.cache_dir))

    def _make_key(self, params: dict) -> str:
        """Create a deterministic hash key from parameter dict."""
        param_str = json.dumps(params, sort_keys=True, default=str)
        return hashlib.sha256(param_str.encode()).hexdigest()[:16]

    def get(self, params: dict) -> Optional[pd.DataFrame]:
        """Retrieve cached result for given parameters."""
        key = self._make_key(params)
        data = self.cache.get(key)
        if data is not None:
            return pd.read_parquet(pd.io.common.BytesIO(data))
        return None

    def put(self, params: dict, df: pd.DataFrame) -> None:
        """Store a result DataFrame for given parameters."""
        key = self._make_key(params)
        buffer = pd.io.common.BytesIO()
        df.to_parquet(buffer, index=False)
        self.cache.set(key, buffer.getvalue())

    def has(self, params: dict) -> bool:
        """Check if a result exists in cache."""
        key = self._make_key(params)
        return key in self.cache

    def clear(self) -> None:
        """Clear all cached results."""
        self.cache.clear()

    def stats(self) -> dict:
        """Return cache statistics."""
        return {
            "size": len(self.cache),
            "volume_bytes": self.cache.volume(),
            "directory": str(self.cache_dir),
        }
