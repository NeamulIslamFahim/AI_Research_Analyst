"""Small cache helpers for Research Explorer API responses."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional


class ExplorerCache:
    """Tiny in-memory + disk cache for explorer responses."""

    def __init__(self, cache_file: str, ttl_seconds: int) -> None:
        self.cache_file = cache_file
        self.ttl_seconds = ttl_seconds
        self.memory_cache: Dict[str, Dict[str, Any]] = {}

    def make_key(self, topic: str, focus: Optional[str], use_live: Optional[bool]) -> str:
        return f"{topic.strip().lower()}|{(focus or '').strip().lower()}|{use_live}"

    def load_disk_cache(self) -> Dict[str, Any]:
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r", encoding="utf-8") as file_obj:
                    return json.load(file_obj)
        except Exception:
            pass
        return {}

    def save_disk_cache(self, cache: Dict[str, Any]) -> None:
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as file_obj:
                json.dump(cache, file_obj)
        except Exception:
            pass
