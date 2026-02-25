import hashlib
import json
import os
from typing import Any, Dict, Optional

CACHE_DIR = "cache"


def ensure_cache_dir():
    os.makedirs(CACHE_DIR, exist_ok=True)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def cache_path(key: str, suffix: str = ".json") -> str:
    ensure_cache_dir()
    return os.path.join(CACHE_DIR, f"{key}{suffix}")


def load_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Dict[str, Any]) -> None:
    ensure_cache_dir()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)