"""
Timing utilities: measure durations and persist simple historical averages
to provide ETA estimates in logs.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

_CACHE_DIR = Path(__file__).resolve().parents[1] / "cache"
_CACHE_DIR.mkdir(exist_ok=True)
_CACHE_FILE = _CACHE_DIR / "timing_cache.json"


def _load():
    try:
        if _CACHE_FILE.exists():
            return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {}


def _save(data: dict):
    try:
        _CACHE_FILE.write_text(json.dumps(data), encoding="utf-8")
    except Exception:
        pass


def record_timing(key: str, duration: float, count: int = 1) -> None:
    """Record a duration (seconds) for a named step. Maintains running average."""
    data = _load()
    entry = data.get(key, {"total": 0.0, "count": 0})
    entry["total"] = entry.get("total", 0.0) + float(duration) * int(count)
    entry["count"] = entry.get("count", 0) + int(count)
    data[key] = entry
    _save(data)


def get_avg_seconds(key: str) -> Optional[float]:
    """Return average seconds per single unit for the key, or None if unknown."""
    data = _load()
    entry = data.get(key)
    if not entry:
        return None
    if entry.get("count", 0) == 0:
        return None
    return float(entry.get("total", 0.0)) / float(entry.get("count", 1))


class Timer:
    def __init__(self, key: str, units: int = 1):
        self.key = key
        self.units = units
        self._start = None

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._start is None:
            return
        dur = time.time() - self._start
        try:
            record_timing(self.key, dur, count=self.units)
        except Exception:
            pass
