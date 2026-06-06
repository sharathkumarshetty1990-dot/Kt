import time
from threading import Lock


class RuntimeCapabilityCache:
    def __init__(self, ttl_seconds):
        self.ttl_seconds = max(0, int(ttl_seconds or 0))
        self._checked_at = 0.0
        self._data = None
        self._lock = Lock()

    def get(self, probe_fn, force=False):
        now = time.time()
        with self._lock:
            if (
                self._data is not None
                and not force
                and self.ttl_seconds
                and now - self._checked_at < self.ttl_seconds
            ):
                return self._data
            data = probe_fn()
            self._checked_at = now
            self._data = data
            return data

    def clear(self):
        with self._lock:
            self._checked_at = 0.0
            self._data = None

    def stats(self):
        with self._lock:
            now = time.time()
            age_seconds = None if self._data is None else round(now - self._checked_at, 3)
            return {
                "ttl_seconds": self.ttl_seconds,
                "has_data": self._data is not None,
                "checked_at_epoch": self._checked_at or None,
                "age_seconds": age_seconds,
                "fresh": bool(
                    self._data is not None
                    and self.ttl_seconds
                    and now - self._checked_at < self.ttl_seconds
                ),
            }
