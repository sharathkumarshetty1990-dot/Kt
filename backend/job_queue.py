from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import BoundedSemaphore, Lock


class ThreadedJobQueue:
    def __init__(self, max_workers, max_pending):
        self.max_workers = max(1, int(max_workers or 1))
        self.max_pending = max(self.max_workers, int(max_pending or self.max_workers))
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._slots = BoundedSemaphore(self.max_pending)
        self._lock = Lock()
        self._submitted = 0
        self._completed = 0
        self._failed = 0
        self._rejected = 0
        self._inflight = 0
        self._running = 0
        self._last_error = None

    def can_accept(self):
        with self._lock:
            return not self.shutdown_started() and self._inflight < self.max_pending

    def shutdown_started(self):
        return bool(getattr(self._executor, "_shutdown", False))

    def submit(self, fn, *args, **kwargs):
        if self.shutdown_started() or not self._slots.acquire(blocking=False):
            with self._lock:
                self._rejected += 1
            return None

        with self._lock:
            self._submitted += 1
            self._inflight += 1

        def run_and_record():
            with self._lock:
                self._running += 1
            try:
                result = fn(*args, **kwargs)
                with self._lock:
                    self._completed += 1
                return result
            except Exception as exc:
                with self._lock:
                    self._failed += 1
                    self._last_error = {
                        "message": str(exc)[:1000],
                        "at": datetime.now(timezone.utc).isoformat(),
                    }
                raise
            finally:
                with self._lock:
                    self._running = max(0, self._running - 1)
                    self._inflight = max(0, self._inflight - 1)
                self._slots.release()

        return self._executor.submit(run_and_record)

    def stats(self):
        with self._lock:
            return {
                "max_workers": self.max_workers,
                "max_pending": self.max_pending,
                "submitted": self._submitted,
                "completed": self._completed,
                "failed": self._failed,
                "rejected": self._rejected,
                "inflight": self._inflight,
                "running": self._running,
                "queued": max(0, self._inflight - self._running),
                "accepting": not self.shutdown_started() and self._inflight < self.max_pending,
                "shutdown_started": self.shutdown_started(),
                "last_error": self._last_error,
            }
