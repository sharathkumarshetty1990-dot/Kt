import json
import time
from threading import Lock


def clone_plan(plan):
    return json.loads(json.dumps(plan))


class PlannerCache:
    def __init__(self, ttl_seconds, fallback_ttl_seconds, max_entries):
        self.ttl_seconds = max(0, int(ttl_seconds or 0))
        self.fallback_ttl_seconds = max(0, int(fallback_ttl_seconds or 0))
        self.max_entries = max(0, int(max_entries or 0))
        self._entries = {}
        self._lock = Lock()

    def enabled(self):
        return bool(self.ttl_seconds and self.max_entries)

    def get(self, cache_key):
        if not self.enabled():
            return None

        now = time.time()
        with self._lock:
            entry = self._entries.get(cache_key)
            if not entry:
                return None
            ttl_seconds = entry.get("ttl_seconds", self.ttl_seconds)
            if not ttl_seconds or now - entry["created_at"] > ttl_seconds:
                self._entries.pop(cache_key, None)
                return None
            entry["last_used_at"] = now
            entry["hits"] = entry.get("hits", 0) + 1
            return {
                "plan": clone_plan(entry["plan"]),
                "source": entry["source"],
                "hits": entry["hits"],
                "architecture_fingerprint": entry.get("architecture_fingerprint"),
                "public_plan_contract_fingerprint": entry.get("public_plan_contract_fingerprint"),
                "special_param_contract_fingerprint": entry.get("special_param_contract_fingerprint"),
            }

    def store(
        self,
        cache_key,
        plan,
        source,
        architecture_fingerprint,
        public_plan_contract_fingerprint,
        special_param_contract_fingerprint,
    ):
        if not self.enabled():
            return
        ttl_seconds = self.fallback_ttl_seconds if source == "heuristic" else self.ttl_seconds
        if not ttl_seconds:
            return

        now = time.time()
        with self._lock:
            self._entries[cache_key] = {
                "plan": clone_plan(plan),
                "source": source,
                "architecture_fingerprint": architecture_fingerprint,
                "public_plan_contract_fingerprint": public_plan_contract_fingerprint,
                "special_param_contract_fingerprint": special_param_contract_fingerprint,
                "ttl_seconds": ttl_seconds,
                "created_at": now,
                "last_used_at": now,
                "hits": 0,
            }
            while len(self._entries) > self.max_entries:
                oldest_key = min(self._entries, key=lambda key: self._entries[key]["last_used_at"])
                self._entries.pop(oldest_key, None)

    def clear(self):
        with self._lock:
            self._entries.clear()

    def stats(
        self,
        architecture_fingerprint,
        public_plan_contract_fingerprint,
        special_param_contract_fingerprint,
    ):
        with self._lock:
            by_source = {}
            by_architecture = {}
            by_plan_contract = {}
            by_special_param_contract = {}
            for entry in self._entries.values():
                by_source[entry.get("source", "unknown")] = by_source.get(entry.get("source", "unknown"), 0) + 1
                fingerprint = entry.get("architecture_fingerprint", "unknown")
                by_architecture[fingerprint] = by_architecture.get(fingerprint, 0) + 1
                contract_fingerprint = entry.get("public_plan_contract_fingerprint", "unknown")
                by_plan_contract[contract_fingerprint] = by_plan_contract.get(contract_fingerprint, 0) + 1
                special_fingerprint = entry.get("special_param_contract_fingerprint", "unknown")
                by_special_param_contract[special_fingerprint] = by_special_param_contract.get(special_fingerprint, 0) + 1
            return {
                "enabled": self.enabled(),
                "entries": len(self._entries),
                "entries_by_source": dict(sorted(by_source.items())),
                "entries_by_architecture": dict(sorted(by_architecture.items())),
                "entries_by_public_plan_contract": dict(sorted(by_plan_contract.items())),
                "entries_by_special_param_contract": dict(sorted(by_special_param_contract.items())),
                "ttl_seconds": self.ttl_seconds,
                "fallback_ttl_seconds": self.fallback_ttl_seconds,
                "max_entries": self.max_entries,
                "architecture_fingerprint": architecture_fingerprint,
                "public_plan_contract_fingerprint": public_plan_contract_fingerprint,
                "special_param_contract_fingerprint": special_param_contract_fingerprint,
            }
