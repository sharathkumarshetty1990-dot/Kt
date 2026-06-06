import json
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock

from job_lifecycle import (
    STATUS_COMPLETE,
    STATUS_ERROR,
    STATUS_PLANNING,
    STATUS_PROCESSING,
    status_accepts_command,
    transition_job_status,
)


class JobStore:
    def __init__(self, upload_root):
        self.upload_root = Path(upload_root)
        self._jobs = {}
        self._lock = RLock()

    def state_path_from_job(self, job):
        return Path(job["video_path"]).parent / "job.json"

    def state_path(self, job_id):
        return self.upload_root / job_id / "job.json"

    def sanitize(self, job, from_disk=False):
        if job.get("status") == STATUS_COMPLETE:
            job.pop("error", None)
            job.pop("failed_at", None)
        if from_disk and job.get("status") in {STATUS_PLANNING, STATUS_PROCESSING}:
            previous_status = job.get("status")
            transition_job_status(job, STATUS_ERROR, reason=f"{previous_status}_interrupted", force=True)
            job["error"] = f"{previous_status} interrupted before completion; retry the command"
            job["failed_at"] = job.get("failed_at") or datetime.now(timezone.utc).isoformat()
        return job

    def save(self, job):
        self.sanitize(job)
        path = self.state_path_from_job(job)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(job, indent=2), encoding="utf-8")
        tmp_path.replace(path)
        return job

    def remember(self, job):
        with self._lock:
            self._jobs[job["job_id"]] = job
            self.save(job)
        return job

    def persist(self, job):
        with self._lock:
            self._jobs[job["job_id"]] = job
            self.save(job)
        return job

    def get(self, job_id):
        with self._lock:
            job = self._jobs.get(job_id)
        if job is not None:
            return job

        state_path = self.state_path(job_id)
        if not state_path.exists():
            return None

        job = self.sanitize(json.loads(state_path.read_text(encoding="utf-8")), from_disk=True)
        with self._lock:
            self._jobs[job_id] = job
            self.save(job)
        return job

    def claim_for_command(self, job_id, command_text):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                state_path = self.state_path(job_id)
                if not state_path.exists():
                    return None, "not_found"
                job = self.sanitize(json.loads(state_path.read_text(encoding="utf-8")), from_disk=True)
                self._jobs[job_id] = job

            if not status_accepts_command(job.get("status")):
                return job, "conflict"

            transition_job_status(job, STATUS_PLANNING, reason="command_claimed_for_planning")
            job["command"] = command_text
            job["planning_started_at"] = datetime.now(timezone.utc).isoformat()
            job["updated_at"] = job["planning_started_at"]
            job.pop("warnings", None)
            self.save(job)
            return job, "claimed"

    def load_persisted(self):
        loaded = 0
        for state_path in self.upload_root.glob("*/job.json"):
            try:
                job = self.sanitize(json.loads(state_path.read_text(encoding="utf-8")), from_disk=True)
                job_id = job.get("job_id")
                if job_id:
                    with self._lock:
                        self._jobs[job_id] = job
                        self.save(job)
                    loaded += 1
            except Exception:
                continue
        return loaded

    def persisted_count(self):
        return sum(1 for _ in self.upload_root.glob("*/job.json"))

    def memory_count(self):
        with self._lock:
            return len(self._jobs)
