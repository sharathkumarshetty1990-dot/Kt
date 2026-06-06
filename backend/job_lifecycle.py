from datetime import datetime, timezone


STATUS_UPLOADED = "uploaded"
STATUS_PLANNING = "planning"
STATUS_PROCESSING = "processing"
STATUS_COMPLETE = "complete"
STATUS_ERROR = "error"
STATUS_PLAN_REJECTED = "plan_rejected"

JOB_STATUS_SPECS = {
    STATUS_UPLOADED: {
        "terminal": False,
        "accepts_command": True,
        "description": "Media uploaded and ready for a natural-language edit command.",
    },
    STATUS_PLANNING: {
        "terminal": False,
        "accepts_command": False,
        "description": "A natural-language command is being planned and validated.",
    },
    STATUS_PROCESSING: {
        "terminal": False,
        "accepts_command": False,
        "description": "The edit plan is queued or executing.",
    },
    STATUS_COMPLETE: {
        "terminal": True,
        "accepts_command": True,
        "description": "A valid output artifact was produced.",
    },
    STATUS_ERROR: {
        "terminal": True,
        "accepts_command": True,
        "description": "Planning or execution failed.",
    },
    STATUS_PLAN_REJECTED: {
        "terminal": True,
        "accepts_command": True,
        "description": "Planner validation blocked execution before media work started.",
    },
}

ALLOWED_TRANSITIONS = {
    None: {STATUS_UPLOADED},
    STATUS_UPLOADED: {STATUS_PLANNING, STATUS_PROCESSING, STATUS_PLAN_REJECTED, STATUS_ERROR},
    STATUS_PLANNING: {STATUS_PROCESSING, STATUS_PLAN_REJECTED, STATUS_ERROR},
    STATUS_PROCESSING: {STATUS_COMPLETE, STATUS_ERROR},
    STATUS_COMPLETE: {STATUS_PLANNING, STATUS_PROCESSING, STATUS_ERROR, STATUS_PLAN_REJECTED},
    STATUS_ERROR: {STATUS_PLANNING, STATUS_PROCESSING, STATUS_PLAN_REJECTED},
    STATUS_PLAN_REJECTED: {STATUS_PLANNING, STATUS_PROCESSING, STATUS_ERROR},
}

MAX_STATUS_HISTORY = 40


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def known_job_status(status):
    return status in JOB_STATUS_SPECS


def status_is_terminal(status):
    return bool(JOB_STATUS_SPECS.get(status, {}).get("terminal"))


def status_accepts_command(status):
    return bool(JOB_STATUS_SPECS.get(status, {}).get("accepts_command"))


def status_transition_allowed(previous_status, next_status):
    return next_status in ALLOWED_TRANSITIONS.get(previous_status, set())


def transition_job_status(job, next_status, reason=None, now=None, force=False):
    if not known_job_status(next_status):
        raise ValueError(f"unknown job status: {next_status}")

    previous_status = job.get("status")
    if previous_status == next_status:
        return job
    if previous_status is not None and not known_job_status(previous_status):
        previous_status = None
    if not force and not status_transition_allowed(previous_status, next_status):
        raise ValueError(f"invalid job status transition: {previous_status} -> {next_status}")

    timestamp = now or utc_now()
    history = list(job.get("status_history") or [])
    history.append({
        "from": previous_status,
        "to": next_status,
        "reason": reason,
        "at": timestamp,
    })
    job["status"] = next_status
    job["status_updated_at"] = timestamp
    job["status_history"] = history[-MAX_STATUS_HISTORY:]
    return job


def job_lifecycle_summary():
    return {
        "statuses": {
            status: dict(spec)
            for status, spec in sorted(JOB_STATUS_SPECS.items())
        },
        "allowed_transitions": {
            str(source): sorted(targets)
            for source, targets in ALLOWED_TRANSITIONS.items()
        },
        "max_status_history": MAX_STATUS_HISTORY,
    }
