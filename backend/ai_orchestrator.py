from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def create_execution_manifest(internal_plan):
    steps = []
    for step in internal_plan.get("steps", []):
        steps.append({
            "id": step.get("id"),
            "type": step.get("type"),
            "worker": step.get("worker"),
            "depends_on": step.get("depends_on", []),
            "legacy_ref": deepcopy(step.get("legacy_ref")),
            "status": "queued",
            "confidence": step.get("confidence"),
            "confidence_band": step.get("confidence_band"),
        })
    return refresh_manifest_progress({
        "schema_version": "linguist-execution-manifest-v1",
        "plan_id": internal_plan.get("plan_id"),
        "status": "queued",
        "created_at": utc_now(),
        "started_at": None,
        "completed_at": None,
        "steps": steps,
    })


def refresh_manifest_progress(manifest):
    updated = dict(manifest or {})
    steps = updated.get("steps") or []
    total = len(steps)
    complete = len([step for step in steps if step.get("status") == "complete"])
    failed = len([step for step in steps if step.get("status") == "error"])
    running = next((step.get("id") for step in steps if step.get("status") == "running"), None)
    updated["progress"] = {
        "total_steps": total,
        "complete_steps": complete,
        "failed_steps": failed,
        "percent": int(round((complete / total) * 100)) if total else 0,
        "active_step_id": running,
    }
    return updated


def mark_manifest_running(manifest):
    updated = deepcopy(manifest or {})
    updated["status"] = "running"
    updated["started_at"] = updated.get("started_at") or utc_now()
    for step in updated.get("steps", []):
        if step.get("status") == "queued":
            step["status"] = "pending"
    return refresh_manifest_progress(updated)


def mark_manifest_complete(manifest):
    updated = deepcopy(manifest or {})
    updated["status"] = "complete"
    updated["completed_at"] = utc_now()
    for step in updated.get("steps", []):
        if step.get("status") in {"queued", "pending", "running"}:
            step["status"] = "complete"
    return refresh_manifest_progress(updated)


def mark_manifest_failed(manifest, error):
    updated = deepcopy(manifest or {})
    updated["status"] = "error"
    updated["failed_at"] = utc_now()
    updated["error"] = str(error)[:1000]
    for step in updated.get("steps", []):
        if step.get("status") == "running":
            step["status"] = "error"
            step["error"] = str(error)[:1000]
            step["failed_at"] = updated["failed_at"]
            break
    return refresh_manifest_progress(updated)


def mark_manifest_step_status(
    manifest,
    status,
    step_type=None,
    legacy_section=None,
    legacy_index=None,
    error=None,
):
    updated = deepcopy(manifest or {})
    now = utc_now()
    for step in updated.get("steps", []):
        legacy_ref = step.get("legacy_ref") or {}
        if step_type and step.get("type") != step_type:
            continue
        if legacy_section and legacy_ref.get("section") != legacy_section:
            continue
        if legacy_index is not None and legacy_ref.get("index") != legacy_index:
            continue
        step["status"] = status
        if status == "running":
            step["started_at"] = step.get("started_at") or now
        elif status == "complete":
            step["completed_at"] = now
            step.pop("error", None)
        elif status == "error":
            step["failed_at"] = now
            step["error"] = str(error or "step failed")[:1000]
    return refresh_manifest_progress(updated)


def inspect_output_artifact(output_path, expected_plan=None, duration_probe=None):
    path = Path(output_path) if output_path else None
    issues = []
    scores = {}

    exists = bool(path and path.exists())
    scores["artifact_exists"] = 1.0 if exists else 0.0
    if not exists:
        issues.append({"code": "missing_output", "message": "Final output artifact was not created."})
        return {
            "schema_version": "linguist-result-inspection-v1",
            "ok": False,
            "checked_at": utc_now(),
            "output_path": str(path) if path else None,
            "scores": scores,
            "issues": issues,
        }

    size = path.stat().st_size
    non_empty = size > 0
    scores["non_empty_file"] = 1.0 if non_empty else 0.0
    if not non_empty:
        issues.append({"code": "empty_output", "message": "Final output artifact is empty."})

    duration = None
    if duration_probe:
        try:
            duration = float(duration_probe(path))
            scores["positive_duration"] = 1.0 if duration > 0 else 0.0
            if duration <= 0:
                issues.append({"code": "zero_duration", "message": "Final output duration is not positive."})
        except Exception as exc:
            scores["positive_duration"] = 0.0
            issues.append({"code": "duration_probe_failed", "message": str(exc)[:500]})

    planned_steps = len((expected_plan or {}).get("steps", []))
    scores["plan_steps_present"] = 1.0 if planned_steps > 0 else 0.0

    return {
        "schema_version": "linguist-result-inspection-v1",
        "ok": not issues,
        "checked_at": utc_now(),
        "output_path": str(path),
        "size_bytes": size,
        "duration_seconds": duration,
        "scores": scores,
        "issues": issues,
    }


def repair_packet_from_exception(error, internal_plan=None):
    validation = (internal_plan or {}).get("validation") or {}
    return {
        "schema_version": "linguist-repair-packet-v1",
        "created_at": utc_now(),
        "failed_step": "unknown",
        "reason": str(error)[:1000],
        "symptom": "pipeline execution failed before producing a valid final artifact",
        "validation_context": {
            "confidence": validation.get("confidence"),
            "confidence_band": validation.get("confidence_band"),
            "issues": validation.get("issues", []),
            "warnings": validation.get("warnings", []),
        },
        "repair_options": [
            "retry failed command with safe fallback settings",
            "remove or simplify the failing filter step",
            "switch complex graph effects into a dedicated special worker",
            "ask for clarification if the plan confidence is below execution threshold",
        ],
    }
