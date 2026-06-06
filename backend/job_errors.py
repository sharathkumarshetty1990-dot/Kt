from datetime import datetime, timezone


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def concise_message(value, limit=1000):
    text = str(value or "unknown error").strip()
    return text[:limit]


def job_error(
    code,
    message,
    phase,
    retryable=False,
    user_action=None,
    details=None,
):
    error = {
        "schema_version": "linguist-job-error-v1",
        "code": code,
        "message": concise_message(message),
        "phase": phase,
        "retryable": bool(retryable),
        "created_at": utc_now(),
    }
    if user_action:
        error["user_action"] = concise_message(user_action, limit=500)
    if details:
        error["details"] = details
    return error


def exception_job_error(code, phase, exc, retryable=False, user_action=None, details=None):
    return job_error(
        code=code,
        message=exc,
        phase=phase,
        retryable=retryable,
        user_action=user_action,
        details=details,
    )


def plan_validation_job_error(validation):
    error_codes = [
        issue.get("code")
        for issue in (validation or {}).get("issues", [])
        if issue.get("severity") == "error"
    ]
    return job_error(
        code="plan_validation_failed",
        message="The edit plan did not pass validation.",
        phase="planning",
        retryable=True,
        user_action="Revise the command or upload the media/capabilities required by the validation errors.",
        details={
            "error_codes": error_codes,
            "confidence": (validation or {}).get("confidence"),
            "confidence_band": (validation or {}).get("confidence_band"),
        },
    )


def apply_job_error(job, error):
    if job is None:
        return None
    job["job_error"] = error
    job["error"] = error.get("message") or "job failed"
    job["failed_at"] = error.get("created_at") or utc_now()
    return job
