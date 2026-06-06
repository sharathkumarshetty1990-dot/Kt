DEFAULT_FINAL_ENCODE = {
    "vcodec": "libx264",
    "crf": 22,
    "preset": "fast",
    "acodec": "aac",
    "audio_bitrate": "192k",
}


def planned_operation_count(plan):
    return (
        len(plan.get("analysis", []))
        + len(plan.get("special", []))
        + len(plan.get("video_filters", []))
        + len(plan.get("audio_filters", []))
    )


def final_encode_changes_output_shape(plan):
    final_encode = plan.get("final_encode") or {}
    return bool(final_encode.get("width") and final_encode.get("height"))


def execution_evidence(job, plan):
    evidence = []

    special_execution = job.get("special_execution") or {}
    if special_execution.get("applied_count", 0) > 0:
        evidence.append({
            "type": "special",
            "applied_count": special_execution.get("applied_count"),
            "planned_count": special_execution.get("planned_count"),
        })

    video_execution = job.get("video_filter_execution") or {}
    if video_execution.get("applied_filter_count", 0) > 0:
        evidence.append({
            "type": "video_filter",
            "applied_count": video_execution.get("applied_filter_count"),
            "planned_count": video_execution.get("planned_filter_count"),
        })

    audio_execution = job.get("audio_filter_execution") or {}
    if audio_execution.get("applied_filter_count", 0) > 0:
        evidence.append({
            "type": "audio_filter",
            "applied_count": audio_execution.get("applied_filter_count"),
            "planned_count": audio_execution.get("planned_filter_count"),
        })

    uploaded_audio_action = job.get("uploaded_audio_action")
    if uploaded_audio_action:
        evidence.append({
            "type": "uploaded_audio",
            "action": uploaded_audio_action.get("type"),
        })

    output_aspect_execution = job.get("output_aspect_execution")
    if output_aspect_execution:
        evidence.append({
            "type": "output_aspect",
            "format": output_aspect_execution.get("format"),
        })

    if final_encode_changes_output_shape(plan):
        evidence.append({
            "type": "final_encode_dimensions",
            "width": plan.get("final_encode", {}).get("width"),
            "height": plan.get("final_encode", {}).get("height"),
        })

    return evidence


def inspect_edit_effectiveness(job, plan):
    planned_count = planned_operation_count(plan)
    evidence = execution_evidence(job, plan)
    issues = []

    if planned_count > 0 and not evidence:
        issues.append({
            "code": "no_effective_edit",
            "message": "The plan contained edit operations, but execution produced no evidence that any edit was applied.",
        })
    elif planned_count == 0 and not final_encode_changes_output_shape(plan):
        issues.append({
            "code": "empty_edit_plan",
            "message": "The plan contained no executable edit operations.",
        })

    return {
        "schema_version": "linguist-edit-effectiveness-v1",
        "ok": not issues,
        "planned_operation_count": planned_count,
        "evidence": evidence,
        "issues": issues,
    }
