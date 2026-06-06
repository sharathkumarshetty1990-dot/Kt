import copy
import re
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from editing_architecture import (
    ANALYSIS_CONTEXTS,
    FILTER_COMPLEX_ONLY,
    REQUIRED_FINAL_ENCODE_KEYS,
    SCHEMA_VERSION,
    SUPPORTED_ANALYSIS_FUNCTIONS,
    SUPPORTED_SPECIAL_TYPES,
    WORKER_BY_STEP_TYPE,
    analysis_function_confidence,
    capability_severity,
    confidence_band,
    enrich_validation_entries,
    external_dependency_prefixes,
    normalize_final_encode_settings,
    operation_required_capabilities,
    retry_policy_for_confidence,
    special_operation_confidence,
    special_operation_requires_uploaded_audio,
    special_operation_worker,
    validation_threshold,
)
from intent_contract import intent_coverage_issues
from plan_contract import normalize_public_plan_shape


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def clone_json(value):
    return copy.deepcopy(value)


def slug(value, fallback="step"):
    text = re.sub(r"[^a-zA-Z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return text or fallback


def list_value(plan, key):
    value = plan.get(key)
    if not value:
        return []
    return value if isinstance(value, list) else [value]


def prompt_terms(prompt):
    return set(re.findall(r"[a-z0-9]+", str(prompt or "").lower()))


def contains_any(text, terms):
    lowered = str(text or "").lower()
    return any(term in lowered for term in terms)


def parse_intent(command_text, plan):
    terms = prompt_terms(command_text)
    domains = []
    if terms & {"beat", "beats", "rhythm", "sync", "music", "drop", "bass"}:
        domains.append("rhythm")
    if terms & {"caption", "captions", "subtitle", "subtitles", "transcribe"}:
        domains.append("subtitles")
    if terms & {"color", "grade", "cinematic", "lofi", "vhs", "underwater", "dream"}:
        domains.append("look")
    if terms & {"shake", "zoom", "pan", "crop", "stabilize", "stabilise"}:
        domains.append("motion")
    if terms & {"silence", "pause", "pauses", "blank", "black", "freeze", "duplicate"}:
        domains.append("cleanup")
    if terms & {"blur", "redact", "license", "plate", "face", "privacy"}:
        domains.append("privacy")
    if list_value(plan, "audio_filters") or any(
        special_operation_requires_uploaded_audio(step.get("type"))
        for step in list_value(plan, "special")
    ):
        domains.append("audio")
    if list_value(plan, "video_filters"):
        domains.append("video")

    ordered_domains = []
    for domain in domains:
        if domain not in ordered_domains:
            ordered_domains.append(domain)

    return {
        "summary": plan.get("intent") or str(command_text or "").strip(),
        "domains": ordered_domains or ["general_edit"],
        "requires_audio_analysis": any(step.get("tool") == "librosa" for step in list_value(plan, "analysis")),
        "requires_uploaded_audio": any(
            special_operation_requires_uploaded_audio(step.get("type"))
            for step in list_value(plan, "special")
        ),
        "source": "natural_language",
    }


def extract_constraints(command_text, job, plan):
    lowered = str(command_text or "").lower()
    aspect_ratio = None
    if any(term in lowered for term in ["9:16", "tiktok", "reels", "shorts", "vertical"]):
        aspect_ratio = "9:16"
    elif "4:5" in lowered:
        aspect_ratio = "4:5"
    elif "1:1" in lowered or "square" in lowered:
        aspect_ratio = "1:1"
    elif any(term in lowered for term in ["2.39", "2.35", "letterbox", "widescreen"]):
        aspect_ratio = "2.39:1"

    quality = "production_default"
    encode = plan.get("final_encode") if isinstance(plan.get("final_encode"), dict) else {}
    if encode.get("crf") is not None:
        quality = f"crf_{encode.get('crf')}"

    return {
        "aspect_ratio": aspect_ratio,
        "quality": quality,
        "dark_launch": False,
        "video_required": True,
        "audio_required": bool(list_value(plan, "audio_filters")),
        "uploaded_audio_available": bool(job and job.get("audio_path")),
        "max_retries_per_step": 1,
    }


def describe_assets(job):
    video_path = Path(job["video_path"]) if job and job.get("video_path") else None
    audio_path = Path(job["audio_path"]) if job and job.get("audio_path") else None
    return {
        "video": {
            "name": job.get("video_name") if job else None,
            "path": str(video_path) if video_path else None,
            "exists": bool(video_path and video_path.exists()),
        },
        "audio": {
            "name": job.get("audio_name") if job else None,
            "path": str(audio_path) if audio_path else None,
            "exists": bool(audio_path and audio_path.exists()),
        },
    }


def analysis_context_index(plan):
    index = {}
    for step in list_value(plan, "analysis"):
        store_as = step.get("store_as")
        if store_as:
            index[store_as] = step.get("function")
            if store_as == "energy_curve":
                index["energy_curve_times"] = step.get("function")
    return index


def ensure_analysis_for_context(plan, context_key):
    if not context_key:
        return False
    function, store_as = ANALYSIS_CONTEXTS.get(context_key, (None, None))
    if not function:
        return False

    analysis = list_value(plan, "analysis")
    if any(step.get("tool") == "librosa" and step.get("function") == function for step in analysis):
        return False

    step = {"tool": "librosa", "function": function, "store_as": store_as}
    if function == "onset_detect":
        step["sensitivity"] = 0.5
    analysis.append(step)
    plan["analysis"] = analysis
    return True


def repair_plan_dependencies(plan):
    repaired = clone_json(plan)
    fixes = []

    for step in list_value(repaired, "video_filters"):
        timing = step.get("timing")
        context_key = step.get("requires_context")
        if not context_key and timing == "per_beat":
            context_key = "beat_times"
        elif not context_key and timing == "per_onset":
            context_key = "onset_times"
        if ensure_analysis_for_context(repaired, context_key):
            fixes.append(f"added missing librosa analysis for {context_key}")

    for step in list_value(repaired, "special"):
        params = step.get("params") if isinstance(step.get("params"), dict) else {}
        context_key = params.get("context")
        if ensure_analysis_for_context(repaired, context_key):
            fixes.append(f"added missing librosa analysis for {context_key}")

    return repaired, fixes


def repair_final_encode(plan):
    repaired = clone_json(plan)
    normalized, changed = normalize_final_encode_settings(repaired.get("final_encode"))
    repaired["final_encode"] = normalized
    fixes = [f"normalized final_encode: {', '.join(changed)}"] if changed else []
    return repaired, fixes


def confidence_for_step(kind, step):
    if kind == "final_encode":
        return 0.98
    if kind == "analysis":
        return analysis_function_confidence(step.get("function"))
    if kind == "special":
        return special_operation_confidence(step.get("type"))
    filter_string = str(step.get("filter") or "")
    if not filter_string:
        return 0.35
    names = filter_names(filter_string)
    if names & FILTER_COMPLEX_ONLY:
        return 0.45
    if "[" in filter_string or "]" in filter_string:
        return 0.50
    return 0.76


def filter_names(filter_string):
    names = set()
    for part in str(filter_string or "").split(","):
        name = part.strip().split("=", 1)[0].split("@", 1)[0].strip()
        if name:
            names.add(name)
    return names


def step_context_requirement(kind, step):
    if kind == "video_filter":
        timing = step.get("timing")
        context_key = step.get("requires_context")
        if not context_key and timing == "per_beat":
            return "beat_times"
        if not context_key and timing == "per_onset":
            return "onset_times"
        return context_key
    if kind == "special":
        params = step.get("params") if isinstance(step.get("params"), dict) else {}
        return params.get("context")
    return None


def build_step(step_id, kind, source, index, payload, depends_on):
    confidence = confidence_for_step(kind, payload)
    worker = WORKER_BY_STEP_TYPE.get(kind, "MediaWorker")
    if kind == "special":
        worker = special_operation_worker(payload.get("type"), worker)
    return {
        "id": step_id,
        "type": kind,
        "worker": worker,
        "depends_on": [item for item in depends_on if item],
        "params": clone_json(payload),
        "confidence": round(confidence, 2),
        "confidence_band": confidence_band(confidence),
        "retry_policy": retry_policy_for_confidence(confidence),
        "legacy_ref": {"section": source, "index": index},
    }


def compile_steps(plan):
    steps = []
    context_owner = {}

    for index, step in enumerate(list_value(plan, "analysis")):
        step_id = f"analysis_{index + 1}_{slug(step.get('function'), 'librosa')}"
        steps.append(build_step(step_id, "analysis", "analysis", index, step, ["asset:audio_or_video"]))
        if step.get("store_as"):
            context_owner[step["store_as"]] = step_id
            if step["store_as"] == "energy_curve":
                context_owner["energy_curve_times"] = step_id

    last_media_step = "asset:video"
    for index, step in enumerate(list_value(plan, "special")):
        special_type = step.get("type") or "special"
        depends_on = [last_media_step]
        context_key = step_context_requirement("special", step)
        if context_key:
            depends_on.append(context_owner.get(context_key))
        if special_operation_requires_uploaded_audio(special_type):
            depends_on.append("asset:audio")
        step_id = f"special_{index + 1}_{slug(special_type)}"
        steps.append(build_step(step_id, "special", "special", index, step, depends_on))
        last_media_step = step_id

    for index, step in enumerate(list_value(plan, "video_filters")):
        depends_on = [last_media_step]
        context_key = step_context_requirement("video_filter", step)
        if context_key:
            depends_on.append(context_owner.get(context_key))
        step_id = f"video_{index + 1}_{slug(step.get('description') or step.get('filter'), 'filter')}"
        steps.append(build_step(step_id, "video_filter", "video_filters", index, step, depends_on))
        last_media_step = step_id

    for index, step in enumerate(list_value(plan, "audio_filters")):
        step_id = f"audio_{index + 1}_{slug(step.get('description') or step.get('filter'), 'filter')}"
        steps.append(build_step(step_id, "audio_filter", "audio_filters", index, step, [last_media_step]))
        last_media_step = step_id

    final_step = plan.get("final_encode") if isinstance(plan.get("final_encode"), dict) else {}
    steps.append(build_step("export_1_final_encode", "final_encode", "final_encode", 0, final_step, [last_media_step]))
    return topological_sort_steps(steps)


def topological_sort_steps(steps):
    by_id = {step["id"]: step for step in steps}
    pending = dict(by_id)
    ordered = []
    external_prefixes = external_dependency_prefixes()

    while pending:
        ready_ids = []
        for step_id, step in pending.items():
            deps = step.get("depends_on", [])
            if all(dep not in pending and (dep in by_id or str(dep).startswith(external_prefixes)) for dep in deps):
                ready_ids.append(step_id)
        if not ready_ids:
            ordered.extend(pending.values())
            break
        for step_id in ready_ids:
            ordered.append(pending.pop(step_id))
    return ordered


def validate_internal_plan(plan, internal_plan, capabilities=None, command_text=None):
    issues = []
    warnings = []
    fixes = list(internal_plan.get("applied_fixes") or [])
    assets = internal_plan["assets"]
    contexts = analysis_context_index(plan)
    executor_caps = (capabilities or {}).get("executor") or {}

    if not assets["video"]["exists"]:
        issues.append({
            "code": "missing_video_asset",
            "message": "Uploaded video is missing or unreadable before planning.",
            "severity": "error",
        })
    if capabilities and not executor_caps.get("ffmpeg_ready"):
        issues.append({
            "code": "ffmpeg_not_ready",
            "message": "FFmpeg/ffprobe are not ready, so media execution cannot be trusted.",
            "severity": "error",
        })

    encode = plan.get("final_encode")
    if not isinstance(encode, dict):
        issues.append({"code": "missing_final_encode", "message": "Plan has no final_encode block.", "severity": "error"})
    else:
        missing = [key for key in REQUIRED_FINAL_ENCODE_KEYS if key not in encode]
        if missing:
            issues.append({
                "code": "incomplete_final_encode",
                "message": f"final_encode missing: {', '.join(missing)}",
                "severity": "error",
            })

    for step in internal_plan["steps"]:
        params = step.get("params") or {}
        names = filter_names(params.get("filter")) if step["type"] in {"video_filter", "audio_filter"} else set()
        if capabilities:
            for capability in operation_required_capabilities(step["type"], params, names):
                if executor_caps.get(capability):
                    continue
                severity = capability_severity(capability)
                entry = {
                    "code": f"{capability}_missing",
                    "message": f"Step requires runtime capability '{capability}', but it is not ready.",
                    "severity": severity,
                    "step_id": step["id"],
                }
                if severity == "error":
                    issues.append(entry)
                else:
                    warnings.append(entry)
        if step["type"] == "analysis" and params.get("function") not in SUPPORTED_ANALYSIS_FUNCTIONS:
            issues.append({
                "code": "unsupported_analysis_function",
                "message": f"Unsupported analysis function: {params.get('function')}",
                "severity": "error",
                "step_id": step["id"],
            })
        if step["type"] == "special" and special_operation_requires_uploaded_audio(params.get("type")) and not assets["audio"]["exists"]:
            warnings.append({
                "code": "uploaded_audio_missing",
                "message": f"{params.get('type')} requested but no uploaded audio asset is present.",
                "step_id": step["id"],
            })
        if step["type"] == "special" and params.get("type") not in SUPPORTED_SPECIAL_TYPES:
            issues.append({
                "code": "unsupported_special_type",
                "message": f"Unsupported special operation: {params.get('type')}",
                "severity": "error",
                "step_id": step["id"],
            })
        context_key = step_context_requirement(step["type"], params)
        if context_key and context_key not in contexts and context_key not in {"asset:audio", "asset:video"}:
            issues.append({
                "code": "missing_analysis_context",
                "message": f"Step requires analysis context '{context_key}' but no producer exists.",
                "severity": "error",
                "step_id": step["id"],
            })
        if step["type"] in {"video_filter", "audio_filter"}:
            blocked = sorted(names & FILTER_COMPLEX_ONLY)
            if blocked:
                issues.append({
                    "code": "filter_complex_not_supported_as_direct_filter",
                    "message": (
                        "Direct video_filters/audio_filters only support single-input filter chains. "
                        f"Use a supported special operation instead of: {', '.join(blocked)}"
                    ),
                    "severity": "error",
                    "step_id": step["id"],
                })

    issues.extend(intent_coverage_issues(command_text, plan))

    score_values = [step.get("confidence", 0.0) for step in internal_plan["steps"]]
    confidence = min(score_values) if score_values else 0.0
    if confidence < validation_threshold("fallback_or_confirm"):
        issues.append({
            "code": "low_confidence_plan",
            "message": "Plan confidence is below the safe execution threshold.",
            "severity": "error",
        })
    elif confidence < validation_threshold("execute_with_guardrails"):
        warnings.append({
            "code": "guardrailed_execution",
            "message": "Plan confidence is in fallback/confirmation range; executor will rely on guardrails.",
        })
    issues = enrich_validation_entries(issues)
    warnings = enrich_validation_entries(warnings)
    ok = not any(issue.get("severity") == "error" for issue in issues)

    return {
        "ok": ok,
        "confidence": round(confidence, 2),
        "confidence_band": confidence_band(confidence),
        "issues": issues,
        "warnings": warnings,
        "fixes": fixes,
        "policy": "model proposes, planner structures, validator gates, executor acts",
    }


def prepare_production_plan(command_text, plan, job=None, capabilities=None):
    normalized_plan, fixes = normalize_public_plan_shape(
        plan,
        intent_fallback=str(command_text or "AI generated video edit"),
    )
    repaired_plan, dependency_fixes = repair_plan_dependencies(normalized_plan)
    fixes.extend(dependency_fixes)
    repaired_plan, encode_fixes = repair_final_encode(repaired_plan)
    fixes.extend(encode_fixes)
    internal_plan = {
        "schema_version": SCHEMA_VERSION,
        "plan_id": uuid4().hex[:12],
        "created_at": utc_now(),
        "goal": repaired_plan.get("intent") or str(command_text or "").strip(),
        "intent": parse_intent(command_text, repaired_plan),
        "constraints": extract_constraints(command_text, job, repaired_plan),
        "assets": describe_assets(job),
        "steps": [],
        "fallbacks": {
            "video_filter": "use deterministic fallback filter chain or skip failed visual step",
            "audio_filter": "use safe loudness normalization fallback or skip failed audio step",
            "special": "emit repair packet and keep previous media artifact when possible",
        },
        "metrics": {
            "operation_count": 0,
            "requires_inspection": True,
            "stores_failures_for_learning": True,
        },
        "applied_fixes": fixes,
        "status": "draft",
    }
    internal_plan["steps"] = compile_steps(repaired_plan)
    internal_plan["metrics"]["operation_count"] = max(0, len(internal_plan["steps"]) - 1)
    validation = validate_internal_plan(repaired_plan, internal_plan, capabilities, command_text)
    internal_plan["validation"] = validation
    internal_plan["status"] = "ready" if validation["ok"] else "blocked"
    return repaired_plan, internal_plan


def validation_allows_execution(internal_plan):
    validation = internal_plan.get("validation") or {}
    return bool(validation.get("ok"))
