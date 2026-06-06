import hashlib
import json
import re


ARCHITECTURE_VERSION = "linguist-editing-architecture-v1"
SCHEMA_VERSION = "linguist-internal-plan-v1"

REQUIRED_FINAL_ENCODE_KEYS = ["vcodec", "crf", "preset", "acodec", "audio_bitrate"]
DEFAULT_FINAL_ENCODE = {
    "vcodec": "libx264",
    "crf": 22,
    "preset": "fast",
    "acodec": "aac",
    "audio_bitrate": "192k",
}

CAPABILITY_SPECS = {
    "ffmpeg_ready": {
        "description": "FFmpeg and ffprobe are available for core media execution.",
        "severity": "error",
    },
    "audio_analysis_ready": {
        "description": "librosa/numpy audio analysis is available for beat, onset, and energy extraction.",
        "severity": "warning",
    },
    "pitch_shift_ready": {
        "description": "rubberband pitch shifting is available through FFmpeg or the rubberband CLI.",
        "severity": "error",
    },
    "asr_ready": {
        "description": "Speech recognition for auto captions is available.",
        "severity": "error",
    },
    "ocr_ready": {
        "description": "OCR tooling is available for text/license redaction.",
        "severity": "error",
    },
    "opencv_ready": {
        "description": "OpenCV semantic detection is available; privacy workers can still fall back to safe regions.",
        "severity": "warning",
    },
    "stabilize_ready": {
        "description": "FFmpeg vidstab filters are available for stabilization.",
        "severity": "error",
    },
    "frei0r_ready": {
        "description": "FFmpeg frei0r plugin bridge is available for plugin effects.",
        "severity": "error",
    },
}

ANALYSIS_FUNCTION_SPECS = {
    "beat_track": {
        "store_as": "beat_times",
        "contexts": ["beat_times"],
        "worker": "AudioAnalysisWorker",
        "confidence": 0.88,
        "requires_audio": True,
        "required_capabilities": ["audio_analysis_ready"],
    },
    "onset_detect": {
        "store_as": "onset_times",
        "contexts": ["onset_times"],
        "worker": "AudioAnalysisWorker",
        "confidence": 0.88,
        "requires_audio": True,
        "required_capabilities": ["audio_analysis_ready"],
    },
    "rms_energy": {
        "store_as": "energy_curve",
        "contexts": ["energy_curve", "energy_curve_times"],
        "worker": "AudioAnalysisWorker",
        "confidence": 0.88,
        "requires_audio": True,
        "required_capabilities": ["audio_analysis_ready"],
    },
}
ANALYSIS_CONTEXTS = {
    context: (function, spec["store_as"])
    for function, spec in ANALYSIS_FUNCTION_SPECS.items()
    for context in spec["contexts"]
}
SUPPORTED_ANALYSIS_FUNCTIONS = frozenset(ANALYSIS_FUNCTION_SPECS)

WORKER_BY_STEP_TYPE = {
    "analysis": "AudioAnalysisWorker",
    "special": "MediaTransformWorker",
    "video_filter": "VideoFilterWorker",
    "audio_filter": "AudioFilterWorker",
    "final_encode": "ExportWorker",
}

SPECIAL_OPERATION_SPECS = {
    "auto_captions": {
        "worker": "SubtitleWorker",
        "domain": "subtitles",
        "confidence": 0.82,
        "required_capabilities": ["asr_ready"],
    },
    "ocr_redact": {
        "worker": "OCRWorker",
        "domain": "privacy",
        "confidence": 0.82,
        "required_capabilities": ["ocr_ready"],
    },
    "face_privacy_blur": {
        "worker": "PrivacyWorker",
        "domain": "privacy",
        "confidence": 0.82,
        "optional_capabilities": ["opencv_ready"],
    },
    "silence_remove": {"worker": "CutWorker", "domain": "cleanup", "confidence": 0.82},
    "black_remove": {"worker": "CutWorker", "domain": "cleanup", "confidence": 0.82},
    "freeze_remove": {"worker": "CutWorker", "domain": "cleanup", "confidence": 0.82},
    "dedupe_frames": {"worker": "CutWorker", "domain": "cleanup", "confidence": 0.82},
    "beat_cut": {"worker": "CutWorker", "domain": "rhythm", "confidence": 0.82, "context_param": True},
    "scene_montage": {"worker": "CutWorker", "domain": "assembly", "confidence": 0.82},
    "energy_montage": {"worker": "CutWorker", "domain": "rhythm", "confidence": 0.82, "context_param": True},
    "replace_audio": {"worker": "AudioWorker", "domain": "audio", "confidence": 0.82, "requires_uploaded_audio": True},
    "mix_uploaded_audio": {"worker": "AudioWorker", "domain": "audio", "confidence": 0.82, "requires_uploaded_audio": True},
    "remove_audio": {"worker": "AudioWorker", "domain": "audio", "confidence": 0.82},
    "pitch_shift": {
        "worker": "AudioWorker",
        "domain": "audio",
        "confidence": 0.82,
        "required_capabilities": ["pitch_shift_ready"],
    },
    "speed_ramp": {"worker": "TimingWorker", "domain": "timing", "confidence": 0.82},
    "stabilize": {
        "worker": "StabilizeWorker",
        "domain": "motion",
        "confidence": 0.82,
        "required_capabilities": ["stabilize_ready"],
    },
    "reverse": {"worker": "TimingWorker", "domain": "timing", "confidence": 0.82},
    "boomerang": {"worker": "TimingWorker", "domain": "timing", "confidence": 0.82},
    "end_reverse": {"worker": "TimingWorker", "domain": "timing", "confidence": 0.82},
    "trim": {"worker": "CutWorker", "domain": "cleanup", "confidence": 0.82},
    "remove_segment": {"worker": "CutWorker", "domain": "cleanup", "confidence": 0.82},
    "blur_background": {"worker": "LayoutWorker", "domain": "layout", "confidence": 0.82},
    "chroma_key": {"worker": "CompositingWorker", "domain": "compositing", "confidence": 0.82},
    "film_damage": {"worker": "LookWorker", "domain": "look", "confidence": 0.82},
    "picture_in_picture": {"worker": "CompositingWorker", "domain": "compositing", "confidence": 0.82},
    "split_screen_mirror": {"worker": "LayoutWorker", "domain": "layout", "confidence": 0.82},
    "crop_borders": {"worker": "LayoutWorker", "domain": "layout", "confidence": 0.82},
}
SPECIAL_WORKERS = {
    operation_type: spec["worker"]
    for operation_type, spec in SPECIAL_OPERATION_SPECS.items()
}
SUPPORTED_SPECIAL_TYPES = frozenset(SPECIAL_WORKERS)

FILTER_CAPABILITY_REQUIREMENTS = {
    "asr": ["asr_ready"],
    "frei0r": ["frei0r_ready"],
    "rubberband": ["pitch_shift_ready"],
    "vidstabdetect": ["stabilize_ready"],
    "vidstabtransform": ["stabilize_ready"],
}

FILTER_COMPLEX_ONLY = {
    "overlay",
    "hstack",
    "vstack",
    "xstack",
    "amix",
    "sidechaincompress",
}

VALIDATION_POLICY = {
    "confidence_bands": {
        "execute": 0.85,
        "execute_with_guardrails": 0.65,
        "fallback_or_confirm": 0.40,
    },
    "retry_policy": {
        "default_max_attempts": 1,
        "guardrailed_max_attempts": 2,
        "retry_threshold": 0.65,
    },
    "external_dependency_prefixes": ["asset:"],
}

PLANNER_FALLBACK_POLICY = {
    "model_unavailable": {
        "mode": "heuristic_guardrailed",
        "allows_execution": True,
        "requires_validation": True,
        "cache_source": "heuristic",
        "user_visible_warning": True,
        "description": "Use deterministic heuristic planning only when NIM is unavailable, then require normal validation before execution.",
    },
    "model_repair_failed": {
        "mode": "reject",
        "allows_execution": False,
        "requires_validation": True,
        "cache_source": None,
        "user_visible_warning": True,
        "description": "Reject the command if a model-correctable invalid plan cannot be repaired by the model.",
    },
    "validation_not_repairable": {
        "mode": "reject",
        "allows_execution": False,
        "requires_validation": True,
        "cache_source": None,
        "user_visible_warning": True,
        "description": "Reject plans blocked by asset or runtime failures instead of inventing an edit.",
    },
}

EXECUTION_FAILURE_POLICY = {
    "special": {
        "mode": "fail_required_step",
        "allows_partial_success": False,
        "description": "A planned special operation is product behavior, not decoration. If it cannot run, fail the job instead of silently returning unchanged media.",
    },
    "video_filter": {
        "mode": "fallback_then_fail_if_none_applied",
        "allows_partial_success": True,
        "description": "Try safe per-filter fallback after a direct FFmpeg filter failure, but fail the phase if none of the planned video filters can be applied.",
    },
    "audio_filter": {
        "mode": "fallback_then_fail_if_none_applied",
        "allows_partial_success": True,
        "description": "Try safe per-filter fallback after a direct FFmpeg audio filter failure, but fail the phase if no planned audio filter can be applied.",
    },
    "final_encode": {
        "mode": "safe_default_encode",
        "allows_partial_success": True,
        "description": "Final encode may fall back to safe defaults because it packages the already-produced edit rather than replacing a requested creative operation.",
    },
}

VALIDATION_ISSUE_SPECS = {
    "missing_video_asset": {
        "owner": "asset",
        "repairable_by_model": False,
        "repair_hint": "Ask the user to upload a readable video before planning.",
    },
    "ffmpeg_not_ready": {
        "owner": "runtime",
        "repairable_by_model": False,
        "repair_hint": "Fix FFmpeg/ffprobe availability before accepting edit commands.",
    },
    "missing_final_encode": {
        "owner": "model",
        "repairable_by_model": True,
        "repair_hint": "Add final_encode with vcodec, crf, preset, acodec, and audio_bitrate.",
    },
    "incomplete_final_encode": {
        "owner": "model",
        "repairable_by_model": True,
        "repair_hint": "Complete final_encode using the required export keys.",
    },
    "runtime_capability_missing": {
        "owner": "runtime",
        "repairable_by_model": True,
        "repair_hint": "Choose a different ready operation or filter that does not require the missing capability.",
    },
    "unsupported_analysis_function": {
        "owner": "model",
        "repairable_by_model": True,
        "repair_hint": "Use only registered analysis functions from ANALYSIS_FUNCTION_SPECS.",
    },
    "uploaded_audio_missing": {
        "owner": "asset",
        "repairable_by_model": True,
        "repair_hint": "Avoid uploaded-audio-only operations unless an uploaded audio asset exists.",
    },
    "unsupported_special_type": {
        "owner": "model",
        "repairable_by_model": True,
        "repair_hint": "Use only registered special operation types from SPECIAL_OPERATION_SPECS.",
    },
    "missing_analysis_context": {
        "owner": "model",
        "repairable_by_model": True,
        "repair_hint": "Add the required analysis producer or remove the context-dependent operation.",
    },
    "filter_complex_not_supported_as_direct_filter": {
        "owner": "model",
        "repairable_by_model": True,
        "repair_hint": "Replace direct filter_complex/multi-stream filters with a supported special operation or single-input chain.",
    },
    "low_confidence_plan": {
        "owner": "model",
        "repairable_by_model": True,
        "repair_hint": "Use higher-confidence supported operations and simpler executable filter chains.",
    },
    "guardrailed_execution": {
        "owner": "model",
        "repairable_by_model": True,
        "repair_hint": "Prefer explicit supported operations over ambiguous or brittle filter chains.",
    },
    "intent_requirement_missing": {
        "owner": "model",
        "repairable_by_model": True,
        "repair_hint": "Add the missing operation, analysis, filter, or special worker required by the user's command.",
    },
}


def default_final_encode():
    return dict(DEFAULT_FINAL_ENCODE)


def normalized_encode_preset(value):
    preset = str(value or DEFAULT_FINAL_ENCODE["preset"]).strip().lower()
    if preset in {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium"}:
        return preset
    return DEFAULT_FINAL_ENCODE["preset"]


def normalized_video_codec(value):
    codec = str(value or DEFAULT_FINAL_ENCODE["vcodec"]).strip().lower()
    aliases = {
        "h264": "libx264",
        "x264": "libx264",
        "h.264": "libx264",
        "hevc": "libx265",
        "h265": "libx265",
        "h.265": "libx265",
        "x265": "libx265",
    }
    codec = aliases.get(codec, codec)
    if codec in {"libx264", "libx265"}:
        return codec
    return DEFAULT_FINAL_ENCODE["vcodec"]


def normalized_audio_codec(value):
    codec = str(value or DEFAULT_FINAL_ENCODE["acodec"]).strip().lower()
    aliases = {
        "mp3": "libmp3lame",
        "mpeg3": "libmp3lame",
        "m4a": "aac",
    }
    codec = aliases.get(codec, codec)
    if codec in {"aac", "libmp3lame"}:
        return codec
    return DEFAULT_FINAL_ENCODE["acodec"]


def normalized_crf(value):
    try:
        crf = int(float(value))
    except (TypeError, ValueError):
        return DEFAULT_FINAL_ENCODE["crf"]
    return max(16, min(35, crf))


def normalized_audio_bitrate(value):
    bitrate = str(value or DEFAULT_FINAL_ENCODE["audio_bitrate"]).strip().lower()
    match = re.fullmatch(r"(\d{2,4})k", bitrate)
    if not match:
        return DEFAULT_FINAL_ENCODE["audio_bitrate"]
    kbps = max(64, min(512, int(match.group(1))))
    return f"{kbps}k"


def normalized_encode_dimension(value):
    try:
        dimension = int(float(value))
    except (TypeError, ValueError):
        return None
    if dimension < 2 or dimension > 7680:
        return None
    if dimension % 2:
        dimension -= 1
    return max(2, dimension)


def normalize_final_encode_settings(final_settings):
    source = final_settings if isinstance(final_settings, dict) else {}
    normalized = {
        "vcodec": normalized_video_codec(source.get("vcodec")),
        "crf": normalized_crf(source.get("crf")),
        "preset": normalized_encode_preset(source.get("preset")),
        "acodec": normalized_audio_codec(source.get("acodec")),
        "audio_bitrate": normalized_audio_bitrate(source.get("audio_bitrate")),
    }

    width = normalized_encode_dimension(source.get("width"))
    height = normalized_encode_dimension(source.get("height"))
    if width and height:
        normalized["width"] = width
        normalized["height"] = height

    changed = {
        key
        for key, value in normalized.items()
        if source.get(key) != value
    }
    if not isinstance(final_settings, dict):
        changed.add("final_encode")
    if ("width" in source or "height" in source) and not (width and height):
        changed.add("dimensions")
    return normalized, sorted(changed)


def analysis_function_spec(function):
    return ANALYSIS_FUNCTION_SPECS.get(function)


def analysis_function_confidence(function):
    spec = analysis_function_spec(function)
    return spec.get("confidence", 0.55) if spec else 0.55


def analysis_function_required_capabilities(function):
    spec = analysis_function_spec(function)
    return tuple(spec.get("required_capabilities", ())) if spec else ()


def special_operation_spec(operation_type):
    return SPECIAL_OPERATION_SPECS.get(operation_type)


def special_operation_worker(operation_type, fallback="MediaTransformWorker"):
    spec = special_operation_spec(operation_type)
    return spec.get("worker", fallback) if spec else fallback


def special_operation_confidence(operation_type):
    spec = special_operation_spec(operation_type)
    return spec.get("confidence", 0.58) if spec else 0.58


def special_operation_requires_uploaded_audio(operation_type):
    spec = special_operation_spec(operation_type)
    return bool(spec and spec.get("requires_uploaded_audio"))


def special_operation_required_capabilities(operation_type):
    spec = special_operation_spec(operation_type)
    return tuple(spec.get("required_capabilities", ())) if spec else ()


def special_operation_optional_capabilities(operation_type):
    spec = special_operation_spec(operation_type)
    return tuple(spec.get("optional_capabilities", ())) if spec else ()


def special_operation_uses_context_param(operation_type):
    spec = special_operation_spec(operation_type)
    return bool(spec and spec.get("context_param"))


def capability_spec(capability):
    return CAPABILITY_SPECS.get(capability, {
        "description": "Runtime capability required by an operation.",
        "severity": "warning",
    })


def capability_severity(capability):
    return capability_spec(capability).get("severity", "warning")


def architecture_required_capabilities():
    return tuple(sorted(CAPABILITY_SPECS))


def architecture_registry_issues():
    known_capabilities = set(CAPABILITY_SPECS)
    issues = []

    def check_capabilities(source_type, source_name, field_name, capabilities):
        for capability in capabilities:
            if capability in known_capabilities:
                continue
            issues.append({
                "code": "unknown_capability_reference",
                "message": (
                    f"{source_type} '{source_name}' references unknown capability "
                    f"'{capability}' in {field_name}."
                ),
                "severity": "error",
                "source": source_type,
                "name": source_name,
                "field": field_name,
            })

    for function, spec in sorted(ANALYSIS_FUNCTION_SPECS.items()):
        check_capabilities(
            "analysis_function",
            function,
            "required_capabilities",
            spec.get("required_capabilities", ()),
        )

    for operation_type, spec in sorted(SPECIAL_OPERATION_SPECS.items()):
        check_capabilities(
            "special_operation",
            operation_type,
            "required_capabilities",
            spec.get("required_capabilities", ()),
        )
        check_capabilities(
            "special_operation",
            operation_type,
            "optional_capabilities",
            spec.get("optional_capabilities", ()),
        )

    for filter_name, capabilities in sorted(FILTER_CAPABILITY_REQUIREMENTS.items()):
        check_capabilities(
            "ffmpeg_filter",
            filter_name,
            "filter_capability_requirements",
            capabilities,
        )

    for capability, spec in sorted(CAPABILITY_SPECS.items()):
        if spec.get("severity") not in {"error", "warning"}:
            issues.append({
                "code": "invalid_capability_severity",
                "message": f"Capability '{capability}' has invalid severity '{spec.get('severity')}'.",
                "severity": "error",
                "source": "capability",
                "name": capability,
                "field": "severity",
            })

    for issue_code, spec in sorted(VALIDATION_ISSUE_SPECS.items()):
        missing_fields = [
            field
            for field in ["owner", "repairable_by_model", "repair_hint"]
            if field not in spec
        ]
        if missing_fields:
            issues.append({
                "code": "incomplete_validation_issue_spec",
                "message": (
                    f"Validation issue spec '{issue_code}' is missing required fields: "
                    f"{', '.join(missing_fields)}."
                ),
                "severity": "error",
                "source": "validation_issue",
                "name": issue_code,
                "field": ",".join(missing_fields),
            })
        if "repairable_by_model" in spec and not isinstance(spec.get("repairable_by_model"), bool):
            issues.append({
                "code": "invalid_validation_repairability",
                "message": f"Validation issue spec '{issue_code}' must use a boolean repairable_by_model value.",
                "severity": "error",
                "source": "validation_issue",
                "name": issue_code,
                "field": "repairable_by_model",
            })

    return issues


def filter_name_required_capabilities(name):
    return tuple(FILTER_CAPABILITY_REQUIREMENTS.get(str(name or "").strip(), ()))


def filter_names_required_capabilities(names):
    capabilities = []
    seen = set()
    for name in names:
        for capability in filter_name_required_capabilities(name):
            if capability not in seen:
                seen.add(capability)
                capabilities.append(capability)
    return tuple(capabilities)


def operation_required_capabilities(kind, params, filter_names=()):
    if kind == "analysis":
        return analysis_function_required_capabilities((params or {}).get("function"))
    if kind == "special":
        return special_operation_required_capabilities((params or {}).get("type"))
    if kind in {"video_filter", "audio_filter"}:
        return filter_names_required_capabilities(filter_names)
    return ()


def readiness_from_capabilities(required_capabilities, optional_capabilities, executor_capabilities):
    executor = executor_capabilities or {}
    missing_required = [
        capability
        for capability in required_capabilities or ()
        if not executor.get(capability)
    ]
    missing_optional = [
        capability
        for capability in optional_capabilities or ()
        if not executor.get(capability)
    ]
    blocking_missing = [
        capability
        for capability in missing_required
        if capability_severity(capability) == "error"
    ]
    warning_missing = [
        capability
        for capability in missing_required
        if capability_severity(capability) != "error"
    ] + missing_optional

    if blocking_missing:
        status = "blocked"
    elif warning_missing:
        status = "degraded"
    else:
        status = "ready"

    return {
        "status": status,
        "required_capabilities": list(required_capabilities or ()),
        "optional_capabilities": list(optional_capabilities or ()),
        "missing_required_capabilities": missing_required,
        "missing_optional_capabilities": missing_optional,
        "blocking_missing_capabilities": blocking_missing,
        "warning_missing_capabilities": warning_missing,
    }


def runtime_operation_contract(executor_capabilities):
    analysis = {}
    for function, spec in sorted(ANALYSIS_FUNCTION_SPECS.items()):
        readiness = readiness_from_capabilities(
            spec.get("required_capabilities", ()),
            spec.get("optional_capabilities", ()),
            executor_capabilities,
        )
        analysis[function] = {
            **readiness,
            "worker": spec.get("worker"),
            "store_as": spec.get("store_as"),
            "contexts": list(spec.get("contexts", ())),
            "requires_audio": bool(spec.get("requires_audio")),
        }

    special = {}
    for operation_type, spec in sorted(SPECIAL_OPERATION_SPECS.items()):
        readiness = readiness_from_capabilities(
            spec.get("required_capabilities", ()),
            spec.get("optional_capabilities", ()),
            executor_capabilities,
        )
        special[operation_type] = {
            **readiness,
            "worker": spec.get("worker"),
            "domain": spec.get("domain"),
            "requires_uploaded_audio": bool(spec.get("requires_uploaded_audio")),
            "context_param": bool(spec.get("context_param")),
        }

    filter_requirements = {}
    for filter_name, capabilities in sorted(FILTER_CAPABILITY_REQUIREMENTS.items()):
        filter_requirements[filter_name] = readiness_from_capabilities(
            capabilities,
            (),
            executor_capabilities,
        )

    return {
        "analysis_functions": analysis,
        "special_operations": special,
        "filter_requirements": filter_requirements,
        "available_analysis_functions": sorted(
            function
            for function, details in analysis.items()
            if details["status"] != "blocked"
        ),
        "blocked_analysis_functions": sorted(
            function
            for function, details in analysis.items()
            if details["status"] == "blocked"
        ),
        "available_special_operations": sorted(
            operation_type
            for operation_type, details in special.items()
            if details["status"] != "blocked"
        ),
        "blocked_special_operations": sorted(
            operation_type
            for operation_type, details in special.items()
            if details["status"] == "blocked"
        ),
    }


def runtime_operation_prompt_contract(executor_capabilities):
    contract = runtime_operation_contract(executor_capabilities)

    def operation_lines(items):
        lines = []
        for name, details in sorted(items.items()):
            traits = [
                f"status={details['status']}",
                f"worker={details.get('worker')}",
            ]
            if details.get("domain"):
                traits.append(f"domain={details.get('domain')}")
            if details.get("required_capabilities"):
                traits.append(f"required={','.join(details['required_capabilities'])}")
            if details.get("optional_capabilities"):
                traits.append(f"optional={','.join(details['optional_capabilities'])}")
            if details.get("blocking_missing_capabilities"):
                traits.append(f"blocking_missing={','.join(details['blocking_missing_capabilities'])}")
            if details.get("warning_missing_capabilities"):
                traits.append(f"warning_missing={','.join(details['warning_missing_capabilities'])}")
            lines.append(f"- {name}: {'; '.join(traits)}")
        return lines or ["- none"]

    filter_lines = []
    for name, details in sorted(contract["filter_requirements"].items()):
        traits = [f"status={details['status']}"]
        if details.get("required_capabilities"):
            traits.append(f"required={','.join(details['required_capabilities'])}")
        if details.get("blocking_missing_capabilities"):
            traits.append(f"blocking_missing={','.join(details['blocking_missing_capabilities'])}")
        filter_lines.append(f"- {name}: {'; '.join(traits)}")

    return "\n".join([
        "Runtime operation readiness contract:",
        "- Use ready operations normally.",
        "- Use degraded operations only when they are the best available match; expect executor guardrails or fallbacks.",
        "- Do not choose blocked operations or blocked FFmpeg filters.",
        f"- Direct filter chains cannot use filter_complex-only filters: {', '.join(sorted(FILTER_COMPLEX_ONLY))}.",
        "- Analysis function readiness:",
        *operation_lines(contract["analysis_functions"]),
        "- Special operation readiness:",
        *operation_lines(contract["special_operations"]),
        "- FFmpeg filter readiness:",
        *(filter_lines or ["- none"]),
    ])


def confidence_band(score):
    bands = VALIDATION_POLICY["confidence_bands"]
    if score >= bands["execute"]:
        return "execute"
    if score >= bands["execute_with_guardrails"]:
        return "execute_with_guardrails"
    if score >= bands["fallback_or_confirm"]:
        return "fallback_or_confirm"
    return "clarify_or_switch_pipeline"


def retry_policy_for_confidence(score):
    retry_policy = VALIDATION_POLICY["retry_policy"]
    max_attempts = (
        retry_policy["guardrailed_max_attempts"]
        if score >= retry_policy["retry_threshold"]
        else retry_policy["default_max_attempts"]
    )
    return {"max_attempts": max_attempts}


def validation_threshold(name):
    return VALIDATION_POLICY["confidence_bands"][name]


def external_dependency_prefixes():
    return tuple(VALIDATION_POLICY["external_dependency_prefixes"])


def planner_fallback_policy(reason):
    policy = PLANNER_FALLBACK_POLICY.get(reason)
    if policy:
        return dict(policy)
    return {
        "mode": "reject",
        "allows_execution": False,
        "requires_validation": True,
        "cache_source": None,
        "user_visible_warning": True,
        "description": "Unknown fallback reason; fail closed.",
    }


def execution_failure_policy(step_type=None):
    if step_type is None:
        return {
            name: dict(policy)
            for name, policy in sorted(EXECUTION_FAILURE_POLICY.items())
        }
    return dict(EXECUTION_FAILURE_POLICY.get(step_type, {
        "mode": "fail_closed",
        "allows_partial_success": False,
        "description": "Unknown execution phases fail closed by default.",
    }))


def validation_issue_spec(code):
    if code in VALIDATION_ISSUE_SPECS:
        return dict(VALIDATION_ISSUE_SPECS[code])
    if str(code or "").endswith("_missing"):
        return dict(VALIDATION_ISSUE_SPECS["runtime_capability_missing"])
    return {
        "owner": "unknown",
        "repairable_by_model": False,
        "repair_hint": "No registered repair strategy exists for this validation issue.",
    }


def enrich_validation_entry(entry):
    enriched = dict(entry or {})
    spec = validation_issue_spec(enriched.get("code"))
    enriched.setdefault("owner", spec.get("owner"))
    enriched.setdefault("repairable_by_model", bool(spec.get("repairable_by_model")))
    enriched.setdefault("repair_hint", spec.get("repair_hint"))
    return enriched


def enrich_validation_entries(entries):
    return [enrich_validation_entry(entry) for entry in entries or []]


def validation_error_codes(validation):
    return [
        issue.get("code")
        for issue in validation.get("issues", [])
        if issue.get("severity") == "error"
    ]


def validation_is_model_repairable(validation):
    issues = [
        enrich_validation_entry(issue)
        for issue in validation.get("issues", [])
        if issue.get("severity") == "error"
    ]
    return bool(issues) and all(issue.get("repairable_by_model") for issue in issues)


def validation_repair_summary(validation):
    issues = enrich_validation_entries(validation.get("issues", []))
    warnings = enrich_validation_entries(validation.get("warnings", []))
    return {
        "repairable_by_model": validation_is_model_repairable({"issues": issues}),
        "error_codes": validation_error_codes({"issues": issues}),
        "issue_hints": [
            {
                "code": issue.get("code"),
                "owner": issue.get("owner"),
                "repairable_by_model": issue.get("repairable_by_model"),
                "repair_hint": issue.get("repair_hint"),
                "step_id": issue.get("step_id"),
            }
            for issue in issues
        ],
        "warning_hints": [
            {
                "code": warning.get("code"),
                "owner": warning.get("owner"),
                "repair_hint": warning.get("repair_hint"),
                "step_id": warning.get("step_id"),
            }
            for warning in warnings
        ],
    }


def architecture_summary():
    return {
        "version": ARCHITECTURE_VERSION,
        "fingerprint": architecture_fingerprint(),
        "internal_plan_schema": SCHEMA_VERSION,
        "supported_analysis_functions": sorted(SUPPORTED_ANALYSIS_FUNCTIONS),
        "supported_special_types": sorted(SUPPORTED_SPECIAL_TYPES),
        "capability_specs": {
            capability: dict(spec)
            for capability, spec in sorted(CAPABILITY_SPECS.items())
        },
        "analysis_function_specs": {
            function: dict(spec)
            for function, spec in sorted(ANALYSIS_FUNCTION_SPECS.items())
        },
        "special_operation_specs": {
            operation_type: dict(spec)
            for operation_type, spec in sorted(SPECIAL_OPERATION_SPECS.items())
        },
        "filter_complex_only": sorted(FILTER_COMPLEX_ONLY),
        "filter_capability_requirements": {
            name: list(requirements)
            for name, requirements in sorted(FILTER_CAPABILITY_REQUIREMENTS.items())
        },
        "default_final_encode": default_final_encode(),
        "required_final_encode_keys": list(REQUIRED_FINAL_ENCODE_KEYS),
        "worker_by_step_type": dict(WORKER_BY_STEP_TYPE),
        "special_workers": dict(sorted(SPECIAL_WORKERS.items())),
        "validation_policy": dict(VALIDATION_POLICY),
        "planner_fallback_policy": {
            reason: dict(policy)
            for reason, policy in sorted(PLANNER_FALLBACK_POLICY.items())
        },
        "execution_failure_policy": execution_failure_policy(),
        "validation_issue_specs": {
            code: dict(spec)
            for code, spec in sorted(VALIDATION_ISSUE_SPECS.items())
        },
        "registry_issues": architecture_registry_issues(),
    }


def architecture_contract():
    return {
        "version": ARCHITECTURE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "capability_specs": {
            capability: dict(spec)
            for capability, spec in sorted(CAPABILITY_SPECS.items())
        },
        "analysis_function_specs": {
            function: dict(spec)
            for function, spec in sorted(ANALYSIS_FUNCTION_SPECS.items())
        },
        "special_operation_specs": {
            operation_type: dict(spec)
            for operation_type, spec in sorted(SPECIAL_OPERATION_SPECS.items())
        },
        "filter_complex_only": sorted(FILTER_COMPLEX_ONLY),
        "filter_capability_requirements": {
            name: list(requirements)
            for name, requirements in sorted(FILTER_CAPABILITY_REQUIREMENTS.items())
        },
        "final_encode": {
            "required_keys": list(REQUIRED_FINAL_ENCODE_KEYS),
            "default": default_final_encode(),
            "supported_video_codecs": ["libx264", "libx265"],
            "supported_audio_codecs": ["aac", "libmp3lame"],
            "crf_range": [16, 35],
            "audio_bitrate_kbps_range": [64, 512],
            "max_dimension": 7680,
        },
        "validation_policy": VALIDATION_POLICY,
        "planner_fallback_policy": {
            reason: dict(policy)
            for reason, policy in sorted(PLANNER_FALLBACK_POLICY.items())
        },
        "execution_failure_policy": execution_failure_policy(),
        "validation_issue_specs": {
            code: dict(spec)
            for code, spec in sorted(VALIDATION_ISSUE_SPECS.items())
        },
    }


def architecture_fingerprint():
    payload = json.dumps(architecture_contract(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def architecture_prompt_contract():
    analysis_lines = []
    for function, spec in sorted(ANALYSIS_FUNCTION_SPECS.items()):
        contexts = ", ".join(spec.get("contexts", []))
        requirements = ", ".join(spec.get("required_capabilities", [])) or "none"
        analysis_lines.append(
            f"- {function}: store_as={spec.get('store_as')}; contexts={contexts}; required_capabilities={requirements}"
        )

    special_lines = []
    for operation_type, spec in sorted(SPECIAL_OPERATION_SPECS.items()):
        traits = [f"worker={spec.get('worker')}", f"domain={spec.get('domain')}"]
        if spec.get("requires_uploaded_audio"):
            traits.append("requires_uploaded_audio=true")
        if spec.get("context_param"):
            traits.append("context_param=true")
        if spec.get("required_capabilities"):
            traits.append(f"required_capabilities={','.join(spec.get('required_capabilities'))}")
        if spec.get("optional_capabilities"):
            traits.append(f"optional_capabilities={','.join(spec.get('optional_capabilities'))}")
        special_lines.append(f"- {operation_type}: {'; '.join(traits)}")

    filter_lines = []
    for name, requirements in sorted(FILTER_CAPABILITY_REQUIREMENTS.items()):
        filter_lines.append(f"- {name}: required_capabilities={','.join(requirements)}")

    return "\n".join([
        "Active Linguist executor architecture contract:",
        f"- architecture_version={ARCHITECTURE_VERSION}",
        f"- architecture_fingerprint={architecture_fingerprint()}",
        "- Direct video_filters/audio_filters execute as single-input -vf/-af chains only.",
        "- Only use analysis functions listed here:",
        *analysis_lines,
        "- Only use special operation types listed here:",
        *special_lines,
        f"- Do not put these filter_complex-only filters directly in video_filters/audio_filters: {', '.join(sorted(FILTER_COMPLEX_ONLY))}.",
        "- Single-input FFmpeg filter capability requirements:",
        *filter_lines,
        "- final_encode must use vcodec libx264/libx265, acodec aac/libmp3lame, crf 16-35, preset ultrafast/superfast/veryfast/faster/fast/medium.",
        "- Prefer video_filters/audio_filters for single-input FFmpeg chains; represent multi-input or timeline-assembly work with supported special operations.",
    ])
