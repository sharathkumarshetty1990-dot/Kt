import hashlib
import json

from editing_architecture import REQUIRED_FINAL_ENCODE_KEYS, default_final_encode, normalize_final_encode_settings
from special_params import (
    normalize_special_params,
    special_param_contract,
    special_param_contract_fingerprint,
)


PUBLIC_PLAN_CONTRACT_VERSION = "linguist-public-plan-contract-v1"
PLAN_LIST_KEYS = ("analysis", "video_filters", "audio_filters", "special")
PUBLIC_PLAN_KEYS = ("intent", *PLAN_LIST_KEYS, "final_encode")


def clone_plain(value):
    if isinstance(value, dict):
        return {key: clone_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_plain(item) for item in value]
    return value


def list_steps(value):
    if not value:
        return []
    return value if isinstance(value, list) else [value]


def normalize_analysis_step(step):
    if isinstance(step, str):
        step = {"tool": "librosa", "function": step}
    if not isinstance(step, dict):
        return None
    normalized = clone_plain(step)
    function = normalized.get("function")
    if function and not normalized.get("store_as"):
        normalized["store_as"] = {
            "beat_track": "beat_times",
            "onset_detect": "onset_times",
            "rms_energy": "energy_curve",
        }.get(function, function)
    if normalized.get("function"):
        normalized.setdefault("tool", "librosa")
        return normalized
    return None


def normalize_filter_step(step):
    if isinstance(step, str):
        step = {"filter": step}
    if not isinstance(step, dict):
        return None
    normalized = clone_plain(step)
    if normalized.get("filter"):
        normalized.setdefault("description", "Generated FFmpeg filter step")
        normalized.setdefault("requires_context", None)
        normalized.setdefault("timing", "continuous")
        return normalized
    return None


def normalize_special_step(step):
    if isinstance(step, str):
        step = {"type": step, "params": {}}
    if not isinstance(step, dict):
        return None
    normalized = clone_plain(step)
    if not normalized.get("type"):
        return None
    params = normalized.get("params")
    normalized["params"] = normalize_special_params(normalized.get("type"), params)
    return normalized


def normalize_plan_list(key, value):
    normalizer = {
        "analysis": normalize_analysis_step,
        "video_filters": normalize_filter_step,
        "audio_filters": normalize_filter_step,
        "special": normalize_special_step,
    }[key]
    normalized = []
    for step in list_steps(value):
        item = normalizer(step)
        if item:
            normalized.append(item)
    return normalized


def normalize_public_plan_shape(plan, intent_fallback="AI generated video edit"):
    source = plan if isinstance(plan, dict) else {}
    normalized = {}
    fixes = []

    intent = str(source.get("intent") or intent_fallback or "").strip()
    if not intent:
        intent = "AI generated video edit"
    if source.get("intent") != intent:
        fixes.append("normalized intent")
    normalized["intent"] = intent

    for key in PLAN_LIST_KEYS:
        values = normalize_plan_list(key, source.get(key))
        if values:
            normalized[key] = values
        if source.get(key) and source.get(key) != values:
            fixes.append(f"normalized {key}")

    final_encode, encode_fixes = normalize_final_encode_settings(source.get("final_encode"))
    normalized["final_encode"] = final_encode
    fixes.extend(f"normalized final_encode: {item}" for item in encode_fixes)

    return normalized, fixes


def normalize_public_plan(plan, intent_fallback="AI generated video edit"):
    normalized, _fixes = normalize_public_plan_shape(plan, intent_fallback)
    return normalized


def public_plan_contract():
    return {
        "version": PUBLIC_PLAN_CONTRACT_VERSION,
        "root_keys": list(PUBLIC_PLAN_KEYS),
        "list_keys": list(PLAN_LIST_KEYS),
        "required_final_encode_keys": list(REQUIRED_FINAL_ENCODE_KEYS),
        "default_final_encode": default_final_encode(),
        "special_param_contract": {
            **special_param_contract(),
            "fingerprint": special_param_contract_fingerprint(),
        },
        "normalization": {
            "non_object_plan": "converted to default plan shell",
            "missing_intent": "filled from command or generic fallback",
            "missing_lists": "treated as empty lists and omitted from public response",
            "string_analysis_step": "converted to librosa analysis step",
            "string_filter_step": "converted to filter object",
            "string_special_step": "converted to special operation with empty params",
            "special_params": "known special operation params are type-normalized and clamped",
            "missing_filter_timing": "defaulted to continuous",
            "missing_final_encode": "normalized to production defaults",
        },
    }


def public_plan_contract_fingerprint():
    payload = json.dumps(public_plan_contract(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
