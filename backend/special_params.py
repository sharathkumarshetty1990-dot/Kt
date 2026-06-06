import hashlib
import json
import re


SPECIAL_PARAM_CONTRACT_VERSION = "linguist-special-param-contract-v1"


def clamp_float(value, default, minimum, maximum):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = float(default)
    return max(float(minimum), min(float(maximum), parsed))


def clamp_int(value, default, minimum, maximum):
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = int(default)
    return max(int(minimum), min(int(maximum), parsed))


def bool_value(value, default=False):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "on"}:
            return True
        if lowered in {"false", "0", "no", "n", "off"}:
            return False
    if value is None:
        return bool(default)
    return bool(value)


def text_value(value, default="", max_length=120):
    text = str(value if value is not None else default).strip()
    return text[:max_length]


def choice_value(value, default, choices):
    text = str(value if value is not None else default).strip().lower()
    return text if text in set(choices) else default


def ffmpeg_color_value(value, default="black"):
    text = text_value(value, default, 32).lower()
    if re.fullmatch(r"#[0-9a-f]{6}", text):
        return text
    if re.fullmatch(r"[a-z_]+", text):
        return text
    return default


def even_dimension(value, default, minimum=120, maximum=7680):
    dimension = clamp_int(value, default, minimum, maximum)
    if dimension % 2:
        dimension -= 1
    return max(2, dimension)


def normalize_special_params(operation_type, params):
    source = params if isinstance(params, dict) else {}
    op = str(operation_type or "")

    if op == "silence_remove":
        return {
            "threshold_db": clamp_float(source.get("threshold_db"), -35, -80, -5),
            "min_silence_duration": clamp_float(source.get("min_silence_duration"), 0.5, 0.1, 5.0),
        }
    if op == "black_remove":
        return {
            "min_black_duration": clamp_float(source.get("min_black_duration"), 0.5, 0.1, 10.0),
            "pixel_threshold": clamp_float(source.get("pixel_threshold", source.get("pix_th")), 0.1, 0.0, 1.0),
            "picture_threshold": clamp_float(source.get("picture_threshold", source.get("pic_th")), 0.98, 0.0, 1.0),
        }
    if op == "freeze_remove":
        return {
            "noise_db": clamp_float(source.get("noise_db", source.get("noise")), -60, -100, 0),
            "min_duration": clamp_float(source.get("min_duration", source.get("duration")), 0.5, 0.1, 10.0),
        }
    if op == "dedupe_frames":
        return {
            "hi": clamp_int(source.get("hi"), 768, 0, 65535),
            "lo": clamp_int(source.get("lo"), 320, 0, 65535),
            "frac": clamp_float(source.get("frac"), 0.33, 0.0, 1.0),
            "max": clamp_int(source.get("max"), 12, 1, 1000),
        }
    if op == "beat_cut":
        return {
            "context": text_value(source.get("context"), "beat_times", 64) or "beat_times",
            "slice_duration": clamp_float(source.get("slice_duration"), 0.35, 0.12, 1.5),
            "max_cuts": clamp_int(source.get("max_cuts"), 24, 1, 96),
        }
    if op == "scene_montage":
        return {
            "threshold": clamp_float(source.get("threshold"), 0.28, 0.05, 0.95),
            "slice_duration": clamp_float(source.get("slice_duration"), 1.2, 0.25, 4.0),
            "max_segments": clamp_int(source.get("max_segments"), 12, 1, 48),
        }
    if op == "energy_montage":
        return {
            "context": text_value(source.get("context"), "energy_curve_times", 64) or "energy_curve_times",
            "slice_duration": clamp_float(source.get("slice_duration"), 1.0, 0.25, 4.0),
            "max_segments": clamp_int(source.get("max_segments"), 12, 1, 48),
            "pre_roll": clamp_float(source.get("pre_roll"), 0.15, 0.0, 1.5),
        }
    if op == "crop_borders":
        return {
            "limit": clamp_int(source.get("limit"), 24, 0, 255),
            "round": clamp_int(source.get("round"), 2, 2, 64),
            "max_frames": clamp_int(source.get("max_frames"), 120, 10, 600),
        }
    if op == "stabilize":
        return {
            "smoothing": clamp_int(source.get("smoothing"), 10, 1, 60),
            "crop_black": bool_value(source.get("crop_black"), True),
        }
    if op == "pitch_shift":
        return {"semitones": clamp_float(source.get("semitones"), 0, -24, 24)}
    if op == "boomerang":
        return {
            "loops": clamp_int(source.get("loops"), 1, 1, 4),
            "mute_reversed_audio": bool_value(source.get("mute_reversed_audio"), True),
        }
    if op == "end_reverse":
        return {"duration": clamp_float(source.get("duration"), 1.5, 0.35, 10.0)}
    if op == "trim":
        normalized = {}
        for key in ["start", "end", "duration", "from_end", "remove_end"]:
            if key in source:
                normalized[key] = clamp_float(source.get(key), 0.0, 0.0, 86400.0)
        return normalized
    if op == "remove_segment":
        return {
            "start": clamp_float(source.get("start"), 0.0, 0.0, 86400.0),
            "end": clamp_float(source.get("end"), 0.0, 0.0, 86400.0),
        }
    if op == "ocr_redact":
        return {
            "sample_fps": clamp_float(source.get("sample_fps"), 1.0, 0.2, 2.0),
            "max_frames": clamp_int(source.get("max_frames"), 20, 1, 60),
            "confidence": clamp_float(source.get("confidence"), 45, 0, 95),
            "padding": clamp_int(source.get("padding"), 12, 0, 80),
        }
    if op == "face_privacy_blur":
        return {
            "target": choice_value(source.get("target"), "faces", {"faces", "face", "person", "people", "humans"}),
            "layout": choice_value(source.get("layout"), "group", {"group", "center", "body"}),
        }
    if op == "auto_captions":
        return {
            "source": choice_value(source.get("source"), "speech", {"speech", "audio"}),
            "language": text_value(source.get("language"), "en", 16) or "en",
            "style": text_value(source.get("style"), "bottom_box", 48) or "bottom_box",
            "max_segments": clamp_int(source.get("max_segments"), 80, 1, 160),
            "font_size": clamp_int(source.get("font_size"), 44, 18, 96),
            "position": choice_value(source.get("position"), "bottom", {"bottom", "top", "upper"}),
        }
    if op == "picture_in_picture":
        return {
            "position": choice_value(source.get("position"), "top_right", {"top_right", "top_left", "bottom_right", "bottom_left"}),
            "scale": clamp_float(source.get("scale"), 0.32, 0.12, 0.5),
            "margin": clamp_int(source.get("margin"), 24, 0, 256),
        }
    if op == "split_screen_mirror":
        return {"divider_color": ffmpeg_color_value(source.get("divider_color"), "white")}
    if op == "blur_background":
        return {
            "width": even_dimension(source.get("width"), 1080),
            "height": even_dimension(source.get("height"), 1920),
            "sigma": clamp_float(source.get("sigma"), 28.0, 4.0, 80.0),
            "background_saturation": clamp_float(source.get("background_saturation"), 1.08, 0.2, 2.0),
            "background_brightness": clamp_float(source.get("background_brightness"), -0.04, -0.35, 0.35),
        }
    if op == "chroma_key":
        return {
            "key_color": choice_value(source.get("key_color"), "green", {"green", "blue", "black", "white"}),
            "replacement_color": ffmpeg_color_value(source.get("replacement_color"), "black"),
            "similarity": clamp_float(source.get("similarity"), 0.20, 0.01, 1.0),
            "blend": clamp_float(source.get("blend"), 0.08, 0.0, 1.0),
        }
    if op == "film_damage":
        return {
            "intensity": clamp_float(source.get("intensity"), 0.7, 0.1, 1.0),
            "grain": clamp_int(source.get("grain"), 32, 10, 54),
            "gate_weave": clamp_int(source.get("gate_weave"), 8, 2, 16),
            "scratch_opacity": clamp_float(source.get("scratch_opacity"), 0.26, 0.06, 0.40),
            "dust_opacity": clamp_float(source.get("dust_opacity"), 0.42, 0.10, 0.60),
        }
    if op == "speed_ramp":
        normalized = {}
        if any(key in source for key in ["factor", "speed_factor", "tempo"]):
            normalized["factor"] = clamp_float(
                source.get("factor", source.get("speed_factor", source.get("tempo"))),
                1.0,
                0.1,
                8.0,
            )
        else:
            normalized["slow_factor"] = clamp_float(source.get("slow_factor"), 0.5, 0.1, 8.0)
            normalized["fast_factor"] = clamp_float(source.get("fast_factor"), 2.0, 0.1, 8.0)
        return normalized
    if op == "mix_uploaded_audio":
        return {
            "original_volume": clamp_float(source.get("original_volume"), 1.0, 0.0, 2.0),
            "music_volume": clamp_float(source.get("music_volume"), 0.35, 0.0, 2.0),
            "duck": bool_value(source.get("duck"), False),
        }
    if op in {"remove_audio", "replace_audio", "reverse"}:
        return {}

    return dict(source)


def special_param_contract():
    return {
        "version": SPECIAL_PARAM_CONTRACT_VERSION,
        "policy": "Known special operation params are type-normalized and clamped before planning/execution.",
        "normalized_special_types": [
            "auto_captions",
            "beat_cut",
            "black_remove",
            "blur_background",
            "boomerang",
            "chroma_key",
            "crop_borders",
            "dedupe_frames",
            "end_reverse",
            "energy_montage",
            "face_privacy_blur",
            "film_damage",
            "freeze_remove",
            "mix_uploaded_audio",
            "ocr_redact",
            "picture_in_picture",
            "pitch_shift",
            "remove_audio",
            "remove_segment",
            "replace_audio",
            "reverse",
            "scene_montage",
            "silence_remove",
            "speed_ramp",
            "split_screen_mirror",
            "stabilize",
            "trim",
        ],
    }


def special_param_contract_fingerprint():
    payload = json.dumps(special_param_contract(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
