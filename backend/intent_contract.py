import re


def normalized_text(value):
    return re.sub(r"\s+", " ", str(value or "").lower()).strip()


def contains_any(text, phrases):
    return any(phrase in text for phrase in phrases)


def plan_list(plan, key):
    value = plan.get(key) if isinstance(plan, dict) else None
    if not value:
        return []
    return value if isinstance(value, list) else [value]


def special_types(plan):
    return {
        step.get("type")
        for step in plan_list(plan, "special")
        if isinstance(step, dict) and step.get("type")
    }


def analysis_functions(plan):
    return {
        step.get("function")
        for step in plan_list(plan, "analysis")
        if isinstance(step, dict) and step.get("function")
    }


def filter_text(plan):
    parts = []
    for key in ["video_filters", "audio_filters"]:
        for step in plan_list(plan, key):
            if isinstance(step, dict):
                parts.extend([
                    str(step.get("description") or ""),
                    str(step.get("filter") or ""),
                    str(step.get("requires_context") or ""),
                    str(step.get("timing") or ""),
                ])
            else:
                parts.append(str(step))
    return normalized_text(" ".join(parts))


def has_beat_logic(plan):
    specials = special_types(plan)
    analyses = analysis_functions(plan)
    text = filter_text(plan)
    return (
        "beat_track" in analyses
        or bool(specials & {"beat_cut", "energy_montage"})
        or "beat_times" in text
        or "per_beat" in text
    )


def has_onset_or_energy_logic(plan):
    analyses = analysis_functions(plan)
    text = filter_text(plan)
    return (
        bool(analyses & {"onset_detect", "rms_energy", "beat_track"})
        or "onset_times" in text
        or "energy_curve" in text
        or "per_onset" in text
    )


def has_audio_cleanup(plan):
    text = filter_text(plan)
    return any(
        token in text
        for token in ["afftdn", "agate", "deesser", "speechnorm", "loudnorm", "highpass", "lowpass"]
    )


def has_drawtext(plan):
    return "drawtext" in filter_text(plan)


def no_audio_sync_requested(text):
    return contains_any(text, [
        "without audio sync",
        "without audio synchronization",
        "no audio sync",
        "no audio synchronization",
        "do not use audio sync",
        "don't use audio sync",
    ])


def requirement_missing(code, message, expected, command_text):
    return {
        "code": "intent_requirement_missing",
        "message": message,
        "severity": "error",
        "intent_code": code,
        "expected": expected,
        "command_excerpt": str(command_text or "")[:240],
    }


def intent_coverage_issues(command_text, plan):
    text = normalized_text(command_text)
    if not text:
        return []

    issues = []
    specials = special_types(plan)
    analyses = analysis_functions(plan)
    filters = filter_text(plan)
    no_audio_sync = no_audio_sync_requested(text)

    if not no_audio_sync and contains_any(text, [
        "on every beat", "every beat", "beat sync", "sync to the beat", "sync with the beat",
        "sync to music", "music sync", "rhythm", "on the drop", "bass hit", "bass hits",
    ]):
        if not has_beat_logic(plan):
            issues.append(requirement_missing(
                "beat_sync",
                "Command asks for beat/music-synced editing, but the plan has no beat analysis or beat-timed operation.",
                "Add beat_track analysis plus beat_cut or per_beat video/audio operations.",
                command_text,
            ))

    if not no_audio_sync and contains_any(text, ["on impact", "on impacts", "on hit", "on hits", "snap cut", "transient"]):
        if not has_onset_or_energy_logic(plan):
            issues.append(requirement_missing(
                "impact_sync",
                "Command asks for impact/transient-based timing, but the plan has no onset, beat, or energy analysis.",
                "Add onset_detect or rms_energy analysis and use context-timed operations.",
                command_text,
            ))

    if contains_any(text, ["auto caption", "auto captions", "caption the speech", "transcribe", "subtitles from speech"]):
        if "auto_captions" not in specials:
            issues.append(requirement_missing(
                "auto_captions",
                "Command asks for generated speech captions/subtitles, but the plan has no auto_captions operation.",
                "Add special operation auto_captions.",
                command_text,
            ))

    if contains_any(text, ["caption that says", "text that says", "add text", "title that says"]):
        if not has_drawtext(plan):
            issues.append(requirement_missing(
                "text_overlay",
                "Command asks for visible text, but the plan has no drawtext video filter.",
                "Add a drawtext video filter with the requested text.",
                command_text,
            ))

    if contains_any(text, ["remove silence", "remove silences", "cut silence", "cut silences", "remove pauses", "cut pauses"]):
        if "silence_remove" not in specials:
            issues.append(requirement_missing(
                "silence_remove",
                "Command asks to remove silence/pauses, but the plan has no silence_remove operation.",
                "Add special operation silence_remove.",
                command_text,
            ))

    if contains_any(text, ["stabilize", "stabilise", "shaky footage", "smooth camera"]):
        if "stabilize" not in specials:
            issues.append(requirement_missing(
                "stabilize",
                "Command asks for stabilization, but the plan has no stabilize operation.",
                "Add special operation stabilize.",
                command_text,
            ))

    if contains_any(text, ["blur face", "blur faces", "censor face", "censor faces", "hide face", "hide faces"]):
        if "face_privacy_blur" not in specials:
            issues.append(requirement_missing(
                "face_privacy_blur",
                "Command asks for face privacy, but the plan has no face_privacy_blur operation.",
                "Add special operation face_privacy_blur.",
                command_text,
            ))

    if contains_any(text, ["license plate", "number plate", "redact text", "blur text", "hide text on screen"]):
        if "ocr_redact" not in specials:
            issues.append(requirement_missing(
                "ocr_redact",
                "Command asks for OCR/text or plate redaction, but the plan has no ocr_redact operation.",
                "Add special operation ocr_redact.",
                command_text,
            ))

    if contains_any(text, ["replace audio", "replace the audio", "replace soundtrack", "replace music"]):
        if not bool(specials & {"replace_audio", "mix_uploaded_audio"}):
            issues.append(requirement_missing(
                "replace_audio",
                "Command asks to replace audio/music, but the plan has no uploaded-audio operation.",
                "Add special operation replace_audio, or mix_uploaded_audio when the request says to mix.",
                command_text,
            ))

    if contains_any(text, ["background music", "add music", "mix audio", "music under", "duck music"]):
        if "mix_uploaded_audio" not in specials and "replace_audio" not in specials:
            issues.append(requirement_missing(
                "mix_uploaded_audio",
                "Command asks to add or mix music/audio, but the plan has no uploaded-audio mix operation.",
                "Add special operation mix_uploaded_audio.",
                command_text,
            ))

    if contains_any(text, ["clean audio", "denoise audio", "remove background noise", "reduce hiss", "make dialogue clearer"]):
        if not has_audio_cleanup(plan):
            issues.append(requirement_missing(
                "audio_cleanup",
                "Command asks for audio cleanup, but the plan has no cleanup audio filter chain.",
                "Add FFmpeg audio cleanup filters such as afftdn, highpass, lowpass, speechnorm, agate, deesser, or loudnorm.",
                command_text,
            ))

    if contains_any(text, ["reverse video", "play backwards", "play backward", "reverse the clip"]):
        if not bool(specials & {"reverse", "boomerang", "end_reverse"}) and "reverse" not in filters:
            issues.append(requirement_missing(
                "reverse",
                "Command asks for reverse playback, but the plan has no reverse operation.",
                "Add special operation reverse, boomerang, or end_reverse as appropriate.",
                command_text,
            ))

    if contains_any(text, ["speed up", "slow down", "slow motion", "faster", "slower", "speed ramp"]):
        if "speed_ramp" not in specials and "setpts" not in filters and "atempo" not in filters:
            issues.append(requirement_missing(
                "speed_change",
                "Command asks for speed change, but the plan has no speed operation.",
                "Add special operation speed_ramp or synchronized setpts/atempo filters.",
                command_text,
            ))

    return issues
