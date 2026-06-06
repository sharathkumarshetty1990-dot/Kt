import re
from functools import lru_cache
from pathlib import Path


AI_KNOWLEDGE_DIR = Path(__file__).with_name("ai_knowledge")
APPROVED_AI_KNOWLEDGE_FILES = (
    "termux_media_stack_v2.txt",
    "termux_media_engineering_masterbook.txt",
)


BASE_CONTEXT = """Local editing runtime:
- Media execution is real, not simulated.
- Primary executor targets FFmpeg filter chains, librosa analysis, frei0r plugins, rubberband, sox-compatible audio ideas, and multi-step special operations.
- Declared local media packages include libx264, libx265, libvpx, libaom, svt-av1, libopus, libvorbis, libmp3lame, libass, libvidstab, libvmaf, zimg, libplacebo, libsoxr, libsamplerate, OpenCV, SciPy, NumPy, Pillow, MoviePy, MLT, editly, vidgear, pytesseract, rapidocr, pysubs2, pycaption, srt, yt-dlp, and GraphicsMagick. Trust the runtime-verified capability note over this inventory when they disagree.
- FFmpeg, ffprobe, and rubberband may run through Ubuntu proot when native Termux binaries are unavailable.
- Termux_Media_Stack_Manual_v2.pdf has been fed to the AI word-for-word as extracted text in backend/ai_knowledge/termux_media_stack_v2.txt.
- Termux_Media_Engineering_Masterbook.pdf has been fed to the AI as extracted text in backend/ai_knowledge/termux_media_engineering_masterbook.txt.
- Manual guidance has been incorporated: use FFmpeg as the central hub for convert/trim/join/filter/encode/mux/demux/subtitle work; use Python packages for repeatable automation, subtitle handling, analysis, OCR, and batch processing; use npm globals such as editly only when a scripted assembly CLI is a better fit.
- Common verified manual patterns: trim/transcode with ffmpeg -ss/-to plus libx264/AAC; extract audio with ffmpeg -vn; burn subtitles with FFmpeg subtitles/libass; normalize or trim audio with SoX; run OCR with Tesseract on extracted frames/screenshots.
- Package roles from the manual: libvidstab is for stabilization, libvmaf for quality comparison, rubberband for high-quality time-stretch/pitch-shift, librosa for beats/tempo/pitch/spectrograms, OpenCV/NumPy/SciPy for frame and signal analysis, Pillow/GraphicsMagick for image assets, pysubs2/pycaption/srt for subtitle files, and yt-dlp/youtube-transcript-api for ingesting online media/transcripts when supported.
"""


CAPABILITY_GROUPS = [
    {
        "name": "ffmpeg_core_filters",
        "keywords": [
            "edit", "filter", "crop", "resize", "scale", "rotate", "blur", "sharpen",
            "overlay", "text", "watermark", "fade", "transition", "speed", "slow",
            "fast", "trim", "cut", "merge", "concat", "reverse", "letterbox",
        ],
        "packages": [
            "ffmpeg", "ffmpeg-python", "imageio-ffmpeg", "libx264", "libx265",
            "libvpx", "libaom", "svt-av1", "libass", "zimg", "libplacebo",
        ],
        "guidance": [
            "Prefer native FFmpeg -vf and -af filters for edits that can be expressed as filter chains.",
            "Use special operations only for multi-step work such as silence removal, stabilization, reverse, boomerang ping-pong loops, blurred-background social layouts, speed ramps, and pitch shifting.",
            "For remove/cut black screens, blank frames, or dark sections, use special type black_remove rather than a visual filter.",
            "For remove/cut frozen frames, stuck frames, or held-frame sections, use special type freeze_remove.",
            "For remove/drop duplicate or repeated video frames, use special type dedupe_frames.",
            "For removing black bars, letterbox bars, pillarbox bars, or empty borders around the picture, use special type crop_borders.",
            "Use final_encode settings for codec, CRF, preset, audio codec, and bitrate instead of inventing output commands.",
        ],
    },
    {
        "name": "color_grade_and_look_design",
        "keywords": [
            "cinematic", "color", "grade", "lut", "teal", "orange", "warm",
            "cold", "moody", "vintage", "lofi", "dream", "film", "grain",
            "contrast", "saturation", "brightness", "underwater", "night",
        ],
        "packages": [
            "ffmpeg", "frei0r-plugins", "OpenCV", "opencv-python", "opencv-contrib-python",
            "numpy", "scipy", "pillow", "libplacebo", "zimg",
        ],
        "guidance": [
            "Compose looks with eq, colorbalance, curves, hue, lut3d, vignette, noise, gblur, rgbashift, lenscorrection, and geq.",
            "For black-and-white except one color, use colorhold or hsvhold instead of hand-written channel threshold expressions.",
            "Map visual adjectives to numeric intensity using the system prompt scale.",
            "For inspectable output, keep filters valid as plain FFmpeg -vf strings.",
        ],
    },
    {
        "name": "frei0r_visual_effects",
        "keywords": [
            "glitch", "crt", "scanline", "pixel", "pixelate", "halftone",
            "distort", "warp", "vhs", "datamosh", "corrupt", "retro",
        ],
        "packages": ["frei0r-plugins", "ffmpeg", "movit", "mlt"],
        "guidance": [
            "Use supported frei0r plugin names such as glitch0r, distort0r, scanline0r, pixeliz0r, colorhalftone, and vignette.",
            "Represent frei0r effects as FFmpeg video filter strings in video_filters.",
            "For beat-triggered glitches, include librosa beat_track analysis and set timing to per_beat.",
        ],
    },
    {
        "name": "audio_music_and_beats",
        "keywords": [
            "audio", "music", "beat", "beats", "rhythm", "sync", "drop", "bass",
            "loud", "quiet", "normalize", "volume", "voice", "speech", "echo",
            "reverb", "muffle", "noise", "denoise", "tempo", "pitch",
        ],
        "packages": [
            "librosa", "sox", "rubberband", "scipy", "numpy", "libsndfile",
            "libsoxr", "libsamplerate", "libebur128", "libopus", "libvorbis",
            "libmp3lame", "webrtc-audio-processing",
        ],
        "guidance": [
            "For beat, rhythm, music sync, drop, or bass-hit requests, include librosa beat_track analysis storing beat_times.",
            "For cut-to-the-beat, jump cuts on beats, hard cuts on every beat, or edit-to-the-music requests, use special type beat_cut with context beat_times.",
            "For impact, snap, transient, or hit-based requests, include librosa onset_detect storing onset_times.",
            "For loud/quiet/energy-based edits, include librosa rms_energy storing energy_curve.",
            "For hype reels or montages from loudest/highest-energy audio moments, use special type energy_montage with context energy_curve_times.",
            "Use FFmpeg audio filters such as loudnorm, equalizer, aecho, atempo, volume, highpass, lowpass, dynaudnorm, sidechaincompress, and rubberband.",
            "For denoise, remove background noise, reduce hiss, clean audio, or make dialogue clearer, use afftdn, highpass, lowpass, speechnorm, agate, deesser, and loudnorm as a real audio cleanup chain.",
            "When only one mixed audio stream is available, prefer acompressor over sidechaincompress because sidechaincompress requires two audio inputs.",
            "For replacing the video's soundtrack with an uploaded audio file, use special type replace_audio.",
            "For adding uploaded audio as background music under dialogue/voice, use special type mix_uploaded_audio with duck=true when the user asks for ducking or music under speech.",
        ],
    },
    {
        "name": "stabilization_motion_and_quality",
        "keywords": [
            "stabilize", "stabilise", "shaky", "shake", "smooth", "jitter",
            "motion", "tracking", "denoise", "quality", "enhance", "repair",
        ],
        "packages": [
            "libvidstab", "opencv", "opencv-python", "opencv-contrib-python",
            "scikit-video", "vidgear", "numpy", "scipy", "ffmpeg",
        ],
        "guidance": [
            "For camera stabilization, use special type stabilize with smoothing and crop_black params.",
            "For intentional shake, use FFmpeg crop/scale or rotate expressions; for beat shake, require beat_times and timing per_beat.",
            "Use unsharp, gblur, boxblur, noise, and lenscorrection for quality and lens-style changes.",
        ],
    },
    {
        "name": "captions_subtitles_and_text",
        "keywords": [
            "caption", "captions", "subtitle", "subtitles", "srt", "karaoke",
            "lyrics", "text", "title", "lower third", "burn", "transcript",
        ],
        "packages": [
            "libass", "pysubs2", "pycaption", "srt", "youtube-transcript-api",
            "pytesseract", "rapidocr", "tesseract", "ffmpeg",
        ],
        "guidance": [
            "Use FFmpeg drawtext for direct text overlays in video_filters when the user provides explicit text.",
            "For generated captions/subtitles from speech, transcription, or caption-the-speech requests, use special type auto_captions with source=speech.",
            "auto_captions currently runs real FFmpeg/pocketsphinx ASR and burns timed drawtext captions; transcript accuracy is best-effort until a stronger ASR model is installed.",
            "For subtitle styling, libass and pysubs2 are available, but current JSON output should still express executable FFmpeg filter intent.",
            "Keep displayed text safely quoted inside drawtext filter strings.",
        ],
    },
    {
        "name": "computer_vision_and_ocr",
        "keywords": [
            "detect", "face", "object", "scene", "ocr", "read", "license",
            "blur face", "track", "tracking", "screen", "document", "scan",
        ],
        "packages": [
            "opencv", "opencv-python", "opencv-contrib-python", "pytesseract",
            "rapidocr", "tesseract", "numpy", "scipy", "pillow", "scikit-video",
        ],
        "guidance": [
            "For license plates, number plates, serial numbers, screen text, document text, and visible text redaction, use special type ocr_redact.",
            "OpenCV, OCR, and frame analysis packages are installed for executor expansion, but trust runtime readiness before using object tracking.",
            "When OpenCV semantic tracking is not runtime-ready and the user asks to blur/censor faces or people, use special type face_privacy_blur as a safe non-tracking privacy-region fallback.",
            "For the current executor, express results as FFmpeg filters or supported special operations unless a matching JSON operation exists.",
            "If the request needs object-specific tracking that the current JSON cannot represent, approximate with global filters and state the intended effect in intent.",
        ],
    },
    {
        "name": "image_graphics_and_thumbnails",
        "keywords": [
            "thumbnail", "poster", "image", "png", "jpg", "gif", "sticker",
            "logo", "overlay", "frame", "contact sheet", "sprite", "title card",
        ],
        "packages": [
            "pillow", "ImageIO", "graphicsmagick", "giflib", "libpng",
            "libjpeg-turbo", "libwebp", "libavif", "libheif", "librsvg",
            "libcairo", "pango", "ffmpeg",
        ],
        "guidance": [
            "Use FFmpeg overlay/drawtext/drawbox where the current JSON can represent the visual result.",
            "Use image tooling as context for future asset generation, but keep output constrained to executable video/audio plan JSON.",
        ],
    },
    {
        "name": "timeline_assembly_and_programmatic_editing",
        "keywords": [
            "montage", "slideshow", "sequence", "timeline", "b-roll", "broll",
            "intro", "outro", "highlight", "recap", "shorts", "reels", "tiktok",
            "youtube", "vertical", "horizontal", "social", "boomerang", "ping-pong",
            "ping pong", "bounce loop", "blurred background", "blur background",
            "blurred sides", "background blur",
        ],
        "packages": [
            "moviepy", "mlt", "editly", "ffmpeg-python", "vidgear", "yt-dlp",
            "opencv", "numpy", "ffmpeg",
        ],
        "guidance": [
            "The machine has higher-level assembly libraries for future timeline generation.",
            "For the current backend, express assembly-like requests with FFmpeg trim, crop, scale, setpts, fade, drawtext, and final encode where possible.",
            "For highlight reel, montage, recap, trailer, quick-cut, or best-moments requests, use special type scene_montage to detect scene changes and concatenate short slices.",
            "For boomerang, ping-pong, bounce-loop, or forward-then-reverse requests, use special type boomerang to concatenate original playback with a reversed copy.",
            "For vertical/reels/social edits with blurred background, blurred sides, or filled side bars, use special type blur_background with width/height target params instead of cropping away the subject.",
            "For green-screen, blue-screen, chroma-key, or remove keyed background requests, use special type chroma_key with key_color and replacement_color params.",
            "For old film, damaged film, 8mm/16mm, scratches, dust, flicker, or gate weave, use special type film_damage.",
            "For comic-book, manga, halftone, inked-outline, or thick-outline looks, include both pixelize and edgedetect in the video filter chain.",
            "For vertical 9:16 social output, prefer this exact final video filter pattern: scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1.",
            "For Instagram 4:5 portrait output, prefer: scale=1080:1350:force_original_aspect_ratio=increase,crop=1080:1350,setsar=1.",
            "For square 1:1 output, prefer: scale=1080:1080:force_original_aspect_ratio=increase,crop=1080:1080,setsar=1.",
            "For cinematic 2.39:1 letterbox output, prefer: scale=1920:804:force_original_aspect_ratio=increase,crop=1920:804,pad=1920:1080:0:(oh-ih)/2:color=black,setsar=1.",
        ],
    },
    {
        "name": "export_codecs_and_delivery",
        "keywords": [
            "export", "compress", "codec", "mp4", "webm", "mov", "h264",
            "h265", "hevc", "av1", "vp9", "size", "bitrate", "1080p",
            "4k", "fps", "framerate", "aac", "opus",
        ],
        "packages": [
            "libx264", "libx265", "libvpx", "libaom", "svt-av1", "libopus",
            "libvorbis", "libmp3lame", "libvmaf", "ffmpeg", "ffprobe",
        ],
        "guidance": [
            "Use final_encode for output settings and keep vcodec/acodec values compatible with FFmpeg.",
            "Default to libx264, CRF 22, fast preset, AAC 192k unless the request asks for a specific delivery target.",
            "Use scale filters for resolution changes and avoid inventing root JSON keys.",
        ],
    },
]


def _score_group(prompt_words, group):
    score = 0
    for keyword in group["keywords"]:
        keyword_words = keyword.lower().split()
        if all(word in prompt_words for word in keyword_words):
            score += 3 if len(keyword_words) > 1 else 1
    return score


def selected_capabilities(prompt, limit=5):
    words = set(re.findall(r"[a-z0-9]+", prompt.lower()))
    scored = [(_score_group(words, group), group) for group in CAPABILITY_GROUPS]
    selected = [group for score, group in sorted(scored, key=lambda item: item[0], reverse=True) if score > 0]
    if not selected:
        selected = [
            group for group in CAPABILITY_GROUPS
            if group["name"] in ["ffmpeg_core_filters", "color_grade_and_look_design", "audio_music_and_beats"]
        ]
    return selected[:limit]


def format_capability_context(prompt):
    sections = [
        BASE_CONTEXT.strip(),
        "Relevant local executor guidance. Full installed package inventory is already in the system prompt."
    ]
    for group in selected_capabilities(prompt):
        sections.append(
            "\n".join([
                f"- {group['name']}",
                "  guidance:",
                *[f"  - {item}" for item in group["guidance"]],
            ])
        )
    knowledge = full_ai_knowledge_context()
    if knowledge:
        sections.append(knowledge)
    return "\n\n".join(sections)


@lru_cache(maxsize=1)
def load_ai_knowledge_files():
    if not AI_KNOWLEDGE_DIR.exists():
        return []
    documents = []
    for name in APPROVED_AI_KNOWLEDGE_FILES:
        path = AI_KNOWLEDGE_DIR / name
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        documents.append((path.name, text))
    return documents


def full_ai_knowledge_context():
    documents = load_ai_knowledge_files()
    if not documents:
        return ""
    sections = [
        "FULL PDF KNOWLEDGE FED TO AI.",
        "Use the following extracted PDF text as authoritative local media-stack knowledge.",
        "The text is included verbatim from the .txt extraction files; do not treat it as a summary.",
    ]
    for name, text in documents:
        sections.append(
            "\n".join([
                f"===== BEGIN FULL PDF EXTRACT: {name} =====",
                text.rstrip(),
                f"===== END FULL PDF EXTRACT: {name} =====",
            ])
        )
    return "\n\n".join(sections)


def augment_prompt_for_capabilities(prompt, runtime_note=None, architecture_note=None):
    context = format_capability_context(prompt)
    runtime_section = f"\n\nRuntime-verified executor state:\n{runtime_note}" if runtime_note else ""
    architecture_section = f"\n\n{architecture_note}" if architecture_note else ""
    return f"""Use this local installed capability context when planning the edit.

{context}
{runtime_section}
{architecture_section}

User editing request:
{prompt}

Return only the required JSON plan. Do not add unsupported root keys."""
