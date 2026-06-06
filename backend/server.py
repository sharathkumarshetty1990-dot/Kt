import json
import hashlib
import importlib
import os
import re
import shutil
import subprocess
import tempfile
import wave
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from flask import Flask, jsonify, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge

from ai_orchestrator import (
    create_execution_manifest,
    inspect_output_artifact,
    mark_manifest_complete,
    mark_manifest_failed,
    mark_manifest_running,
    mark_manifest_step_status,
    repair_packet_from_exception,
)
from ai_planner import prepare_production_plan, validation_allows_execution
from editing_capabilities import augment_prompt_for_capabilities
from editing_architecture import (
    SUPPORTED_SPECIAL_TYPES,
    architecture_fingerprint,
    architecture_prompt_contract,
    architecture_registry_issues,
    architecture_required_capabilities,
    architecture_summary,
    default_final_encode as architecture_default_final_encode,
    execution_failure_policy,
    normalize_final_encode_settings as architecture_normalize_final_encode_settings,
    planner_fallback_policy,
    runtime_operation_contract,
    runtime_operation_prompt_contract,
    validation_error_codes,
    validation_is_model_repairable,
    validation_repair_summary,
)
from job_lifecycle import (
    STATUS_COMPLETE,
    STATUS_ERROR,
    STATUS_PLAN_REJECTED,
    STATUS_PROCESSING,
    STATUS_UPLOADED,
    job_lifecycle_summary,
    transition_job_status,
)
from job_store import JobStore
from job_queue import ThreadedJobQueue
from llm_provider import NimChatProvider
from media_runner import MediaCommandRunner
from plan_contract import (
    normalize_public_plan_shape,
    public_plan_contract,
    public_plan_contract_fingerprint,
)
from planner_cache import PlannerCache, clone_plan
from runtime_cache import RuntimeCapabilityCache
from special_params import special_param_contract, special_param_contract_fingerprint


app = Flask(__name__)

ALLOWED_ORIGIN = os.environ.get("LINGUIST_ALLOWED_ORIGIN", "http://127.0.0.1:8000")
REQUESTED_UPLOAD_ROOT = Path(os.environ.get("LINGUIST_UPLOAD_ROOT", "/tmp/linguist"))
FALLBACK_UPLOAD_ROOT = Path(tempfile.gettempdir()) / "linguist"
NIM_API_URL = os.environ.get("NIM_API_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
NIM_MODEL = os.environ.get("NIM_MODEL", "meta/llama-3.1-70b-instruct")
NIM_API_KEY = (os.environ.get("NIM_API_KEY") or os.environ.get("NGC_API_KEY") or "").strip()
NIM_TIMEOUT_SECONDS = int(os.environ.get("NIM_TIMEOUT_SECONDS", "18"))
NIM_MAX_ATTEMPTS = max(1, int(os.environ.get("NIM_MAX_ATTEMPTS", "1")))
SERVER_HOST = os.environ.get("LINGUIST_HOST", "127.0.0.1")
SERVER_PORT = int(os.environ.get("LINGUIST_PORT", "5000"))
MAX_UPLOAD_MB = int(os.environ.get("LINGUIST_MAX_UPLOAD_MB", "2048"))
WORKER_COUNT = max(1, int(os.environ.get("LINGUIST_WORKERS", "2")))
QUEUE_MAX_PENDING = max(WORKER_COUNT, int(os.environ.get("LINGUIST_QUEUE_MAX_PENDING", str(WORKER_COUNT * 4))))
COMMAND_TIMEOUT_SECONDS = max(30, int(os.environ.get("LINGUIST_COMMAND_TIMEOUT_SECONDS", "300")))
job_queue = ThreadedJobQueue(max_workers=WORKER_COUNT, max_pending=QUEUE_MAX_PENDING)
MEDIA_COMMANDS = {"ffmpeg", "ffprobe", "rubberband"}
PROOT_DISTRO = shutil.which("proot-distro")
PROOT_ROOTFS = Path(os.environ.get(
    "LINGUIST_PROOT_ROOTFS",
    "/data/data/com.termux/files/usr/var/lib/proot-distro/containers/ubuntu/rootfs",
))
OPENCV_DETECTOR_SCRIPT = Path(__file__).with_name("opencv_privacy_detector.py")
media_runner = MediaCommandRunner(
    timeout_seconds=COMMAND_TIMEOUT_SECONDS,
    media_commands=MEDIA_COMMANDS,
    proot_distro=PROOT_DISTRO,
)
PROOT_CV2_PYTHON_CANDIDATES = [
    value.strip()
    for value in os.environ.get(
        "LINGUIST_PROOT_CV2_PYTHONS",
        "/root/myenv/bin/python:/root/video-editor/backend/venv/bin/python",
    ).split(":")
    if value.strip()
]
POCKETSPHINX_HMM = os.environ.get(
    "LINGUIST_POCKETSPHINX_HMM",
    "/usr/share/pocketsphinx/model/en-us/en-us",
)
POCKETSPHINX_DICT = os.environ.get(
    "LINGUIST_POCKETSPHINX_DICT",
    "/usr/share/pocketsphinx/model/en-us/cmudict-en-us.dict",
)
POCKETSPHINX_LM = os.environ.get(
    "LINGUIST_POCKETSPHINX_LM",
    "/usr/share/pocketsphinx/model/en-us/en-us.lm.bin",
)
CAPABILITY_CACHE_SECONDS = int(os.environ.get("LINGUIST_CAPABILITY_CACHE_SECONDS", "300"))
PLAN_CACHE_SECONDS = max(0, int(os.environ.get("LINGUIST_PLAN_CACHE_SECONDS", "900")))
FALLBACK_PLAN_CACHE_SECONDS = max(0, int(os.environ.get("LINGUIST_FALLBACK_PLAN_CACHE_SECONDS", "60")))
PLAN_CACHE_MAX_ENTRIES = max(0, int(os.environ.get("LINGUIST_PLAN_CACHE_MAX_ENTRIES", "128")))
runtime_capability_cache = RuntimeCapabilityCache(CAPABILITY_CACHE_SECONDS)
planner_cache = PlannerCache(
    ttl_seconds=PLAN_CACHE_SECONDS,
    fallback_ttl_seconds=FALLBACK_PLAN_CACHE_SECONDS,
    max_entries=PLAN_CACHE_MAX_ENTRIES,
)

app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

NIM_SYSTEM_PROMPT = """You are LINGUIST ENGINE — the AI brain of a professional video editing system running on a machine with FFmpeg, frei0r-plugins, librosa, rubberband, and sox installed.

Your job: translate ANY natural language video editing request into a precise, executable JSON plan using real FFmpeg filter syntax, real librosa API calls, and real frei0r plugin names.

You are NOT limited to preset effects. You reason from first principles using your deep knowledge of FFmpeg filters and compose any effect the user describes.

══════════════════════
OUTPUT RULES
══════════════════════
1. Output ONLY valid JSON. No markdown. No backticks. No explanation.
2. Root keys: intent, analysis, video_filters, audio_filters, special, final_encode
3. All filter strings must be valid FFmpeg -vf or -af syntax
4. Never output empty arrays — omit a key entirely if unused
5. Always include final_encode
6. When the user mentions beats, rhythm, or music sync — add librosa beat_track to analysis

══════════════════════════════════════
AVAILABLE LOCAL EDITING STACK
══════════════════════════════════════
This machine has a broad Termux, Python, and npm media stack installed. Use it as implementation context, but keep the response in the required JSON schema. Map requests back to FFmpeg filters, librosa analysis, sox/rubberband-compatible audio processing, frei0r plugins, and supported special operations.

Termux packages include: alsa-lib, ffmpeg, fftw, fontconfig, freetype, frei0r-plugins, fribidi, game-music-emu, gdk-pixbuf, giflib, graphicsmagick, harfbuzz, jack, jack2, leptonica, libao, libaom, libass, libavif, libbluray, libbs2b, libcairo, libdav1d, libde265, libebur128, libexif, libflac, libgd, libheif, libid3tag, libjasper, libjpeg-turbo, libjxl, libmad, libmp3lame, libmpg123, libogg, libopencore-amr, libopenmpt, libopus, libplacebo, libpng, librav1e, librsvg, libsamplerate, libsixel, libsndfile, libsoxr, libsrt, libtheora, libtiff, libudfread, libv4l, libvidstab, libvmaf, libvo-amrwbenc, libvorbis, libvpx, libwebp, libwebrtc-audio-processing, libx264, libx265, libzimg, mesa, mesa-vulkan-icd-swrast, mlt, movit, opencv, opencv-python, opengl, openjpeg, openjpeg-tools, opusfile, pango, pulseaudio, python-numpy, python-opencv-python, python-scipy, qt6-qtsvg, rubberband, sdl, sdl2, sox, speexdsp, svt-av1, termimage, tesseract, timg, ttf-dejavu, vulkan-icd, vulkan-loader, vulkan-loader-generic, vulkan-tools, xcb-util-image, xvidcore.

Python packages include: ffmpeg-python, fonttools, ImageIO, imageio-ffmpeg, librosa, moviepy, numpy, opencv-contrib-python, opencv-python, pillow, pycaption, pyclipper, pysubs2, pytesseract, rapidocr, scikit-video, scipy, shapely, srt, vidgear, youtube-transcript-api, yt-dlp.

npm global tools include: editly, freebuff.

Executor note: FFmpeg, ffprobe, and rubberband may be executed through Ubuntu proot when native Termux media binaries are unavailable. The executor applies each video_filters item as a chained single-input -vf filter graph, not filter_complex. Avoid multi-stream filters such as split, hstack, vstack, xstack, and overlay unless they are represented as a single-input approximation.

══════════════════════════════════════
YOUR FFMPEG VIDEO FILTER KNOWLEDGE
══════════════════════════════════════

USE THESE FILTERS TO COMPOSE ANY VISUAL EFFECT:

GEOMETRY & MOTION:
  crop=w:h:x:y
    → cut a region. Use math expressions for animated shake:
      crop=iw-40:ih-40:20+20*sin(10*t):20*cos(10*t),scale=iw+40:ih+40
  zoompan=z='zoom+0.001':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1920x1080
    → smooth zoom/pan over time. z is zoom level expression, d is duration frames
  scale=w:h
    → resize. scale=-1:1080 preserves aspect
  rotate=angle:fillcolor=black
    → rotate. angle in radians, use t for time-based: rotate=0.1*sin(t)
  perspective=x0:y0:x1:y1:x2:y2:x3:y3
    → perspective warp

COLOR:
  eq=contrast=1.2:brightness=0.0:saturation=1.3:gamma=1.0
    → fundamental color adjustments
  hue=h=0:s=1.0
    → hue rotation (radians) and saturation. hue=h=t for animated hue spin
  colorhold=color=red:similarity=0.25:blend=0.12
    → keep one color visible while turning everything else grayscale. Use for "black and white except red/blue/green/etc."
  hsvhold=hue=0.0:sat=0.7:val=0.8:similarity=0.08
    → isolate a hue range while graying out the rest
  colorbalance=rs=-0.1:rm=0.2:rh=0.3:bs=0.1:bm=0.1:bh=-0.2
    → shadow/midtone/highlight balance per channel: rs/gs/bs, rm/gm/bm, rh/gh/bh
  colorchannelmixer=rr=1:rg=0:rb=0:gr=0:gg=1:gb=0:br=0:bg=0:bb=1
    → full color matrix. Use to isolate colors or create cross-processing
  curves=r='0/0 0.5/0.6 1/1':g='0/0 0.5/0.5 1/1':b='0/0 0.5/0.4 1/1'
    → curves adjustment per channel, points as 'input/output' pairs
  lut3d=file=lut.cube
    → apply .cube LUT file
  negate
    → color invert
  geq=r='p(X,Y)':g='p(X,Y)':b='p(X,Y)'
    → per-pixel math. p(X,Y) = pixel value at X,Y. Use for custom effects

BLUR & DISTORTION:
  gblur=sigma=3
    → gaussian blur. sigma controls radius
  boxblur=5:1
    → box blur, luma_radius:luma_power
  unsharp=5:5:1.0
    → sharpen. luma_msize_x:luma_msize_y:luma_amount
  rgbashift=rh=5:bh=-5:rv=3:bv=-3
    → chromatic aberration. rh/rv = red horizontal/vertical shift. bh/bv = blue
  vignette=angle=PI/4
    → vignette darkening. angle controls spread
  lenscorrection=k1=-0.3:k2=0
    → barrel/pincushion distortion. k1 negative=barrel, positive=pincushion

NOISE & TEXTURE:
  noise=alls=25:allf=t+u
    → film grain/noise. alls=strength, allf=t (temporal) u (uniform) or t+u
  drawgrid=width=1:color=white@0.05:thickness=1
    → subtle grid overlay for texture

BLEND & OVERLAY:
  fade=t=in:st=0:d=1:color=black
    → fade in/out. t=in or out, st=start time, d=duration
  drawbox=x=0:y=0:w=iw:h=ih:color=white@0.9:t=fill
    → color flash/overlay. Use enable= for triggered flashes
  blend=all_expr='A*(1-0.5)+B*0.5'
    → blend two streams

SPEED & TIME:
  setpts=0.5*PTS
    → speed up (0.5 = 2x speed, 2.0 = half speed)
  reverse
    → reverse video. Use with areverse for audio

TEXT:
  drawtext=text='YOUR TEXT':fontcolor=white:fontsize=48:x=(w-text_w)/2:y=(h-text_h)/2
    → burn text. Supports fontfile, box, boxcolor, shadowcolor, enable=

ANALYSIS-DEPENDENT (use when beat_times or onset_times from librosa):
  For beat-synced effects use enable= with between(t, start, end) expressions
  Example shake on beats (beat_times = [0.5, 1.0, 1.5, 2.0]):
    crop=iw-40:ih-40:20+20*if(BETWEEN_EXPR,1,0):10,scale=iw+40:ih+40
  where BETWEEN_EXPR = between(t,0.49,0.55)+between(t,0.99,1.05)+...

  For flash on beats:
    drawbox=x=0:y=0:w=iw:h=ih:color=white@0.9:t=fill:enable='BETWEEN_EXPR'

FREI0R PLUGINS (use as: frei0r=filter_name=NAME:filter_params=VALUE):
  glitch0r   filter_params=0.0-1.0 (intensity)   → digital glitch corruption
  distort0r  filter_params=0.0-1.0 (amount)      → geometric warp distortion
  scanline0r filter_params=0.0-1.0 (opacity)     → CRT scanlines
  pixeliz0r  filter_params=0.0-1.0 (block size)  → pixelation
  colorhalftone                            → halftone dot pattern
  vignette   filter_params=0.0-1.0               → circular vignette (alternative)

══════════════════════════════════════
YOUR FFMPEG AUDIO FILTER KNOWLEDGE
══════════════════════════════════════

  loudnorm=I=-14:TP=-1.5:LRA=11
    → EBU R128 loudness normalization. I=target LUFS
  equalizer=f=80:width_type=o:width=2:g=6
    → parametric EQ band. f=frequency, g=gain dB. Chain multiple for full EQ
  aecho=0.8:0.5:500:0.5
    → reverb/echo. in_gain:out_gain:delay_ms:decay
  atempo=1.5
    → time stretch audio (0.5-2.0 range, chain for wider range)
  asetpts=0.5*PTS
    → change audio speed (use with setpts for sync)
  volume=2.0
    → volume adjustment. Can use expressions: volume=1+0.5*sin(t)
  highpass=f=200
    → high pass filter, removes low frequencies below f Hz
  lowpass=f=3000
    → low pass filter
  dynaudnorm=f=150:g=15
    → dynamic audio normalization, smooths out loud/quiet sections
  sidechaincompress=threshold=0.02:ratio=4:release=200
    → sidechain compression for ducking music under voice
  afftdn=nr=18:nf=-35
    → FFT denoise for hiss, background noise, fan noise, and room tone
  speechnorm=e=4:c=2:r=0.0005:l=1
    → normalize speech dynamics so dialogue becomes clearer and more even
  deesser=i=0.45:m=0.55:f=0.5
    → reduce harsh sibilance and sharp "s" sounds
  agate=threshold=0.035:ratio=2.5:attack=8:release=120
    → reduce low-level background noise between speech phrases
  rubberband=tempo=1.0:pitch=1.0
    → high quality time stretch + pitch shift via librubberband

══════════════════════════════════════
YOUR LIBROSA KNOWLEDGE
══════════════════════════════════════

When you add to "analysis", use these exact function references:

  { "tool": "librosa", "function": "beat_track", "store_as": "beat_times" }
    → librosa.beat.beat_track() → returns array of beat timestamps in seconds
    → USE when: user mentions beat, rhythm, music, sync, drop, bass hit

  { "tool": "librosa", "function": "onset_detect", "store_as": "onset_times", "sensitivity": 0.5 }
    → librosa.onset.onset_detect() → returns audio transient timestamps
    → USE when: user mentions snap cut, on hit, on impact, sharp transition

  { "tool": "librosa", "function": "rms_energy", "store_as": "energy_curve" }
    → librosa.feature.rms() → returns per-frame loudness values
    → USE when: user mentions loud parts, quiet parts, energy-based effects

══════════════════════════════════════
SPECIAL OPERATIONS
══════════════════════════════════════

Use the "special" array for operations that need multi-step handling:

  { "type": "silence_remove", "params": { "threshold_db": -35, "min_silence_duration": 0.5 } }
  { "type": "black_remove", "params": { "min_black_duration": 0.5, "pixel_threshold": 0.1, "picture_threshold": 0.98 } }
    → run FFmpeg blackdetect, cut detected black/blank visual sections, and concatenate the remaining segments
  { "type": "freeze_remove", "params": { "noise_db": -60, "min_duration": 0.5 } }
    → run FFmpeg freezedetect, cut frozen/stuck-frame sections, and concatenate the remaining moving segments
  { "type": "dedupe_frames", "params": { "hi": 768, "lo": 320, "frac": 0.33, "max": 12 } }
    → run FFmpeg mpdecimate to drop duplicate or near-duplicate frames and rebuild video timing
  { "type": "beat_cut", "params": { "context": "beat_times", "slice_duration": 0.35, "max_cuts": 24 } }
    → create real jump cuts by keeping short video/audio slices around detected beat timestamps and concatenating them
  { "type": "scene_montage", "params": { "threshold": 0.28, "slice_duration": 1.2, "max_segments": 12 } }
    → detect visual scene changes with FFmpeg scene scoring, keep short highlight slices, and concatenate a montage
  { "type": "energy_montage", "params": { "context": "energy_curve_times", "slice_duration": 1.0, "max_segments": 12 } }
    → cut a hype/high-energy montage by keeping short video/audio slices around loudest audio moments
  { "type": "crop_borders", "params": { "limit": 24, "round": 2, "max_frames": 120 } }
    → run FFmpeg cropdetect to remove black borders/letterbox bars/pillarbox bars around the picture
  { "type": "stabilize", "params": { "smoothing": 10, "crop_black": true } }
  { "type": "reverse", "params": {} }
  { "type": "boomerang", "params": { "loops": 1, "mute_reversed_audio": true } }
    → create a ping-pong loop by concatenating original playback with a reversed copy
  { "type": "trim", "params": { "start": 2.0, "end": 12.0 } }
  { "type": "trim", "params": { "start": 0, "duration": 5.0 } }
  { "type": "remove_segment", "params": { "start": 4.0, "end": 7.0 } }
  { "type": "remove_audio", "params": {} }
  { "type": "face_privacy_blur", "params": { "target": "faces", "layout": "group" } }
    → apply non-tracking safe privacy redaction regions for faces/people when OpenCV semantic tracking is unavailable
  { "type": "ocr_redact", "params": { "sample_fps": 1.0, "confidence": 45 } }
    → sample frames, run pytesseract OCR, and apply timed delogo redaction boxes over detected text. Use for license plates, number plates, serial numbers, screen text, document text, and visible text.
  { "type": "auto_captions", "params": { "source": "speech", "language": "en", "style": "bottom_box" } }
    → run FFmpeg/pocketsphinx speech recognition and burn recognized speech phrases as timed bottom captions. Use for auto captions, speech subtitles, generated subtitles, transcription, and caption-the-speech requests. For explicit text like "caption that says ...", use drawtext instead.
  { "type": "blur_background", "params": { "width": 1080, "height": 1920, "sigma": 28 } }
    → fit the original video over a blurred duplicate background for vertical/reels/social layouts without cropping the subject
  { "type": "chroma_key", "params": { "key_color": "green", "replacement_color": "black", "similarity": 0.2, "blend": 0.08 } }
    → remove green-screen/blue-screen/chroma-key backgrounds and composite over a solid replacement color with FFmpeg chromakey + overlay
  { "type": "film_damage", "params": { "intensity": 0.7, "grain": 32, "gate_weave": 8, "scratch_opacity": 0.2, "dust_opacity": 0.35 } }
    → create old damaged 8mm/16mm film using real FFmpeg grain, faded color, flicker, vertical scratches, dust marks, and animated gate weave
  { "type": "picture_in_picture", "params": { "position": "top_right", "scale": 0.32 } }
  { "type": "split_screen_mirror", "params": { "divider_color": "white" } }
  { "type": "speed_ramp", "params": { "factor": 1.25 } }
  { "type": "speed_ramp", "params": { "slow_factor": 0.5, "fast_factor": 2.0 } }
  { "type": "pitch_shift", "params": { "semitones": -3 } }
  { "type": "replace_audio", "params": {} }
    → replace the video's audio with the uploaded audio file, looping the uploaded audio if needed
  { "type": "mix_uploaded_audio", "params": { "original_volume": 1.0, "music_volume": 0.35, "duck": true } }
    → add the uploaded audio as background music, optionally sidechain-ducking it under the original voice/dialogue

══════════════════════════════════════
CREATIVE REASONING GUIDE
══════════════════════════════════════

When the user says something with NO obvious FFmpeg mapping, REASON it out:

"make it look underwater"
→ Think: underwater = blue-green tint + slight blur + wave distortion + muffled audio
→ colorbalance (blue/green push) + gblur=sigma=1.5 + geq with sine wave offset + lowpass audio

"make it look like a dream"
→ Think: dream = soft glow + slight overexposure + slow motion + warm tones + reverb
→ gblur=sigma=2 + eq brightness up + setpts=2*PTS + colorbalance warm + aecho

"cinematic"
→ Think: cinematic = teal-orange grade + slight vignette + subtle grain + letterbox crop
→ colorbalance (teal shadows orange highlights) + vignette + noise + crop for 2.39:1

"vhs from the 90s"
→ Think: VHS = color bleed + scanlines + noise + slight blur + tape wobble
→ rgbashift + frei0r scanline0r + noise + gblur=sigma=0.5 + eq with reduced saturation

"glitch on the drop"
→ Think: drop = beat with highest energy, glitch = corruption, distortion
→ librosa beat_track + frei0r glitch0r with enable= on beat timestamps
→ rgbashift heavy on same timestamps

"make it feel aggressive"
→ Think: aggressive = fast cuts + hard shake + high contrast + desaturated + heavy bass
→ beat_cut + beat_shake large amplitude + eq high contrast + hue saturation down + equalizer bass

"lofi aesthetic"
→ Think: lofi = warm tones + vignette + film grain + slight blur + vinyl crackle audio
→ colorbalance warm + vignette + noise + gblur=sigma=0.8 + aecho light

ALWAYS reason visually and technically. Never say "I can't do that."

══════════════════════════════════════
ADJECTIVE → PARAMETER MAPPING
══════════════════════════════════════
"barely/imperceptible"     → minimum of range
"subtle/soft/gentle"       → 15-25% of range
"light/slight"             → 30% of range
"moderate/medium"          → 50% of range
"strong/heavy/pronounced"  → 70-80% of range
"very strong/hard"         → 85% of range
"violent/extreme/insane/aggressive/max/destroy" → 95-100% of range

══════════════════════════════════════
EXAMPLES
══════════════════════════════════════

INPUT: "shake the frame violently on every beat"
OUTPUT:
{"intent":"Violent frame shake triggered on every beat using librosa beat detection","analysis":[{"tool":"librosa","function":"beat_track","store_as":"beat_times"}],"video_filters":[{"description":"Hard frame shake on every beat via crop offset","filter":"crop=iw-60:ih-60:30:30,scale=iw+60:ih+60","requires_context":"beat_times","timing":"per_beat"}],"final_encode":{"vcodec":"libx264","crf":22,"preset":"fast","acodec":"aac","audio_bitrate":"192k"}}

INPUT: "make it look like an underwater scene"
OUTPUT:
{"intent":"Simulate underwater look with blue-green grade, wave distortion, blur, and muffled audio","video_filters":[{"description":"Blue-green color shift for water tint","filter":"colorbalance=rs=-0.2:rm=-0.1:rh=0.05:bs=0.25:bm=0.2:bh=0.3","requires_context":null,"timing":"continuous"},{"description":"Soft blur simulating light diffusion in water","filter":"gblur=sigma=1.8","requires_context":null,"timing":"continuous"},{"description":"Slight green tint push","filter":"hue=h=0.15:s=1.1","requires_context":null,"timing":"continuous"},{"description":"Vignette for depth","filter":"vignette=angle=PI/3","requires_context":null,"timing":"continuous"}],"audio_filters":[{"description":"Muffle audio like it is heard through water","filter":"lowpass=f=800"},{"description":"Add slight reverb for underwater space","filter":"aecho=0.8:0.4:300:0.4"}],"final_encode":{"vcodec":"libx264","crf":22,"preset":"fast","acodec":"aac","audio_bitrate":"192k"}}

INPUT: "lofi aesthetic with warm tones"
OUTPUT:
{"intent":"Lofi look: warm color grade, film grain, vignette, soft blur, vinyl-style audio","video_filters":[{"description":"Warm orange-yellow color push","filter":"colorbalance=rs=0.15:rm=0.2:rh=0.25:bs=-0.15:bm=-0.05:bh=-0.2","requires_context":null,"timing":"continuous"},{"description":"Reduce saturation slightly for faded film look","filter":"hue=s=0.75","requires_context":null,"timing":"continuous"},{"description":"Film grain texture","filter":"noise=alls=18:allf=t+u","requires_context":null,"timing":"continuous"},{"description":"Soft lens blur","filter":"gblur=sigma=0.9","requires_context":null,"timing":"continuous"},{"description":"Vignette edges","filter":"vignette=angle=PI/3.5","requires_context":null,"timing":"continuous"}],"audio_filters":[{"description":"Gentle high frequency rolloff like vinyl","filter":"lowpass=f=12000"},{"description":"Subtle warmth EQ","filter":"equalizer=f=200:width_type=o:width=2:g=3"}],"final_encode":{"vcodec":"libx264","crf":22,"preset":"fast","acodec":"aac","audio_bitrate":"192k"}}

INPUT: "remove all silences, normalize audio"
OUTPUT:
{"intent":"Cut all silent gaps and normalize loudness to -14 LUFS","special":[{"type":"silence_remove","params":{"threshold_db":-35,"min_silence_duration":0.5}}],"audio_filters":[{"description":"EBU R128 loudness normalization","filter":"loudnorm=I=-14:TP=-1.5:LRA=11"}],"final_encode":{"vcodec":"libx264","crf":22,"preset":"fast","acodec":"aac","audio_bitrate":"192k"}}

INPUT: "make it feel aggressive and intense"
OUTPUT:
{"intent":"Aggressive feel: high contrast, desaturated, violent shake on beat, heavy bass, fast glitch","analysis":[{"tool":"librosa","function":"beat_track","store_as":"beat_times"}],"video_filters":[{"description":"High contrast desaturated look","filter":"eq=contrast=1.6:brightness=-0.05:saturation=0.4","requires_context":null,"timing":"continuous"},{"description":"Hard vignette for intensity","filter":"vignette=angle=PI/2.5","requires_context":null,"timing":"continuous"},{"description":"Glitch corruption on beats","filter":"frei0r=filter_name=glitch0r:filter_params=0.7","requires_context":"beat_times","timing":"per_beat"},{"description":"Chromatic aberration on beats","filter":"rgbashift=rh=12:bh=-12:rv=6:bv=-6","requires_context":"beat_times","timing":"per_beat"}],"audio_filters":[{"description":"Heavy bass boost for impact","filter":"equalizer=f=60:width_type=o:width=2:g=10"},{"description":"Dynamic loudness push","filter":"dynaudnorm=f=150:g=15"}],"final_encode":{"vcodec":"libx264","crf":22,"preset":"fast","acodec":"aac","audio_bitrate":"192k"}}
"""

nim_provider = NimChatProvider(
    api_url=NIM_API_URL,
    model=NIM_MODEL,
    api_key=NIM_API_KEY,
    timeout_seconds=NIM_TIMEOUT_SECONDS,
    max_attempts=NIM_MAX_ATTEMPTS,
    system_prompt=NIM_SYSTEM_PROMPT,
)


def get_upload_root():
    candidates = [REQUESTED_UPLOAD_ROOT]
    if FALLBACK_UPLOAD_ROOT != REQUESTED_UPLOAD_ROOT:
        candidates.append(FALLBACK_UPLOAD_ROOT)

    for root in candidates:
        try:
            root.mkdir(parents=True, exist_ok=True)
            probe = root / f".write-test-{uuid4().hex}"
            probe.write_text("", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return root
        except OSError:
            continue

    raise RuntimeError("No writable upload directory available")


UPLOAD_ROOT = get_upload_root()
job_store = JobStore(UPLOAD_ROOT)


def job_state_path_from_job(job):
    return job_store.state_path_from_job(job)


def job_state_path(job_id):
    return job_store.state_path(job_id)


def sanitize_job_state(job, from_disk=False):
    return job_store.sanitize(job, from_disk=from_disk)


def save_job(job):
    return job_store.save(job)


def remember_job(job):
    return job_store.remember(job)


def persist_job(job):
    return job_store.persist(job)


def get_job_record(job_id):
    return job_store.get(job_id)


def claim_job_for_command(job_id, command_text):
    return job_store.claim_for_command(job_id, command_text)


def load_persisted_jobs():
    return job_store.load_persisted()


def persisted_job_count():
    return job_store.persisted_count()


def in_memory_job_count():
    return job_store.memory_count()


def upload_root_health():
    probe = UPLOAD_ROOT / f".ready-{uuid4().hex}"
    try:
        UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
        probe.write_text("", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return {"ok": True, "path": str(UPLOAD_ROOT)}
    except Exception as exc:
        return {"ok": False, "path": str(UPLOAD_ROOT), "error": concise_error(exc)}


load_persisted_jobs()


def original_filename(file_storage):
    filename = (file_storage.filename or "").replace("\\", "/")
    return Path(filename).name


def media_stream_types(path):
    try:
        result = run_command([
            "ffprobe", "-v", "error",
            "-show_entries", "stream=codec_type",
            "-of", "json",
            str(path),
        ])
        payload = json.loads(result.stdout)
        return {stream.get("codec_type") for stream in payload.get("streams", [])}
    except Exception:
        return set()


def uploaded_media_has_stream(path, stream_type):
    return stream_type in media_stream_types(path)


def validate_uploaded_media(video_path, audio_path=None):
    if not uploaded_media_has_stream(video_path, "video"):
        raise ValueError("valid video file required")
    if audio_path and not uploaded_media_has_stream(audio_path, "audio"):
        raise ValueError("valid audio file required")


def prepare_command(args):
    return media_runner.prepare(args)


def command_label(args):
    return media_runner.label(args)


def compact_command_output(text, limit=800):
    return media_runner.compact_output(text, limit=limit)


def run_command(args, timeout=None):
    return media_runner.run(args, timeout=timeout)


def run_command_result(args, timeout=None):
    return media_runner.run(args, timeout=timeout, check=False)


def first_output_line(text):
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("proot warning:"):
            continue
        if stripped:
            return stripped[:240]
    return ""


def probe_command(command, version_arg="--version"):
    status = {
        "available": bool(shutil.which(command)),
        "ok": False,
        "via_proot": bool(command in MEDIA_COMMANDS and PROOT_DISTRO),
    }
    try:
        result = run_command([command, version_arg])
        status["ok"] = True
        status["version"] = first_output_line(result.stdout or result.stderr)
    except Exception as exc:
        status["error"] = concise_error(exc)
    return status


def proot_host_path(container_path):
    raw_path = Path(str(container_path))
    if raw_path.exists():
        return raw_path
    if raw_path.is_absolute():
        candidate = PROOT_ROOTFS / str(raw_path).lstrip("/")
        if candidate.exists():
            return candidate
    return raw_path


def probe_file_or_directory(path):
    host_path = proot_host_path(path)
    return {
        "container_path": str(path),
        "host_path": str(host_path),
        "exists": host_path.exists(),
    }


def probe_pocketsphinx_models():
    paths = {
        "hmm": probe_file_or_directory(POCKETSPHINX_HMM),
        "dict": probe_file_or_directory(POCKETSPHINX_DICT),
        "lm": probe_file_or_directory(POCKETSPHINX_LM),
    }
    return {
        "ok": all(item["exists"] for item in paths.values()),
        "paths": paths,
    }


def probe_ffmpeg_filter(filter_name_value):
    status = {"ok": False}
    try:
        result = run_command(["ffmpeg", "-hide_banner", "-h", f"filter={filter_name_value}"])
        output = f"{result.stdout}\n{result.stderr}"
        status["ok"] = bool(re.search(rf"\bfilter\s+{re.escape(filter_name_value)}\b", output, re.IGNORECASE))
        status["summary"] = first_output_line(output)
    except Exception as exc:
        status["error"] = concise_error(exc)
    return status


def probe_python_module(module_name):
    status = {
        "available": bool(importlib.util.find_spec(module_name)),
        "ok": False,
    }
    try:
        module = importlib.import_module(module_name)
        status["ok"] = True
        version = getattr(module, "__version__", None)
        if version:
            status["version"] = str(version)
    except Exception as exc:
        status["error"] = concise_error(exc)
    return status


def probe_proot_cv2_python():
    status = {
        "available": bool(PROOT_DISTRO),
        "ok": False,
        "via_proot": True,
    }
    if not PROOT_DISTRO:
        status["error"] = "proot-distro unavailable"
        return status

    last_error = None
    for python_path in PROOT_CV2_PYTHON_CANDIDATES:
        try:
            result = run_command_result([
                PROOT_DISTRO, "login", "ubuntu", "--",
                python_path,
                "-c",
                "import cv2; print(cv2.__version__)",
            ], timeout=20)
            if result.returncode == 0:
                status["ok"] = True
                status["python"] = python_path
                status["version"] = first_output_line(result.stdout)
                return status
            last_error = result.stderr.strip() or result.stdout.strip() or "cv2 probe failed"
        except Exception as exc:
            last_error = exc

    status["error"] = concise_error(last_error)
    return status


def collect_runtime_capabilities():
    commands = {
        "ffmpeg": probe_command("ffmpeg", "-version"),
        "ffprobe": probe_command("ffprobe", "-version"),
        "rubberband": probe_command("rubberband", "--version"),
        "sox": probe_command("sox", "--version"),
        "tesseract": probe_command("tesseract", "--version"),
    }
    python_modules = {
        "librosa": probe_python_module("librosa"),
        "numpy": probe_python_module("numpy"),
        "cv2": probe_python_module("cv2"),
        "pytesseract": probe_python_module("pytesseract"),
        "rapidocr": probe_python_module("rapidocr"),
        "pysubs2": probe_python_module("pysubs2"),
        "srt": probe_python_module("srt"),
    }
    cv2_proot = probe_proot_cv2_python()
    python_modules["cv2_proot"] = cv2_proot
    ffmpeg_filter_names = ["asr", "rubberband", "frei0r", "vidstabdetect", "vidstabtransform"]
    ffmpeg_filters = {
        name: probe_ffmpeg_filter(name) if commands["ffmpeg"]["ok"] else {"ok": False}
        for name in ffmpeg_filter_names
    }
    pocketsphinx_models = probe_pocketsphinx_models()
    return {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "media_commands": commands,
        "ffmpeg_filters": ffmpeg_filters,
        "pocketsphinx_models": pocketsphinx_models,
        "python_modules": python_modules,
        "executor": {
            "media_runs_via_proot": bool(PROOT_DISTRO),
            "ffmpeg_ready": commands["ffmpeg"]["ok"] and commands["ffprobe"]["ok"],
            "audio_analysis_ready": python_modules["librosa"]["ok"] or python_modules["numpy"]["ok"],
            "pitch_shift_ready": commands["rubberband"]["ok"] or ffmpeg_filters["rubberband"]["ok"],
            "stabilize_ready": ffmpeg_filters["vidstabdetect"]["ok"] and ffmpeg_filters["vidstabtransform"]["ok"],
            "frei0r_ready": ffmpeg_filters["frei0r"]["ok"],
            "asr_ready": commands["ffmpeg"]["ok"] and ffmpeg_filters["asr"]["ok"] and pocketsphinx_models["ok"],
            "ocr_ready": commands["tesseract"]["ok"] and (
                python_modules["pytesseract"]["ok"] or python_modules["rapidocr"]["ok"]
            ),
            "opencv_ready": python_modules["cv2"]["ok"] or cv2_proot["ok"],
            "opencv_mode": "native_python" if python_modules["cv2"]["ok"] else (
                "ubuntu_proot_python" if cv2_proot["ok"] else "unavailable"
            ),
        },
    }


def runtime_capabilities(force=False):
    return runtime_capability_cache.get(collect_runtime_capabilities, force=force)


def runtime_capability_prompt(capabilities=None):
    capabilities = capabilities or runtime_capabilities()
    executor = capabilities["executor"]
    media = capabilities["media_commands"]
    modules = capabilities["python_modules"]
    lines = [
        f"FFmpeg execution: {'ready' if executor['ffmpeg_ready'] else 'not ready'}"
        + (" through Ubuntu proot." if executor["media_runs_via_proot"] else "."),
        f"librosa audio analysis: {'ready' if modules['librosa']['ok'] else 'not ready'}.",
        f"rubberband pitch/time processing: {'ready' if executor['pitch_shift_ready'] else 'not ready'} via FFmpeg filter or CLI.",
        f"vidstab stabilization: {'ready' if executor['stabilize_ready'] else 'not ready'}.",
        f"frei0r plugin effects: {'ready' if executor['frei0r_ready'] else 'not ready'}.",
        f"Speech auto-captions: {'ready' if executor['asr_ready'] else 'not ready'} via special type auto_captions using FFmpeg/pocketsphinx.",
        f"OCR text redaction: {'ready' if executor['ocr_ready'] else 'not ready'} via special type ocr_redact.",
        f"OpenCV/cv2 semantic detection: {'ready' if executor['opencv_ready'] else 'not ready'}"
        + (f" via {executor.get('opencv_mode')}." if executor.get("opencv_mode") else ".")
    ]
    if not executor["opencv_ready"]:
        lines.append(
            "Do not invent face/license/object-tracking operations. For faces/people, use special type face_privacy_blur as a non-tracking safe-region privacy fallback; for license/text, use ocr_redact."
        )
        error = modules["cv2"].get("error")
        if error:
            lines.append(f"cv2 import error: {error}")
    else:
        lines.append(
            "For faces/people, use special type face_privacy_blur; the executor can run OpenCV cascade detection and falls back to safe regions if nothing is detected."
        )
    if not media["ffmpeg"]["ok"]:
        error = media["ffmpeg"].get("error")
        lines.append(f"FFmpeg error: {error or 'unavailable'}")
    return "\n".join(lines)


def runtime_planning_prompt(capabilities=None):
    capabilities = capabilities or runtime_capabilities()
    return "\n\n".join([
        runtime_capability_prompt(capabilities),
        runtime_operation_prompt_contract(capabilities.get("executor") or {}),
    ])


def append_job_warning(job, message):
    if job is None:
        return
    warnings = job.setdefault("warnings", [])
    warnings.append(str(message)[:1000])


def record_planner_fallback(job, reason, detail=None):
    if job is None:
        return
    policy = planner_fallback_policy(reason)
    entry = {
        "reason": reason,
        "mode": policy.get("mode"),
        "allows_execution": bool(policy.get("allows_execution")),
        "requires_validation": bool(policy.get("requires_validation")),
        "user_visible_warning": bool(policy.get("user_visible_warning")),
        "description": policy.get("description"),
    }
    if detail:
        entry["detail"] = str(detail)[:1000]
    job["planner_fallback"] = entry


def concise_error(message):
    text = str(message or "unknown error").strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text[:240]


def next_media_path(job_dir, label, suffix=".mp4"):
    return job_dir / f"{label}_{uuid4().hex[:8]}{suffix}"


def extract_json(text):
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    return json.loads(stripped)


COLORBALANCE_OPTION_ALIASES = {
    "ss": "rs",
    "ms": "rm",
    "hs": "rh",
    "sb": "bs",
    "mb": "bm",
    "hb": "bh",
}


def repair_colorbalance_filter(filter_string):
    def replace_options(match):
        options = match.group(1).split(":")
        repaired = []
        for option in options:
            key, separator, value = option.partition("=")
            repaired_key = COLORBALANCE_OPTION_ALIASES.get(key, key)
            repaired.append(f"{repaired_key}{separator}{value}" if separator else repaired_key)
        return f"colorbalance={':'.join(repaired)}"

    return re.sub(r"colorbalance=([^,]+)", replace_options, filter_string)


FREI0R_NATIVE_FALLBACKS = {
    "glitch0r": "rgbashift=rh=14:bh=-14:rv=8:bv=-8,noise=alls=40:allf=t+u",
    "distort0r": "lenscorrection=k1=-0.35:k2=0.08",
    "scanline0r": "drawgrid=width=iw:height=4:thickness=1:color=black@0.35",
    "pixeliz0r": "pixelize=width=20:height=20",
    "colorhalftone": "pixelize=width=10:height=10,hue=s=0.75",
    "gaussianblur": "gblur=sigma=10",
    "vignette": "vignette=angle=PI/4",
}


def replace_frei0r_with_native_fallback(filter_part):
    if not isinstance(filter_part, str) or not filter_part.startswith("frei0r="):
        return filter_part

    options = split_filter_options(filter_part[len("frei0r="):])
    option_map = {}
    for option in options:
        key, separator, value = option.partition("=")
        if separator:
            option_map[key] = value
    plugin_name = option_map.get("filter_name")
    return FREI0R_NATIVE_FALLBACKS.get(plugin_name, "noise=alls=18:allf=t+u")


def repair_frei0r_filter(filter_string):
    repaired = re.sub(r"(frei0r=[^,]*?):param\d+=", r"\1:filter_params=", filter_string)
    return ",".join(replace_frei0r_with_native_fallback(part) for part in split_filter_chain(repaired))


def repair_geq_filter(filter_string):
    parts = []
    for part in split_filter_chain(filter_string):
        if part.startswith("geq="):
            if re.search(r"[<>]=?", part):
                parts.append(
                    "delogo=x=__PRIVACY_X__:y=__PRIVACY_Y__:w=__PRIVACY_W__:h=__PRIVACY_H__:show=0,"
                    "vignette=angle=PI/3,eq=brightness=-0.04"
                )
            elif re.search(r"\b[RGB]\b", part):
                parts.append("eq=contrast=1.05:saturation=0.75")
            else:
                parts.append(re.sub(r"\bt\b", "T", part))
        else:
            parts.append(part)
    return ",".join(parts)


def repair_lut3d_filter(filter_string):
    cinematic_grade = (
        "colorbalance=rs=-0.15:rm=-0.05:rh=0.18:bs=0.18:bm=0.05:bh=-0.12,"
        "eq=contrast=1.12:saturation=1.08"
    )
    return re.sub(r"lut3d=([^,]+)", cinematic_grade, filter_string)


def repair_chromakey_filter(filter_string):
    def replace_chromakey(match):
        value = match.group(1)
        parts = split_filter_options(value)
        if len(parts) <= 3 and not all(re.fullmatch(r"[0-9.]+", part) for part in parts):
            return match.group(0)
        return "chromakey=0x00ff00:0.18:0.05,format=yuv420p"

    return re.sub(r"chromakey=([^,]+)", replace_chromakey, filter_string)


def repair_colorchannelmixer_filter(filter_string):
    def replace_options(match):
        options = match.group(1).split(":")
        repaired = []
        for option in options:
            key, separator, value = option.partition("=")
            if key == "bm":
                key = "bg"
            repaired.append(f"{key}{separator}{value}" if separator else key)
        return f"colorchannelmixer={':'.join(repaired)}"

    return re.sub(r"colorchannelmixer=([^,]+)", replace_options, filter_string)


COLOR_HOLD_FILTERS = {
    "red": "format=rgb24,colorhold=color=red:similarity=0.25:blend=0.12,format=yuv420p",
    "blue": "format=rgb24,colorhold=color=blue:similarity=0.25:blend=0.12,format=yuv420p",
    "green": "format=rgb24,colorhold=color=green:similarity=0.25:blend=0.12,format=yuv420p",
    "yellow": "format=rgb24,colorhold=color=yellow:similarity=0.25:blend=0.12,format=yuv420p",
    "cyan": "format=rgb24,colorhold=color=cyan:similarity=0.25:blend=0.12,format=yuv420p",
    "magenta": "format=rgb24,colorhold=color=magenta:similarity=0.25:blend=0.12,format=yuv420p",
    "orange": "format=rgb24,colorhold=color=orange:similarity=0.25:blend=0.18,format=yuv420p",
}
SAFE_DRAWTEXT_ALPHA = "'min(1,t/3)'"
DRAWTEXT_OPTION_MARKERS = [
    ":fontfile=", ":fontcolor=", ":fontsize=", ":x=", ":y=", ":box=",
    ":boxcolor=", ":shadowcolor=", ":shadowx=", ":shadowy=", ":enable=",
    ":alpha=", ":borderw=", ":bordercolor=", ":line_spacing=",
]


def requested_color_hold_filter(text):
    lowered = text.lower()
    intent_words = [
        "except", "only", "isolate", "isolation", "hold", "preserve",
        "keep", "black and white", "black-and-white", "grayscale", "greyscale",
    ]
    if not any(word in lowered for word in intent_words):
        return None

    for color, filter_string in COLOR_HOLD_FILTERS.items():
        if color in lowered:
            return filter_string
    return None


def has_broken_channel_threshold(filter_string):
    if not isinstance(filter_string, str):
        return False
    if "colorchannelmixer=" not in filter_string and "geq=" not in filter_string:
        return False
    return bool(
        re.search(r"\b[RGB]\s*[<>]=?\s*[0-9]", filter_string)
        or re.search(r"(^|,)\s*(?:min|max)\(", filter_string)
    )


def repair_color_hold_filter(filter_string, description=""):
    requested = requested_color_hold_filter(f"{description} {filter_string}")
    if requested:
        return requested
    if has_broken_channel_threshold(filter_string):
        return COLOR_HOLD_FILTERS["red"]
    return filter_string


def strip_generated_metadata_fragments(filter_string):
    repaired = []
    for part in split_filter_chain(filter_string):
        name = filter_name(part).strip("'\"").lower()
        if name in {"description", "intent", "analysis"} or name.startswith("description"):
            continue
        repaired.append(part)
    return ",".join(repaired)


def quote_drawtext_filter_text(value):
    value = value.strip()
    if len(value) >= 2 and value[0] in {"'", '"'} and value[-1] == value[0]:
        value = value[1:-1]
    value = value.replace("\\", "\\\\").replace('"', r"\"")
    return f'"{value}"'


def normalize_drawtext_text_quoting(filter_string):
    if not isinstance(filter_string, str) or "drawtext=text=" not in filter_string:
        return filter_string

    prefix = "drawtext=text="
    result = []
    cursor = 0
    while True:
        start = filter_string.find(prefix, cursor)
        if start == -1:
            result.append(filter_string[cursor:])
            break

        value_start = start + len(prefix)
        option_index = min(
            [index for marker in DRAWTEXT_OPTION_MARKERS if (index := filter_string.find(marker, value_start)) != -1],
            default=-1,
        )
        if option_index == -1:
            result.append(filter_string[cursor:])
            break

        result.append(filter_string[cursor:value_start])
        result.append(quote_drawtext_filter_text(filter_string[value_start:option_index]))
        cursor = option_index

    return "".join(result)


def quote_filter_value(value):
    value = value.strip()
    quote = value[0] if len(value) >= 2 and value[0] in {"'", '"'} and value[-1] == value[0] else "'"
    if quote in {"'", '"'} and len(value) >= 2 and value[0] == quote and value[-1] == quote:
        inner = value[1:-1]
    else:
        inner = value
    inner = inner.replace(r"\,", ",")
    return f"{quote}{inner}{quote}"


def replace_or_append_filter_option(filter_part, option_name, option_value):
    name, separator, option_string = filter_part.partition("=")
    if not separator:
        return filter_part

    replaced = False
    options = []
    for option in split_filter_options(option_string):
        key, option_separator, _value = option.partition("=")
        if option_separator and key == option_name:
            options.append(f"{option_name}={option_value}")
            replaced = True
        else:
            options.append(option)

    if not replaced:
        options.append(f"{option_name}={option_value}")

    return f"{name}={':'.join(options)}"


def unquote_filter_value(value):
    value = value.strip()
    if len(value) >= 2 and value[0] in {"'", '"'} and value[-1] == value[0]:
        return value[1:-1]
    return value


def malformed_filter_expression(value):
    expression = unquote_filter_value(value).replace(r"\,", ",")
    balance = 0
    for char in expression:
        if char == "(":
            balance += 1
        elif char == ")":
            balance -= 1
        if balance < 0:
            return True
    return balance != 0 or expression.endswith((",", "+", "-", "*", "/", "("))


def repair_drawtext_options(filter_part):
    name, separator, option_string = filter_part.partition("=")
    if not separator:
        return filter_part, False

    repaired = []
    fixed_malformed_alpha = False
    for option in split_filter_options(option_string):
        key, option_separator, value = option.partition("=")
        if option_separator and key == "alpha":
            if malformed_filter_expression(value) or "if(" in unquote_filter_value(value):
                repaired.append(f"alpha={SAFE_DRAWTEXT_ALPHA}")
                fixed_malformed_alpha = True
            else:
                repaired.append(f"alpha={quote_filter_value(value)}")
        else:
            repaired.append(option)

    return f"{name}={':'.join(repaired)}", fixed_malformed_alpha


def repair_drawtext_pseudo_filters(filter_string):
    repaired = []
    skip_malformed_tail = False
    for part in split_filter_chain(filter_string):
        malformed_tail = re.match(r"^[^=]*\):(.+)$", part)
        if malformed_tail and repaired and filter_name(repaired[-1]) == "drawtext":
            for option in split_filter_options(malformed_tail.group(1)):
                key, separator, value = option.partition("=")
                if separator and key in {"x", "y", "enable", "alpha"}:
                    if key == "alpha":
                        value = quote_filter_value(value)
                    repaired[-1] = replace_or_append_filter_option(repaired[-1], key, value)
            continue

        if skip_malformed_tail and "=" not in part:
            continue
        skip_malformed_tail = False

        if filter_name(part) == "drawtext":
            fixed_part, fixed_malformed_alpha = repair_drawtext_options(part)
            repaired.append(fixed_part)
            skip_malformed_tail = fixed_malformed_alpha
            continue

        if part.startswith("alpha=") and repaired and filter_name(repaired[-1]) == "drawtext":
            alpha_options = split_filter_options(part[len("alpha="):])
            if alpha_options:
                repaired[-1] = replace_or_append_filter_option(
                    repaired[-1],
                    "alpha",
                    quote_filter_value(alpha_options[0]),
                )
                for option in alpha_options[1:]:
                    key, separator, value = option.partition("=")
                    if separator and key == "y":
                        repaired[-1] = replace_or_append_filter_option(repaired[-1], "y", value)
            continue
        repaired.append(part)
    return ",".join(repaired)


def repair_drawbox_filter(filter_string):
    repaired = []
    for part in split_filter_chain(filter_string):
        if filter_name(part) != "drawbox":
            repaired.append(part)
            continue

        option_map = drawbox_option_map(part)
        is_full_frame = is_full_frame_drawbox(part)
        color = option_map.get("color", "")
        opacity_match = re.search(r"@([0-9.]+)", color)
        opacity = float(opacity_match.group(1)) if opacity_match else 1.0

        if is_full_frame and "enable" not in option_map and opacity >= 0.4:
            repaired.append(
                "drawbox=x=0:y=0:w=iw:h=ih:color=white@0.14:t=fill:"
                "enable='lt(mod(t,1.2),0.06)'"
            )
        elif is_full_frame and "enable" in option_map and opacity >= 0.4:
            enable_expression = unquote_filter_value(option_map["enable"]).replace(r"\,", ",")
            between_mod_match = re.search(
                r"between\(mod\(t,\s*([0-9.]+)\)\s*,\s*0\s*,\s*([0-9.]+)\)",
                enable_expression,
            )
            lt_mod_match = re.search(
                r"lt\(mod\(t,\s*([0-9.]+)\)\s*,\s*([0-9.]+)\)",
                enable_expression,
            )
            comparison_mod_match = re.search(
                r"mod\(t,\s*([0-9.]+)\)\s*<\s*([0-9.]+)",
                enable_expression,
            )
            mod_match = between_mod_match or lt_mod_match or comparison_mod_match
            if mod_match:
                period = float(mod_match.group(1))
                flash_width = min(0.08, max(0.04, period * 0.08))
                enable_expression = f"lt(mod(t,{period:g}),{flash_width:g})"
            else:
                raw_mod_match = re.fullmatch(r"mod\(t,\s*([0-9.]+)\)", enable_expression.strip())
                if raw_mod_match:
                    period = float(raw_mod_match.group(1))
                    flash_width = min(0.08, max(0.04, period * 0.08))
                    enable_expression = f"lt(mod(t,{period:g}),{flash_width:g})"
            repaired.append(
                "drawbox=x=0:y=0:w=iw:h=ih:color=white@0.45:t=fill:"
                f"enable='{enable_expression}'"
            )
        else:
            repaired.append(part)
    return ",".join(repaired)


def repair_eq_expression_filter(filter_string):
    repaired = []
    for part in split_filter_chain(filter_string):
        if filter_name(part) == "eq" and re.search(r"brightness\s*=\s*if\(", part):
            repaired.append("eq=brightness=0.18:contrast=1.05:saturation=1.05")
        else:
            repaired.append(part)
    return ",".join(repaired)


def repair_single_input_video_filters(filter_string):
    replacements = {
        "blend": "gblur=sigma=1.2,eq=brightness=0.05:saturation=1.08",
        "glow": "gblur=sigma=2,eq=brightness=0.08:saturation=1.15",
        "overlay": "drawbox=x=0:y=0:w=iw:h=ih:color=white@0.08:t=fill",
    }
    repaired = []
    for part in split_filter_chain(filter_string):
        filter_name = part.split("=", 1)[0].strip()
        repaired.append(replacements.get(filter_name, part))
    return ",".join(repaired)


def repair_audio_filter_string(filter_string):
    if not isinstance(filter_string, str):
        return filter_string
    repaired = filter_string.replace("sidechaincompress=", "acompressor=")
    repaired = re.sub(r"\bspeakernorm=", "speechnorm=", repaired, flags=re.IGNORECASE)
    repaired_parts = []
    for part in split_filter_chain(repaired):
        name = filter_name(part)
        if name == "noise":
            repaired_parts.append("acrusher=bits=8:mode=log:mix=0.18")
        else:
            repaired_parts.append(part)
    return ",".join(repaired_parts)


def repair_letterbox_filter_string(filter_string):
    lowered = filter_string.lower()
    if ("2.39" in lowered or "2.35" in lowered) and ("crop=" in lowered or "pad=" in lowered):
        return ""
    return filter_string


def repair_filter_string(filter_string, description=""):
    if not isinstance(filter_string, str):
        return filter_string
    repaired = normalize_drawtext_text_quoting(filter_string)
    repaired = strip_generated_metadata_fragments(repaired)
    repaired = repair_lut3d_filter(repair_geq_filter(repair_frei0r_filter(repair_colorbalance_filter(repaired))))
    repaired = repair_chromakey_filter(repair_colorchannelmixer_filter(repaired))
    repaired = repair_color_hold_filter(repaired, description)
    repaired = repair_drawtext_pseudo_filters(repaired)
    repaired = repair_eq_expression_filter(repaired)
    repaired = repair_drawbox_filter(repaired)
    repaired = repair_single_input_video_filters(repaired)
    return repair_letterbox_filter_string(repaired)


def normalize_filter_steps(value, filter_kind):
    if not value:
        return []
    raw_steps = value if isinstance(value, list) else [value]
    normalized_steps = []
    for step in raw_steps:
        if isinstance(step, str):
            step = {"filter": step}
        if not isinstance(step, dict):
            continue
        filter_string = step.get("filter")
        if filter_kind == "video":
            step["filter"] = repair_filter_string(filter_string, step.get("description", ""))
        else:
            step["filter"] = repair_audio_filter_string(repair_filter_string(filter_string, step.get("description", "")))
        if step.get("filter"):
            normalized_steps.append(step)
    return normalized_steps


def normalize_plan(plan):
    plan, _fixes = normalize_public_plan_shape(plan)

    plan["video_filters"] = normalize_filter_steps(plan.get("video_filters", []), "video")
    plan["audio_filters"] = normalize_filter_steps(plan.get("audio_filters", []), "audio")

    normalized = {key: value for key, value in plan.items() if not (isinstance(value, list) and len(value) == 0)}
    return normalized


def no_audio_sync_requested(command_text):
    lowered = command_text.lower()
    phrases = [
        "without relying on audio synchronization",
        "without audio synchronization",
        "without audio sync",
        "no audio synchronization",
        "no audio sync",
        "do not use audio sync",
        "don't use audio sync",
        "not synced to audio",
    ]
    return any(phrase in lowered for phrase in phrases)


def audio_dependent_video_filter(step):
    timing = step.get("timing")
    requires_context = str(step.get("requires_context") or "")
    filter_string = str(step.get("filter") or "")
    if timing in {"per_beat", "per_onset"}:
        return True
    if requires_context in {"beat_times", "onset_times", "energy_curve", "energy_curve_times"}:
        return True
    return any(token in filter_string for token in ["beat_times", "onset_times", "energy_curve"])


def strip_audio_dependent_video_filters(video_filters):
    return [
        step for step in video_filters
        if not audio_dependent_video_filter(step)
    ]


def enforce_no_audio_sync_constraints(plan):
    plan.pop("analysis", None)
    plan.pop("audio_filters", None)
    plan["video_filters"] = strip_audio_dependent_video_filters(plan.get("video_filters", []))
    plan["special"] = [
        step for step in plan.get("special", [])
        if step.get("type") not in {
            "beat_cut",
            "energy_montage",
            "pitch_shift",
            "silence_remove",
            "mix_uploaded_audio",
            "replace_audio",
        }
    ]
    return plan


def impact_timing_requested(command_text):
    if no_audio_sync_requested(command_text):
        return False
    lowered = command_text.lower()
    return re.search(r"\b(?:impact|impacts|onset|onsets|hit|hits|snap|snaps)\b", lowered) is not None


def end_reverse_requested(command_text):
    lowered = command_text.lower()
    if "reverse" not in lowered:
        return False
    end_phrases = ["end with", "ends with", "ending", "at the end", "final"]
    return any(phrase in lowered for phrase in end_phrases)


def boomerang_requested(command_text):
    lowered = command_text.lower()
    direct_terms = ["boomerang", "ping-pong", "ping pong", "bounce loop", "bouncing loop"]
    if any(term in lowered for term in direct_terms):
        return True
    phrase_terms = [
        "forward then reverse",
        "forwards then backwards",
        "forward then backward",
        "play forward then backward",
        "play forwards then backwards",
        "loop forward and backward",
        "loop forwards and backwards",
        "reverse loop",
        "rewind loop",
    ]
    return any(term in lowered for term in phrase_terms)


def text_rollout_requested(command_text):
    lowered = command_text.lower()
    return (
        "text" in lowered
        and any(phrase in lowered for phrase in ["roll out", "rolling out", "rollout", "disappear with"])
    )


def crop_borders_requested(command_text):
    lowered = command_text.lower()
    action_terms = ["remove", "delete", "crop", "trim", "cut"]
    target_terms = [
        "black bars", "black border", "black borders", "letterbox bars",
        "letterboxing", "letterbox border", "letterbox borders",
        "pillarbox", "pillarboxing", "side bars", "empty borders",
        "border around", "borders around",
    ]
    return any(term in lowered for term in action_terms) and any(term in lowered for term in target_terms)


def output_aspect_requested(command_text):
    if crop_borders_requested(command_text):
        return False
    lowered = command_text.lower()
    return any(term in lowered for term in [
        "9:16", "tiktok", "reels", "shorts", "vertical",
        "4:5", "instagram portrait",
        "1:1", "square",
        "2.39", "2.35", "letterbox", "widescreen",
    ])


def blurred_background_requested(command_text):
    lowered = command_text.lower()
    if privacy_blur_requested(command_text) or ocr_redact_requested(command_text):
        return False
    direct_terms = [
        "blurred background",
        "blur background",
        "background blur",
        "blurred bg",
        "blur bg",
        "bokeh background",
    ]
    if any(term in lowered for term in direct_terms):
        return True
    side_terms = ["blur the sides", "blurred sides", "blur side bars", "blurred side bars"]
    fill_terms = ["fill the sides", "fill side bars", "fill the empty space", "fill background"]
    social_terms = ["vertical", "reels", "reel", "tiktok", "shorts", "9:16", "portrait"]
    return (
        any(term in lowered for term in side_terms + fill_terms)
        and any(term in lowered for term in social_terms)
    )


def blurred_background_params(command_text):
    lowered = command_text.lower()
    if "4:5" in lowered or ("instagram" in lowered and "portrait" in lowered):
        width, height = 1080, 1350
    elif "1:1" in lowered or "square" in lowered:
        width, height = 1080, 1080
    elif "2.39" in lowered or "2.35" in lowered or "widescreen" in lowered:
        width, height = 1920, 1080
    else:
        width, height = 1080, 1920
    return {"width": width, "height": height, "sigma": 28}


def requested_output_dimensions(command_text):
    lowered = command_text.lower()
    resolution = None
    if re.search(r"\b(?:8k|4320p)\b", lowered):
        resolution = (7680, 4320)
    elif re.search(r"\b(?:4k|uhd|ultra\s*hd|2160p)\b", lowered):
        resolution = (3840, 2160)
    elif re.search(r"\b(?:2k|qhd|1440p)\b", lowered):
        resolution = (2560, 1440)
    elif re.search(r"\b(?:full\s*hd|fhd|1080p)\b", lowered):
        resolution = (1920, 1080)
    elif re.search(r"\b720p\b", lowered):
        resolution = (1280, 720)
    if not resolution:
        return None

    width, height = resolution
    if any(term in lowered for term in ["9:16", "tiktok", "reels", "shorts", "vertical"]):
        return {"width": height, "height": width}
    if "4:5" in lowered or ("instagram" in lowered and "portrait" in lowered):
        return {"width": height, "height": int(round(height * 1.25))}
    if "1:1" in lowered or "square" in lowered:
        return {"width": height, "height": height}
    return {"width": width, "height": height}


def apply_requested_output_dimensions(plan, command_text):
    dimensions = requested_output_dimensions(command_text)
    if not dimensions:
        return
    final_encode_settings = dict(plan.get("final_encode") or default_final_encode())
    final_encode_settings.update(dimensions)
    plan["final_encode"] = final_encode_settings


def chroma_key_requested(command_text):
    lowered = command_text.lower()
    target_terms = [
        "green screen", "greenscreen", "green-screen",
        "blue screen", "bluescreen", "blue-screen",
        "chroma key", "chromakey", "key out", "keying",
    ]
    if not any(term in lowered for term in target_terms):
        return False
    action_terms = ["remove", "replace", "key", "cut out", "transparent", "background", "screen"]
    return any(term in lowered for term in action_terms)


SOLID_COLOR_VALUES = {
    "black": "black",
    "white": "white",
    "red": "red",
    "green": "green",
    "blue": "blue",
    "yellow": "yellow",
    "cyan": "cyan",
    "magenta": "magenta",
    "purple": "purple",
    "orange": "orange",
    "gray": "gray",
    "grey": "gray",
}


def chroma_key_params(command_text):
    lowered = command_text.lower()
    key_color = "blue" if any(term in lowered for term in ["blue screen", "bluescreen", "blue-screen"]) else "green"
    replacement_color = "black"

    replace_match = re.search(
        r"\b(?:replace|with|onto|over)\s+(?:it\s+)?(?:with\s+)?(?:a\s+)?([a-z]+)(?:\s+background)?\b",
        lowered,
    )
    if replace_match:
        candidate = replace_match.group(1)
        replacement_color = SOLID_COLOR_VALUES.get(candidate, replacement_color)
    for color_name, ffmpeg_color in SOLID_COLOR_VALUES.items():
        if f"with {color_name}" in lowered or f"to {color_name}" in lowered or f"{color_name} background" in lowered:
            replacement_color = ffmpeg_color

    similarity = 0.24 if any(term in lowered for term in ["spill", "rough", "uneven", "heavy"]) else 0.20
    blend = 0.10 if any(term in lowered for term in ["soft", "spill", "smooth"]) else 0.08
    return {
        "key_color": key_color,
        "replacement_color": replacement_color,
        "similarity": similarity,
        "blend": blend,
    }


def chroma_key_additional_visual_edits_requested(command_text):
    lowered = command_text.lower()
    style_terms = [
        "cinematic", "grade", "color", "contrast", "saturation", "brightness",
        "grain", "film", "vhs", "lofi", "dream", "memory", "underwater",
        "blur", "sharpen", "glow", "glitch", "flash", "strobe", "zoom",
        "shake", "vertical", "reels", "tiktok", "square", "portrait",
        "caption", "text", "subtitle", "black and white", "grayscale",
    ]
    return any(term in lowered for term in style_terms)


def film_damage_requested(command_text):
    lowered = command_text.lower()
    direct_terms = [
        "old film", "damaged film", "16mm", "8mm", "super 8",
        "gate weave", "film burn", "film damage", "scratchy film",
    ]
    if any(term in lowered for term in direct_terms):
        return True
    film_terms = ["film", "celluloid", "reel"]
    damage_terms = ["scratches", "scratch", "dust", "flicker", "weave", "damaged", "aged"]
    return any(term in lowered for term in film_terms) and any(term in lowered for term in damage_terms)


def visual_age_texture_requested(command_text):
    lowered = command_text.lower()
    direct_terms = [
        "film grain", "grainy footage", "old footage", "old video",
        "aged footage", "vintage footage", "retro footage", "vhs",
        "8mm", "16mm", "super 8", "dusty film", "scratched film",
    ]
    if any(term in lowered for term in direct_terms):
        return True
    if "old" not in lowered:
        return any(term in lowered for term in ["grain", "film grain", "dusty footage", "scratchy footage"])
    visual_terms = ["look", "footage", "video", "visual", "film", "tape", "camera", "style", "aesthetic"]
    audio_terms = ["telephone", "phone", "call", "audio", "voice", "speech", "dialogue", "sound"]
    return any(term in lowered for term in visual_terms) and not any(term in lowered for term in audio_terms)


def film_damage_params(command_text):
    lowered = command_text.lower()
    intensity = 0.7
    if any(term in lowered for term in ["barely", "imperceptible", "subtle", "soft", "gentle", "light", "slight"]):
        intensity = 0.35
    if any(term in lowered for term in ["strong", "heavy", "pronounced", "damaged", "destroyed", "extreme", "violent", "aggressive"]):
        intensity = 0.9

    grain = int(round(12 + intensity * 32))
    gate_weave = int(round(2 + intensity * 12))
    scratch_opacity = round(0.08 + intensity * 0.24, 2)
    dust_opacity = round(0.16 + intensity * 0.38, 2)

    return {
        "intensity": round(intensity, 2),
        "grain": clamp_int(grain, 12, 48),
        "gate_weave": clamp_int(gate_weave, 3, 14),
        "scratch_opacity": max(0.08, min(0.34, scratch_opacity)),
        "dust_opacity": max(0.16, min(0.56, dust_opacity)),
    }


def comic_halftone_requested(command_text):
    lowered = command_text.lower()
    style_terms = [
        "comic book", "comic-book", "comic style", "comic", "halftone",
        "manga", "ink outline", "inked outline", "thick outline", "thick outlines",
    ]
    return any(term in lowered for term in style_terms)


def comic_halftone_filter_step(command_text):
    lowered = command_text.lower()
    block = 8 if any(term in lowered for term in ["detailed", "fine", "subtle", "small dots"]) else 12
    high = 0.14 if any(term in lowered for term in ["thick", "bold", "strong", "heavy"]) else 0.18
    low = 0.04 if any(term in lowered for term in ["thick", "bold", "strong", "heavy"]) else 0.06
    return {
        "description": "Comic halftone blocks with thick color-mixed outlines",
        "filter": (
            f"pixelize=width={block}:height={block},"
            "eq=contrast=1.28:saturation=1.25,"
            f"edgedetect=mode=colormix:low={low:.2f}:high={high:.2f}"
        ),
    }


def underwater_requested(command_text):
    lowered = command_text.lower()
    return "underwater" in lowered or "under water" in lowered or "submerged" in lowered


def underwater_video_filter_steps():
    return [
        {"description": "Blue-green underwater grade", "filter": "colorbalance=rs=-0.2:rm=-0.1:rh=0.05:bs=0.25:bm=0.2:bh=0.3"},
        {"description": "Bounded underwater wave drift", "filter": "crop=iw-24:ih-24:12+6*sin(2*t):12+6*cos(1.7*t),scale=iw+24:ih+24,setsar=1"},
        {"description": "Soft underwater diffusion blur", "filter": "gblur=sigma=1.2"},
        {"description": "Underwater depth vignette", "filter": "vignette=angle=PI/3"},
    ]


def underwater_audio_filter_steps():
    return [
        {"description": "Muffle audio like sound passing through water", "filter": "lowpass=f=800"},
        {"description": "Short underwater space echo", "filter": "aecho=0.8:0.4:300:0.4"},
    ]


def ocr_redact_requested(command_text):
    lowered = command_text.lower()
    action_terms = ["blur", "redact", "hide", "cover", "remove", "censor", "mask"]
    target_terms = [
        "license plate", "license plates", "number plate", "number plates",
        "plate number", "plate numbers", "screen text", "onscreen text",
        "on-screen text", "visible text", "all text", "any text", "ocr",
        "document text", "serial number", "serial numbers",
    ]
    return any(term in lowered for term in action_terms) and any(term in lowered for term in target_terms)


def face_privacy_requested(command_text):
    lowered = command_text.lower()
    action_terms = ["blur", "redact", "hide", "cover", "censor", "mask", "anonymize", "anonymise"]
    target_terms = [
        "face", "faces", "person", "people", "human", "humans",
        "identity", "identities", "head", "heads",
    ]
    return any(term in lowered for term in action_terms) and any(term in lowered for term in target_terms)


def face_privacy_params(command_text):
    lowered = command_text.lower()
    target = "people" if any(term in lowered for term in ["person", "people", "human", "humans"]) else "faces"
    group_terms = ["faces", "people", "humans", "all", "everyone", "crowd", "multiple"]
    layout = "group" if any(term in lowered for term in group_terms) else "center"
    if any(term in lowered for term in ["full body", "whole body", "entire person", "entire people"]):
        target = "people"
        layout = "body"
    return {"target": target, "layout": layout}


def black_remove_requested(command_text):
    lowered = command_text.lower()
    if "black and white" in lowered or "black-and-white" in lowered:
        return False
    action_terms = ["remove", "delete", "cut", "trim", "skip", "drop"]
    target_terms = [
        "black screen", "black screens", "black frame", "black frames",
        "black section", "black sections", "black part", "black parts",
        "blank screen", "blank screens", "blank frame", "blank frames",
        "blank section", "blank sections", "empty screen", "empty screens",
        "dark screen", "dark screens",
    ]
    return any(term in lowered for term in action_terms) and any(term in lowered for term in target_terms)


def freeze_remove_requested(command_text):
    lowered = command_text.lower()
    if any(term in lowered for term in ["freeze frame effect", "add freeze frame", "make a freeze frame"]):
        return False
    action_terms = ["remove", "delete", "cut", "trim", "skip", "drop"]
    target_terms = [
        "frozen frame", "frozen frames", "freeze frame", "freeze frames",
        "frozen section", "frozen sections", "frozen part", "frozen parts",
        "frozen video", "stuck frame", "stuck frames", "stuck section",
        "stuck sections", "stalled frame", "stalled frames", "frame holds",
        "held frame", "held frames",
    ]
    return any(term in lowered for term in action_terms) and any(term in lowered for term in target_terms)


def dedupe_frames_requested(command_text):
    lowered = command_text.lower()
    action_terms = ["remove", "delete", "drop", "dedupe", "de-duplicate", "deduplicate", "clean"]
    target_terms = [
        "duplicate frame", "duplicate frames", "duplicated frame", "duplicated frames",
        "repeated frame", "repeated frames", "near duplicate frame", "near duplicate frames",
        "duplicate video frames", "frame duplicates",
    ]
    return any(term in lowered for term in action_terms) and any(term in lowered for term in target_terms)


def beat_cut_requested(command_text):
    lowered = command_text.lower()
    if no_audio_sync_requested(command_text):
        return False
    beat_terms = ["beat", "beats", "rhythm", "drop", "music"]
    if not any(term in lowered for term in beat_terms):
        return False
    structural_terms = [
        "cut to the beat", "cut on every beat", "cuts on every beat",
        "cut every beat", "jump cut", "jump cuts", "hard cut", "hard cuts",
        "beat cut", "beat cuts", "edit to the beat", "cut with the music",
        "cuts with the music", "sync cuts", "sync the cuts", "make cuts",
    ]
    return any(term in lowered for term in structural_terms)


def scene_montage_requested(command_text):
    lowered = command_text.lower()
    if any(term in lowered for term in ["beat", "rhythm", "audio sync", "music sync"]):
        return False
    action_terms = ["make", "create", "cut", "edit", "build", "generate"]
    target_terms = [
        "highlight reel", "highlights", "best moments", "montage",
        "recap", "trailer", "quick cut", "quick cuts", "fast cut",
        "fast cuts", "scene montage", "scene changes", "every scene",
    ]
    return any(term in lowered for term in action_terms) and any(term in lowered for term in target_terms)


def energy_montage_requested(command_text):
    lowered = command_text.lower()
    action_terms = ["make", "create", "cut", "edit", "build", "generate"]
    target_terms = [
        "hype reel", "energy montage", "energetic montage", "high energy montage",
        "high-energy montage", "loudest moments", "loudest parts", "loud parts",
        "loud moments", "highest energy", "high energy moments",
        "high-energy moments", "most energetic moments", "biggest moments",
    ]
    if not any(term in lowered for term in action_terms) or not any(term in lowered for term in target_terms):
        return False
    return any(term in lowered for term in ["audio", "music", "sound", "loud", "energy", "energetic", "hype"])


def energy_reactive_effects_requested(command_text):
    lowered = command_text.lower()
    audio_terms = [
        "music gets louder", "audio gets louder", "sound gets louder",
        "loudest moment", "loudest moments", "loud parts", "loud moments",
        "louder", "loudness", "energy curve", "high energy", "highest energy",
    ]
    effect_terms = [
        "glitch", "flash", "strobe", "shake", "pulse", "zoom",
        "brighter", "brightness", "harder", "more intense", "intensity",
    ]
    return any(term in lowered for term in audio_terms) and any(term in lowered for term in effect_terms)


def energy_reactive_video_filter_steps(command_text):
    lowered = command_text.lower()
    steps = []
    if any(term in lowered for term in ["glitch", "harder", "intense", "intensity"]):
        steps.append({
            "description": "Glitch bursts on loudest audio moments",
            "filter": FREI0R_NATIVE_FALLBACKS["glitch0r"],
            "requires_context": "energy_curve",
            "timing": "per_energy",
        })
    if any(term in lowered for term in ["flash", "strobe", "brighter", "brightness", "pulse"]):
        steps.append({
            "description": "White flash on loudest audio moments",
            "filter": "drawbox=x=0:y=0:w=iw:h=ih:color=white@0.55:t=fill",
            "requires_context": "energy_curve",
            "timing": "per_energy",
        })
    if any(term in lowered for term in ["shake", "jolt", "hit"]):
        steps.append({
            "description": "Frame jolt on loudest audio moments",
            "filter": "crop=iw-36:ih-36:18+18*sin(30*t):18+18*cos(24*t),scale=iw+36:ih+36",
            "requires_context": "energy_curve",
            "timing": "per_energy",
        })
    return steps


def security_camera_requested(command_text):
    lowered = command_text.lower()
    return (
        any(term in lowered for term in ["security camera", "cctv", "surveillance camera", "surveillance footage"])
        or ("timestamp" in lowered and "scanline" in lowered)
    )


def security_camera_filter_steps():
    return [
        {
            "description": "Security camera low-saturation contrast",
            "filter": "eq=contrast=1.08:saturation=0.35",
        },
        {
            "description": "Security camera scanlines",
            "filter": "drawgrid=width=iw:height=4:thickness=1:color=black@0.35",
        },
        {
            "description": "Security timestamp overlay",
            "filter": (
                f"drawtext=text={quote_drawtext_filter_text('REC 00:00:00')}:"
                "fontcolor=white@0.85:fontsize=28:x=24:y=24:"
                "box=1:boxcolor=black@0.35"
            ),
        },
    ]


def has_equivalent_energy_filter(video_filters, required):
    required_description = required.get("description")
    required_parts = split_filter_chain(required.get("filter", ""))
    required_name = filter_name(required_parts[0]) if required_parts else ""
    for step in video_filters:
        if step.get("requires_context") != "energy_curve":
            continue
        if required_description and step.get("description") == required_description:
            return True
        existing_parts = split_filter_chain(step.get("filter", ""))
        existing_name = filter_name(existing_parts[0]) if existing_parts else ""
        if required_name and existing_name == required_name:
            return True
    return False


def privacy_blur_requested(command_text):
    lowered = command_text.lower()
    if ocr_redact_requested(command_text) and not any(term in lowered for term in ["face", "faces", "person", "people", "center", "centre"]):
        return False
    return "blur" in lowered and any(term in lowered for term in ["privacy", "center", "centre", "face", "faces"])


def remove_audio_requested(command_text):
    lowered = command_text.lower()
    direct_phrases = [
        "remove audio",
        "remove the audio",
        "delete audio",
        "delete the audio",
        "drop audio",
        "drop the audio",
        "strip audio",
        "strip the audio",
        "mute audio",
        "mute the audio",
        "mute this clip",
        "mute the clip",
        "mute video",
        "mute the video",
        "remove sound",
        "remove the sound",
        "delete sound",
        "delete the sound",
        "no sound",
        "without sound",
        "zero audio",
        "soundless video",
        "silent video",
    ]
    if any(phrase in lowered for phrase in direct_phrases):
        return True

    silence_edit_terms = [
        "remove silences",
        "remove all silences",
        "cut silences",
        "cut all silences",
        "delete silences",
        "silent gaps",
        "silent pauses",
        "silent sections",
        "silent parts",
    ]
    if any(term in lowered for term in silence_edit_terms):
        return False

    return bool(re.search(
        r"\bmake\s+(?:it|this|the\s+clip|this\s+clip|the\s+video|video|clip)\s+"
        r"(?:completely\s+|totally\s+)?(?:silent|soundless)\b",
        lowered,
    ))


def mix_uploaded_audio_requested(command_text):
    lowered = command_text.lower()
    if not any(term in lowered for term in ["uploaded audio", "uploaded music", "audio file", "music", "song", "track", "soundtrack"]):
        return False
    mix_terms = [
        "add music", "add the music", "add background", "background music",
        "mix in", "mix the", "under dialogue", "under the dialogue",
        "under voice", "under the voice", "under speech", "under the speech",
        "duck", "lower the music", "behind the voice", "behind dialogue",
    ]
    return any(term in lowered for term in mix_terms)


def replace_uploaded_audio_requested(command_text):
    lowered = command_text.lower()
    if remove_audio_requested(command_text) or mix_uploaded_audio_requested(command_text):
        return False
    explicit_terms = [
        "replace audio", "replace the audio", "replace original audio",
        "replace the original audio", "replace sound", "replace the sound",
        "swap audio", "swap the audio", "use uploaded audio",
        "use the uploaded audio", "use uploaded music", "use the uploaded music",
        "set audio to", "set the audio to", "make the uploaded audio the soundtrack",
    ]
    if any(term in lowered for term in explicit_terms):
        return True
    return "uploaded audio" in lowered and any(term in lowered for term in ["sync", "beat", "rhythm", "music"])


def audio_cleanup_requested(command_text):
    lowered = command_text.lower()
    if remove_audio_requested(command_text):
        return False
    if any(term in lowered for term in ["add noise", "add film noise", "film grain", "visual noise", "noise texture"]):
        return False

    cleanup_terms = [
        "denoise", "de-noise", "remove noise", "reduce noise", "background noise",
        "clean audio", "clean up audio", "cleanup audio", "clean the audio",
        "hiss", "hum", "buzz", "fan noise", "room noise", "wind noise",
        "audio cleanup", "noise reduction",
    ]
    dialogue_terms = [
        "dialogue clearer", "dialog clearer", "make speech clear", "make the speech clear",
        "make voice clear", "make the voice clear", "enhance dialogue", "enhance dialog",
        "enhance voice", "enhance speech", "clearer voice", "clear speech",
        "voice clearer", "speech clearer",
    ]
    deess_terms = ["de-ess", "deess", "sibilance", "harsh s", "sharp s"]
    return any(term in lowered for term in cleanup_terms + dialogue_terms + deess_terms)


def audio_cleanup_filter_step(command_text):
    lowered = command_text.lower()
    heavy = any(term in lowered for term in ["heavy", "strong", "very noisy", "lots of noise", "loud background"])
    light = any(term in lowered for term in ["subtle", "gentle", "light"])
    noise_reduction = 12 if light else 24 if heavy else 18
    noise_floor = -40 if light else -30 if heavy else -35

    parts = []
    if any(term in lowered for term in ["rumble", "wind", "hum", "low frequency", "low-frequency"]):
        parts.append("highpass=f=100")
    else:
        parts.append("highpass=f=80")

    parts.append(f"afftdn=nr={noise_reduction}:nf={noise_floor}")
    parts.append("lowpass=f=12000")

    if any(term in lowered for term in ["dialogue", "dialog", "speech", "voice", "vocal", "interview"]):
        parts.append("speechnorm=e=4:c=2:r=0.0005:l=1")
        parts.append("agate=threshold=0.035:ratio=2.5:attack=8:release=120")

    if any(term in lowered for term in ["de-ess", "deess", "sibilance", "harsh s", "sharp s"]):
        parts.append("deesser=i=0.45:m=0.55:f=0.5")

    parts.append("loudnorm=I=-16:TP=-1.5:LRA=9")
    return {
        "description": "Denoise and clarify dialogue with FFmpeg audio cleanup filters",
        "filter": ",".join(parts),
    }


def audio_filter_step_names(audio_filters):
    names = set()
    for step in audio_filters or []:
        filter_string = step.get("filter") if isinstance(step, dict) else None
        if not filter_string:
            continue
        for part in split_filter_chain(filter_string):
            names.add(filter_name(part))
    return names


def audio_filter_steps_contain(audio_filters, names):
    return bool(audio_filter_step_names(audio_filters) & set(names))


def audio_cleanup_satisfied(command_text, audio_filters):
    lowered = command_text.lower()
    names = audio_filter_step_names(audio_filters)
    if "afftdn" not in names:
        return False
    if any(term in lowered for term in ["dialogue", "dialog", "speech", "voice", "vocal", "interview"]):
        if "speechnorm" not in names:
            return False
    if any(term in lowered for term in ["de-ess", "deess", "sibilance", "harsh s", "sharp s"]):
        if "deesser" not in names:
            return False
    return True


def uploaded_audio_mix_params(command_text):
    lowered = command_text.lower()
    duck = any(term in lowered for term in ["duck", "under dialogue", "under the dialogue", "under voice", "under speech", "behind the voice"])
    music_volume = 0.28 if duck else 0.35
    if any(term in lowered for term in ["subtle", "soft", "quiet", "low"]):
        music_volume = min(music_volume, 0.22)
    elif any(term in lowered for term in ["loud", "strong", "heavy"]):
        music_volume = 0.5
    return {"original_volume": 1.0, "music_volume": music_volume, "duck": duck}


def picture_in_picture_requested(command_text):
    lowered = command_text.lower()
    return any(term in lowered for term in [
        "picture in picture",
        "picture-in-picture",
        "pip",
    ]) or (
        "duplicate" in lowered
        and any(term in lowered for term in ["top right", "top-right", "corner"])
    )


def split_screen_mirror_requested(command_text):
    lowered = command_text.lower()
    return (
        any(term in lowered for term in ["split screen", "split-screen"])
        and any(term in lowered for term in ["mirror", "mirrored", "flipped", "flip"])
    )


def auto_captions_requested(command_text):
    if extracted_text_prompt(command_text):
        return False

    lowered = command_text.lower()
    phrases = [
        "auto caption", "auto captions", "automatic caption", "automatic captions",
        "generate captions", "generate subtitles", "create captions", "create subtitles",
        "add subtitles", "add captions", "burn subtitles", "burn captions",
        "caption the speech", "caption speech", "speech captions", "speech subtitles",
        "captions from speech", "subtitles from speech", "captions for speech",
        "subtitles for speech", "caption what they say", "subtitle what they say",
        "what they say", "what is being said", "transcribe", "transcription",
        "make subtitles", "make captions",
    ]
    if any(phrase in lowered for phrase in phrases):
        return True

    return (
        any(term in lowered for term in ["caption", "captions", "subtitle", "subtitles"])
        and any(term in lowered for term in ["speech", "voice", "dialogue", "dialog", "spoken", "transcript"])
    )


def auto_caption_params(command_text):
    lowered = command_text.lower()
    style = "bottom_box"
    if any(term in lowered for term in ["karaoke", "word by word", "highlight words"]):
        style = "bottom_box"
    return {"source": "speech", "language": "en", "style": style}


def normalized_speed_factor(value, default=None):
    try:
        factor = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.1, min(8.0, factor))


NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


def natural_pitch_requested(command_text):
    lowered = command_text.lower()
    return any(term in lowered for term in [
        "keep audio pitch",
        "keep the audio pitch",
        "keep pitch",
        "preserve pitch",
        "natural pitch",
        "pitch natural",
        "without changing pitch",
    ])


def parse_number_or_word(value):
    text = str(value or "").strip().lower()
    if text in NUMBER_WORDS:
        return float(NUMBER_WORDS[text])
    try:
        return float(text)
    except ValueError:
        return None


def pitch_special_from_command(command_text):
    lowered = command_text.lower()
    if natural_pitch_requested(command_text):
        return None
    if not any(term in lowered for term in ["pitch", "semitone", "semitones", "octave", "chipmunk", "deep voice"]):
        return None

    semitones = None
    semitone_match = re.search(
        r"\b(\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+semitones?\b",
        lowered,
    )
    if semitone_match:
        semitones = parse_number_or_word(semitone_match.group(1))
    elif "octave" in lowered:
        semitones = 12.0
    elif any(term in lowered for term in ["chipmunk", "higher", "raise", "pitch up"]):
        semitones = 3.0
    elif any(term in lowered for term in ["lower", "deeper", "deep voice", "pitch down", "drop pitch"]):
        semitones = 3.0

    if semitones is None:
        return None

    if any(term in lowered for term in ["lower", "down", "deeper", "deep voice", "drop"]):
        semitones = -abs(semitones)
    elif any(term in lowered for term in ["raise", "up", "higher", "chipmunk"]):
        semitones = abs(semitones)

    if abs(semitones) < 0.01:
        return None
    return {"type": "pitch_shift", "params": {"semitones": round(semitones, 3)}}


def speed_special_from_command(command_text):
    lowered = command_text.lower()
    has_plain_slow = re.search(r"\bslow\b", lowered) is not None
    if not any(term in lowered for term in ["speed", "faster", "fast", "slower", "slow down", "slow motion", "slo mo", "slomo"]) and not has_plain_slow:
        return None

    factor = None
    faster_percent = (
        re.search(r"\b(\d+(?:\.\d+)?)\s*(?:%|percent)\s+(?:faster|quicker)\b", lowered)
        or re.search(r"\b(?:speed\s+up|make\s+(?:it|the\s+clip|the\s+video)\s+faster)\s+by\s+(\d+(?:\.\d+)?)\s*(?:%|percent)\b", lowered)
    )
    slower_percent = (
        re.search(r"\b(\d+(?:\.\d+)?)\s*(?:%|percent)\s+(?:slower)\b", lowered)
        or re.search(r"\b(?:slow\s+(?:it\s+|the\s+clip\s+|the\s+video\s+)?down)\s+by\s+(\d+(?:\.\d+)?)\s*(?:%|percent)\b", lowered)
    )
    if faster_percent:
        factor = 1.0 + (float(faster_percent.group(1)) / 100.0)
    elif slower_percent:
        factor = 1.0 - (float(slower_percent.group(1)) / 100.0)
    else:
        x_match = re.search(r"\b(\d+(?:\.\d+)?)\s*x\s*(?:faster|speed|as\s+fast)?\b", lowered)
        if x_match:
            factor = float(x_match.group(1))
        elif re.search(r"\b(?:twice|double)\s+(?:as\s+)?(?:fast|speed)\b", lowered):
            factor = 2.0
        elif re.search(r"\b(?:half|1/2|0\.5x)\s+(?:speed|as\s+fast)\b", lowered):
            factor = 0.5
        elif "slow motion" in lowered or "slo mo" in lowered or "slomo" in lowered:
            factor = 0.5
        elif "slower" in lowered or "slow down" in lowered:
            factor = 0.75
        elif has_plain_slow and not re.search(r"\bslow\s+(?:fade|transition|zoom|pan|push|pull|dissolve)\b", lowered):
            factor = 0.75
        elif "faster" in lowered or "speed up" in lowered:
            factor = 1.25

    factor = normalized_speed_factor(factor)
    if factor is None or abs(factor - 1.0) < 0.01:
        return None
    return {"type": "speed_ramp", "params": {"factor": round(factor, 4)}}


TIME_VALUE_PATTERN = (
    r"(\d+(?:\.\d+)?(?::\d{1,2}(?:\.\d+)?)?(?::\d{1,2}(?:\.\d+)?)?)"
    r"\s*(milliseconds?|msecs?|ms|seconds?|secs?|sec|s|minutes?|mins?|min|m|hours?|hrs?|hr|h)?"
)


def parse_time_value(value, unit=None):
    text = str(value or "").strip().lower()
    if not text:
        return None
    if ":" in text:
        parts = [float(part) for part in text.split(":")]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        return None

    try:
        number = float(text)
    except ValueError:
        return None

    normalized_unit = (unit or "seconds").lower()
    if normalized_unit.startswith("ms") or normalized_unit.startswith("millisecond") or normalized_unit.startswith("msec"):
        return number / 1000
    if normalized_unit in {"m", "min", "mins", "minute", "minutes"}:
        return number * 60
    if normalized_unit in {"h", "hr", "hrs", "hour", "hours"}:
        return number * 3600
    return number


def matched_time(match, group_index=1):
    return parse_time_value(match.group(group_index), match.group(group_index + 1))


def strobe_period_from_command(command_text):
    lowered = command_text.lower().replace("-", " ")
    if not any(term in lowered for term in ["strobe", "flash", "flashes", "flicker", "flickers", "pulse", "pulses"]):
        return None
    if not any(term in lowered for term in ["every", "each", "per second", "twice a second", "once a second"]):
        return None
    if any(term in lowered for term in ["every beat", "each beat", "per beat", "on every beat"]):
        return None

    if re.search(r"\b(?:every|each)\s+(?:a\s+)?half(?:\s+a)?\s+second\b", lowered):
        return 0.5
    if "twice a second" in lowered or "two times a second" in lowered:
        return 0.5
    if "once a second" in lowered or re.search(r"\b(?:every|each)\s+second\b", lowered):
        return 1.0

    match = re.search(r"\b(?:every|each)\s+" + TIME_VALUE_PATTERN, lowered)
    if match:
        period = parse_time_value(match.group(1), match.group(2))
        if period is not None:
            return max(0.1, min(10.0, period))

    return None


def trim_special_from_command(command_text):
    lowered = command_text.lower()

    range_patterns = [
        (
            r"\b(?:remove|delete|cut\s+out)\s+(?:the\s+)?(?:part\s+)?(?:from|between)\s+"
            + TIME_VALUE_PATTERN
            + r"\s+(?:to|and|-)\s+"
            + TIME_VALUE_PATTERN,
            "remove_segment",
        ),
        (
            r"\b(?:keep|use|select|export|trim)\s+(?:only\s+)?(?:the\s+)?(?:part\s+)?(?:from|between)\s+"
            + TIME_VALUE_PATTERN
            + r"\s+(?:to|and|-)\s+"
            + TIME_VALUE_PATTERN,
            "trim",
        ),
    ]
    for pattern, action_type in range_patterns:
        match = re.search(pattern, lowered)
        if not match:
            continue
        start = matched_time(match, 1)
        end = matched_time(match, 3)
        if start is None or end is None:
            continue
        if action_type == "remove_segment":
            return {"type": "remove_segment", "params": {"start": start, "end": end}}
        return {"type": "trim", "params": {"start": start, "end": end}}

    first_match = re.search(
        r"\b(?:remove|delete|cut|trim|skip)\s+(?:off\s+)?(?:the\s+)?(?:first|opening|beginning)\s+"
        + TIME_VALUE_PATTERN,
        lowered,
    )
    if first_match:
        start = matched_time(first_match, 1)
        if start is not None:
            return {"type": "trim", "params": {"start": start}}

    last_remove_match = re.search(
        r"\b(?:remove|delete|cut|trim)\s+(?:off\s+)?(?:the\s+)?(?:last|ending|end)\s+"
        + TIME_VALUE_PATTERN,
        lowered,
    )
    if last_remove_match:
        remove_end = matched_time(last_remove_match, 1)
        if remove_end is not None:
            return {"type": "trim", "params": {"remove_end": remove_end}}

    keep_first_match = re.search(
        r"\b(?:keep|use|export|make)\s+(?:only\s+)?(?:the\s+)?first\s+"
        + TIME_VALUE_PATTERN,
        lowered,
    )
    if keep_first_match:
        duration = matched_time(keep_first_match, 1)
        if duration is not None:
            return {"type": "trim", "params": {"start": 0, "duration": duration}}

    keep_last_match = re.search(
        r"\b(?:keep|use|export|make)\s+(?:only\s+)?(?:the\s+)?last\s+"
        + TIME_VALUE_PATTERN,
        lowered,
    )
    if keep_last_match:
        duration = matched_time(keep_last_match, 1)
        if duration is not None:
            return {"type": "trim", "params": {"from_end": duration}}

    return None


def privacy_blur_filter_step():
    return {
        "description": "Blur the center region and darken the edges",
        "filter": (
            "delogo=x=__PRIVACY_X__:y=__PRIVACY_Y__:w=__PRIVACY_W__:h=__PRIVACY_H__:show=0,"
            "vignette=angle=PI/3,eq=brightness=-0.04"
        ),
        "requires_context": None,
        "timing": "continuous",
    }


def clamp_int(value, minimum, maximum):
    return max(minimum, min(maximum, int(round(float(value)))))


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalized_ocr_detection(raw_detection, width, height, frame_time, padding):
    x = clamp_int(raw_detection.get("x", 0) - padding, 1, max(1, width - 3))
    y = clamp_int(raw_detection.get("y", 0) - padding, 1, max(1, height - 3))
    right = clamp_int(raw_detection.get("x", 0) + raw_detection.get("w", 0) + padding, x + 2, max(x + 2, width - 1))
    bottom = clamp_int(raw_detection.get("y", 0) + raw_detection.get("h", 0) + padding, y + 2, max(y + 2, height - 1))
    start = max(0.0, float(frame_time) - 0.65)
    end = float(frame_time) + 1.0
    return {
        "x": x,
        "y": y,
        "w": max(2, right - x),
        "h": max(2, bottom - y),
        "start": round(start, 3),
        "end": round(end, 3),
        "text": str(raw_detection.get("text", ""))[:80],
        "confidence": round(safe_float(raw_detection.get("confidence"), 0.0), 2),
    }


def ocr_redact_filter_chain(detections, max_filters=80):
    filters = []
    for detection in detections[:max_filters]:
        x = int(detection["x"])
        y = int(detection["y"])
        w = int(detection["w"])
        h = int(detection["h"])
        start = float(detection["start"])
        end = float(detection["end"])
        filters.append(
            f"delogo=x={x}:y={y}:w={w}:h={h}:show=0:enable='between(t,{start:.3f},{end:.3f})'"
        )
    return ",".join(filters)


def drawbox_option_map(filter_part):
    options = split_filter_options(filter_part[len("drawbox="):])
    option_map = {}
    for option in options:
        key, separator, value = option.partition("=")
        if separator:
            option_map[key] = value
    return option_map


def is_full_frame_drawbox(filter_part):
    if filter_name(filter_part) != "drawbox":
        return False
    option_map = drawbox_option_map(filter_part)
    return (
        option_map.get("x") == "0"
        and option_map.get("y") == "0"
        and option_map.get("w") == "iw"
        and option_map.get("h") == "ih"
        and option_map.get("t") == "fill"
    )


def drawtext_text_value(filter_part):
    if filter_name(filter_part) != "drawtext" or not filter_part.startswith("drawtext="):
        return None
    for option in split_filter_options(filter_part[len("drawtext="):]):
        key, separator, value = option.partition("=")
        if separator and key == "text":
            return unquote_filter_value(value)
    return None


def quote_drawtext_text(value):
    return "'" + value.replace("\\", "\\\\").replace("'", r"\'") + "'"


def strip_reverse_filters(video_filters):
    stripped = []
    removed_reverse = False
    for step in video_filters:
        filter_string = step.get("filter")
        if not filter_string:
            continue
        parts = []
        for part in split_filter_chain(filter_string):
            if filter_name(part) == "reverse":
                removed_reverse = True
                continue
            parts.append(part)
        if parts:
            repaired_step = dict(step)
            repaired_step["filter"] = ",".join(parts)
            stripped.append(repaired_step)
        else:
            removed_reverse = True
    return stripped, removed_reverse


def strobe_filter_for_period(period):
    period = max(0.1, min(10.0, float(period)))
    flash_width = min(0.08, max(0.04, period * 0.08))
    return (
        "drawbox=x=0:y=0:w=iw:h=ih:color=white@0.45:t=fill:"
        f"enable='lt(mod(t,{period:g}),{flash_width:g})'"
    )


def align_strobe_filters(video_filters, period):
    desired_filter = strobe_filter_for_period(period)
    aligned = []
    replaced = False

    for step in video_filters:
        filter_string = step.get("filter")
        if not filter_string:
            continue

        description = step.get("description", "").lower()
        is_strobe_step = any(term in description for term in ["strobe", "flash", "flicker", "pulse"])
        parts = []
        replaced_in_step = False
        for part in split_filter_chain(filter_string):
            if is_full_frame_drawbox(part):
                if not replaced:
                    parts.append(desired_filter)
                    replaced = True
                replaced_in_step = True
                continue
            parts.append(part)

        if is_strobe_step and not replaced_in_step:
            repaired_step = dict(step)
            repaired_step["filter"] = desired_filter
            aligned.append(repaired_step)
            replaced = True
            continue

        if parts:
            repaired_step = dict(step)
            repaired_step["filter"] = ",".join(parts)
            aligned.append(repaired_step)

    if not replaced:
        aligned.append({
            "description": "Timed strobe flashes",
            "filter": desired_filter,
        })
    return aligned


def align_text_rollout_filters(video_filters):
    aligned = []
    text_value = None
    removed_overlay = False
    replaced_rollout = False

    for step in video_filters:
        filter_string = step.get("filter")
        if not filter_string:
            continue

        kept_parts = []
        for part in split_filter_chain(filter_string):
            if text_value is None:
                text_value = drawtext_text_value(part) or text_value
            if filter_name(part) == "drawtext":
                enable_expression = extract_enable_expression(part) or ""
                if re.search(r"(?:between\(t,3,6\)|gte\(t,3\))", enable_expression.replace(" ", "")):
                    replaced_rollout = True
                    continue
            if is_full_frame_drawbox(part):
                removed_overlay = True
                continue
            kept_parts.append(part)

        if kept_parts:
            repaired_step = dict(step)
            repaired_step["filter"] = ",".join(kept_parts)
            aligned.append(repaired_step)

    if text_value and (removed_overlay or replaced_rollout or aligned):
        aligned.append({
            "description": "Roll text out after the opening fade",
            "filter": (
                f"drawtext=text={quote_drawtext_text(text_value)}:"
                "fontcolor=white:fontsize=48:"
                "x='(w-text_w)/2+(t-3)*220':y=(h-text_h)/2:"
                "enable='between(t,3,6)'"
            ),
            "requires_context": None,
            "timing": "continuous",
        })

    return aligned


def model_output_format_filter_part(filter_part):
    name = filter_name(filter_part)
    if name in {"pad", "setsar", "setdar"}:
        return True
    if name == "scale":
        return (
            "force_original_aspect_ratio" in filter_part
            or bool(re.search(r"\b(?:1080|1350|1920|2160|3840)\b", filter_part))
            or bool(re.search(r"scale=-?\d+:1\b", filter_part))
            or "iw" in filter_part
            or "ih" in filter_part
        )
    if name == "crop":
        return (
            bool(re.search(r"crop=(?:1080|1920|3840):(?:804|1080|1350|1920|2160)", filter_part))
            or "crop=iw" in filter_part
            or "crop=ih" in filter_part
            or "iw/" in filter_part
            or "ih/" in filter_part
        )
    return False


def strip_model_output_format_filters(video_filters):
    stripped = []
    for step in video_filters:
        filter_string = step.get("filter")
        if not filter_string:
            continue
        kept_parts = [
            part for part in split_filter_chain(filter_string)
            if not model_output_format_filter_part(part)
        ]
        if kept_parts:
            repaired_step = dict(step)
            repaired_step["filter"] = ",".join(kept_parts)
            stripped.append(repaired_step)
    return stripped


def strip_time_selection_filters(steps):
    stripped = []
    for step in steps or []:
        filter_string = step.get("filter")
        if not filter_string:
            continue
        kept_parts = []
        for part in split_filter_chain(filter_string):
            name = filter_name(part)
            if name in {"trim", "atrim", "select", "aselect"}:
                continue
            if name in {"setpts", "asetpts"} and "STARTPTS" in part.upper():
                continue
            kept_parts.append(part)
        if kept_parts:
            repaired_step = dict(step)
            repaired_step["filter"] = ",".join(kept_parts)
            stripped.append(repaired_step)
    return stripped


def layout_filter_part(filter_part, remove_divider=False):
    name = filter_name(filter_part)
    if ";" in filter_part or "[" in filter_part or "]" in filter_part:
        return True
    if name in {"split", "overlay", "hstack", "vstack", "xstack", "concat"}:
        return True
    if name in {"hflip", "vflip"}:
        return True
    if remove_divider and name == "drawbox" and "iw/2" in filter_part:
        return True
    return False


def strip_layout_filters(steps, remove_divider=False):
    stripped = []
    for step in steps or []:
        filter_string = step.get("filter")
        if not filter_string:
            continue
        kept_parts = [
            part for part in split_filter_chain(filter_string)
            if not layout_filter_part(part, remove_divider)
        ]
        if kept_parts:
            repaired_step = dict(step)
            repaired_step["filter"] = ",".join(kept_parts)
            stripped.append(repaired_step)
    return stripped


def generated_redaction_filter_part(filter_part):
    name = filter_name(filter_part)
    lowered = filter_part.lower()
    if name in {"delogo", "removelogo", "cover_rect", "find_rect"}:
        return True
    if name in {"boxblur", "avgblur", "pixelize"} and any(term in lowered for term in ["enable=", "x=", "y=", "w=", "h="]):
        return True
    return False


def strip_generated_redaction_filters(steps):
    stripped = []
    for step in steps or []:
        filter_string = step.get("filter")
        if not filter_string:
            continue
        kept_parts = [
            part for part in split_filter_chain(filter_string)
            if not generated_redaction_filter_part(part)
        ]
        if kept_parts:
            repaired_step = dict(step)
            repaired_step["filter"] = ",".join(kept_parts)
            stripped.append(repaired_step)
    return stripped


def caption_filter_part(filter_part):
    name = filter_name(filter_part)
    lowered = filter_part.lower()
    return name in {"drawtext", "subtitles", "ass"} or any(
        term in lowered for term in ["caption", "subtitle", "transcript"]
    )


def strip_caption_filters(steps):
    stripped = []
    for step in steps or []:
        filter_string = step.get("filter")
        if not filter_string:
            continue
        kept_parts = [
            part for part in split_filter_chain(filter_string)
            if not caption_filter_part(part)
        ]
        if kept_parts:
            repaired_step = dict(step)
            repaired_step["filter"] = ",".join(kept_parts)
            stripped.append(repaired_step)
    return stripped


def chroma_key_filter_part(filter_part):
    name = filter_name(filter_part)
    lowered = filter_part.lower()
    return name in {"chromakey", "colorkey", "backgroundkey"} or "green screen" in lowered or "chroma key" in lowered


def strip_chroma_key_filters(steps):
    stripped = []
    for step in steps or []:
        filter_string = step.get("filter")
        if not filter_string:
            continue
        kept_parts = [
            part for part in split_filter_chain(filter_string)
            if not chroma_key_filter_part(part)
        ]
        if kept_parts:
            repaired_step = dict(step)
            repaired_step["filter"] = ",".join(kept_parts)
            stripped.append(repaired_step)
    return stripped


def film_damage_description(description):
    lowered = str(description or "").lower()
    return any(term in lowered for term in [
        "old film", "damaged film", "film damage", "film scratch", "scratches",
        "dust", "flicker", "gate weave", "16mm", "8mm", "celluloid",
    ])


def film_damage_filter_part(filter_part):
    name = filter_name(filter_part)
    lowered = filter_part.lower()
    if name in {"noise", "deflicker"}:
        return True
    if name == "drawbox" and any(term in lowered for term in ["mod(t", "sin(", "color=white@", "color=black@"]):
        return True
    if name in {"crop", "scale"} and "sin(" in lowered:
        return True
    return False


def strip_film_damage_filters(steps):
    stripped = []
    for step in steps or []:
        filter_string = step.get("filter")
        if not filter_string:
            continue
        if film_damage_description(step.get("description")):
            continue
        kept_parts = [
            part for part in split_filter_chain(filter_string)
            if not film_damage_filter_part(part)
        ]
        if kept_parts:
            repaired_step = dict(step)
            repaired_step["filter"] = ",".join(kept_parts)
            stripped.append(repaired_step)
    return stripped


def distortion_filter_part(filter_part):
    return filter_name(filter_part) in {"lenscorrection", "perspective"} or filter_part.lower().startswith("frei0r=filter_name=distort0r")


def strip_unrequested_distortion_filters(steps):
    stripped = []
    for step in steps or []:
        filter_string = step.get("filter")
        if not filter_string:
            continue
        kept_parts = [
            part for part in split_filter_chain(filter_string)
            if not distortion_filter_part(part)
        ]
        if kept_parts:
            repaired_step = dict(step)
            repaired_step["filter"] = ",".join(kept_parts)
            stripped.append(repaired_step)
    return stripped


def identity_geq_filter_part(filter_part):
    lowered = filter_part.replace(" ", "").lower()
    return (
        lowered.startswith("geq=")
        and "r='p(x,y)'" in lowered
        and "g='p(x,y)'" in lowered
        and "b='p(x,y)'" in lowered
    )


def strip_identity_geq_filters(steps):
    stripped = []
    for step in steps or []:
        filter_string = step.get("filter")
        if not filter_string:
            continue
        kept_parts = [
            part for part in split_filter_chain(filter_string)
            if not identity_geq_filter_part(part)
        ]
        if kept_parts:
            repaired_step = dict(step)
            repaired_step["filter"] = ",".join(kept_parts)
            stripped.append(repaired_step)
    return stripped


def speed_filter_part(filter_part):
    name = filter_name(filter_part)
    lowered = filter_part.lower()
    if name in {"setpts", "asetpts", "atempo"}:
        return True
    if name == "rubberband" and ("tempo=" in lowered or "pitch=1" in lowered):
        return True
    return False


def strip_speed_filters(steps):
    stripped = []
    for step in steps or []:
        filter_string = step.get("filter")
        if not filter_string:
            continue
        kept_parts = [
            part for part in split_filter_chain(filter_string)
            if not speed_filter_part(part)
        ]
        if kept_parts:
            repaired_step = dict(step)
            repaired_step["filter"] = ",".join(kept_parts)
            stripped.append(repaired_step)
    return stripped


def pitch_filter_part(filter_part):
    name = filter_name(filter_part)
    lowered = filter_part.lower()
    if name == "rubberband" and "pitch=" in lowered:
        return True
    if name in {"asetrate", "aresample"} and "pitch" in lowered:
        return True
    return False


def strip_pitch_filters(steps):
    stripped = []
    for step in steps or []:
        filter_string = step.get("filter")
        if not filter_string:
            continue
        kept_parts = [
            part for part in split_filter_chain(filter_string)
            if not pitch_filter_part(part)
        ]
        if kept_parts:
            repaired_step = dict(step)
            repaired_step["filter"] = ",".join(kept_parts)
            stripped.append(repaired_step)
    return stripped


def ensure_special(plan, special_type, params=None):
    special = list(plan.get("special", []))
    if not any(step.get("type") == special_type for step in special):
        special.append({"type": special_type, "params": params or {}})
    plan["special"] = special


def ensure_analysis(plan, function, store_as, extra=None):
    analysis = list(plan.get("analysis", []))
    if not any(step.get("tool") == "librosa" and step.get("function") == function for step in analysis):
        step = {"tool": "librosa", "function": function, "store_as": store_as}
        if extra:
            step.update(extra)
        analysis.append(step)
    plan["analysis"] = analysis


def align_plan_with_command(plan, command_text):
    aligned = json.loads(json.dumps(plan))
    trim_special = trim_special_from_command(command_text)
    speed_special = speed_special_from_command(command_text)
    pitch_special = pitch_special_from_command(command_text)
    wants_privacy_blur = privacy_blur_requested(command_text)
    wants_ocr_redact = ocr_redact_requested(command_text)
    wants_face_privacy = face_privacy_requested(command_text)
    wants_black_remove = black_remove_requested(command_text)
    wants_crop_borders = crop_borders_requested(command_text)
    wants_freeze_remove = freeze_remove_requested(command_text)
    wants_dedupe_frames = dedupe_frames_requested(command_text)
    wants_beat_cut = beat_cut_requested(command_text)
    wants_scene_montage = scene_montage_requested(command_text)
    wants_energy_montage = energy_montage_requested(command_text)
    wants_energy_effects = energy_reactive_effects_requested(command_text)
    wants_boomerang = boomerang_requested(command_text)
    wants_blurred_background = blurred_background_requested(command_text)
    wants_mix_uploaded_audio = mix_uploaded_audio_requested(command_text)
    wants_replace_uploaded_audio = replace_uploaded_audio_requested(command_text)
    wants_auto_captions = auto_captions_requested(command_text)
    wants_audio_cleanup = audio_cleanup_requested(command_text)
    wants_chroma_key = chroma_key_requested(command_text)
    wants_film_damage = film_damage_requested(command_text)
    wants_comic_halftone = comic_halftone_requested(command_text)
    wants_underwater = underwater_requested(command_text)
    wants_security_camera = security_camera_requested(command_text)
    strobe_period = strobe_period_from_command(command_text)

    if no_audio_sync_requested(command_text):
        enforce_no_audio_sync_constraints(aligned)

    if remove_audio_requested(command_text):
        aligned.pop("audio_filters", None)
        aligned["special"] = [
            step for step in aligned.get("special", [])
            if step.get("type") not in {"remove_audio", "silence_remove", "pitch_shift", "replace_audio", "mix_uploaded_audio"}
        ]
        ensure_special(aligned, "remove_audio")

    if wants_mix_uploaded_audio:
        aligned["special"] = [
            step for step in aligned.get("special", [])
            if step.get("type") not in {"replace_audio", "mix_uploaded_audio"}
        ]
        ensure_special(aligned, "mix_uploaded_audio", uploaded_audio_mix_params(command_text))
    elif wants_replace_uploaded_audio:
        aligned["special"] = [
            step for step in aligned.get("special", [])
            if step.get("type") not in {"replace_audio", "mix_uploaded_audio"}
        ]
        ensure_special(aligned, "replace_audio")

    if wants_audio_cleanup:
        audio_filters = list(aligned.get("audio_filters", []))
        if not audio_cleanup_satisfied(command_text, audio_filters):
            audio_filters.append(audio_cleanup_filter_step(command_text))
        aligned["audio_filters"] = audio_filters

    if end_reverse_requested(command_text):
        aligned["video_filters"], removed_reverse = strip_reverse_filters(aligned.get("video_filters", []))
        has_reverse_special = any(step.get("type") == "reverse" for step in aligned.get("special", []))
        if removed_reverse or has_reverse_special:
            aligned["special"] = [
                step for step in aligned.get("special", [])
                if step.get("type") != "reverse"
            ]
            ensure_special(aligned, "end_reverse", {"duration": 1.5})

    if wants_boomerang:
        aligned["video_filters"], _ = strip_reverse_filters(aligned.get("video_filters", []))
        aligned["special"] = [
            step for step in aligned.get("special", [])
            if step.get("type") not in {"boomerang", "reverse", "end_reverse"}
        ]
        ensure_special(aligned, "boomerang", {"loops": 1, "mute_reversed_audio": True})

    if text_rollout_requested(command_text):
        aligned["video_filters"] = align_text_rollout_filters(aligned.get("video_filters", []))

    if wants_auto_captions:
        aligned["video_filters"] = strip_caption_filters(aligned.get("video_filters", []))
        aligned["special"] = [
            step for step in aligned.get("special", [])
            if step.get("type") != "auto_captions"
        ]
        ensure_special(aligned, "auto_captions", auto_caption_params(command_text))

    if strobe_period is not None:
        aligned["video_filters"] = align_strobe_filters(aligned.get("video_filters", []), strobe_period)

    if output_aspect_requested(command_text):
        aligned["video_filters"] = strip_model_output_format_filters(aligned.get("video_filters", []))

    if wants_blurred_background:
        aligned["video_filters"] = strip_layout_filters(strip_model_output_format_filters(aligned.get("video_filters", [])))
        aligned["special"] = [
            step for step in aligned.get("special", [])
            if step.get("type") != "blur_background"
        ]
        ensure_special(aligned, "blur_background", blurred_background_params(command_text))

    if wants_chroma_key:
        aligned["video_filters"] = strip_chroma_key_filters(strip_layout_filters(aligned.get("video_filters", [])))
        if not chroma_key_additional_visual_edits_requested(command_text):
            aligned["video_filters"] = []
        aligned["special"] = [
            step for step in aligned.get("special", [])
            if step.get("type") != "chroma_key"
        ]
        ensure_special(aligned, "chroma_key", chroma_key_params(command_text))

    if wants_film_damage:
        aligned["video_filters"] = strip_film_damage_filters(aligned.get("video_filters", []))
        aligned["special"] = [
            step for step in aligned.get("special", [])
            if step.get("type") != "film_damage"
        ]
        ensure_special(aligned, "film_damage", film_damage_params(command_text))

    if wants_comic_halftone:
        if not any(term in command_text.lower() for term in ["fisheye", "lens", "barrel", "warp", "distort"]):
            aligned["video_filters"] = strip_unrequested_distortion_filters(aligned.get("video_filters", []))
        filters_text = ",".join(step.get("filter", "") for step in aligned.get("video_filters", []))
        if "pixelize" not in filters_text or "edgedetect" not in filters_text:
            video_filters = list(aligned.get("video_filters", []))
            video_filters.append(comic_halftone_filter_step(command_text))
            aligned["video_filters"] = video_filters

    if wants_underwater:
        aligned["video_filters"] = strip_identity_geq_filters(aligned.get("video_filters", []))
        video_filters = list(aligned.get("video_filters", []))
        filters_text = ",".join(step.get("filter", "") for step in video_filters)
        for required in underwater_video_filter_steps():
            name = filter_name(required["filter"])
            if name not in {filter_name(part) for part in split_filter_chain(filters_text)}:
                video_filters.append(required)
                filters_text = ",".join(step.get("filter", "") for step in video_filters)
        aligned["video_filters"] = video_filters

        audio_filters = list(aligned.get("audio_filters", []))
        audio_text = ",".join(step.get("filter", "") for step in audio_filters)
        for required in underwater_audio_filter_steps():
            name = filter_name(required["filter"])
            if name not in {filter_name(part) for part in split_filter_chain(audio_text)}:
                audio_filters.append(required)
                audio_text = ",".join(step.get("filter", "") for step in audio_filters)
        aligned["audio_filters"] = audio_filters

    if wants_security_camera:
        video_filters = [
            step for step in aligned.get("video_filters", [])
            if not ("drawtext" in step.get("filter", "").lower() and "scanlines" in step.get("filter", "").lower())
        ]
        filters_text = ",".join(step.get("filter", "") for step in video_filters)
        existing_names = {
            filter_name(part)
            for step in video_filters
            for part in split_filter_chain(step.get("filter", ""))
        }
        for required in security_camera_filter_steps():
            required_name = filter_name(split_filter_chain(required["filter"])[0])
            if required_name == "eq":
                if "saturation=0." not in filters_text:
                    video_filters.append(required)
                    filters_text = ",".join(step.get("filter", "") for step in video_filters)
                continue
            if required_name not in existing_names:
                video_filters.append(required)
                existing_names.add(required_name)
        aligned["video_filters"] = video_filters

    if wants_face_privacy:
        aligned["video_filters"] = strip_generated_redaction_filters(aligned.get("video_filters", []))
        aligned["special"] = [
            step for step in aligned.get("special", [])
            if step.get("type") != "face_privacy_blur"
        ]
        ensure_special(aligned, "face_privacy_blur", face_privacy_params(command_text))

    if wants_privacy_blur and not wants_face_privacy:
        aligned["video_filters"] = [privacy_blur_filter_step()]

    if wants_ocr_redact:
        if not wants_privacy_blur and not wants_face_privacy:
            aligned["video_filters"] = strip_generated_redaction_filters(aligned.get("video_filters", []))
        aligned["special"] = [
            step for step in aligned.get("special", [])
            if step.get("type") != "ocr_redact"
        ]
        ensure_special(aligned, "ocr_redact", {"sample_fps": 1.0, "confidence": 45})

    if wants_black_remove:
        aligned["special"] = [
            step for step in aligned.get("special", [])
            if step.get("type") != "black_remove"
        ]
        ensure_special(aligned, "black_remove", {
            "min_black_duration": 0.5,
            "pixel_threshold": 0.1,
            "picture_threshold": 0.98,
        })

    if wants_freeze_remove:
        aligned["special"] = [
            step for step in aligned.get("special", [])
            if step.get("type") != "freeze_remove"
        ]
        ensure_special(aligned, "freeze_remove", {"noise_db": -60, "min_duration": 0.5})

    if wants_dedupe_frames:
        aligned["special"] = [
            step for step in aligned.get("special", [])
            if step.get("type") != "dedupe_frames"
        ]
        ensure_special(aligned, "dedupe_frames", {"hi": 768, "lo": 320, "frac": 0.33, "max": 12})

    if wants_beat_cut:
        ensure_analysis(aligned, "beat_track", "beat_times")
        aligned["special"] = [
            step for step in aligned.get("special", [])
            if step.get("type") != "beat_cut"
        ]
        ensure_special(aligned, "beat_cut", {"context": "beat_times", "slice_duration": 0.35, "max_cuts": 24})

    if wants_scene_montage:
        aligned["special"] = [
            step for step in aligned.get("special", [])
            if step.get("type") != "scene_montage"
        ]
        ensure_special(aligned, "scene_montage", {"threshold": 0.28, "slice_duration": 1.2, "max_segments": 12})

    if wants_energy_montage:
        ensure_analysis(aligned, "rms_energy", "energy_curve")
        aligned["special"] = [
            step for step in aligned.get("special", [])
            if step.get("type") != "energy_montage"
        ]
        ensure_special(aligned, "energy_montage", {"context": "energy_curve_times", "slice_duration": 1.0, "max_segments": 12})

    if wants_energy_effects:
        ensure_analysis(aligned, "rms_energy", "energy_curve")
        video_filters = list(aligned.get("video_filters", []))
        for required in energy_reactive_video_filter_steps(command_text):
            if not has_equivalent_energy_filter(video_filters, required):
                video_filters.append(required)
        aligned["video_filters"] = video_filters

    if wants_crop_borders:
        aligned["special"] = [
            step for step in aligned.get("special", [])
            if step.get("type") != "crop_borders"
        ]
        ensure_special(aligned, "crop_borders", {"limit": 24, "round": 2, "max_frames": 120})

    if trim_special:
        aligned["video_filters"] = strip_time_selection_filters(aligned.get("video_filters", []))
        aligned["audio_filters"] = strip_time_selection_filters(aligned.get("audio_filters", []))
        aligned["special"] = [
            step for step in aligned.get("special", [])
            if step.get("type") not in {"trim", "remove_segment"}
        ]
        aligned["special"].append(trim_special)

    if speed_special:
        aligned["video_filters"] = strip_speed_filters(aligned.get("video_filters", []))
        aligned["audio_filters"] = strip_speed_filters(aligned.get("audio_filters", []))
        aligned["special"] = [
            step for step in aligned.get("special", [])
            if step.get("type") != "speed_ramp"
        ]
        aligned["special"].append(speed_special)

    if pitch_special:
        aligned["audio_filters"] = strip_pitch_filters(aligned.get("audio_filters", []))
        aligned["special"] = [
            step for step in aligned.get("special", [])
            if step.get("type") != "pitch_shift"
        ]
        aligned["special"].append(pitch_special)

    if picture_in_picture_requested(command_text):
        aligned["video_filters"] = strip_layout_filters(aligned.get("video_filters", []))
        aligned["special"] = [
            step for step in aligned.get("special", [])
            if step.get("type") != "picture_in_picture"
        ]
        ensure_special(aligned, "picture_in_picture", {"position": "top_right", "scale": 0.32})

    if split_screen_mirror_requested(command_text):
        aligned["video_filters"] = strip_layout_filters(aligned.get("video_filters", []), remove_divider=True)
        aligned["special"] = [
            step for step in aligned.get("special", [])
            if step.get("type") != "split_screen_mirror"
        ]
        ensure_special(aligned, "split_screen_mirror", {"divider_color": "white"})

    if no_audio_sync_requested(command_text):
        enforce_no_audio_sync_constraints(aligned)

    apply_requested_output_dimensions(aligned, command_text)

    return normalize_plan(aligned)


def call_nim(prompt):
    data = nim_provider.chat_json(prompt)
    if isinstance(data, str):
        return normalize_plan(extract_json(data))
    return normalize_plan(data)


def default_final_encode():
    return architecture_default_final_encode()


def extracted_text_prompt(prompt):
    match = re.search(
        r"(?:caption\s+that\s+says|text\s+that\s+says|says?|caption(?:\s+saying)?|text(?:\s+saying)?)\s+(.+)",
        prompt,
        re.IGNORECASE,
    )
    if not match:
        return None
    value = match.group(1)
    value = re.split(
        r"\s+(?:and|then|with|for|at|after|before|while)\s+",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return value.strip(" .,:;\"'")[:120] or None


def heuristic_plan(prompt):
    lowered = prompt.lower()
    wants_removed_audio = remove_audio_requested(prompt)
    wants_mix_uploaded_audio = mix_uploaded_audio_requested(prompt)
    wants_replace_uploaded_audio = replace_uploaded_audio_requested(prompt)
    wants_boomerang = boomerang_requested(prompt)
    wants_blurred_background = blurred_background_requested(prompt)
    wants_auto_captions = auto_captions_requested(prompt)
    wants_audio_cleanup = audio_cleanup_requested(prompt)
    wants_chroma_key = chroma_key_requested(prompt)
    wants_film_damage = film_damage_requested(prompt)
    wants_comic_halftone = comic_halftone_requested(prompt)
    wants_underwater = underwater_requested(prompt)
    wants_energy_effects = energy_reactive_effects_requested(prompt)
    wants_security_camera = security_camera_requested(prompt)
    speed_special = speed_special_from_command(prompt)
    pitch_special = pitch_special_from_command(prompt)
    plan = {
        "intent": f"Fallback executable edit plan for: {prompt[:140]}",
        "video_filters": [],
        "audio_filters": [],
        "special": [],
        "final_encode": default_final_encode(),
    }

    if any(term in lowered for term in ["beat", "rhythm", "music sync", "bass hit"]):
        plan["analysis"] = [{"tool": "librosa", "function": "beat_track", "store_as": "beat_times"}]
        if beat_cut_requested(prompt):
            plan["special"].append({"type": "beat_cut", "params": {"context": "beat_times", "slice_duration": 0.35, "max_cuts": 24}})
        if "shake" in lowered:
            plan["video_filters"].append({
                "description": "Beat-synced frame shake",
                "filter": "crop=iw-60:ih-60:30+30*sin(24*t):30+30*cos(18*t),scale=iw+60:ih+60",
                "requires_context": "beat_times",
                "timing": "per_beat",
            })
        if any(term in lowered for term in ["flash", "strobe", "pulse", "bright"]):
            plan["video_filters"].append({
                "description": "Beat-synced brightness pulse",
                "filter": "eq=brightness=0.18:contrast=1.05:saturation=1.05",
                "requires_context": "beat_times",
                "timing": "per_beat",
            })
        if "zoom" in lowered:
            plan["video_filters"].append({
                "description": "Beat-synced subtle zoom pulse",
                "filter": "zoompan=z='min(zoom+0.001,1.08)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1920x1080",
                "requires_context": "beat_times",
                "timing": "per_beat",
            })

    if impact_timing_requested(prompt):
        plan["analysis"] = [{"tool": "librosa", "function": "onset_detect", "store_as": "onset_times", "sensitivity": 0.5}]
        plan["video_filters"].append({
            "description": "Impact-triggered chromatic glitch",
            "filter": "rgbashift=rh=12:bh=-12:rv=5:bv=-5,drawbox=x=0:y=0:w=iw:h=ih:color=white@0.45:t=fill",
            "requires_context": "onset_times",
            "timing": "per_onset",
        })

    if face_privacy_requested(prompt):
        plan["special"].append({"type": "face_privacy_blur", "params": face_privacy_params(prompt)})
    elif privacy_blur_requested(prompt):
        plan["video_filters"].append(privacy_blur_filter_step())
    if ocr_redact_requested(prompt):
        plan["special"].append({"type": "ocr_redact", "params": {"sample_fps": 1.0, "confidence": 45}})

    color_hold = requested_color_hold_filter(prompt)
    if color_hold:
        plan["video_filters"].append({"description": "Color isolation", "filter": color_hold})

    if wants_underwater:
        plan["video_filters"].extend(underwater_video_filter_steps())
        plan["audio_filters"].extend(underwater_audio_filter_steps())

    if "vhs" in lowered:
        plan["video_filters"].extend([
            {"description": "VHS color bleed", "filter": "rgbashift=rh=6:bh=-6:rv=2:bv=-2"},
            {"description": "VHS scanlines", "filter": "drawgrid=width=iw:height=4:thickness=1:color=black@0.35"},
            {"description": "VHS noise", "filter": "noise=alls=24:allf=t+u"},
        ])

    if "cinematic" in lowered or "teal" in lowered:
        plan["video_filters"].extend([
            {"description": "Teal-orange cinematic grade", "filter": "colorbalance=rs=-0.18:rm=-0.05:rh=0.18:bs=0.18:bm=0.05:bh=-0.12"},
            {"description": "Cinematic contrast", "filter": "eq=contrast=1.12:saturation=1.08"},
        ])

    if any(term in lowered for term in ["dream", "memory", "lofi"]):
        plan["video_filters"].append({
            "description": "Soft warm dream grade with blur and vignette",
            "filter": "colorbalance=rs=0.12:rm=0.18:rh=0.22:bs=-0.12:bm=-0.06:bh=-0.18,gblur=sigma=0.9,vignette=angle=PI/3.5",
        })
        plan["audio_filters"].append({"description": "Light reverb", "filter": "aecho=0.8:0.4:350:0.35"})

    if wants_comic_halftone:
        plan["video_filters"].append(comic_halftone_filter_step(prompt))

    if wants_security_camera:
        plan["video_filters"].extend(security_camera_filter_steps())

    if not wants_film_damage and visual_age_texture_requested(prompt):
        plan["video_filters"].append({"description": "Film grain", "filter": "noise=alls=18:allf=t+u"})

    if any(term in lowered for term in ["neon", "cyberpunk"]):
        plan["video_filters"].append({"description": "Neon cyberpunk color", "filter": "eq=contrast=1.25:saturation=1.55,hue=h=0.08:s=1.25"})

    text_value = extracted_text_prompt(prompt)
    if text_value and not wants_security_camera:
        y_expr = "h-(2*text_h)" if any(term in lowered for term in ["lower third", "bottom"]) else "(h-text_h)/2"
        plan["video_filters"].append({
            "description": "Text overlay",
            "filter": f"drawtext=text={quote_drawtext_filter_text(text_value)}:fontcolor=white:fontsize=48:x=(w-text_w)/2:y={y_expr}:enable='between(t,0,4)'",
        })
    if wants_auto_captions:
        plan["special"].append({"type": "auto_captions", "params": auto_caption_params(prompt)})

    if "silence" in lowered and not wants_removed_audio:
        plan["special"].append({"type": "silence_remove", "params": {"threshold_db": -35, "min_silence_duration": 0.5}})
    if wants_removed_audio:
        plan["audio_filters"] = []
        plan["special"].append({"type": "remove_audio", "params": {}})
    elif wants_mix_uploaded_audio:
        plan["special"].append({"type": "mix_uploaded_audio", "params": uploaded_audio_mix_params(prompt)})
    elif wants_replace_uploaded_audio:
        plan["special"].append({"type": "replace_audio", "params": {}})
    if black_remove_requested(prompt):
        plan["special"].append({
            "type": "black_remove",
            "params": {
                "min_black_duration": 0.5,
                "pixel_threshold": 0.1,
                "picture_threshold": 0.98,
            },
        })
    if freeze_remove_requested(prompt):
        plan["special"].append({"type": "freeze_remove", "params": {"noise_db": -60, "min_duration": 0.5}})
    if dedupe_frames_requested(prompt):
        plan["special"].append({"type": "dedupe_frames", "params": {"hi": 768, "lo": 320, "frac": 0.33, "max": 12}})
    if scene_montage_requested(prompt):
        plan["special"].append({"type": "scene_montage", "params": {"threshold": 0.28, "slice_duration": 1.2, "max_segments": 12}})
    if energy_montage_requested(prompt):
        ensure_analysis(plan, "rms_energy", "energy_curve")
        plan["special"].append({"type": "energy_montage", "params": {"context": "energy_curve_times", "slice_duration": 1.0, "max_segments": 12}})
    if wants_energy_effects:
        ensure_analysis(plan, "rms_energy", "energy_curve")
        plan["video_filters"].extend(energy_reactive_video_filter_steps(prompt))
    if crop_borders_requested(prompt):
        plan["special"].append({"type": "crop_borders", "params": {"limit": 24, "round": 2, "max_frames": 120}})
    trim_special = trim_special_from_command(prompt)
    if trim_special:
        plan["special"].append(trim_special)
    if "stabilize" in lowered or "stabilise" in lowered:
        plan["special"].append({"type": "stabilize", "params": {"smoothing": 10, "crop_black": True}})
    if wants_boomerang:
        plan["special"].append({"type": "boomerang", "params": {"loops": 1, "mute_reversed_audio": True}})
    elif "reverse" in lowered:
        plan["special"].append({"type": "reverse", "params": {}})
    if speed_special:
        plan["special"].append(speed_special)
    if pitch_special:
        plan["special"].append(pitch_special)
    if picture_in_picture_requested(prompt):
        plan["special"].append({"type": "picture_in_picture", "params": {"position": "top_right", "scale": 0.32}})
    if split_screen_mirror_requested(prompt):
        plan["special"].append({"type": "split_screen_mirror", "params": {"divider_color": "white"}})
    if wants_blurred_background:
        plan["special"].append({"type": "blur_background", "params": blurred_background_params(prompt)})
    if wants_chroma_key:
        plan["special"].append({"type": "chroma_key", "params": chroma_key_params(prompt)})
    if wants_film_damage:
        plan["special"].append({"type": "film_damage", "params": film_damage_params(prompt)})

    if "bass" in lowered:
        plan["audio_filters"].append({"description": "Bass boost", "filter": "equalizer=f=60:width_type=o:width=2:g=6"})
    if "normalize" in lowered or "normalise" in lowered:
        plan["audio_filters"].append({"description": "Loudness normalization", "filter": "loudnorm=I=-14:TP=-1.5:LRA=11"})
    if "telephone" in lowered:
        plan["audio_filters"].extend([
            {"description": "Telephone high pass", "filter": "highpass=f=300"},
            {"description": "Telephone low pass", "filter": "lowpass=f=3400"},
        ])
    if "echo" in lowered or "cave" in lowered:
        plan["audio_filters"].append({"description": "Echo", "filter": "aecho=0.8:0.45:500:0.45"})
    if wants_audio_cleanup and not audio_cleanup_satisfied(prompt, plan["audio_filters"]):
        plan["audio_filters"].append(audio_cleanup_filter_step(prompt))

    if not plan["video_filters"] and not plan["audio_filters"] and not plan["special"]:
        plan["video_filters"].append({"description": "Subtle clarity edit", "filter": "eq=contrast=1.08:saturation=1.05"})

    apply_requested_output_dimensions(plan, prompt)

    return normalize_plan(plan)


def sha256_text(value):
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def normalized_plan_cache_prompt(prompt):
    return re.sub(r"\s+", " ", str(prompt or "").strip().lower())


def planner_cache_key(prompt, runtime_note, architecture_hash=None):
    prompt_version = hashlib.sha256(NIM_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    payload = "\n".join([
        NIM_MODEL,
        prompt_version,
        architecture_hash or architecture_fingerprint(),
        public_plan_contract_fingerprint(),
        special_param_contract_fingerprint(),
        normalized_plan_cache_prompt(prompt),
        str(runtime_note or ""),
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_cached_plan(cache_key):
    return planner_cache.get(cache_key)


def store_cached_plan(cache_key, plan, source, architecture_hash=None):
    planner_cache.store(
        cache_key,
        plan,
        source,
        architecture_hash or architecture_fingerprint(),
        public_plan_contract_fingerprint(),
        special_param_contract_fingerprint(),
    )


def clear_plan_cache():
    planner_cache.clear()


def plan_cache_stats():
    return planner_cache.stats(
        architecture_fingerprint(),
        public_plan_contract_fingerprint(),
        special_param_contract_fingerprint(),
    )


def planner_context_metadata(prompt, runtime_note, architecture_hash, cache_key):
    normalized_prompt = normalized_plan_cache_prompt(prompt)
    return {
        "architecture_fingerprint": architecture_hash,
        "public_plan_contract_fingerprint": public_plan_contract_fingerprint(),
        "special_param_contract_fingerprint": special_param_contract_fingerprint(),
        "system_prompt_hash": sha256_text(NIM_SYSTEM_PROMPT),
        "model": NIM_MODEL,
        "runtime_note_hash": sha256_text(runtime_note),
        "normalized_command_hash": sha256_text(normalized_prompt),
        "cache_key": cache_key,
    }


def build_plan(prompt, job=None):
    capabilities = runtime_capabilities()
    runtime_note = runtime_planning_prompt(capabilities)
    architecture_hash = architecture_fingerprint()
    cache_key = planner_cache_key(prompt, runtime_note, architecture_hash)
    planner_context = planner_context_metadata(prompt, runtime_note, architecture_hash, cache_key)
    use_cache = job is not None
    cached = get_cached_plan(cache_key) if use_cache else None
    if cached:
        if job is not None:
            job["planner"] = cached["source"]
            job["planner_context"] = dict(planner_context)
            job["planner_cache"] = {
                "status": "hit",
                "hits": cached["hits"],
                "architecture_fingerprint": cached.get("architecture_fingerprint"),
                "public_plan_contract_fingerprint": cached.get("public_plan_contract_fingerprint"),
                "special_param_contract_fingerprint": cached.get("special_param_contract_fingerprint"),
            }
            if cached["source"] == "heuristic":
                record_planner_fallback(job, "model_unavailable", "cached heuristic plan")
            else:
                job.pop("planner_fallback", None)
        return cached["plan"]

    try:
        planner_prompt = augment_prompt_for_capabilities(
            prompt,
            runtime_note,
            architecture_prompt_contract(),
        )
        plan = call_nim(planner_prompt)
        source = "nim"
    except Exception as exc:
        record_planner_fallback(job, "model_unavailable", concise_error(exc))
        append_job_warning(
            job,
            f"NIM planning failed; used heuristic fallback: {concise_error(exc)}",
        )
        plan = heuristic_plan(prompt)
        source = "heuristic"

    if use_cache:
        store_cached_plan(cache_key, plan, source, architecture_hash)
    if job is not None:
        job["planner"] = source
        job["planner_context"] = planner_context
        if use_cache:
            job["planner_cache"] = {
                "status": "miss",
                "architecture_fingerprint": architecture_hash,
                "public_plan_contract_fingerprint": public_plan_contract_fingerprint(),
                "special_param_contract_fingerprint": special_param_contract_fingerprint(),
            }
        if source != "heuristic":
            job.pop("planner_fallback", None)
    return clone_plan(plan)



def source_audio_for_analysis(job, job_dir, current_input):
    audio_path = job.get("audio_path")
    if audio_path and Path(audio_path).exists():
        return Path(audio_path)

    analysis_audio = job_dir / "analysis_audio.wav"
    run_command([
        "ffmpeg", "-y", "-i", str(current_input),
        "-vn", "-ac", "1", "-ar", "22050",
        str(analysis_audio),
    ])
    return analysis_audio


def synthetic_analysis_times(duration):
    duration = max(0.5, float(duration or 6.0))
    count = max(1, int(duration / 0.5))
    return [round(min(duration, 0.5 + index * 0.5), 3) for index in range(count)]


def synthetic_analysis_context(plan, current_input, job, reason):
    append_job_warning(
        job,
        f"audio analysis fallback used synthetic timing because no readable audio was available: {concise_error(reason)}",
    )
    try:
        duration = ffprobe_duration(current_input)
    except Exception:
        duration = 6.0

    times = synthetic_analysis_times(duration)
    context = {}
    for step in plan.get("analysis", []):
        store_as = step.get("store_as")
        function = step.get("function")
        if not store_as:
            continue
        if function in {"beat_track", "onset_detect"}:
            context[store_as] = times
        elif function == "rms_energy":
            context[store_as] = [1.0 for _ in times]
            context[f"{store_as}_times"] = times
    return context


def load_audio_for_fallback(audio_source, job_dir):
    import numpy as np

    wav_path = job_dir / f"analysis_fallback_{uuid4().hex[:8]}.wav"
    run_command([
        "ffmpeg", "-y", "-i", str(audio_source),
        "-vn", "-ac", "1", "-ar", "22050",
        "-sample_fmt", "s16",
        str(wav_path),
    ])

    with wave.open(str(wav_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        raw = wav_file.readframes(wav_file.getnframes())

    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sample_rate


def fallback_rms(y, frame_length=2048, hop_length=512):
    import numpy as np

    if len(y) == 0:
        return np.array([], dtype=np.float32)
    if len(y) < frame_length:
        y = np.pad(y, (0, frame_length - len(y)))

    values = []
    for start in range(0, max(1, len(y) - frame_length + 1), hop_length):
        frame = y[start:start + frame_length]
        values.append(float(np.sqrt(np.mean(frame * frame))))
    return np.array(values, dtype=np.float32)


def energy_peak_times(rms_values, sr, hop_length=512):
    import numpy as np

    values = np.asarray(rms_values, dtype=np.float32)
    if values.size == 0:
        return []
    if values.max() > 0:
        values = values / values.max()

    threshold = max(0.35, float(np.percentile(values, 85)))
    min_distance = max(1, int(0.25 * sr / hop_length))
    peaks = []
    last_peak = -min_distance
    for index in range(1, len(values) - 1):
        if index - last_peak < min_distance:
            continue
        if values[index] >= threshold and values[index] >= values[index - 1] and values[index] >= values[index + 1]:
            peaks.append(index)
            last_peak = index

    return [round(index * hop_length / sr, 3) for index in peaks]


def fallback_peak_times(y, sr, sensitivity=0.5):
    import numpy as np

    hop_length = 512
    envelope = fallback_rms(y, hop_length=hop_length)
    if envelope.size == 0:
        return []

    if envelope.max() > 0:
        envelope = envelope / envelope.max()

    threshold = min(0.95, max(0.18, float(envelope.mean() + envelope.std() * (0.8 + sensitivity))))
    min_distance = max(1, int(0.25 * sr / hop_length))
    peaks = []
    last_peak = -min_distance
    for index in range(1, len(envelope) - 1):
        if index - last_peak < min_distance:
            continue
        if envelope[index] >= threshold and envelope[index] >= envelope[index - 1] and envelope[index] >= envelope[index + 1]:
            peaks.append(index)
            last_peak = index

    return [round(index * hop_length / sr, 3) for index in peaks]


def fallback_analysis(function, store_as, step, y, sr):
    if function == "beat_track":
        return fallback_peak_times(y, sr, sensitivity=0.25)
    if function == "onset_detect":
        return fallback_peak_times(y, sr, sensitivity=float(step.get("sensitivity", 0.5)))
    if function == "rms_energy":
        return [float(value) for value in fallback_rms(y).tolist()]
    return None


def run_analysis(plan, job, job_dir, current_input):
    context = {}
    analysis_steps = plan.get("analysis", [])
    if not analysis_steps:
        return context

    try:
        audio_source = source_audio_for_analysis(job, job_dir, current_input)
    except Exception as exc:
        return synthetic_analysis_context(plan, current_input, job, exc)

    librosa = None
    y = None
    sr = None
    try:
        import librosa as librosa_module

        librosa = librosa_module
        y, sr = librosa.load(str(audio_source), sr=None, mono=True)
    except Exception:
        try:
            y, sr = load_audio_for_fallback(audio_source, job_dir)
        except Exception as exc:
            return synthetic_analysis_context(plan, current_input, job, exc)

    for step in analysis_steps:
        if step.get("tool") != "librosa":
            continue

        function = step.get("function")
        store_as = step.get("store_as")
        if not store_as:
            continue

        try:
            if librosa and function == "beat_track":
                _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
                context[store_as] = librosa.frames_to_time(beat_frames, sr=sr).tolist()
                continue
            if librosa and function == "onset_detect":
                sensitivity = float(step.get("sensitivity", 0.5))
                onset_frames = librosa.onset.onset_detect(y=y, sr=sr, delta=sensitivity)
                context[store_as] = librosa.frames_to_time(onset_frames, sr=sr).tolist()
                continue
            if librosa and function == "rms_energy":
                rms_values = librosa.feature.rms(y=y).flatten()
                context[store_as] = rms_values.tolist()
                context[f"{store_as}_times"] = energy_peak_times(rms_values, sr)
                continue
        except Exception:
            pass

        fallback_value = fallback_analysis(function, store_as, step, y, sr)
        if fallback_value is not None:
            context[store_as] = fallback_value
            if function == "rms_energy":
                context[f"{store_as}_times"] = energy_peak_times(fallback_value, sr)

    return context


def parse_ffprobe_duration_value(value):
    try:
        duration = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if duration <= 0:
        return None
    return duration


def ffprobe_duration(input_path):
    result = run_command([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(input_path),
    ])
    duration = parse_ffprobe_duration_value(result.stdout)
    if duration is not None:
        return duration

    result = run_command([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration:stream=duration:stream_tags=DURATION",
        "-of", "json",
        str(input_path),
    ])
    probe = json.loads(result.stdout or "{}")
    candidates = [probe.get("format", {}).get("duration")]
    for stream in probe.get("streams", []):
        candidates.append(stream.get("duration"))
        candidates.append(stream.get("tags", {}).get("DURATION"))

    parsed = [value for value in (parse_ffprobe_duration_value(candidate) for candidate in candidates) if value is not None]
    if parsed:
        return max(parsed)
    raise ValueError(f"ffprobe could not determine a positive duration for {input_path}")


def ffprobe_video_dimensions(input_path):
    result = run_command([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "json",
        str(input_path),
    ])
    data = json.loads(result.stdout)
    stream = (data.get("streams") or [{}])[0]
    return int(stream.get("width") or 1280), int(stream.get("height") or 720)


def ffprobe_streams(input_path):
    result = run_command([
        "ffprobe", "-v", "error",
        "-show_entries", "stream=index,codec_type,codec_name",
        "-of", "json",
        str(input_path),
    ])
    try:
        return json.loads(result.stdout).get("streams", [])
    except json.JSONDecodeError:
        return []


def ffprobe_has_audio(input_path):
    result = run_command([
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=index",
        "-of", "csv=p=0",
        str(input_path),
    ])
    return bool(result.stdout.strip())


def ffprobe_video_frame_count(input_path):
    result = run_command([
        "ffprobe", "-v", "error",
        "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames,nb_frames",
        "-of", "json",
        str(input_path),
    ])
    data = json.loads(result.stdout)
    stream = (data.get("streams") or [{}])[0]
    value = stream.get("nb_read_frames") or stream.get("nb_frames") or 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def apply_silence_remove(input_path, job_dir, params):
    threshold_db = params.get("threshold_db", -35)
    min_silence_duration = params.get("min_silence_duration", 0.5)
    result = run_command_result([
        "ffmpeg", "-i", str(input_path),
        "-af", f"silencedetect=noise={threshold_db}dB:d={min_silence_duration}",
        "-f", "null", "-",
    ])

    duration = ffprobe_duration(input_path)
    starts = [float(value) for value in re.findall(r"silence_start: ([0-9.]+)", result.stderr)]
    ends = [float(value) for value in re.findall(r"silence_end: ([0-9.]+)", result.stderr)]
    if len(ends) < len(starts):
        ends.append(duration)

    keep_segments = []
    cursor = 0.0
    for start, end in zip(starts, ends):
        if start > cursor:
            keep_segments.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        keep_segments.append((cursor, duration))

    if not keep_segments:
        return input_path

    segment_paths = []
    for index, (start, end) in enumerate(keep_segments):
        segment_path = job_dir / f"nonsilent_{index:03d}.mp4"
        run_command([
            "ffmpeg", "-y",
            "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
            "-i", str(input_path),
            "-c", "copy",
            str(segment_path),
        ])
        segment_paths.append(segment_path)

    concat_file = job_dir / "concat_segments.txt"
    concat_file.write_text(
        "\n".join(f"file '{path.as_posix()}'" for path in segment_paths),
        encoding="utf-8",
    )
    output_path = next_media_path(job_dir, "silence_removed")
    run_command([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(output_path),
    ])
    return output_path


def parse_blackdetect_segments(stderr_text, duration):
    starts = [float(value) for value in re.findall(r"black_start:\s*([0-9.]+)", stderr_text or "")]
    ends = [float(value) for value in re.findall(r"black_end:\s*([0-9.]+)", stderr_text or "")]
    if len(ends) < len(starts):
        ends.append(duration)
    segments = []
    for start, end in zip(starts, ends):
        start = min(max(0.0, start), duration)
        end = min(max(0.0, end), duration)
        if end > start + 0.05:
            segments.append((start, end))
    return segments


def keep_segments_excluding(duration, removed_segments):
    keep_segments = []
    cursor = 0.0
    for start, end in sorted(removed_segments):
        if start > cursor + 0.05:
            keep_segments.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration - 0.05:
        keep_segments.append((cursor, duration))
    return keep_segments


def concat_encoded_segments(input_path, job_dir, keep_segments, label):
    segment_paths = []
    for index, (segment_start, segment_end) in enumerate(keep_segments):
        segment_path = job_dir / f"{label}_keep_{index:03d}.mp4"
        encode_segment(input_path, segment_path, segment_start, segment_end - segment_start)
        segment_paths.append(segment_path)

    if not segment_paths:
        return input_path
    if len(segment_paths) == 1:
        return segment_paths[0]

    concat_file = job_dir / f"{label}_concat_{uuid4().hex[:8]}.txt"
    concat_file.write_text(
        "\n".join(f"file '{path.as_posix()}'" for path in segment_paths),
        encoding="utf-8",
    )
    output_path = next_media_path(job_dir, label)
    run_command([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(output_path),
    ])
    return output_path


def beat_cut_segments(beat_times, duration, slice_duration, max_cuts):
    segments = []
    last_start = -999.0
    for raw_time in beat_times:
        try:
            beat_time = float(raw_time)
        except (TypeError, ValueError):
            continue
        start = min(max(0.0, beat_time), max(0.0, duration - 0.05))
        if start < last_start + 0.08:
            continue
        end = min(duration, start + slice_duration)
        if end > start + 0.05:
            segments.append((start, end))
            last_start = start
        if len(segments) >= max_cuts:
            break
    return segments


def apply_beat_cut(input_path, job_dir, params, context, job=None):
    context_key = str(params.get("context", "beat_times") or "beat_times")
    beat_times = context.get(context_key, []) if isinstance(context, dict) else []
    if not beat_times:
        append_job_warning(job, f"beat_cut skipped because analysis context {context_key} is unavailable")
        return input_path

    duration = ffprobe_duration(input_path)
    slice_duration = max(0.12, min(1.5, float(params.get("slice_duration", 0.35) or 0.35)))
    max_cuts = max(1, min(96, int(params.get("max_cuts", 24) or 24)))
    segments = beat_cut_segments(beat_times, duration, slice_duration, max_cuts)
    if len(segments) < 2:
        append_job_warning(job, "beat_cut found fewer than two usable beat segments; skipped structural cut")
        return input_path

    if job is not None:
        job["beat_cuts"] = [
            {"start": round(start, 3), "end": round(end, 3)}
            for start, end in segments
        ]
    return concat_encoded_segments(input_path, job_dir, segments, "beat_cut")


def detect_scene_change_times(input_path, threshold):
    result = run_command_result([
        "ffmpeg", "-hide_banner", "-i", str(input_path),
        "-vf", f"select='gt(scene,{threshold:.3f})',showinfo",
        "-an", "-f", "null", "-",
    ])
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "scene detection failed")

    times = []
    for value in re.findall(r"pts_time:([0-9.]+)", result.stderr or ""):
        try:
            times.append(float(value))
        except ValueError:
            continue
    return sorted(set(round(time_value, 3) for time_value in times))


def evenly_spaced_montage_times(duration, max_segments):
    count = max(1, min(max_segments, int(duration // 1.0) or 1))
    if count == 1:
        return [0.0]
    step = duration / count
    return [round(index * step, 3) for index in range(count)]


def scene_montage_segments(scene_times, duration, slice_duration, max_segments):
    anchors = [0.0]
    anchors.extend(time_value for time_value in scene_times if 0.05 <= time_value < duration - 0.05)
    deduped = []
    for anchor in sorted(anchors):
        if deduped and anchor < deduped[-1] + 0.2:
            continue
        deduped.append(anchor)
        if len(deduped) >= max_segments:
            break

    segments = []
    for anchor in deduped:
        start = min(max(0.0, anchor), max(0.0, duration - 0.05))
        end = min(duration, start + slice_duration)
        if end > start + 0.05:
            segments.append((start, end))
    return segments


def apply_scene_montage(input_path, job_dir, params, job=None):
    duration = ffprobe_duration(input_path)
    threshold = max(0.05, min(0.95, float(params.get("threshold", 0.28) or 0.28)))
    slice_duration = max(0.25, min(4.0, float(params.get("slice_duration", 1.2) or 1.2)))
    max_segments = max(1, min(48, int(params.get("max_segments", 12) or 12)))

    try:
        scene_times = detect_scene_change_times(input_path, threshold)
    except Exception as exc:
        append_job_warning(job, f"scene_montage scene detection failed; using timed fallback: {concise_error(exc)}")
        scene_times = []

    if not scene_times:
        append_job_warning(job, "scene_montage found no strong scene changes; using evenly spaced slices")
        scene_times = evenly_spaced_montage_times(duration, max_segments)

    segments = scene_montage_segments(scene_times, duration, slice_duration, max_segments)
    if not segments:
        append_job_warning(job, "scene_montage found no usable segments; skipped montage")
        return input_path

    if job is not None:
        job["scene_montage"] = [
            {"start": round(start, 3), "end": round(end, 3)}
            for start, end in segments
        ]
    return concat_encoded_segments(input_path, job_dir, segments, "scene_montage")


def energy_montage_segments(peak_times, duration, slice_duration, max_segments, pre_roll=0.15):
    segments = []
    last_end = -999.0
    for raw_time in peak_times:
        try:
            peak_time = float(raw_time)
        except (TypeError, ValueError):
            continue
        start = min(max(0.0, peak_time - pre_roll), max(0.0, duration - 0.05))
        if start < last_end - 0.05:
            continue
        end = min(duration, start + slice_duration)
        if end > start + 0.05:
            segments.append((start, end))
            last_end = end
        if len(segments) >= max_segments:
            break
    return segments


def apply_energy_montage(input_path, job_dir, params, context, job=None):
    context_key = str(params.get("context", "energy_curve_times") or "energy_curve_times")
    peak_times = context.get(context_key, []) if isinstance(context, dict) else []
    if not peak_times and context_key.endswith("_times"):
        peak_times = context.get(context_key.removesuffix("_times"), [])
    if not peak_times:
        append_job_warning(job, f"energy_montage skipped because analysis context {context_key} is unavailable")
        return input_path

    duration = ffprobe_duration(input_path)
    slice_duration = max(0.25, min(4.0, float(params.get("slice_duration", 1.0) or 1.0)))
    max_segments = max(1, min(48, int(params.get("max_segments", 12) or 12)))
    pre_roll = max(0.0, min(1.5, float(params.get("pre_roll", 0.15) or 0.15)))
    segments = energy_montage_segments(peak_times, duration, slice_duration, max_segments, pre_roll)
    if not segments:
        append_job_warning(job, "energy_montage found no usable high-energy segments; skipped montage")
        return input_path

    if job is not None:
        job["energy_montage"] = [
            {"start": round(start, 3), "end": round(end, 3)}
            for start, end in segments
        ]
    return concat_encoded_segments(input_path, job_dir, segments, "energy_montage")


def apply_black_remove(input_path, job_dir, params, job=None):
    min_black_duration = max(0.1, min(10.0, float(params.get("min_black_duration", 0.5) or 0.5)))
    pixel_threshold = max(0.0, min(1.0, float(params.get("pixel_threshold", params.get("pix_th", 0.1)) or 0.1)))
    picture_threshold = max(0.0, min(1.0, float(params.get("picture_threshold", params.get("pic_th", 0.98)) or 0.98)))
    result = run_command_result([
        "ffmpeg", "-i", str(input_path),
        "-vf", (
            f"blackdetect=d={min_black_duration:.3f}:"
            f"pix_th={pixel_threshold:.3f}:pic_th={picture_threshold:.3f}"
        ),
        "-an", "-f", "null", "-",
    ])

    duration = ffprobe_duration(input_path)
    black_segments = parse_blackdetect_segments(result.stderr, duration)
    if job is not None:
        job["black_segments_removed"] = [
            {"start": round(start, 3), "end": round(end, 3)}
            for start, end in black_segments
        ]
    if not black_segments:
        append_job_warning(job, "black_remove found no black or blank sections; skipped visual cut")
        return input_path

    keep_segments = keep_segments_excluding(duration, black_segments)
    if not keep_segments:
        append_job_warning(job, "black_remove detected the whole clip as black; skipped visual cut")
        return input_path

    return concat_encoded_segments(input_path, job_dir, keep_segments, "black_removed")


def parse_freezedetect_segments(stderr_text, duration):
    starts = [float(value) for value in re.findall(r"freeze_start:\s*([0-9.]+)", stderr_text or "")]
    ends = [float(value) for value in re.findall(r"freeze_end:\s*([0-9.]+)", stderr_text or "")]
    if len(ends) < len(starts):
        ends.append(duration)
    segments = []
    for start, end in zip(starts, ends):
        start = min(max(0.0, start), duration)
        end = min(max(0.0, end), duration)
        if end > start + 0.05:
            segments.append((start, end))
    return segments


def apply_freeze_remove(input_path, job_dir, params, job=None):
    noise_db = max(-100.0, min(0.0, float(params.get("noise_db", params.get("noise", -60)) or -60)))
    min_duration = max(0.1, min(10.0, float(params.get("min_duration", params.get("duration", 0.5)) or 0.5)))
    result = run_command_result([
        "ffmpeg", "-i", str(input_path),
        "-vf", f"freezedetect=n={noise_db:.1f}dB:d={min_duration:.3f}",
        "-an", "-f", "null", "-",
    ])

    duration = ffprobe_duration(input_path)
    frozen_segments = parse_freezedetect_segments(result.stderr, duration)
    if job is not None:
        job["freeze_segments_removed"] = [
            {"start": round(start, 3), "end": round(end, 3)}
            for start, end in frozen_segments
        ]
    if not frozen_segments:
        append_job_warning(job, "freeze_remove found no frozen or stuck sections; skipped visual cut")
        return input_path

    keep_segments = keep_segments_excluding(duration, frozen_segments)
    if not keep_segments:
        append_job_warning(job, "freeze_remove detected the whole clip as frozen; skipped visual cut")
        return input_path

    return concat_encoded_segments(input_path, job_dir, keep_segments, "freeze_removed")


def apply_dedupe_frames(input_path, job_dir, params, job=None):
    hi = max(0, min(65535, int(params.get("hi", 768) or 768)))
    lo = max(0, min(65535, int(params.get("lo", 320) or 320)))
    frac = max(0.0, min(1.0, float(params.get("frac", 0.33) or 0.33)))
    max_drop = max(1, min(1000, int(params.get("max", 12) or 12)))
    before_frames = ffprobe_video_frame_count(input_path)
    has_audio = ffprobe_has_audio(input_path)
    output_path = next_media_path(job_dir, "dedupe_frames")
    command = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", f"mpdecimate=max={max_drop}:hi={hi}:lo={lo}:frac={frac},setpts=N/FRAME_RATE/TB",
        "-fps_mode", "vfr",
        "-map", "0:v:0",
    ]
    if has_audio:
        command.extend(["-map", "0:a:0", "-shortest"])
    else:
        command.append("-an")
    command.extend([
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
    ])
    if has_audio:
        command.extend(["-c:a", "aac", "-b:a", "192k"])
    command.append(str(output_path))
    run_command(command)
    after_frames = ffprobe_video_frame_count(output_path)
    if job is not None:
        job["dedupe_frames"] = {
            "before": before_frames,
            "after": after_frames,
            "removed": max(0, before_frames - after_frames),
        }
        if has_audio:
            append_job_warning(job, "dedupe_frames shortened duplicate-video timing and trimmed audio to the new video duration")
    if before_frames and after_frames >= before_frames:
        append_job_warning(job, "dedupe_frames found no duplicate frames to remove")
    return output_path


def parse_cropdetect_crops(stderr_text):
    crops = []
    for match in re.finditer(r"\bcrop=(\d+):(\d+):(\d+):(\d+)\b", stderr_text or ""):
        crops.append(tuple(int(value) for value in match.groups()))
    return crops


def best_cropdetect_crop(crops, dimensions):
    if not crops:
        return None
    width, height = dimensions
    counts = {}
    for crop in crops:
        crop_width, crop_height, x, y = crop
        if crop_width <= 0 or crop_height <= 0:
            continue
        if crop_width > width or crop_height > height or x < 0 or y < 0:
            continue
        counts[crop] = counts.get(crop, 0) + 1
    if not counts:
        return None
    crop = max(counts, key=lambda item: (counts[item], item[0] * item[1]))
    crop_width, crop_height, _x, _y = crop
    if crop_width >= width - 2 and crop_height >= height - 2:
        return None
    return crop


def apply_crop_borders(input_path, job_dir, params, job=None):
    limit = max(0, min(255, int(params.get("limit", 24) or 24)))
    round_value = max(2, min(64, int(params.get("round", 2) or 2)))
    max_frames = max(10, min(600, int(params.get("max_frames", 120) or 120)))
    result = run_command_result([
        "ffmpeg", "-i", str(input_path),
        "-vf", f"cropdetect=limit={limit}:round={round_value}:reset=0",
        "-frames:v", str(max_frames),
        "-an", "-f", "null", "-",
    ])

    dimensions = ffprobe_video_dimensions(input_path)
    crop = best_cropdetect_crop(parse_cropdetect_crops(result.stderr), dimensions)
    if crop is None:
        append_job_warning(job, "crop_borders found no removable black border; skipped crop")
        return input_path

    crop_width, crop_height, x, y = crop
    if job is not None:
        job["crop_borders"] = {
            "width": crop_width,
            "height": crop_height,
            "x": x,
            "y": y,
        }
    output_path = next_media_path(job_dir, "crop_borders")
    run_video_filter_step(input_path, output_path, f"crop={crop_width}:{crop_height}:{x}:{y}")
    return output_path


def apply_stabilize(input_path, job_dir, params):
    transforms = job_dir / "vidstab.trf"
    smoothing = int(params.get("smoothing", 10))
    crop = "black" if params.get("crop_black", True) else "keep"
    run_command([
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", f"vidstabdetect=shakiness=5:accuracy=15:result={transforms.as_posix()}",
        "-f", "null", "-",
    ])
    output_path = next_media_path(job_dir, "stabilized")
    run_command([
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", f"vidstabtransform=input={transforms.as_posix()}:smoothing={smoothing}:crop={crop}",
        "-c:a", "copy",
        str(output_path),
    ])
    return output_path


def atempo_chain(speed_factor):
    factors = []
    factor = float(speed_factor)
    while factor > 2.0:
        factors.append(2.0)
        factor /= 2.0
    while factor < 0.5:
        factors.append(0.5)
        factor /= 0.5
    factors.append(round(factor, 4))
    return ",".join(f"atempo={value}" for value in factors)


def pitch_ratio_from_semitones(semitones):
    return 2 ** (float(semitones) / 12.0)


def apply_pitch_shift(input_path, job_dir, params, job=None):
    semitones = float(params.get("semitones", 0))
    if not ffprobe_has_audio(input_path):
        append_job_warning(job, "pitch_shift skipped because no audio stream is available")
        return input_path

    output_path = next_media_path(job_dir, "pitch_shift")
    pitch_ratio = max(0.01, min(100.0, pitch_ratio_from_semitones(semitones)))
    try:
        run_command([
            "ffmpeg", "-y", "-i", str(input_path),
            "-map", "0:v:0", "-map", "0:a:0",
            "-c:v", "copy",
            "-af", f"rubberband=pitch={pitch_ratio:.8f}:pitchq=quality",
            "-c:a", "aac", "-b:a", "192k",
            str(output_path),
        ])
        if job is not None:
            job["pitch_shift"] = {
                "mode": "ffmpeg_rubberband_filter",
                "semitones": semitones,
                "pitch_ratio": round(pitch_ratio, 8),
            }
        return output_path
    except Exception as exc:
        append_job_warning(
            job,
            f"pitch_shift FFmpeg rubberband filter failed; retrying rubberband CLI: {concise_error(exc)}",
        )

    extracted_audio = job_dir / f"pitch_source_{uuid4().hex[:8]}.wav"
    shifted_audio = job_dir / f"pitch_shifted_{uuid4().hex[:8]}.wav"
    run_command(["ffmpeg", "-y", "-i", str(input_path), "-vn", str(extracted_audio)])
    run_command(["rubberband", "-p", str(semitones), str(extracted_audio), str(shifted_audio)])
    run_command([
        "ffmpeg", "-y", "-i", str(input_path), "-i", str(shifted_audio),
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy", "-c:a", "aac",
        str(output_path),
    ])
    if job is not None:
        job["pitch_shift"] = {
            "mode": "rubberband_cli_fallback",
            "semitones": semitones,
            "pitch_ratio": round(pitch_ratio, 8),
        }
    return output_path


def apply_boomerang(input_path, job_dir, params, job=None):
    loops = max(1, min(4, int(safe_float(params.get("loops", 1), 1))))
    mute_value = params.get("mute_reversed_audio", True)
    if isinstance(mute_value, bool):
        mute_reversed_audio = mute_value
    elif isinstance(mute_value, str):
        mute_reversed_audio = mute_value.lower() not in {"false", "0", "no"}
    else:
        mute_reversed_audio = bool(mute_value)
    has_audio = ffprobe_has_audio(input_path)
    pair_path = next_media_path(job_dir, "boomerang_pair")

    if has_audio:
        reversed_audio_filter = "volume=0" if mute_reversed_audio else "areverse"
        filter_complex = (
            "[0:v]split=2[v0][v1];"
            "[v0]setpts=PTS-STARTPTS[vf];"
            "[v1]reverse,setpts=PTS-STARTPTS[vr];"
            "[0:a]asplit=2[a0][a1];"
            "[a0]asetpts=PTS-STARTPTS[af];"
            f"[a1]{reversed_audio_filter},asetpts=PTS-STARTPTS[ar];"
            "[vf][af][vr][ar]concat=n=2:v=1:a=1[vc][a];"
            "[vc]format=yuv420p[v]"
        )
        command = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "[a]",
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            "-c:a", "aac", "-b:a", "192k",
            str(pair_path),
        ]
    else:
        filter_complex = (
            "[0:v]split=2[v0][v1];"
            "[v0]setpts=PTS-STARTPTS[vf];"
            "[v1]reverse,setpts=PTS-STARTPTS[vr];"
            "[vf][vr]concat=n=2:v=1:a=0[vc];"
            "[vc]format=yuv420p[v]"
        )
        command = [
            "ffmpeg", "-y", "-i", str(input_path),
            "-filter_complex", filter_complex,
            "-map", "[v]", "-an",
            "-c:v", "libx264", "-preset", "fast", "-crf", "22",
            str(pair_path),
        ]
    run_command(command)

    output_path = pair_path
    if loops > 1:
        concat_file = job_dir / f"boomerang_loop_concat_{uuid4().hex[:8]}.txt"
        concat_file.write_text(
            "\n".join(f"file '{pair_path.as_posix()}'" for _ in range(loops)),
            encoding="utf-8",
        )
        output_path = next_media_path(job_dir, "boomerang")
        run_command([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(output_path),
        ])

    if job is not None:
        job["boomerang"] = {
            "loops": loops,
            "mute_reversed_audio": mute_reversed_audio,
            "has_audio": has_audio,
        }
    return output_path


def apply_end_reverse(input_path, job_dir, params):
    duration = ffprobe_duration(input_path)
    segment_duration = float(params.get("duration", 1.5))
    segment_duration = min(max(0.35, segment_duration), max(0.35, duration))
    start = max(0.0, duration - segment_duration)
    has_audio = ffprobe_has_audio(input_path)

    tail_path = next_media_path(job_dir, "end_reverse_tail")
    tail_command = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", str(input_path),
        "-t", f"{segment_duration:.3f}",
        "-vf", "reverse",
    ]
    if has_audio:
        tail_command.extend(["-af", "areverse"])
    else:
        tail_command.append("-an")
    tail_command.extend([
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac", "-b:a", "192k",
        str(tail_path),
    ])
    run_command(tail_command)

    if start <= 0.1:
        return tail_path

    main_path = next_media_path(job_dir, "end_reverse_main")
    main_command = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-t", f"{start:.3f}",
        "-map", "0:v:0",
    ]
    if has_audio:
        main_command.extend(["-map", "0:a:0"])
    else:
        main_command.append("-an")
    main_command.extend([
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac", "-b:a", "192k",
        str(main_path),
    ])
    run_command(main_command)

    concat_file = job_dir / f"end_reverse_concat_{uuid4().hex[:8]}.txt"
    concat_file.write_text(
        "\n".join([f"file '{main_path.as_posix()}'", f"file '{tail_path.as_posix()}'"]),
        encoding="utf-8",
    )
    output_path = next_media_path(job_dir, "end_reverse")
    run_command([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(output_path),
    ])
    return output_path


def encode_segment(input_path, output_path, start, segment_duration):
    has_audio = ffprobe_has_audio(input_path)
    command = [
        "ffmpeg", "-y",
        "-ss", f"{max(0.0, start):.3f}",
        "-i", str(input_path),
        "-t", f"{max(0.01, segment_duration):.3f}",
        "-map", "0:v:0",
    ]
    if has_audio:
        command.extend(["-map", "0:a:0"])
    else:
        command.append("-an")
    command.extend([
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "aac", "-b:a", "192k",
        str(output_path),
    ])
    run_command(command)


def apply_trim(input_path, job_dir, params):
    duration = ffprobe_duration(input_path)
    start = float(params.get("start", 0) or 0)

    if "from_end" in params:
        segment_duration = max(0.0, float(params.get("from_end") or 0))
        start = max(0.0, duration - segment_duration)
        end = duration
    elif "duration" in params:
        end = start + max(0.0, float(params.get("duration") or 0))
    else:
        end = float(params.get("end", duration) or duration)

    if "remove_end" in params:
        end = duration - max(0.0, float(params.get("remove_end") or 0))

    start = min(max(0.0, start), duration)
    end = min(max(0.0, end), duration)
    if end <= start + 0.05:
        return input_path

    output_path = next_media_path(job_dir, "trim")
    encode_segment(input_path, output_path, start, end - start)
    return output_path


def apply_remove_segment(input_path, job_dir, params):
    duration = ffprobe_duration(input_path)
    start = min(max(0.0, float(params.get("start", 0) or 0)), duration)
    end = min(max(0.0, float(params.get("end", duration) or duration)), duration)
    if end <= start + 0.05:
        return input_path

    keep_segments = []
    if start > 0.05:
        keep_segments.append((0.0, start))
    if end < duration - 0.05:
        keep_segments.append((end, duration))
    if not keep_segments:
        return input_path

    segment_paths = []
    for index, (segment_start, segment_end) in enumerate(keep_segments):
        segment_path = job_dir / f"removed_segment_keep_{index:03d}.mp4"
        encode_segment(input_path, segment_path, segment_start, segment_end - segment_start)
        segment_paths.append(segment_path)

    if len(segment_paths) == 1:
        return segment_paths[0]

    concat_file = job_dir / f"remove_segment_concat_{uuid4().hex[:8]}.txt"
    concat_file.write_text(
        "\n".join(f"file '{path.as_posix()}'" for path in segment_paths),
        encoding="utf-8",
    )
    output_path = next_media_path(job_dir, "remove_segment")
    run_command([
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_file),
        "-c", "copy",
        str(output_path),
    ])
    return output_path


def apply_remove_audio(input_path, job_dir):
    output_path = next_media_path(job_dir, "remove_audio")
    try:
        run_command([
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-map", "0:v:0",
            "-an",
            "-c:v", "copy",
            str(output_path),
        ])
        return output_path
    except Exception:
        fallback_path = next_media_path(job_dir, "remove_audio_encoded")
        run_command([
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-map", "0:v:0",
            "-an",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "22",
            str(fallback_path),
        ])
        return fallback_path


def extract_ocr_frames(input_path, job_dir, sample_fps, max_frames):
    frame_dir = job_dir / f"ocr_frames_{uuid4().hex[:8]}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = frame_dir / "frame_%04d.png"
    run_command([
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-vf", f"fps={sample_fps}",
        "-frames:v", str(max_frames),
        str(output_pattern),
    ])
    return sorted(frame_dir.glob("frame_*.png"))


def ocr_frame_detections(frame_path, frame_time, confidence_threshold, padding):
    from PIL import Image
    import pytesseract

    with Image.open(frame_path) as image:
        width, height = image.size
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config="--psm 6")

    detections = []
    for index, text in enumerate(data.get("text", [])):
        cleaned = str(text or "").strip()
        if len(cleaned) < 2 or not re.search(r"[A-Za-z0-9]", cleaned):
            continue
        confidence = safe_float(data.get("conf", [0])[index], -1)
        if confidence < confidence_threshold:
            continue
        raw = {
            "x": int(data.get("left", [0])[index]),
            "y": int(data.get("top", [0])[index]),
            "w": int(data.get("width", [0])[index]),
            "h": int(data.get("height", [0])[index]),
            "text": cleaned,
            "confidence": confidence,
        }
        detections.append(normalized_ocr_detection(raw, width, height, frame_time, padding))
    return detections


def detect_ocr_text_regions(input_path, job_dir, params):
    sample_fps = max(0.2, min(2.0, float(params.get("sample_fps", 1.0) or 1.0)))
    max_frames = max(1, min(60, int(params.get("max_frames", 20) or 20)))
    confidence = max(0, min(95, float(params.get("confidence", 45) or 45)))
    padding = max(0, min(80, int(params.get("padding", 12) or 12)))
    frames = extract_ocr_frames(input_path, job_dir, sample_fps, max_frames)
    detections = []
    for index, frame_path in enumerate(frames):
        frame_time = index / sample_fps
        detections.extend(ocr_frame_detections(frame_path, frame_time, confidence, padding))
    return detections


def apply_ocr_redact(input_path, job_dir, params, job=None):
    detections = detect_ocr_text_regions(input_path, job_dir, params)
    if job is not None:
        job["ocr_redactions"] = detections[:80]
    if not detections:
        append_job_warning(job, "ocr_redact found no readable text regions; skipped OCR redaction")
        return input_path

    chain = ocr_redact_filter_chain(detections)
    if not chain:
        return input_path
    output_path = next_media_path(job_dir, "ocr_redact")
    run_video_filter_step(input_path, output_path, chain)
    return output_path


def normalized_region_box(width, height, x_frac, y_frac, w_frac, h_frac):
    x = clamp_int(width * x_frac, 0, max(0, width - 4))
    y = clamp_int(height * y_frac, 0, max(0, height - 4))
    box_width = clamp_int(width * w_frac, 4, width - x)
    box_height = clamp_int(height * h_frac, 4, height - y)
    return {"x": x, "y": y, "w": box_width, "h": box_height}


def face_privacy_regions(width, height, params):
    target = str(params.get("target", "faces") or "faces").lower()
    layout = str(params.get("layout", "group") or "group").lower()

    if layout == "body" or target in {"person", "people", "humans"} and layout != "center":
        specs = [
            (0.18, 0.05, 0.64, 0.82),
            (0.02, 0.08, 0.32, 0.76),
            (0.66, 0.08, 0.32, 0.76),
        ] if layout == "group" else [(0.18, 0.05, 0.64, 0.82)]
    elif layout == "center":
        specs = [(0.30, 0.08, 0.40, 0.38)]
    else:
        specs = [
            (0.06, 0.08, 0.28, 0.38),
            (0.36, 0.07, 0.28, 0.40),
            (0.66, 0.08, 0.28, 0.38),
        ]

    return [normalized_region_box(width, height, *spec) for spec in specs]


def face_privacy_filter_chain(regions):
    filters = []
    for region in regions:
        base = f"delogo=x={region['x']}:y={region['y']}:w={region['w']}:h={region['h']}:show=0"
        if "start" in region and "end" in region:
            base += f":enable='between(t,{float(region['start']):.3f},{float(region['end']):.3f})'"
        filters.append(base)
    return ",".join(filters)


def proot_cv2_python_path():
    status = runtime_capabilities().get("python_modules", {}).get("cv2_proot", {})
    return status.get("python") if status.get("ok") else None


def detect_face_privacy_regions_with_opencv(input_path, params):
    python_path = proot_cv2_python_path()
    if not python_path or not PROOT_DISTRO or not OPENCV_DETECTOR_SCRIPT.exists():
        return {"mode": "unavailable", "detections": [], "error": "OpenCV detector unavailable"}

    result = run_command_result([
        PROOT_DISTRO, "login", "ubuntu", "--",
        python_path,
        str(OPENCV_DETECTOR_SCRIPT),
        str(input_path),
        json.dumps(params),
    ], timeout=45)
    if result.returncode != 0:
        return {
            "mode": "opencv_cascade",
            "detections": [],
            "error": concise_error(result.stderr or result.stdout or "OpenCV detection failed"),
        }
    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError as exc:
        return {"mode": "opencv_cascade", "detections": [], "error": concise_error(exc)}


def apply_face_privacy_blur(input_path, job_dir, params, job=None):
    width, height = ffprobe_video_dimensions(input_path)
    detection = detect_face_privacy_regions_with_opencv(input_path, params)
    regions = detection.get("detections", [])[:80]
    mode = "opencv_cascade"
    if not regions:
        regions = face_privacy_regions(width, height, params)
        mode = "safe_regions_no_detection"
    chain = face_privacy_filter_chain(regions)
    if not chain:
        append_job_warning(job, "face_privacy_blur found no privacy regions; skipped")
        return input_path
    output_path = next_media_path(job_dir, "face_privacy_blur")
    run_video_filter_step(input_path, output_path, chain)
    if job is not None:
        job["face_privacy_blur"] = {
            "mode": mode,
            "regions": regions,
            "target": str(params.get("target", "faces") or "faces"),
            "layout": str(params.get("layout", "group") or "group"),
            "opencv_detection": {
                "mode": detection.get("mode"),
                "sampled_frames": detection.get("sampled_frames", 0),
                "detected_regions": len(detection.get("detections", [])),
                "error": detection.get("error"),
            },
        }
        if mode != "opencv_cascade":
            append_job_warning(
                job,
                "face_privacy_blur used safe regions because OpenCV found no face/person regions",
            )
    return output_path


def clean_caption_text(text):
    cleaned = re.sub(r"\s+", " ", str(text or "")).strip()
    cleaned = cleaned.replace("\x00", "")
    if cleaned.lower() in {"<unk>", "[unk]", "[sil]", "<sil>"}:
        return ""
    return cleaned[:160]


def caption_wrap_text(text, line_length=34, max_lines=2):
    words = clean_caption_text(text).split()
    if not words:
        return ""
    lines = []
    current = []
    current_length = 0
    for word in words:
        proposed = current_length + len(word) + (1 if current else 0)
        if current and proposed > line_length and len(lines) < max_lines - 1:
            lines.append(" ".join(current))
            current = [word]
            current_length = len(word)
        else:
            current.append(word)
            current_length = proposed
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines[:max_lines])


def parse_asr_metadata_segments(metadata_text, max_segments=80):
    segments = []
    current_time = None
    for raw_line in str(metadata_text or "").splitlines():
        line = raw_line.strip()
        time_match = re.search(r"\bpts_time:([0-9.]+)", line)
        if time_match:
            try:
                current_time = float(time_match.group(1))
            except ValueError:
                current_time = None
            continue

        if "lavfi.asr.text=" not in line:
            continue
        text = clean_caption_text(line.split("lavfi.asr.text=", 1)[1])
        if not text:
            continue
        if current_time is None:
            current_time = segments[-1]["end"] + 2.0 if segments else 2.0

        previous_end = segments[-1]["end"] if segments else 0.0
        start = max(0.0, previous_end)
        end = max(float(current_time), start + 0.6)
        if end - start > 4.0:
            start = max(0.0, end - 4.0)
        segments.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "text": text,
        })
        current_time = None
        if len(segments) >= max_segments:
            break
    return segments


def run_asr_transcription(input_path, job_dir, params, job=None):
    if not ffprobe_has_audio(input_path):
        append_job_warning(job, "auto_captions skipped because no audio stream is available")
        return []

    model_status = probe_pocketsphinx_models()
    if not model_status["ok"]:
        append_job_warning(job, "auto_captions skipped because pocketsphinx English model files are unavailable")
        return []

    max_segments = max(1, min(160, int(params.get("max_segments", 80) or 80)))
    metadata_path = job_dir / f"asr_metadata_{uuid4().hex[:8]}.txt"
    log_path = job_dir / f"asr_{uuid4().hex[:8]}.log"
    filter_chain = (
        "aresample=16000,"
        f"asr=hmm={POCKETSPHINX_HMM}:dict={POCKETSPHINX_DICT}:lm={POCKETSPHINX_LM}:"
        f"logfn={log_path.as_posix()},"
        f"ametadata=print:key=lavfi.asr.text:file={metadata_path.as_posix()}"
    )

    try:
        run_command([
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vn",
            "-af", filter_chain,
            "-f", "null", "-",
        ])
    except Exception as exc:
        append_job_warning(job, f"auto_captions ASR failed; skipped captions: {concise_error(exc)}")
        return []

    if not metadata_path.exists():
        append_job_warning(job, "auto_captions produced no ASR metadata; skipped captions")
        return []

    segments = parse_asr_metadata_segments(metadata_path.read_text(encoding="utf-8"), max_segments=max_segments)
    if job is not None:
        job["auto_captions"] = {
            "engine": "ffmpeg_asr_pocketsphinx",
            "segments": segments,
            "metadata_path": str(metadata_path),
            "accuracy": "best_effort",
        }
    if segments:
        append_job_warning(job, "auto_captions used pocketsphinx ASR; transcript accuracy is best-effort")
    else:
        append_job_warning(job, "auto_captions found no recognized speech; skipped captions")
    return segments


def caption_drawtext_chain(segments, job_dir, params):
    font_size = max(18, min(96, int(safe_float(params.get("font_size"), 44))))
    y_expr = "h-(text_h*2.2)"
    if str(params.get("position", "")).lower() in {"top", "upper"}:
        y_expr = "text_h*1.3"
    filters = []
    for index, segment in enumerate(segments, start=1):
        text = caption_wrap_text(segment.get("text", ""))
        if not text:
            continue
        text_path = job_dir / f"caption_{index:03d}_{uuid4().hex[:8]}.txt"
        text_path.write_text(text, encoding="utf-8")
        start = max(0.0, safe_float(segment.get("start"), 0.0))
        end = max(start + 0.4, safe_float(segment.get("end"), start + 2.0))
        filters.append(
            f"drawtext=textfile={text_path.as_posix()}:fontcolor=white:fontsize={font_size}:"
            f"x=(w-text_w)/2:y={y_expr}:box=1:boxcolor=black@0.58:boxborderw=18:"
            f"line_spacing=8:enable='between(t,{start:.3f},{end:.3f})'"
        )
    return ",".join(filters)


def apply_auto_captions(input_path, job_dir, params, job=None):
    segments = run_asr_transcription(input_path, job_dir, params, job)
    if not segments:
        return input_path
    chain = caption_drawtext_chain(segments, job_dir, params)
    if not chain:
        append_job_warning(job, "auto_captions produced no drawable caption text")
        return input_path
    output_path = next_media_path(job_dir, "auto_captions")
    run_video_filter_step(input_path, output_path, chain)
    return output_path


def run_filter_complex_video(input_path, output_path, filter_complex):
    has_audio = ffprobe_has_audio(input_path)
    command = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-filter_complex", filter_complex,
        "-map", "[v]",
    ]
    if has_audio:
        command.extend(["-map", "0:a:0"])
    else:
        command.append("-an")
    command.extend([
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
    ])
    if has_audio:
        command.extend(["-c:a", "aac", "-b:a", "192k"])
    command.append(str(output_path))
    run_command(command)


def apply_picture_in_picture(input_path, job_dir, params):
    scale = max(0.12, min(0.5, float(params.get("scale", 0.32) or 0.32)))
    margin = max(0, int(params.get("margin", 24) or 24))
    position = str(params.get("position", "top_right") or "top_right").lower()
    x_expr = f"W-w-{margin}" if "right" in position else str(margin)
    y_expr = f"H-h-{margin}" if "bottom" in position else str(margin)
    output_path = next_media_path(job_dir, "picture_in_picture")
    filter_complex = (
        "[0:v]split=2[base][pip];"
        f"[pip]scale=trunc(iw*{scale:.3f}/2)*2:-2[pip];"
        f"[base][pip]overlay={x_expr}:{y_expr}:format=auto[v]"
    )
    run_filter_complex_video(input_path, output_path, filter_complex)
    return output_path


def apply_split_screen_mirror(input_path, job_dir, params):
    divider_color = str(params.get("divider_color", "white") or "white")
    if not re.match(r"^[A-Za-z0-9_#@.]+$", divider_color):
        divider_color = "white"
    output_path = next_media_path(job_dir, "split_screen_mirror")
    filter_complex = (
        "[0:v]split=2[left][right];"
        "[left]scale=trunc(iw/4)*2:ih[leftout];"
        "[right]hflip,scale=trunc(iw/4)*2:ih[rightout];"
        "[leftout][rightout]hstack=inputs=2[stack];"
        f"[stack]drawbox=x=iw/2-1:y=0:w=2:h=ih:color={divider_color}@1:t=fill[v]"
    )
    run_filter_complex_video(input_path, output_path, filter_complex)
    return output_path


def even_dimension(value, default, minimum=320, maximum=4096):
    dimension = max(minimum, min(maximum, int(safe_float(value, default))))
    if dimension % 2:
        dimension += 1 if dimension < maximum else -1
    return dimension


def apply_blur_background(input_path, job_dir, params, job=None):
    width = even_dimension(params.get("width"), 1080)
    height = even_dimension(params.get("height"), 1920)
    sigma = max(4.0, min(80.0, safe_float(params.get("sigma"), 28.0)))
    bg_saturation = max(0.2, min(2.0, safe_float(params.get("background_saturation"), 1.08)))
    bg_brightness = max(-0.35, min(0.35, safe_float(params.get("background_brightness"), -0.04)))
    output_path = next_media_path(job_dir, "blur_background")
    filter_complex = (
        "[0:v]split=2[bg][fg];"
        f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},gblur=sigma={sigma:g},"
        f"eq=brightness={bg_brightness:g}:saturation={bg_saturation:g}[bgout];"
        f"[fg]scale={width}:{height}:force_original_aspect_ratio=decrease[fgout];"
        "[bgout][fgout]overlay=(W-w)/2:(H-h)/2:format=auto,setsar=1,format=yuv420p[v]"
    )
    run_filter_complex_video(input_path, output_path, filter_complex)
    if job is not None:
        job["blur_background"] = {
            "width": width,
            "height": height,
            "sigma": sigma,
        }
    return output_path


CHROMA_KEY_COLOR_VALUES = {
    "green": "0x00ff00",
    "blue": "0x0000ff",
    "red": "0xff0000",
}


def safe_ffmpeg_color(value, default="black"):
    raw = str(value or default).strip().lower()
    if raw in SOLID_COLOR_VALUES.values():
        return raw
    if re.fullmatch(r"0x[0-9a-fA-F]{6}", raw) or re.fullmatch(r"#[0-9a-fA-F]{6}", raw):
        return raw.replace("#", "0x")
    return default


def apply_chroma_key(input_path, job_dir, params, job=None):
    width, height = ffprobe_video_dimensions(input_path)
    duration = ffprobe_duration(input_path)
    key_name = str(params.get("key_color", "green") or "green").lower()
    key_color = CHROMA_KEY_COLOR_VALUES.get(key_name, key_name)
    if not re.fullmatch(r"0x[0-9a-fA-F]{6}", key_color):
        key_color = CHROMA_KEY_COLOR_VALUES["green"]
        key_name = "green"
    replacement_color = safe_ffmpeg_color(params.get("replacement_color", "black"), "black")
    similarity = max(0.01, min(1.0, safe_float(params.get("similarity"), 0.20)))
    blend = max(0.0, min(1.0, safe_float(params.get("blend"), 0.08)))
    output_path = next_media_path(job_dir, "chroma_key")
    filter_complex = (
        f"[0:v]chromakey={key_color}:{similarity:.3f}:{blend:.3f}[fg];"
        f"color=c={replacement_color}:s={width}x{height}:d={duration:.3f}[bg];"
        "[bg][fg]overlay=format=auto,format=yuv420p[v]"
    )
    run_filter_complex_video(input_path, output_path, filter_complex)
    if job is not None:
        job["chroma_key"] = {
            "key_color": key_name,
            "replacement_color": replacement_color,
            "similarity": similarity,
            "blend": blend,
            "width": width,
            "height": height,
        }
    return output_path


def film_damage_filter_chain(params, width=1280, height=720):
    intensity = max(0.1, min(1.0, safe_float(params.get("intensity"), 0.7)))
    grain = clamp_int(params.get("grain", 12 + intensity * 32), 10, 54)
    gate_weave = clamp_int(params.get("gate_weave", 2 + intensity * 12), 2, 16)
    max_pad = max(4, min(width, height) // 8)
    pad = clamp_int(max(8, gate_weave * 3), 4, max_pad)
    gate_weave = min(gate_weave, max(1, pad - 1))
    scratch_opacity = max(0.06, min(0.40, safe_float(params.get("scratch_opacity"), 0.08 + intensity * 0.24)))
    dust_opacity = max(0.10, min(0.60, safe_float(params.get("dust_opacity"), 0.16 + intensity * 0.38)))
    contrast = 1.04 + intensity * 0.20
    saturation = 0.82 - intensity * 0.24
    flicker = 0.010 + intensity * 0.025

    return ",".join([
        f"crop=iw-{pad * 2}:ih-{pad * 2}:{pad}+{gate_weave}*sin(7*t):{pad}+{gate_weave}*cos(5*t)",
        f"scale=trunc((iw+{pad * 2})/2)*2:trunc((ih+{pad * 2})/2)*2",
        "setsar=1",
        f"eq=contrast={contrast:.2f}:brightness='0.010+{flicker:.3f}*sin(18*t)':saturation={saturation:.2f}",
        f"noise=alls={grain}:allf=t+u",
        f"drawbox=x='iw*0.16+20*sin(3*t)':y=0:w=2:h=ih:color=white@{scratch_opacity:.2f}:t=fill:enable='lt(mod(t,1.7),0.35)'",
        f"drawbox=x='iw*0.72+15*sin(4*t)':y=0:w=2:h=ih:color=black@{scratch_opacity * 0.9:.2f}:t=fill:enable='lt(mod(t,2.1),0.28)'",
        f"drawbox=x='iw*0.30':y='ih*0.18':w=6:h=6:color=white@{dust_opacity:.2f}:t=fill:enable='lt(mod(t,0.43),0.05)'",
        f"drawbox=x='iw*0.64':y='ih*0.58':w=4:h=4:color=black@{dust_opacity * 0.8:.2f}:t=fill:enable='lt(mod(t,0.61),0.05)'",
    ])


def apply_film_damage(input_path, job_dir, params, job=None):
    width, height = ffprobe_video_dimensions(input_path)
    chain = film_damage_filter_chain(params, width, height)
    output_path = next_media_path(job_dir, "film_damage")
    run_video_filter_step(input_path, output_path, chain)
    if job is not None:
        job["film_damage"] = {
            "intensity": max(0.1, min(1.0, safe_float(params.get("intensity"), 0.7))),
            "grain": clamp_int(params.get("grain", 32), 10, 54),
            "gate_weave": clamp_int(params.get("gate_weave", 8), 2, 16),
        }
    return output_path


def apply_reverse(current, job_dir, _params=None, _job=None):
    output_path = next_media_path(job_dir, "reversed")
    command = ["ffmpeg", "-y", "-i", str(current), "-vf", "reverse"]
    if ffprobe_has_audio(current):
        command.extend(["-af", "areverse"])
    else:
        command.append("-an")
    command.append(str(output_path))
    run_command(command)
    return output_path


def apply_speed_ramp_special(current, job_dir, params, _job=None):
    speed_factor = normalized_speed_factor(
        params.get("factor", params.get("speed_factor", params.get("tempo"))),
        None,
    )
    if speed_factor is None:
        slow_factor = float(params.get("slow_factor", 0.5))
        fast_factor = float(params.get("fast_factor", 2.0))
        speed_factor = normalized_speed_factor((slow_factor + fast_factor) / 2, 1.0)
    output_path = next_media_path(job_dir, "speed_ramp")
    command = [
        "ffmpeg", "-y", "-i", str(current),
        "-vf", f"setpts={1 / speed_factor:.6f}*PTS",
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
    ]
    if ffprobe_has_audio(current):
        command.extend(["-af", atempo_chain(speed_factor), "-c:a", "aac", "-b:a", "192k"])
    else:
        command.append("-an")
    command.append(str(output_path))
    run_command(command)
    return output_path


def special_executor_map():
    return {
        "silence_remove": lambda current, job_dir, params, job, context: apply_silence_remove(current, job_dir, params),
        "black_remove": lambda current, job_dir, params, job, context: apply_black_remove(current, job_dir, params, job),
        "freeze_remove": lambda current, job_dir, params, job, context: apply_freeze_remove(current, job_dir, params, job),
        "dedupe_frames": lambda current, job_dir, params, job, context: apply_dedupe_frames(current, job_dir, params, job),
        "beat_cut": lambda current, job_dir, params, job, context: apply_beat_cut(current, job_dir, params, context, job),
        "scene_montage": lambda current, job_dir, params, job, context: apply_scene_montage(current, job_dir, params, job),
        "energy_montage": lambda current, job_dir, params, job, context: apply_energy_montage(current, job_dir, params, context, job),
        "crop_borders": lambda current, job_dir, params, job, context: apply_crop_borders(current, job_dir, params, job),
        "stabilize": lambda current, job_dir, params, job, context: apply_stabilize(current, job_dir, params),
        "reverse": lambda current, job_dir, params, job, context: apply_reverse(current, job_dir, params, job),
        "boomerang": lambda current, job_dir, params, job, context: apply_boomerang(current, job_dir, params, job),
        "end_reverse": lambda current, job_dir, params, job, context: apply_end_reverse(current, job_dir, params),
        "trim": lambda current, job_dir, params, job, context: apply_trim(current, job_dir, params),
        "remove_segment": lambda current, job_dir, params, job, context: apply_remove_segment(current, job_dir, params),
        "remove_audio": lambda current, job_dir, params, job, context: apply_remove_audio(current, job_dir),
        "replace_audio": lambda current, job_dir, params, job, context: apply_replace_audio(current, job, job_dir, params),
        "mix_uploaded_audio": lambda current, job_dir, params, job, context: apply_mix_uploaded_audio(current, job, job_dir, params),
        "face_privacy_blur": lambda current, job_dir, params, job, context: apply_face_privacy_blur(current, job_dir, params, job),
        "ocr_redact": lambda current, job_dir, params, job, context: apply_ocr_redact(current, job_dir, params, job),
        "auto_captions": lambda current, job_dir, params, job, context: apply_auto_captions(current, job_dir, params, job),
        "picture_in_picture": lambda current, job_dir, params, job, context: apply_picture_in_picture(current, job_dir, params),
        "split_screen_mirror": lambda current, job_dir, params, job, context: apply_split_screen_mirror(current, job_dir, params),
        "blur_background": lambda current, job_dir, params, job, context: apply_blur_background(current, job_dir, params, job),
        "chroma_key": lambda current, job_dir, params, job, context: apply_chroma_key(current, job_dir, params, job),
        "film_damage": lambda current, job_dir, params, job, context: apply_film_damage(current, job_dir, params, job),
        "speed_ramp": lambda current, job_dir, params, job, context: apply_speed_ramp_special(current, job_dir, params, job),
        "pitch_shift": lambda current, job_dir, params, job, context: apply_pitch_shift(current, job_dir, params, job),
    }


SPECIAL_EXECUTORS = special_executor_map()


def special_executor_summary():
    implemented = set(SPECIAL_EXECUTORS)
    registered = set(SUPPORTED_SPECIAL_TYPES)
    return {
        "implemented_special_types": sorted(implemented),
        "registered_but_unimplemented": sorted(registered - implemented),
        "implemented_but_unregistered": sorted(implemented - registered),
    }


def architecture_integrity():
    executor_summary = special_executor_summary()
    required_capabilities = set(architecture_required_capabilities())
    executor_capabilities = set((runtime_capabilities() or {}).get("executor") or {})
    issues = []
    issues.extend(architecture_registry_issues())
    for operation_type in executor_summary["registered_but_unimplemented"]:
        issues.append({
            "code": "registered_special_without_executor",
            "message": f"Registered special operation has no executor: {operation_type}",
            "severity": "error",
        })
    for operation_type in executor_summary["implemented_but_unregistered"]:
        issues.append({
            "code": "executor_special_without_registry",
            "message": f"Executor special operation is not registered: {operation_type}",
            "severity": "error",
        })
    for capability in sorted(required_capabilities - executor_capabilities):
        issues.append({
            "code": "registered_capability_without_probe",
            "message": f"Registered runtime capability is not published by executor probes: {capability}",
            "severity": "error",
        })
    return {
        "ok": not issues,
        "issues": issues,
        "capability_contract": {
            "registered_capabilities": sorted(required_capabilities),
            "published_capabilities": sorted(executor_capabilities),
            "missing_probe_keys": sorted(required_capabilities - executor_capabilities),
        },
    }


def apply_special_step(current, job_dir, step, job=None, context=None):
    step_type = step.get("type")
    params = step.get("params", {})
    executor = SPECIAL_EXECUTORS.get(step_type)
    if not executor:
        return current
    return executor(current, job_dir, params, job, context or {})


class ExecutionPolicyError(RuntimeError):
    pass


def fail_execution_phase(job, phase, message):
    policy = execution_failure_policy(phase)
    detail = f"{message} (policy={policy.get('mode')})"
    append_job_warning(job, detail)
    raise ExecutionPolicyError(detail)


def apply_special(input_path, job_dir, special, job=None, context=None):
    current = input_path
    for index, step in enumerate(special, start=1):
        step_type = step.get("type")
        try:
            next_current = apply_special_step(current, job_dir, step, job, context)
            if next_current != current:
                current = next_current
            elif step_type in SUPPORTED_SPECIAL_TYPES and step_type not in SPECIAL_EXECUTORS:
                fail_execution_phase(
                    job,
                    "special",
                    f"special step {index} registered but has no executor: {step_type}",
                )
            elif step_type not in SUPPORTED_SPECIAL_TYPES:
                fail_execution_phase(
                    job,
                    "special",
                    f"special step {index} has unsupported type: {step_type}",
                )
        except Exception as exc:
            if isinstance(exc, ExecutionPolicyError):
                raise
            fail_execution_phase(
                job,
                "special",
                f"special step {index} ({step_type}) failed: {concise_error(exc)}",
            )
    return current


def split_filter_chain(filter_chain):
    parts = []
    current = []
    quote = None
    escaped = False
    paren_depth = 0
    for char in filter_chain:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\":
            current.append(char)
            escaped = True
            continue
        if char in ["'", '"']:
            quote = None if quote == char else char if quote is None else quote
        elif quote is None and char == "(":
            paren_depth += 1
        elif quote is None and char == ")" and paren_depth > 0:
            paren_depth -= 1
        if char == "," and quote is None and paren_depth == 0:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current).strip())
    return parts


def split_filter_options(value):
    parts = []
    current = []
    quote = None
    escaped = False
    for char in value:
        if escaped:
            current.append(char)
            escaped = False
            continue
        if char == "\\":
            current.append(char)
            escaped = True
            continue
        if char in ["'", '"']:
            quote = None if quote == char else char if quote is None else quote
        if char == ":" and quote is None:
            parts.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        parts.append("".join(current).strip())
    return parts


def escape_filter_expression(expression):
    return expression.replace(",", r"\,")


def add_timing_to_crop_filter(filter_part, expression):
    args = split_filter_options(filter_part[len("crop="):])
    if len(args) < 4:
        return filter_part

    width, height, x_expr, y_expr = args[:4]
    rest = args[4:]
    gate = escape_filter_expression(expression)
    center_x = f"(iw-({width}))/2"
    center_y = f"(ih-({height}))/2"
    timed_x = (
        f"max(0\\,min(iw-({width})\\,"
        f"({center_x})+(({x_expr})-({center_x}))*({gate})))"
    )
    timed_y = (
        f"max(0\\,min(ih-({height})\\,"
        f"({center_y})+(({y_expr})-({center_y}))*({gate})))"
    )
    return ":".join(["crop=" + width, height, timed_x, timed_y, *rest])


NO_TIMELINE_FILTERS = {"scale", "setsar", "setdar", "setpts", "fps", "format", "zoompan"}
MULTI_STREAM_FILTERS = {"split", "hstack", "vstack", "xstack", "concat"}


def filter_name(filter_part):
    name = filter_part.split("=", 1)[0].strip()
    name = name.split(":", 1)[0].strip()
    return re.sub(r"\[.*?\]", "", name)


def extract_enable_expression(filter_part):
    match = re.search(r":enable=(['\"])(.*?)\1", filter_part)
    if match:
        return match.group(2)
    match = re.search(r":enable=([^:]+)", filter_part)
    return match.group(1) if match else None


def remove_enable_option(filter_part):
    return re.sub(r":enable=(['\"]).*?\1", "", filter_part)


def strip_unsupported_timeline_options(filter_chain):
    repaired = []
    for part in split_filter_chain(filter_chain):
        name = filter_name(part)
        expression = extract_enable_expression(part)
        if name in NO_TIMELINE_FILTERS and expression:
            if repaired and filter_name(repaired[-1]) == "crop":
                repaired[-1] = add_timing_to_crop_filter(repaired[-1], expression)
            repaired.append(remove_enable_option(part))
        else:
            repaired.append(part)
    return ",".join(repaired)


def collapse_multistream_filter_chain(filter_chain):
    parts = [part for part in split_filter_chain(filter_chain) if part]
    multistream_indexes = [
        index for index, part in enumerate(parts)
        if filter_name(part) in MULTI_STREAM_FILTERS
    ]
    if not multistream_indexes and ";" not in filter_chain:
        return filter_chain

    tail_start = max(multistream_indexes) + 1 if multistream_indexes else len(parts)
    tail = []
    for part in parts[tail_start:]:
        if ";" in part or "[" in part or "]" in part:
            continue
        if filter_name(part) not in MULTI_STREAM_FILTERS:
            tail.append(part)
    fallback = ["boxblur=10:1", "vignette=angle=PI/3"]
    return ",".join([*fallback, *tail])


def dedupe_filter_chain(filter_chain):
    parts = [part for part in split_filter_chain(filter_chain) if part]
    seen = set()
    deduped = []
    for part in parts:
        if part in seen:
            continue
        seen.add(part)
        deduped.append(part)
    return ",".join(deduped)


def add_timing_to_filter_chain(filter_chain, expression):
    enabled_parts = []
    for part in split_filter_chain(filter_chain):
        name = filter_name(part)
        if name == "crop":
            enabled_parts.append(add_timing_to_crop_filter(part, expression))
            continue
        if name == "zoompan":
            enabled_parts.append(add_timing_to_crop_filter("crop=iw-48:ih-48:24:24", expression))
            enabled_parts.append("scale=iw+48:ih+48")
            continue
        if name in NO_TIMELINE_FILTERS:
            enabled_parts.append(part)
            continue
        if "enable=" in part:
            enabled_parts.append(part)
        else:
            enabled_parts.append(f"{part}:enable='{expression}'")
    return ",".join(enabled_parts)


def between_expression(times):
    windows = [f"between(t,{max(0, time - 0.05):.3f},{time + 0.15:.3f})" for time in times]
    return "+".join(windows) if windows else "0"


def context_window_expression(times, start_offset=0.0, end_offset=0.05):
    windows = []
    for time in times:
        start = max(0, float(time) + start_offset)
        end = max(start + 0.01, float(time) + end_offset)
        windows.append(f"between(t,{start:.3f},{end:.3f})")
    return "+".join(windows) if windows else "0"


def replace_context_time_references(filter_string, context):
    if not isinstance(filter_string, str) or not context:
        return filter_string

    result = filter_string
    for context_key, times in context.items():
        if not isinstance(times, list):
            continue

        if context_key.endswith("energy_curve") and context_key in result:
            energy_times = context.get(f"{context_key}_times", [])
            expression = between_expression(energy_times)
            result = re.sub(r"enable=(['\"])[^'\"]*energy_curve[^'\"]*\1", f"enable='{expression}'", result)
            result = re.sub(rf",\s*\(?{re.escape(context_key)}\)?", "", result)
            result = re.sub(rf"\(?{re.escape(context_key)}\)?\s*,", "", result)
            result = result.strip(",")
            continue

        pattern = re.compile(
            rf"between\(t\s*,\s*{re.escape(context_key)}\[\d+\]\s*"
            rf"(?:(?P<start_op>[+-])\s*(?P<start_offset>[0-9.]+))?\s*,\s*"
            rf"{re.escape(context_key)}\[\d+\]\s*"
            rf"(?:(?P<end_op>[+-])\s*(?P<end_offset>[0-9.]+))?\s*\)"
        )

        def replace_match(match):
            start_offset = float(match.group("start_offset") or 0)
            end_offset = float(match.group("end_offset") or 0.05)
            if match.group("start_op") == "-":
                start_offset = -start_offset
            if match.group("end_op") == "-":
                end_offset = -end_offset
            return context_window_expression(times, start_offset, end_offset)

        result = pattern.sub(replace_match, result)

        def replace_index_reference(match):
            try:
                index = int(match.group(1))
                return f"{float(times[index]):.3f}"
            except (IndexError, TypeError, ValueError):
                return "0"

        result = re.sub(rf"{re.escape(context_key)}\[(\d+)\]", replace_index_reference, result)

    return result


def quote_audio_filter_expressions(filter_string):
    if not isinstance(filter_string, str):
        return filter_string
    if filter_string.startswith("volume=") and not filter_string.startswith("volume='"):
        expression = filter_string[len("volume="):]
        if "if(" in expression or "between(" in expression:
            return f"volume='{expression}'"
    return filter_string


def repair_duration_references(filter_string, duration):
    if not isinstance(filter_string, str) or duration is None:
        return filter_string

    def replace_start(match):
        offset = float(match.group(1))
        return f"st={max(0.0, duration - offset):.3f}"

    return re.sub(r"st=(?:d|duration)-([0-9.]+)", replace_start, filter_string)


def repair_dimension_references(filter_string, dimensions):
    if not isinstance(filter_string, str) or "__PRIVACY_" not in filter_string:
        return filter_string

    width, height = dimensions or (1280, 720)
    box_width = max(80, min(320, int(width * 0.35)))
    box_height = max(60, min(220, int(height * 0.32)))
    x = max(0, (width - box_width) // 2)
    y = max(0, (height - box_height) // 2)
    return (
        filter_string
        .replace("__PRIVACY_X__", str(x))
        .replace("__PRIVACY_Y__", str(y))
        .replace("__PRIVACY_W__", str(box_width))
        .replace("__PRIVACY_H__", str(box_height))
    )


def materialize_drawtext_filter(filter_string, job_dir):
    if not isinstance(filter_string, str) or "drawtext=text=" not in filter_string:
        return filter_string

    prefix = "drawtext=text="
    option_index = min(
        [index for marker in DRAWTEXT_OPTION_MARKERS if (index := filter_string.find(marker, len(prefix))) != -1],
        default=-1,
    )
    if option_index == -1:
        return filter_string

    text_value = filter_string[len(prefix):option_index]
    if len(text_value) >= 2 and text_value[0] in {"'", '"'} and text_value[-1] == text_value[0]:
        text_value = text_value[1:-1]

    text_path = job_dir / f"drawtext_{uuid4().hex[:8]}.txt"
    text_path.write_text(text_value, encoding="utf-8")
    return f"drawtext=textfile={text_path.as_posix()}{filter_string[option_index:]}"


def video_filter_chain(video_filters, context, job_dir, duration=None, dimensions=None):
    filters = []
    for step in video_filters:
        filter_string = step.get("filter")
        if not filter_string:
            continue

        filter_string = replace_context_time_references(filter_string, context)
        filter_string = repair_duration_references(filter_string, duration)
        filter_string = repair_dimension_references(filter_string, dimensions)
        filter_string = materialize_drawtext_filter(filter_string, job_dir)
        filter_string = strip_unsupported_timeline_options(filter_string)
        timing = step.get("timing")
        if timing in ["per_beat", "per_onset"]:
            context_key = step.get("requires_context") or ("beat_times" if timing == "per_beat" else "onset_times")
            if context_key not in context:
                raise RuntimeError(f"missing analysis context: {context_key}")
            filters.append(add_timing_to_filter_chain(filter_string, between_expression(context[context_key])))
        elif str(step.get("requires_context") or "").endswith("energy_curve"):
            context_key = step.get("requires_context")
            peak_key = f"{context_key}_times"
            if peak_key in context:
                filters.append(add_timing_to_filter_chain(filter_string, between_expression(context[peak_key])))
            else:
                filters.append(filter_string)
        else:
            filters.append(filter_string)
    return dedupe_filter_chain(collapse_multistream_filter_chain(",".join(filters)))


def fallback_video_filter_chain(step, failed_chain):
    text = " ".join([
        str(step.get("description", "")),
        str(step.get("filter", "")),
        str(failed_chain or ""),
    ]).lower()
    if any(term in text for term in ["blur", "privacy", "face", "license"]):
        return "delogo=x=480:y=250:w=320:h=220:show=0,vignette=angle=PI/3"
    if any(term in text for term in ["text", "caption", "subtitle", "drawtext"]):
        return "drawbox=x=0:y=ih-96:w=iw:h=96:color=black@0.45:t=fill"
    if any(term in text for term in ["flash", "strobe", "pulse"]):
        return "drawbox=x=0:y=0:w=iw:h=ih:color=white@0.20:t=fill:enable='lt(mod(t,0.5),0.05)'"
    if any(term in text for term in ["glitch", "rgb", "chromatic"]):
        return "rgbashift=rh=8:bh=-8:rv=3:bv=-3,noise=alls=18:allf=t+u"
    if any(term in text for term in ["grain", "film", "vhs", "old"]):
        return "noise=alls=18:allf=t+u"
    return "eq=contrast=1.06:saturation=1.04"


def run_video_filter_step(current, output_path, chain):
    run_command([
        "ffmpeg", "-y", "-i", str(current),
        "-vf", chain,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "copy",
        str(output_path),
    ])


def planned_filter_count(filters):
    return len([
        step for step in filters or []
        if step.get("filter")
    ])


def apply_video_filters(input_path, job_dir, video_filters, context, job=None):
    current = input_path
    filter_count = planned_filter_count(video_filters)
    if not filter_count:
        return current

    duration = None
    dimensions = None
    try:
        duration = ffprobe_duration(current)
    except Exception:
        duration = None
    try:
        dimensions = ffprobe_video_dimensions(current)
    except Exception:
        dimensions = None

    try:
        combined_chain = video_filter_chain(video_filters, context, job_dir, duration, dimensions)
    except Exception as exc:
        combined_chain = ""
        append_job_warning(job, f"combined video filter chain could not be prepared; retrying step-by-step: {concise_error(exc)}")

    if combined_chain:
        output_path = next_media_path(job_dir, "video_filters")
        try:
            run_video_filter_step(current, output_path, combined_chain)
            if job is not None:
                job["video_filter_execution"] = {
                    "mode": "combined",
                    "planned_filter_count": filter_count,
                    "applied_filter_count": filter_count,
                }
            return output_path
        except Exception as exc:
            append_job_warning(
                job,
                f"combined video filter chain failed; retrying step-by-step: {concise_error(exc)}",
            )

    applied_count = 0
    for index, step in enumerate(video_filters, start=1):
        duration = None
        dimensions = None
        try:
            duration = ffprobe_duration(current)
        except Exception:
            duration = None
        try:
            dimensions = ffprobe_video_dimensions(current)
        except Exception:
            dimensions = None

        chain = video_filter_chain([step], context, job_dir, duration, dimensions)
        if not chain:
            continue

        output_path = next_media_path(job_dir, f"video_filter_{index:02d}")
        try:
            run_video_filter_step(current, output_path, chain)
            current = output_path
            applied_count += 1
            if job is not None:
                job["video_filter_execution"] = {
                    "mode": "step_by_step",
                    "planned_filter_count": filter_count,
                    "applied_filter_count": applied_count,
                    "last_successful_step": index,
                }
            continue
        except Exception as exc:
            append_job_warning(
                job,
                f"video filter step {index} failed; attempting fallback: {concise_error(exc)}",
            )

        fallback_chain = fallback_video_filter_chain(step, chain)
        fallback_output_path = next_media_path(job_dir, f"video_filter_{index:02d}_fallback")
        try:
            run_video_filter_step(current, fallback_output_path, fallback_chain)
            append_job_warning(job, f"video filter step {index} used fallback: {fallback_chain}")
            current = fallback_output_path
            applied_count += 1
            if job is not None:
                job["video_filter_execution"] = {
                    "mode": "step_by_step",
                    "planned_filter_count": filter_count,
                    "applied_filter_count": applied_count,
                    "last_successful_step": index,
                }
        except Exception as fallback_exc:
            append_job_warning(
                job,
                f"video filter step {index} fallback failed; skipped step: {concise_error(fallback_exc)}",
            )

    if filter_count and not applied_count:
        fail_execution_phase(
            job,
            "video_filter",
            f"video filter phase planned {filter_count} filter(s), but none could be applied",
        )

    return current


def requested_output_format(job, plan):
    text = " ".join([
        str(job.get("command", "")),
        str(plan.get("intent", "")),
    ]).lower()
    if crop_borders_requested(text):
        return None
    if "blur_background" in plan_special_types(plan):
        return None
    if any(term in text for term in ["9:16", "tiktok", "reels", "shorts", "vertical"]):
        return ("vertical_9x16", "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1")
    if "4:5" in text or ("instagram" in text and "portrait" in text):
        return ("portrait_4x5", "scale=1080:1350:force_original_aspect_ratio=increase,crop=1080:1350,setsar=1")
    if "1:1" in text or "square" in text:
        return ("square_1x1", "scale=1080:1080:force_original_aspect_ratio=increase,crop=1080:1080,setsar=1")
    if "2.39" in text or "2.35" in text or "letterbox" in text or "widescreen" in text:
        return ("letterbox_239", "scale=1920:804:force_original_aspect_ratio=increase,crop=1920:804,pad=1920:1080:0:(oh-ih)/2:color=black,setsar=1")
    return None


def apply_output_aspect_enforcement(input_path, job_dir, job, plan):
    requested_format = requested_output_format(job, plan)
    if not requested_format:
        return input_path

    label, filter_chain = requested_format
    output_path = next_media_path(job_dir, label)
    run_command([
        "ffmpeg", "-y", "-i", str(input_path),
        "-vf", filter_chain,
        "-c:v", "libx264", "-preset", "fast", "-crf", "22",
        "-c:a", "copy",
        str(output_path),
    ])
    return output_path


def fallback_audio_filter_chain(step, failed_filter):
    text = " ".join([
        str(step.get("description", "")),
        str(step.get("filter", "")),
        str(failed_filter or ""),
    ]).lower()
    if any(term in text for term in ["bass", "low"]):
        return "equalizer=f=80:width_type=o:width=2:g=4"
    if any(term in text for term in ["echo", "reverb", "cave"]):
        return "aecho=0.8:0.4:350:0.35"
    if any(term in text for term in ["telephone", "phone"]):
        return "highpass=f=300,lowpass=f=3400"
    return "loudnorm=I=-14:TP=-1.5:LRA=11"


def run_audio_filter_step(current, output_path, filter_string):
    run_command([
        "ffmpeg", "-y", "-i", str(current),
        "-af", filter_string,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        str(output_path),
    ])


def audio_filter_chain(audio_filters, context=None):
    filters = []
    for step in audio_filters or []:
        filter_string = step.get("filter")
        if not filter_string:
            continue
        filter_string = quote_audio_filter_expressions(
            replace_context_time_references(filter_string, context or {})
        )
        if filter_string:
            filters.append(filter_string)
    return ",".join(filters)


def apply_audio_filters(input_path, job_dir, audio_filters, context=None, job=None):
    current = input_path
    filter_count = planned_filter_count(audio_filters)
    if filter_count and not ffprobe_has_audio(current):
        fail_execution_phase(
            job,
            "audio_filter",
            "audio filter phase was requested, but no audio stream is available",
        )

    combined_chain = audio_filter_chain(audio_filters, context)
    if combined_chain:
        output_path = next_media_path(job_dir, "audio_filters")
        try:
            run_audio_filter_step(current, output_path, combined_chain)
            if job is not None:
                job["audio_filter_execution"] = {
                    "mode": "combined",
                    "planned_filter_count": filter_count,
                    "applied_filter_count": filter_count,
                }
            return output_path
        except Exception as exc:
            append_job_warning(
                job,
                f"combined audio filter chain failed; retrying step-by-step: {concise_error(exc)}",
            )

    applied_count = 0
    for index, step in enumerate(audio_filters, start=1):
        filter_string = step.get("filter")
        if not filter_string:
            continue

        filter_string = quote_audio_filter_expressions(
            replace_context_time_references(filter_string, context or {})
        )
        if not filter_string:
            continue

        output_path = next_media_path(job_dir, f"audio_filter_{index:02d}")
        try:
            run_audio_filter_step(current, output_path, filter_string)
            current = output_path
            applied_count += 1
            if job is not None:
                job["audio_filter_execution"] = {
                    "mode": "step_by_step",
                    "planned_filter_count": filter_count,
                    "applied_filter_count": applied_count,
                    "last_successful_step": index,
                }
            continue
        except Exception as exc:
            append_job_warning(
                job,
                f"audio filter step {index} failed; attempting fallback: {concise_error(exc)}",
            )

        fallback_filter = fallback_audio_filter_chain(step, filter_string)
        fallback_output_path = next_media_path(job_dir, f"audio_filter_{index:02d}_fallback")
        try:
            run_audio_filter_step(current, fallback_output_path, fallback_filter)
            append_job_warning(job, f"audio filter step {index} used fallback: {fallback_filter}")
            current = fallback_output_path
            applied_count += 1
            if job is not None:
                job["audio_filter_execution"] = {
                    "mode": "step_by_step",
                    "planned_filter_count": filter_count,
                    "applied_filter_count": applied_count,
                    "last_successful_step": index,
                }
        except Exception as fallback_exc:
            append_job_warning(
                job,
                f"audio filter step {index} fallback failed; skipped step: {concise_error(fallback_exc)}",
            )

    if filter_count and not applied_count:
        fail_execution_phase(
            job,
            "audio_filter",
            f"audio filter phase planned {filter_count} filter(s), but none could be applied",
        )

    return current


def uploaded_audio_path(job):
    audio_path = job.get("audio_path")
    if not audio_path or not Path(audio_path).exists():
        return None
    return Path(audio_path)


def normalized_volume(value, default):
    try:
        volume = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(3.0, volume))


def apply_replace_audio(input_path, job, job_dir, params=None):
    audio_path = uploaded_audio_path(job)
    if not audio_path:
        append_job_warning(job, "replace_audio skipped because no uploaded audio file is available")
        return input_path

    output_path = next_media_path(job_dir, "uploaded_audio")
    run_command([
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-stream_loop", "-1",
        "-i", str(audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        str(output_path),
    ])
    if job is not None:
        job["uploaded_audio_action"] = {"type": "replace_audio"}
    return output_path


def apply_mix_uploaded_audio(input_path, job, job_dir, params):
    audio_path = uploaded_audio_path(job)
    if not audio_path:
        append_job_warning(job, "mix_uploaded_audio skipped because no uploaded audio file is available")
        return input_path
    if not ffprobe_has_audio(input_path):
        append_job_warning(job, "mix_uploaded_audio found no original audio; using uploaded audio as the only soundtrack")
        return apply_replace_audio(input_path, job, job_dir, params)

    original_volume = normalized_volume(params.get("original_volume"), 1.0)
    music_volume = normalized_volume(params.get("music_volume"), 0.35)
    duck = bool(params.get("duck"))
    output_path = next_media_path(job_dir, "mixed_uploaded_audio")
    if duck:
        filter_complex = (
            f"[0:a]volume={original_volume:.3f},asplit=2[voice][side];"
            f"[1:a]volume={music_volume:.3f}[music];"
            "[music][side]sidechaincompress=threshold=0.03:ratio=6:release=500[ducked];"
            "[voice][ducked]amix=inputs=2:duration=first:dropout_transition=2,"
            "loudnorm=I=-14:TP=-1.5:LRA=11[a]"
        )
    else:
        filter_complex = (
            f"[0:a]volume={original_volume:.3f}[original];"
            f"[1:a]volume={music_volume:.3f}[music];"
            "[original][music]amix=inputs=2:duration=first:dropout_transition=2,"
            "loudnorm=I=-14:TP=-1.5:LRA=11[a]"
        )
    run_command([
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-stream_loop", "-1",
        "-i", str(audio_path),
        "-filter_complex", filter_complex,
        "-map", "0:v:0",
        "-map", "[a]",
        "-shortest",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        str(output_path),
    ])
    if job is not None:
        job["uploaded_audio_action"] = {
            "type": "mix_uploaded_audio",
            "original_volume": original_volume,
            "music_volume": music_volume,
            "duck": duck,
        }
    return output_path


def plan_special_types(plan):
    return {
        step.get("type")
        for step in plan.get("special", [])
        if isinstance(step, dict)
    }


def should_attach_uploaded_audio_by_default(input_path, job, plan):
    if not uploaded_audio_path(job):
        return False
    if plan_special_types(plan) & {"remove_audio", "replace_audio", "mix_uploaded_audio"}:
        return False
    return not ffprobe_has_audio(input_path)


def apply_default_uploaded_audio(input_path, job, job_dir):
    output_path = apply_replace_audio(input_path, job, job_dir, {})
    if job is not None and job.get("uploaded_audio_action", {}).get("type") == "replace_audio":
        job["uploaded_audio_action"] = {"type": "default_uploaded_audio"}
    return output_path


def normalized_final_encode_settings(final_settings, job=None):
    normalized, changed = architecture_normalize_final_encode_settings(final_settings)
    if changed:
        append_job_warning(
            job,
            f"final encode settings normalized: {', '.join(changed)}",
        )
    return normalized


def final_encode_command(input_path, output_path, final_settings):
    command = ["ffmpeg", "-y", "-i", str(input_path)]

    width = final_settings.get("width")
    height = final_settings.get("height")
    if width and height:
        command.extend([
            "-vf",
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1",
        ])

    command.extend([
        "-c:v", final_settings["vcodec"],
        "-crf", str(final_settings["crf"]),
        "-preset", final_settings["preset"],
        "-c:a", final_settings["acodec"],
        "-b:a", final_settings["audio_bitrate"],
        str(output_path),
    ])
    return command


def final_stream_copy_command(input_path, output_path):
    return [
        "ffmpeg", "-y", "-i", str(input_path),
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c", "copy",
        "-movflags", "+faststart",
        str(output_path),
    ]


def final_settings_are_default_copy_safe(final_settings):
    default_settings = default_final_encode()
    return all(final_settings.get(key) == value for key, value in default_settings.items()) and not (
        final_settings.get("width") or final_settings.get("height")
    )


def final_streams_are_copy_compatible(input_path, final_settings):
    if not final_settings_are_default_copy_safe(final_settings):
        return False

    streams = ffprobe_streams(input_path)
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if not video_streams:
        return False
    if video_streams[0].get("codec_name") != "h264":
        return False
    return not audio_streams or audio_streams[0].get("codec_name") == "aac"


def final_encode(input_path, job_dir, final_settings, job=None):
    output_path = job_dir / "linguist_output.mp4"
    settings = normalized_final_encode_settings(final_settings, job)
    try:
        if final_streams_are_copy_compatible(input_path, settings):
            try:
                run_command(final_stream_copy_command(input_path, output_path))
                if job is not None:
                    job["final_encode_execution"] = {"mode": "stream_copy"}
                return output_path
            except Exception as copy_exc:
                append_job_warning(
                    job,
                    f"final stream copy failed; retrying encode: {concise_error(copy_exc)}",
                )
        run_command(final_encode_command(input_path, output_path, settings))
        if job is not None:
            job["final_encode_execution"] = {"mode": "encode"}
    except Exception as exc:
        append_job_warning(
            job,
            f"final encode failed; retrying safe defaults: {concise_error(exc)}",
        )
        safe_settings = default_final_encode()
        run_command(final_encode_command(input_path, output_path, safe_settings))
        if job is not None:
            job["final_encode_execution"] = {"mode": "safe_default_encode"}
    return output_path


def set_manifest_phase(job, status, step_type=None, legacy_section=None, legacy_index=None, error=None):
    if job is None or not job.get("execution_manifest"):
        return
    job["execution_manifest"] = mark_manifest_step_status(
        job.get("execution_manifest"),
        status,
        step_type=step_type,
        legacy_section=legacy_section,
        legacy_index=legacy_index,
        error=error,
    )
    persist_job(job)


def run_manifest_phase(job, step_type, legacy_section, callback):
    set_manifest_phase(job, "running", step_type=step_type, legacy_section=legacy_section)
    try:
        result = callback()
    except Exception as exc:
        set_manifest_phase(job, "error", step_type=step_type, legacy_section=legacy_section, error=exc)
        raise
    set_manifest_phase(job, "complete", step_type=step_type, legacy_section=legacy_section)
    return result


def execute_pipeline(job, plan):
    job_dir = Path(job["video_path"]).parent
    current = Path(job["video_path"])
    if should_attach_uploaded_audio_by_default(current, job, plan):
        current = apply_default_uploaded_audio(current, job, job_dir)
    if plan.get("analysis"):
        context = run_manifest_phase(
            job,
            "analysis",
            "analysis",
            lambda: run_analysis(plan, job, job_dir, current),
        )
    else:
        context = run_analysis(plan, job, job_dir, current)
    if plan.get("special"):
        current = run_manifest_phase(
            job,
            "special",
            "special",
            lambda: apply_special(current, job_dir, plan.get("special", []), job, context),
        )
    if plan.get("video_filters"):
        current = run_manifest_phase(
            job,
            "video_filter",
            "video_filters",
            lambda: apply_video_filters(current, job_dir, plan.get("video_filters", []), context, job),
        )
    current = apply_output_aspect_enforcement(current, job_dir, job, plan)
    if plan.get("audio_filters"):
        current = run_manifest_phase(
            job,
            "audio_filter",
            "audio_filters",
            lambda: apply_audio_filters(current, job_dir, plan.get("audio_filters", []), context, job),
        )
    output_path = run_manifest_phase(
        job,
        "final_encode",
        "final_encode",
        lambda: final_encode(current, job_dir, plan["final_encode"], job),
    )
    return str(output_path.resolve())


def operation_count(plan):
    return (
        len(plan.get("analysis", []))
        + len(plan.get("video_filters", []))
        + len(plan.get("audio_filters", []))
        + len(plan.get("special", []))
    )


def clear_previous_execution_fields(job):
    for key in [
        "error",
        "failed_at",
        "completed_at",
        "output_path",
        "output_name",
        "output_size_bytes",
        "processing_seconds",
        "repair_packet",
        "result_inspection",
        "final_encode_execution",
        "video_filter_execution",
        "audio_filter_execution",
        "uploaded_audio_action",
    ]:
        job.pop(key, None)


def record_plan_rejection(job, command_text, plan, internal_plan, validation):
    job["command"] = command_text
    job["plan"] = plan
    job["internal_plan"] = internal_plan
    job["plan_validation"] = validation
    transition_job_status(job, STATUS_PLAN_REJECTED, reason="plan_validation_failed", force=True)
    job["error"] = "plan validation failed"
    job["updated_at"] = datetime.now(timezone.utc).isoformat()
    persist_job(job)


def record_plan_acceptance(job, command_text, plan, internal_plan, validation):
    job["command"] = command_text
    job["plan"] = plan
    job["internal_plan"] = internal_plan
    job["plan_validation"] = validation
    job["execution_manifest"] = create_execution_manifest(internal_plan)
    transition_job_status(job, STATUS_PROCESSING, reason="command_accepted_for_execution", force=True)
    job["updated_at"] = datetime.now(timezone.utc).isoformat()
    job["processing_started_at"] = datetime.now(timezone.utc).isoformat()
    job["operations_count"] = operation_count(plan)
    clear_previous_execution_fields(job)
    persist_job(job)


def validation_repair_prompt(command_text, invalid_plan, validation):
    feedback = {
        "issues": validation.get("issues", []),
        "warnings": validation.get("warnings", []),
        "confidence": validation.get("confidence"),
        "confidence_band": validation.get("confidence_band"),
        "repair_summary": validation_repair_summary(validation),
    }
    return "\n\n".join([
        "Repair the rejected Linguist edit plan.",
        "Keep the user's intent, but return a corrected plan that passes the active architecture contract.",
        "Do not use unsupported special operations, unsupported analysis functions, blocked capabilities, or filter_complex-only filters in direct video_filters/audio_filters.",
        "Original user editing request:",
        str(command_text or ""),
        "Rejected plan JSON:",
        json.dumps(invalid_plan, ensure_ascii=False, sort_keys=True),
        "Validation feedback JSON:",
        json.dumps(feedback, ensure_ascii=False, sort_keys=True),
    ])


def repair_plan_with_validation_feedback(command_text, invalid_plan, validation, job=None):
    capabilities = runtime_capabilities()
    runtime_note = runtime_planning_prompt(capabilities)
    architecture_hash = architecture_fingerprint()
    repair_prompt = augment_prompt_for_capabilities(
        validation_repair_prompt(command_text, invalid_plan, validation),
        runtime_note,
        architecture_prompt_contract(),
    )
    repaired_plan = call_nim(repair_prompt)
    repaired_plan = align_plan_with_command(repaired_plan, command_text)
    cache_key = planner_cache_key(command_text, runtime_note, architecture_hash)
    store_cached_plan(cache_key, repaired_plan, "nim_repair", architecture_hash)
    if job is not None:
        job["planner_repair"] = {
            "attempted": True,
            "source": "nim_repair",
            "previous_error_codes": validation_error_codes(validation),
            "architecture_fingerprint": architecture_hash,
        }
        job["planner"] = "nim_repair"
        job["planner_cache"] = {
            "status": "stored_repaired_plan",
            "architecture_fingerprint": architecture_hash,
            "public_plan_contract_fingerprint": public_plan_contract_fingerprint(),
            "special_param_contract_fingerprint": special_param_contract_fingerprint(),
        }
    return repaired_plan


def prepare_command_plan_with_repair(command_text, job, capabilities):
    proposed_plan = align_plan_with_command(build_plan(command_text, job), command_text)
    plan, internal_plan = prepare_production_plan(
        command_text,
        proposed_plan,
        job,
        capabilities,
    )
    validation = internal_plan["validation"]
    if validation_allows_execution(internal_plan):
        if job is not None:
            job.pop("planner_repair", None)
        return plan, internal_plan

    if not validation_is_model_repairable(validation):
        record_planner_fallback(job, "validation_not_repairable", validation_error_codes(validation))
        if job is not None:
            job["planner_repair"] = {
                "attempted": False,
                "reason": "validation_error_not_model_repairable",
                "previous_error_codes": validation_error_codes(validation),
            }
        return plan, internal_plan

    try:
        repaired_plan = repair_plan_with_validation_feedback(command_text, proposed_plan, validation, job)
    except Exception as exc:
        record_planner_fallback(job, "model_repair_failed", concise_error(exc))
        append_job_warning(job, f"planner validation repair failed: {concise_error(exc)}")
        if job is not None:
            job["planner_repair"] = {
                "attempted": True,
                "source": "nim_repair",
                "ok": False,
                "error": concise_error(exc),
                "previous_error_codes": validation_error_codes(validation),
            }
        return plan, internal_plan

    repaired_plan, repaired_internal_plan = prepare_production_plan(
        command_text,
        repaired_plan,
        job,
        runtime_capabilities(),
    )
    repaired_validation = repaired_internal_plan["validation"]
    if job is not None:
        job.setdefault("planner_repair", {}).update({
            "ok": validation_allows_execution(repaired_internal_plan),
            "repaired_error_codes": validation_error_codes(repaired_validation),
        })
    if validation_allows_execution(repaired_internal_plan):
        if job is not None:
            job.pop("planner_fallback", None)
        append_job_warning(job, "planner validation repair succeeded")
    else:
        record_planner_fallback(job, "model_repair_failed", validation_error_codes(repaired_validation))
    return repaired_plan, repaired_internal_plan


def execute_job_async(job_id):
    job = get_job_record(job_id)
    if job is None:
        return

    try:
        plan = job["plan"]
        job["execution_manifest"] = mark_manifest_running(job.get("execution_manifest"))
        persist_job(job)
        output_path = Path(execute_pipeline(job, plan))
        inspection = inspect_output_artifact(
            output_path,
            expected_plan=job.get("internal_plan"),
            duration_probe=ffprobe_duration,
        )
        job["result_inspection"] = inspection
        if not inspection.get("ok"):
            raise RuntimeError(f"output inspection failed: {inspection.get('issues')}")

        job["output_path"] = str(output_path)
        job["output_name"] = output_path.name
        job["output_size_bytes"] = output_path.stat().st_size if output_path.exists() else None
        transition_job_status(job, STATUS_COMPLETE, reason="execution_complete")
        job["execution_manifest"] = mark_manifest_complete(job.get("execution_manifest"))
        job.pop("error", None)
        job.pop("failed_at", None)
        job.pop("repair_packet", None)
        job["completed_at"] = datetime.now(timezone.utc).isoformat()
        started_at = datetime.fromisoformat(job["processing_started_at"])
        completed_at = datetime.fromisoformat(job["completed_at"])
        job["processing_seconds"] = round((completed_at - started_at).total_seconds(), 2)
        persist_job(job)
    except Exception as exc:
        transition_job_status(job, STATUS_ERROR, reason="execution_exception", force=True)
        job["error"] = str(exc)
        job["execution_manifest"] = mark_manifest_failed(job.get("execution_manifest"), exc)
        job["repair_packet"] = repair_packet_from_exception(exc, job.get("internal_plan"))
        job["failed_at"] = datetime.now(timezone.utc).isoformat()
        persist_job(job)


@app.after_request
def add_cors_headers(response):
    if request.headers.get("Origin") == ALLOWED_ORIGIN:
        response.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
        response.headers["Vary"] = "Origin"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(_exc):
    return jsonify({"error": f"upload too large; max {MAX_UPLOAD_MB} MB"}), 413


@app.route("/upload", methods=["POST", "OPTIONS"])
def upload():
    if request.method == "OPTIONS":
        return ("", 204)

    try:
        video = request.files.get("video")
        if video is None or not video.filename:
            return jsonify({"error": "video file required"}), 400

        video_name = original_filename(video)
        if not video_name:
            return jsonify({"error": "video file required"}), 400

        audio = request.files.get("audio")
        audio_name = original_filename(audio) if audio and audio.filename else None

        job_id = uuid4().hex[:10]
        job_dir = UPLOAD_ROOT / job_id
        job_dir.mkdir(parents=True, exist_ok=False)

        video_path = job_dir / video_name
        video.save(video_path)

        audio_path = None
        if audio and audio_name:
            audio_path = job_dir / audio_name
            audio.save(audio_path)

        try:
            validate_uploaded_media(video_path, audio_path)
        except ValueError as exc:
            shutil.rmtree(job_dir, ignore_errors=True)
            return jsonify({"error": str(exc)}), 400

        job = {
            "job_id": job_id,
            "video_path": str(video_path.resolve()),
            "audio_path": str(audio_path.resolve()) if audio_path else None,
            "video_name": video_name,
            "audio_name": audio_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        transition_job_status(job, STATUS_UPLOADED, reason="upload_accepted")
        remember_job(job)

        return jsonify(
            {
                "job_id": job_id,
                "status": STATUS_UPLOADED,
                "video_name": video_name,
                "audio_name": audio_name,
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/command", methods=["POST", "OPTIONS"])
def command():
    if request.method == "OPTIONS":
        return ("", 204)

    active_job = None
    try:
        data = request.get_json(silent=True) or {}
        job_id = data.get("job_id")
        command_text = data.get("command") or data.get("prompt") or data.get("text")

        if not job_id:
            return jsonify({"error": "job_id required"}), 400
        if not command_text:
            return jsonify({"error": "command required"}), 400

        integrity = architecture_integrity()
        if not integrity["ok"]:
            return jsonify({"error": "backend architecture integrity failed", "architecture_integrity": integrity}), 503
        if not job_queue.can_accept():
            return jsonify({
                "error": "execution queue is full",
                "queue": job_queue.stats(),
                "retry_after": "try again after current jobs finish",
            }), 503

        active_job, claim_status = claim_job_for_command(job_id, command_text)
        if claim_status == "not_found":
            return jsonify({"error": "job not found"}), 404
        if claim_status == "conflict":
            return jsonify({
                "error": "job is not ready for a new command",
                "job_id": job_id,
                "status": active_job.get("status"),
                "retry_after": "wait until the current job is complete or failed",
            }), 409

        plan, internal_plan = prepare_command_plan_with_repair(
            command_text,
            active_job,
            runtime_capabilities(),
        )
        validation = internal_plan["validation"]
        for fix in validation.get("fixes", []):
            append_job_warning(active_job, f"planner repair: {fix}")
        for warning in validation.get("warnings", []):
            append_job_warning(active_job, f"planner warning: {warning.get('message', warning)}")
        if not validation_allows_execution(internal_plan):
            record_plan_rejection(active_job, command_text, plan, internal_plan, validation)
            return jsonify({"error": "plan validation failed", "validation": validation, "plan": plan}), 400

        record_plan_acceptance(active_job, command_text, plan, internal_plan, validation)
        future = job_queue.submit(execute_job_async, job_id)
        if future is None:
            error = RuntimeError("execution queue is full")
            transition_job_status(active_job, STATUS_ERROR, reason="execution_queue_full", force=True)
            active_job["error"] = str(error)
            active_job["execution_manifest"] = mark_manifest_failed(active_job.get("execution_manifest"), error)
            active_job["failed_at"] = datetime.now(timezone.utc).isoformat()
            persist_job(active_job)
            return jsonify({
                "error": "execution queue is full",
                "queue": job_queue.stats(),
                "retry_after": "try again after current jobs finish",
            }), 503
        return jsonify(plan)
    except Exception as exc:
        if active_job is not None:
            transition_job_status(active_job, STATUS_ERROR, reason="command_exception", force=True)
            active_job["error"] = str(exc)
            persist_job(active_job)
        return jsonify({"error": str(exc)}), 500


@app.route("/job/<job_id>", methods=["GET", "OPTIONS"])
def get_job(job_id):
    if request.method == "OPTIONS":
        return ("", 204)

    try:
        job = get_job_record(job_id)
        if job is None:
            return jsonify({"error": "job not found"}), 404
        return jsonify(job)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/download/<job_id>", methods=["GET", "OPTIONS"])
def download(job_id):
    if request.method == "OPTIONS":
        return ("", 204)

    try:
        job = get_job_record(job_id)
        if job is None:
            return jsonify({"error": "job not found"}), 404
        output_path = job.get("output_path")
        if not output_path or not Path(output_path).exists():
            return jsonify({"error": "output not available"}), 404
        return send_file(output_path, as_attachment=True, download_name=Path(output_path).name)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/live", methods=["GET", "OPTIONS"])
def live():
    if request.method == "OPTIONS":
        return ("", 204)

    return jsonify({
        "status": "alive",
        "server_time": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/ready", methods=["GET", "OPTIONS"])
def ready():
    if request.method == "OPTIONS":
        return ("", 204)

    try:
        upload_root = upload_root_health()
        provider = nim_provider.metadata()
        integrity = architecture_integrity()
        queue = job_queue.stats()
        worker_pool_ready = bool(queue.get("accepting"))
        ok = bool(
            upload_root.get("ok")
            and provider.get("configured")
            and integrity.get("ok")
            and worker_pool_ready
        )
        status_code = 200 if ok else 503
        return jsonify({
            "status": "ready" if ok else "not_ready",
            "upload_root": upload_root,
            "ai_provider": provider,
            "architecture_integrity": integrity,
            "worker_pool_ready": worker_pool_ready,
            "execution_queue": queue,
        }), status_code
    except Exception as exc:
        return jsonify({"status": "not_ready", "error": str(exc)}), 503


@app.route("/health", methods=["GET", "OPTIONS"])
def health():
    if request.method == "OPTIONS":
        return ("", 204)

    try:
        capabilities = runtime_capabilities(force=request.args.get("refresh") == "1")
        return jsonify({
            "status": "ok",
            "jobs_in_memory": in_memory_job_count(),
            "jobs_persisted": persisted_job_count(),
            "upload_root": str(UPLOAD_ROOT),
            "max_upload_mb": MAX_UPLOAD_MB,
            "worker_count": WORKER_COUNT,
            "queue_max_pending": QUEUE_MAX_PENDING,
            "execution_queue": job_queue.stats(),
            "command_timeout_seconds": COMMAND_TIMEOUT_SECONDS,
            "media_runner": media_runner.stats(),
            "ai_provider": nim_provider.metadata(),
            "runtime_capability_cache": runtime_capability_cache.stats(),
            "plan_cache": plan_cache_stats(),
            "architecture": architecture_summary(),
            "public_plan_contract": {
                **public_plan_contract(),
                "fingerprint": public_plan_contract_fingerprint(),
            },
            "special_param_contract": {
                **special_param_contract(),
                "fingerprint": special_param_contract_fingerprint(),
            },
            "executor_implementation": special_executor_summary(),
            "architecture_integrity": architecture_integrity(),
            "job_lifecycle": job_lifecycle_summary(),
            "capabilities": capabilities,
            "runtime_operation_contract": runtime_operation_contract(capabilities.get("executor") or {}),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    print(f"Linguist Backend running on port {SERVER_PORT}")
    app.run(host=SERVER_HOST, port=SERVER_PORT)
