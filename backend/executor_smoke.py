#!/usr/bin/env python3
import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import server


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = ROOT / "test_assets" / "suite_video.mp4"
DEFAULT_AUDIO = ROOT / "test_assets" / "suite_beats.mp3"
DEFAULT_REPORT_DIR = Path(__file__).resolve().with_name("test_reports")


BASE_ENCODE = {
    "vcodec": "libx264",
    "crf": 22,
    "preset": "fast",
    "acodec": "aac",
    "audio_bitrate": "192k",
}


CASES = {
    "red_isolation_repair": {
        "command": "make everything black and white except red tones",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "plan": {
            "intent": "Black and white except red tones",
            "video_filters": [
                {
                    "description": "Color grade to black and white except red tones",
                    "filter": (
                        "colorchannelmixer=rr=0.2126:rg=0.7152:rb=0.0722:"
                        "gr=0.2126:gg=0.7152:gb=0.0722:br=0.2126:bg=0.7152:bb=0.0722,"
                        "eq=contrast=1.05:saturation=0.75,min(255,255*(G>128))):"
                        "b='max(0,min(255,255*(B>128))'"
                    ),
                }
            ],
            "final_encode": BASE_ENCODE,
        },
    },
    "privacy_blur_repair": {
        "command": "blur the center like a privacy blur and darken everything outside it",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "plan": {
            "intent": "Blur center like privacy blur and darken outside",
            "video_filters": [
                {"filter": "pixelize=width=20:height=20"},
                {
                    "filter": (
                        "geq=r='255*(X/W-0.5)*(X/W-0.5)+Y*(Y/H-0.5)*(Y/H-0.5)<0.25*0.25':"
                        "g='255*(X/W-0.5)*(X/W-0.5)+Y*(Y/H-0.5)*(Y/H-0.5)<0.25*0.25':"
                        "b='255*(X/W-0.5)*(X/W-0.5)+Y*(Y/H-0.5)*(Y/H-0.5)<0.25*0.25'"
                    )
                },
            ],
            "final_encode": BASE_ENCODE,
        },
    },
    "face_privacy_blur_special": {
        "command": "blur all faces for privacy",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "expect_face_privacy_regions_min": 3,
        "plan": {
            "intent": "Apply safe non-tracking privacy regions over likely face areas",
            "video_filters": [
                {"description": "Generated face blur that should be stripped", "filter": "delogo=x=450:y=120:w=300:h=240:show=0"},
                {"description": "Keep contrast", "filter": "eq=contrast=1.05"},
            ],
            "special": [{"type": "face_privacy_blur", "params": {"target": "faces", "layout": "group"}}],
            "final_encode": BASE_ENCODE,
        },
    },
    "lower_third_punctuation_repair": {
        "command": "add a lower third caption that says Dr. Rao: we're live, don't blink and fade it out after four seconds",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "plan": {
            "intent": "Lower third caption with fade out",
            "video_filters": [
                {
                    "filter": (
                        "drawtext=text='Dr. Rao: we're live,don't blink':fontcolor=white:fontsize=48:"
                        "x=(w-text_w)/2:y=h-(2*text_h):enable='between(t,0,4)':"
                        "alpha=1-if(gte(t,4),1,0.25*(4-t))"
                    )
                }
            ],
            "final_encode": BASE_ENCODE,
        },
    },
    "blue_square_format_repair": {
        "command": "make everything black and white except blue lights, turn it into a square 1:1 post, and add a clean bass boost",
        "expect": {"video": True, "audio": True, "width": 1080, "height": 1080},
        "plan": {
            "intent": "Black and white except blue lights, square 1:1 post, clean bass boost",
            "video_filters": [
                {"description": "Black and white except blue lights", "filter": server.COLOR_HOLD_FILTERS["blue"]},
                {
                    "description": "Square 1:1 post",
                    "filter": "crop=iw/2:ih/2:(iw-iw/2)/2:(ih-ih/2)/2,scale=-1:1",
                },
            ],
            "audio_filters": [
                {"description": "Clean bass boost", "filter": "equalizer=f=60:width_type=o:width=2:g=6"}
            ],
            "final_encode": BASE_ENCODE,
        },
    },
    "beat_shake": {
        "command": "shake the frame violently on every beat of the uploaded audio",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "plan": {
            "intent": "Violent frame shake triggered on every detected beat",
            "analysis": [
                {"tool": "librosa", "function": "beat_track", "store_as": "beat_times"}
            ],
            "video_filters": [
                {
                    "description": "Hard frame shake on every beat via crop offset",
                    "filter": "crop=iw-60:ih-60:30+30*sin(30*t):30+30*cos(24*t),scale=iw+60:ih+60",
                    "requires_context": "beat_times",
                    "timing": "per_beat",
                }
            ],
            "audio_filters": [
                {"description": "Add bass impact", "filter": "equalizer=f=60:width_type=o:width=2:g=6"}
            ],
            "final_encode": BASE_ENCODE,
        },
    },
    "beat_cut_jump_cuts": {
        "command": "make hard jump cuts on every beat of the uploaded audio",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "expect_duration_min": 0.5,
        "expect_duration_max": 5.5,
        "expect_beat_cuts_min": 2,
        "plan": {
            "intent": "Create structural jump cuts on detected beats",
            "analysis": [
                {"tool": "librosa", "function": "beat_track", "store_as": "beat_times"}
            ],
            "video_filters": [{"description": "Slight clarity lift", "filter": "eq=contrast=1.04:saturation=1.02"}],
            "final_encode": BASE_ENCODE,
        },
    },
    "beat_shake_no_audio_fallback": {
        "command": "shake the frame violently on every beat even though this upload has no audio",
        "use_audio": False,
        "strip_audio": True,
        "expect": {"video": True, "width": 1280, "height": 720},
        "expect_warning": "audio analysis fallback used synthetic timing",
        "plan": {
            "intent": "Synthetic beat timing fallback for a silent upload",
            "analysis": [
                {"tool": "librosa", "function": "beat_track", "store_as": "beat_times"}
            ],
            "video_filters": [
                {
                    "description": "Hard frame shake on synthetic beat timing",
                    "filter": "crop=iw-60:ih-60:30+30*sin(30*t):30+30*cos(24*t),scale=iw+60:ih+60",
                    "requires_context": "beat_times",
                    "timing": "per_beat",
                }
            ],
            "final_encode": BASE_ENCODE,
        },
    },
    "invalid_video_filter_recovery": {
        "command": "apply a strange impossible visual effect and still produce an edited video",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "expect_warning": "video filter step 1 used fallback",
        "plan": {
            "intent": "Recover from a bad generated video filter",
            "video_filters": [
                {"description": "Impossible generated visual filter", "filter": "not_a_real_filter=insane=1"}
            ],
            "final_encode": BASE_ENCODE,
        },
    },
    "invalid_audio_filter_recovery": {
        "command": "apply a strange impossible audio effect and still produce an edited video",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "expect_warning": "audio filter step 1 used fallback",
        "plan": {
            "intent": "Recover from a bad generated audio filter",
            "audio_filters": [
                {"description": "Impossible generated audio filter", "filter": "notarealaudiofilter=1"}
            ],
            "final_encode": BASE_ENCODE,
        },
    },
    "audio_noise_filter_repair": {
        "command": "make the audio sound like an old noisy telephone line",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "forbid_warning": "audio filter step",
        "plan": {
            "intent": "Repair generated video noise filter inside audio chain",
            "audio_filters": [
                {"description": "Old telephone line noise", "filter": "noise=alls=10:allf=t+u"}
            ],
            "final_encode": BASE_ENCODE,
        },
    },
    "audio_denoise_dialogue": {
        "command": "remove background noise and make the dialogue clearer",
        "fixture": "noisy_dialogue",
        "use_audio": False,
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "plan": {
            "intent": "Clean noisy dialogue",
            "video_filters": [
                {"description": "Keep picture unchanged except slight clarity", "filter": "eq=contrast=1.02"}
            ],
            "final_encode": BASE_ENCODE,
        },
    },
    "audio_filter_no_audio_skip": {
        "command": "normalize the audio even though this upload has no audio stream",
        "use_audio": False,
        "strip_audio": True,
        "expect": {"video": True, "width": 1280, "height": 720},
        "expect_warning": "audio filters skipped because no audio stream is available",
        "plan": {
            "intent": "Skip audio processing cleanly when no audio stream exists",
            "audio_filters": [
                {"description": "Normalize missing audio", "filter": "loudnorm=I=-14:TP=-1.5:LRA=11"}
            ],
            "final_encode": BASE_ENCODE,
        },
    },
    "replace_uploaded_audio_special": {
        "command": "replace the original audio with the uploaded music track",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "expect_uploaded_audio_action": "replace_audio",
        "plan": {
            "intent": "Replace original audio with uploaded music",
            "video_filters": [{"description": "Keep picture unchanged", "filter": "eq=contrast=1.02"}],
            "final_encode": BASE_ENCODE,
        },
    },
    "mix_uploaded_audio_special": {
        "command": "add the uploaded audio as background music under the dialogue and duck it behind the voice",
        "fixture": "video_with_audio",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "expect_uploaded_audio_action": "mix_uploaded_audio",
        "plan": {
            "intent": "Mix uploaded music under original dialogue",
            "video_filters": [{"description": "Keep picture unchanged", "filter": "eq=contrast=1.02"}],
            "special": [{"type": "replace_audio", "params": {}}],
            "final_encode": BASE_ENCODE,
        },
    },
    "remove_audio_special": {
        "command": "make the clip completely silent and add a little contrast",
        "expect": {"video": True, "audio": False, "width": 1280, "height": 720},
        "plan": {
            "intent": "Remove all audio and keep a visible contrast edit",
            "video_filters": [{"description": "Mild contrast", "filter": "eq=contrast=1.08:saturation=1.02"}],
            "audio_filters": [{"description": "Generated audio edit that should be stripped", "filter": "loudnorm=I=-14:TP=-1.5:LRA=11"}],
            "special": [{"type": "remove_audio", "params": {}}],
            "final_encode": BASE_ENCODE,
        },
    },
    "picture_in_picture_special": {
        "command": "create a picture in picture duplicate of the video in the top right corner",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "plan": {
            "intent": "Render a duplicate picture-in-picture overlay in the top right",
            "special": [{"type": "picture_in_picture", "params": {"position": "top_right", "scale": 0.32}}],
            "final_encode": BASE_ENCODE,
        },
    },
    "split_screen_mirror_special": {
        "command": "create a split screen mirrored duplicate, left side normal and right side flipped",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "plan": {
            "intent": "Render left normal and right mirrored split screen with divider",
            "special": [{"type": "split_screen_mirror", "params": {"divider_color": "white"}}],
            "final_encode": BASE_ENCODE,
        },
    },
    "speed_25_percent_faster": {
        "command": "make the whole clip 25 percent faster but keep audio pitch natural",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "expect_duration_min": 4.4,
        "expect_duration_max": 5.2,
        "plan": {
            "intent": "Increase speed by 25 percent while keeping natural pitch",
            "video_filters": [{"filter": "setpts=0.8*PTS,eq=contrast=1.02"}],
            "audio_filters": [{"filter": "rubberband=tempo=1.25:pitch=1.0"}],
            "special": [{"type": "speed_ramp", "params": {"slow_factor": 0.75, "fast_factor": 1.25}}],
            "final_encode": BASE_ENCODE,
        },
    },
    "pitch_cave_alignment": {
        "command": "make the audio sound like a cave echo and lower the pitch three semitones",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "plan": {
            "intent": "Cave echo with a lower voice pitch",
            "audio_filters": [
                {"description": "Cave-like echo", "filter": "aecho=0.8:0.4:500:0.4"},
                {"description": "Generated pitch filter to normalize", "filter": "rubberband=tempo=1.0:pitch=0.75"},
            ],
            "final_encode": BASE_ENCODE,
        },
    },
    "dream_memory_slow_motion": {
        "command": "make it feel like a memory fading, warm, dreamlike, slow, with reverb",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "expect_duration_min": 7.0,
        "expect_duration_max": 9.5,
        "expect_plan_substrings": ["speed_ramp"],
        "plan": {
            "intent": "Dreamlike memory look missing speed change",
            "video_filters": [
                {"description": "Warm memory grade", "filter": "colorbalance=rs=0.12:rm=0.18:rh=0.22:bs=-0.12:bm=-0.06:bh=-0.18"},
                {"description": "Soft dream blur", "filter": "gblur=sigma=0.9,vignette=angle=PI/3.5"},
            ],
            "audio_filters": [{"description": "Light reverb", "filter": "aecho=0.8:0.4:350:0.35"}],
            "final_encode": BASE_ENCODE,
        },
    },
    "trim_remove_first_seconds": {
        "command": "remove the first 2 seconds and add a little contrast",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "expect_duration_min": 3.5,
        "expect_duration_max": 4.5,
        "plan": {
            "intent": "Trim away the opening two seconds",
            "video_filters": [{"description": "Mild contrast", "filter": "eq=contrast=1.08:saturation=1.02"}],
            "special": [{"type": "trim", "params": {"start": 2.0}}],
            "final_encode": BASE_ENCODE,
        },
    },
    "trim_keep_range": {
        "command": "keep only from 1 second to 3 seconds",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "expect_duration_min": 1.5,
        "expect_duration_max": 2.5,
        "plan": {
            "intent": "Keep a selected time range",
            "special": [{"type": "trim", "params": {"start": 1.0, "end": 3.0}}],
            "final_encode": BASE_ENCODE,
        },
    },
    "invalid_special_recovery": {
        "command": "try an unsupported special operation and still produce an edited video",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "expect_warning": "ignored unsupported type",
        "plan": {
            "intent": "Recover from unsupported special operation",
            "special": [{"type": "teleport_frames", "params": {}}],
            "video_filters": [{"description": "Visible edit", "filter": "eq=contrast=1.08:saturation=1.05"}],
            "final_encode": BASE_ENCODE,
        },
    },
    "broken_pitch_shift_recovery": {
        "command": "try a broken pitch shift and still produce an edited video",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "expect_warning": "special step 1 (pitch_shift) failed",
        "plan": {
            "intent": "Recover from broken pitch shift params",
            "special": [{"type": "pitch_shift", "params": {"semitones": "not-a-number"}}],
            "video_filters": [{"description": "Visible edit", "filter": "eq=contrast=1.08:saturation=1.05"}],
            "final_encode": BASE_ENCODE,
        },
    },
    "invalid_final_encode_recovery": {
        "command": "use impossible export settings and still produce a downloadable video",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "expect_warning": "final encode settings normalized",
        "plan": {
            "intent": "Recover from invalid generated final encode settings",
            "video_filters": [{"description": "Visible edit", "filter": "eq=contrast=1.08:saturation=1.05"}],
            "final_encode": {
                "vcodec": "not-a-codec",
                "crf": "cinematic",
                "preset": "impossible",
                "acodec": "not-audio",
                "audio_bitrate": "huge",
                "width": "no-width",
                "height": "720",
            },
        },
    },
    "beat_flash_zoom_repair": {
        "command": "make the video pulse brighter and zoom slightly on every beat of the uploaded audio",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "plan": {
            "intent": "Pulse brighter and zoom on every beat",
            "analysis": [
                {"tool": "librosa", "function": "beat_track", "store_as": "beat_times"}
            ],
            "video_filters": [
                {
                    "filter": "eq=brightness=if(between(t,beat_times[0]-0.05,beat_times[0]+0.05),1.2,1):contrast=1:saturation=1",
                    "requires_context": "beat_times",
                    "timing": "per_beat",
                },
                {
                    "filter": "zoompan=z='min(zoom+0.001,1.1)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1920x1080",
                    "requires_context": "beat_times",
                    "timing": "per_beat",
                },
            ],
            "final_encode": BASE_ENCODE,
        },
    },
    "drawtext_animation_repair": {
        "command": "fade in the text Hello World at first 3 seconds and roll it out",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "plan": {
            "intent": "Fade in text and roll it out",
            "video_filters": [
                {
                    "description": "Generated text animation with alpha fragment",
                    "filter": (
                        "drawtext=text='Hello World':fontcolor=white:fontsize=48:"
                        "x=(w-text_w)/2:y=(h-text_h)/2:enable='between(t,0,3)',"
                        "alpha='min(1,t/3)':y='h-(h-t*50)'"
                    ),
                    "requires_context": None,
                    "timing": "continuous",
                }
            ],
            "final_encode": BASE_ENCODE,
        },
    },
    "drawtext_malformed_alpha_repair": {
        "command": "fade in the text Hello World with generated alpha expression",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "plan": {
            "intent": "Fade in text with malformed generated alpha expression",
            "video_filters": [
                {
                    "description": "Generated drawtext alpha expression with broken commas",
                    "filter": (
                        "drawtext=text='Hello World':fontcolor=white:fontsize=48:"
                        "x=(w-text_w)/2:y=(h-text_h)/2:enable='between(t,0,3)':"
                        "alpha='1+if(lt(t',1),t,1-if(gt(t,2),t-2,0))"
                    ),
                }
            ],
            "final_encode": BASE_ENCODE,
        },
    },
    "drawtext_malformed_tail_repair": {
        "command": "fade in the text Hello World and roll it out",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "plan": {
            "intent": "Fade in text and roll out with malformed generated tail",
            "video_filters": [
                {
                    "description": "Generated drawtext tail with broken comma fragment",
                    "filter": (
                        "drawtext=text='Hello World':fontcolor=white:fontsize=48:"
                        "x=(w-text_w)/2:y=(h-text_h)/2:enable='between(t,0,3)':"
                        "alpha='if(lt(t\\,3)\\,t/3\\,1)',0):"
                        "y='h-(h-t*100)':enable='between(t,3,6)'"
                    ),
                }
            ],
            "final_encode": BASE_ENCODE,
        },
    },
    "text_rollout_alignment": {
        "command": "fade in the text Hello World at first 3 seconds and let the text disappear with a rolling out animation",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "plan": {
            "intent": "Fade in text then roll out",
            "video_filters": [
                {
                    "description": "Generated drawbox instead of rolling text",
                    "filter": (
                        "drawtext=text='Hello World':fontcolor=white:fontsize=48:"
                        "x=(w-text_w)/2:y=(h-text_h)/2:enable='between(t,0,3)',"
                        "drawbox=x=0:y=0:w=iw:h=ih:color=white@0.9:t=fill:enable='between(t,3,6)'"
                    ),
                }
            ],
            "final_encode": BASE_ENCODE,
        },
    },
    "auto_speech_captions": {
        "command": "generate captions from the speech and burn them at the bottom",
        "fixture": "speech_caption",
        "use_audio": False,
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "expect_auto_captions_min": 1,
        "expect_warning": "auto_captions used pocketsphinx ASR",
        "plan": {
            "intent": "Recognize speech and burn timed captions at the bottom",
            "video_filters": [
                {"description": "Generated static caption that should be stripped", "filter": "drawtext=text='captions':fontcolor=white:fontsize=48:x=20:y=20"}
            ],
            "special": [{"type": "auto_captions", "params": {"source": "speech", "language": "en", "style": "bottom_box", "max_segments": 4}}],
            "final_encode": BASE_ENCODE,
        },
    },
    "vertical_export": {
        "command": "turn it into a vertical TikTok edit",
        "expect": {"video": True, "audio": True, "width": 1080, "height": 1920},
        "plan": {
            "intent": "Vertical TikTok export",
            "video_filters": [
                {"description": "Slight contrast lift", "filter": "eq=contrast=1.08:saturation=1.05"}
            ],
            "final_encode": BASE_ENCODE,
        },
    },
    "vertical_blurred_background": {
        "command": "make it vertical for reels with a blurred background instead of cropping the subject",
        "expect": {"video": True, "audio": True, "width": 1080, "height": 1920},
        "expect_blur_background": {"width": 1080, "height": 1920},
        "plan": {
            "intent": "Vertical social layout with blurred background",
            "video_filters": [
                {
                    "description": "Generated unsafe crop-based vertical layout",
                    "filter": "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,setsar=1",
                },
                {"description": "Keep contrast", "filter": "eq=contrast=1.05"},
            ],
            "special": [{"type": "blur_background", "params": {"width": 1080, "height": 1920, "sigma": 28}}],
            "final_encode": BASE_ENCODE,
        },
    },
    "instagram_portrait_format_repair": {
        "command": "export this as Instagram portrait 4:5 with warm color grade",
        "expect": {"video": True, "audio": True, "width": 1080, "height": 1350},
        "plan": {
            "intent": "Instagram portrait export",
            "video_filters": [
                {
                    "description": "Warm color grade",
                    "filter": "colorbalance=rs=0.15:rm=0.2:rh=0.25:bs=-0.15:bm=-0.05:bh=-0.2",
                },
                {
                    "description": "Generated unsafe 4:5 format chain",
                    "filter": "scale=-1:1080:force_original_aspect_ratio=decrease,pad=1080:1350:(ow-iw)/2:(oh-ih)/2:color=black",
                },
            ],
            "final_encode": BASE_ENCODE,
        },
    },
    "reverse": {
        "command": "reverse the entire clip including audio",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "plan": {
            "intent": "Reverse the full clip including audio",
            "special": [{"type": "reverse", "params": {}}],
            "final_encode": BASE_ENCODE,
        },
    },
    "boomerang_loop": {
        "command": "make this a boomerang ping-pong loop",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "expect_duration_min": 11.0,
        "expect_duration_max": 13.2,
        "expect_boomerang": {"loops": 1, "mute_reversed_audio": True},
        "plan": {
            "intent": "Create a forward then reverse ping-pong loop",
            "special": [{"type": "boomerang", "params": {"loops": 1, "mute_reversed_audio": True}}],
            "final_encode": BASE_ENCODE,
        },
    },
    "end_reverse": {
        "command": "end with a dramatic reverse-motion sequence without relying on audio synchronization",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "plan": {
            "intent": "Visual stress test with end reverse only",
            "analysis": [{"tool": "librosa", "function": "beat_track", "store_as": "beat_times"}],
            "audio_filters": [{"description": "Unwanted bass boost", "filter": "equalizer=f=60:g=10"}],
            "video_filters": [
                {
                    "description": "Reverse should become end-only reverse",
                    "filter": "reverse,eq=contrast=1.2:saturation=1.2",
                }
            ],
            "special": [
                {"type": "speed_ramp", "params": {"slow_factor": 0.5, "fast_factor": 2.0}},
                {"type": "pitch_shift", "params": {"semitones": 2}},
            ],
            "final_encode": BASE_ENCODE,
        },
    },
    "full_frame_flash_repair": {
        "command": "add glitch flickers and light leaks",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "plan": {
            "intent": "Glitch flickers and light leaks without washing out the frame",
            "video_filters": [
                {
                    "description": "Generated full-frame flash overlay",
                    "filter": "rgbashift=rh=14:bh=-14,drawbox=x=0:y=0:w=iw:h=ih:color=white@0.9:t=fill",
                }
            ],
            "final_encode": BASE_ENCODE,
        },
    },
    "raw_mod_strobe_repair": {
        "command": "add harsh strobe light flashes every half second",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "plan": {
            "intent": "Repair raw mod strobe enable expression into short flashes",
            "video_filters": [
                {
                    "description": "Generated raw strobe expression",
                    "filter": "drawbox=x=0:y=0:w=iw:h=ih:color=white@0.45:t=fill:enable='mod(t,0.5)'",
                }
            ],
            "final_encode": BASE_ENCODE,
        },
    },
    "ocr_text_redaction": {
        "command": "blur every visible license plate number and screen text",
        "fixture": "ocr_text",
        "expect": {"video": True, "width": 1280, "height": 720},
        "expect_ocr_redactions_min": 1,
        "plan": {
            "intent": "OCR-detect and redact visible plate text",
            "special": [{"type": "ocr_redact", "params": {"sample_fps": 1.0, "confidence": 35, "max_frames": 4}}],
            "final_encode": BASE_ENCODE,
        },
    },
    "green_screen_chroma_key": {
        "command": "remove green screen background and replace it with black",
        "fixture": "green_screen",
        "use_audio": False,
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "expect_chroma_key": {"key_color": "green", "replacement_color": "black"},
        "expect_pixel": {"x": 12, "y": 12, "rgb": [0, 0, 0], "tolerance": 18},
        "plan": {
            "intent": "Remove green-screen background and composite over black",
            "video_filters": [
                {"description": "Generated direct chroma filter that should be moved to special", "filter": "chromakey=0x00ff00:0.18:0.05,format=yuv420p"}
            ],
            "final_encode": BASE_ENCODE,
        },
    },
    "old_film_damage": {
        "command": "make it look like old damaged 16mm film with scratches dust flicker and gate weave",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "expect_film_damage": {"intensity": 0.9},
        "plan": {
            "intent": "Generated old film damage that should be normalized to the real special",
            "video_filters": [
                {
                    "description": "Generated old film scratches and dust",
                    "filter": "noise=alls=24:allf=t+u,drawbox=x=120:y=0:w=2:h=ih:color=white@0.2:t=fill",
                }
            ],
            "final_encode": BASE_ENCODE,
        },
    },
    "comic_halftone_alignment": {
        "command": "make it look like a comic book halftone with thick outlines",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "expect_plan_substrings": ["pixelize", "edgedetect"],
        "forbid_plan_substrings": ["lenscorrection"],
        "plan": {
            "intent": "Comic book halftone with thick outlines",
            "video_filters": [
                {"filter": "pixelize=width=10:height=10,hue=s=0.75"},
                {"filter": "lenscorrection=k1=-0.35:k2=0.08"},
            ],
            "final_encode": BASE_ENCODE,
        },
    },
    "underwater_alignment": {
        "command": "make it look underwater with wavy distortion and muffled audio",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "expect_plan_substrings": ["colorbalance", "crop=iw-24:ih-24", "lowpass=f=800"],
        "forbid_plan_substrings": ["geq=r='p(X,Y)'"],
        "plan": {
            "intent": "Simulate underwater look with model-generated identity geq",
            "video_filters": [
                {"filter": "colorbalance=rs=-0.2:rm=-0.1:rh=0.05:bs=0.25:bm=0.2:bh=0.3"},
                {"filter": "geq=r='p(X,Y)':g='p(X,Y)':b='p(X,Y)'"},
                {"filter": "gblur=sigma=1.8"},
            ],
            "audio_filters": [{"filter": "lowpass=f=800"}],
            "final_encode": BASE_ENCODE,
        },
    },
    "black_screen_removal": {
        "command": "remove all black screens and blank frames",
        "fixture": "black_middle",
        "expect": {"video": True, "width": 1280, "height": 720},
        "expect_duration_min": 1.7,
        "expect_duration_max": 2.3,
        "expect_black_segments_min": 1,
        "plan": {
            "intent": "Detect and remove black visual sections",
            "special": [
                {
                    "type": "black_remove",
                    "params": {
                        "min_black_duration": 0.4,
                        "pixel_threshold": 0.1,
                        "picture_threshold": 0.98,
                    },
                }
            ],
            "final_encode": BASE_ENCODE,
        },
    },
    "freeze_frame_removal": {
        "command": "remove all frozen frames and stuck sections",
        "fixture": "freeze_middle",
        "expect": {"video": True, "width": 1280, "height": 720},
        "expect_duration_min": 1.6,
        "expect_duration_max": 2.5,
        "expect_freeze_segments_min": 1,
        "plan": {
            "intent": "Detect and remove frozen visual sections",
            "special": [{"type": "freeze_remove", "params": {"noise_db": -60, "min_duration": 0.4}}],
            "final_encode": BASE_ENCODE,
        },
    },
    "duplicate_frame_removal": {
        "command": "remove all duplicate and repeated frames",
        "fixture": "duplicate_frames",
        "expect": {"video": True, "width": 1280, "height": 720},
        "expect_dedupe_removed_min": 10,
        "plan": {
            "intent": "Drop duplicate and repeated frames",
            "special": [{"type": "dedupe_frames", "params": {"hi": 768, "lo": 320, "frac": 0.33, "max": 12}}],
            "final_encode": BASE_ENCODE,
        },
    },
    "scene_highlight_montage": {
        "command": "create a fast highlight reel from the best moments",
        "fixture": "scene_changes",
        "expect": {"video": True, "width": 1280, "height": 720},
        "expect_scene_montage_min": 2,
        "expect_duration_min": 1.0,
        "expect_duration_max": 5.0,
        "plan": {
            "intent": "Create a scene-change highlight montage",
            "special": [{"type": "scene_montage", "params": {"threshold": 0.18, "slice_duration": 0.8, "max_segments": 8}}],
            "final_encode": BASE_ENCODE,
        },
    },
    "energy_hype_montage": {
        "command": "create a hype reel from the loudest moments in the music",
        "expect": {"video": True, "audio": True, "width": 1280, "height": 720},
        "expect_energy_montage_min": 2,
        "expect_duration_min": 0.5,
        "expect_duration_max": 6.0,
        "plan": {
            "intent": "Create a high-energy montage from loudest audio moments",
            "analysis": [{"tool": "librosa", "function": "rms_energy", "store_as": "energy_curve"}],
            "special": [{"type": "energy_montage", "params": {"context": "energy_curve_times", "slice_duration": 0.8, "max_segments": 8}}],
            "final_encode": BASE_ENCODE,
        },
    },
    "crop_black_borders": {
        "command": "remove the black bars around the video",
        "fixture": "black_borders",
        "expect": {"video": True, "width": 960, "height": 540},
        "expect_crop_borders": {"width": 960, "height": 540, "x": 160, "y": 90},
        "plan": {
            "intent": "Detect and crop black borders",
            "special": [{"type": "crop_borders", "params": {"limit": 24, "round": 2, "max_frames": 120}}],
            "final_encode": BASE_ENCODE,
        },
    },
}


def ffprobe(path):
    result = server.run_command([
        "ffprobe",
        "-v", "error",
        "-show_entries", "stream=index,codec_type,codec_name,width,height",
        "-of", "json",
        str(path),
    ])
    return json.loads(result.stdout)


def validate_output(expect, probe):
    streams = probe.get("streams") or []
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    errors = []

    if expect.get("video") and not video:
        errors.append("missing video stream")
    if expect.get("audio") and not audio:
        errors.append("missing audio stream")
    if expect.get("audio") is False and audio:
        errors.append("unexpected audio stream")
    if video:
        for key in ["width", "height"]:
            if key in expect and video.get(key) != expect[key]:
                errors.append(f"expected {key}={expect[key]}, got {video.get(key)}")
    return errors


def validate_duration(case, duration):
    errors = []
    minimum = case.get("expect_duration_min")
    maximum = case.get("expect_duration_max")
    if minimum is not None and duration < minimum:
        errors.append(f"expected duration >= {minimum}, got {duration:.3f}")
    if maximum is not None and duration > maximum:
            errors.append(f"expected duration <= {maximum}, got {duration:.3f}")
    return errors


def sample_output_pixel(video_path, target_dir, x, y):
    from PIL import Image

    frame_path = target_dir / f"sample_{uuid4().hex[:8]}.png"
    server.run_command([
        "ffmpeg", "-y",
        "-ss", "0.2",
        "-i", str(video_path),
        "-frames:v", "1",
        str(frame_path),
    ])
    with Image.open(frame_path) as image:
        return image.convert("RGB").getpixel((int(x), int(y)))


def validate_expected_pixel(case, output_path, job_dir):
    expected = case.get("expect_pixel")
    if not expected:
        return []
    actual = sample_output_pixel(output_path, job_dir, expected.get("x", 0), expected.get("y", 0))
    target = tuple(int(value) for value in expected.get("rgb", [0, 0, 0]))
    tolerance = int(expected.get("tolerance", 12))
    if any(abs(actual[index] - target[index]) > tolerance for index in range(3)):
        return [f"expected pixel near {target} at ({expected.get('x')},{expected.get('y')}), got {actual}"]
    return []


def copy_asset(source, target_dir, fallback_name):
    name = source.name if source.name else fallback_name
    target = target_dir / name
    shutil.copy2(source, target)
    return target


def create_ocr_text_fixture(target_dir):
    target = target_dir / "ocr_text_fixture.mp4"
    server.run_command([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "color=c=black:s=1280x720:d=3:r=24",
        "-vf",
        (
            "drawtext=text='PLATE123':fontcolor=white:fontsize=82:"
            "x=420:y=300:box=1:boxcolor=black@0.8:boxborderw=18"
        ),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(target),
    ])
    return target


def create_green_screen_fixture(target_dir):
    target = target_dir / "green_screen_fixture.mp4"
    server.run_command([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=0x00ff00:s=1280x720:d=3:r=24",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-vf", "drawbox=x=440:y=210:w=400:h=300:color=red:t=fill",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        str(target),
    ])
    return target


def create_black_middle_fixture(target_dir):
    target = target_dir / "black_middle_fixture.mp4"
    server.run_command([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=red:s=1280x720:d=1:r=24",
        "-f", "lavfi", "-i", "color=c=black:s=1280x720:d=1:r=24",
        "-f", "lavfi", "-i", "color=c=blue:s=1280x720:d=1:r=24",
        "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
        "-map", "[v]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(target),
    ])
    return target


def create_freeze_middle_fixture(target_dir):
    target = target_dir / "freeze_middle_fixture.mp4"
    server.run_command([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc2=s=1280x720:d=1:r=24",
        "-f", "lavfi", "-i", "color=c=red:s=1280x720:d=1:r=24",
        "-f", "lavfi", "-i", "testsrc2=s=1280x720:d=1:r=24",
        "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0[v]",
        "-map", "[v]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(target),
    ])
    return target


def create_duplicate_frames_fixture(target_dir):
    target = target_dir / "duplicate_frames_fixture.mp4"
    server.run_command([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "color=c=red:s=1280x720:d=2:r=24",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(target),
    ])
    return target


def create_black_borders_fixture(target_dir):
    target = target_dir / "black_borders_fixture.mp4"
    server.run_command([
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "color=c=red:s=960x540:d=2:r=24",
        "-vf", "pad=1280:720:160:90:color=black",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(target),
    ])
    return target


def create_scene_changes_fixture(target_dir):
    target = target_dir / "scene_changes_fixture.mp4"
    server.run_command([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "color=c=red:s=1280x720:d=1:r=24",
        "-f", "lavfi", "-i", "color=c=green:s=1280x720:d=1:r=24",
        "-f", "lavfi", "-i", "color=c=blue:s=1280x720:d=1:r=24",
        "-f", "lavfi", "-i", "color=c=yellow:s=1280x720:d=1:r=24",
        "-filter_complex", "[0:v][1:v][2:v][3:v]concat=n=4:v=1:a=0[v]",
        "-map", "[v]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(target),
    ])
    return target


def create_video_with_audio_fixture(target_dir):
    target = target_dir / "video_with_audio_fixture.mp4"
    server.run_command([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc2=s=1280x720:d=3:r=24",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        str(target),
    ])
    return target


def create_noisy_dialogue_fixture(target_dir):
    target = target_dir / "noisy_dialogue_fixture.mp4"
    server.run_command([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc2=s=1280x720:d=4:r=24",
        "-f", "lavfi", "-i", "flite=text='this is a noisy dialogue test':voice=slt",
        "-f", "lavfi", "-i", "anoisesrc=color=white:amplitude=0.055:d=4",
        "-filter_complex",
        "[1:a]aresample=44100,volume=1.0[speech];"
        "[2:a]aresample=44100,volume=0.28[noise];"
        "[speech][noise]amix=inputs=2:duration=longest:dropout_transition=0[a]",
        "-map", "0:v:0",
        "-map", "[a]",
        "-shortest",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        str(target),
    ])
    return target


def create_speech_caption_fixture(target_dir):
    target = target_dir / "speech_caption_fixture.mp4"
    server.run_command([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc2=s=1280x720:d=4:r=24",
        "-f", "lavfi", "-i", "flite=text='one two three four five':voice=slt",
        "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=16000:d=1",
        "-filter_complex", "[1:a]aresample=16000[a0];[a0][2:a]concat=n=2:v=0:a=1[a]",
        "-map", "0:v:0",
        "-map", "[a]",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        str(target),
    ])
    return target


def run_case(case_id, case, video_path, audio_path):
    started = time.time()
    job_id = f"smoke-{uuid4().hex[:8]}"
    job_dir = server.UPLOAD_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=False)

    if case.get("fixture") == "ocr_text":
        copied_video = create_ocr_text_fixture(job_dir)
    elif case.get("fixture") == "green_screen":
        copied_video = create_green_screen_fixture(job_dir)
    elif case.get("fixture") == "black_middle":
        copied_video = create_black_middle_fixture(job_dir)
    elif case.get("fixture") == "freeze_middle":
        copied_video = create_freeze_middle_fixture(job_dir)
    elif case.get("fixture") == "duplicate_frames":
        copied_video = create_duplicate_frames_fixture(job_dir)
    elif case.get("fixture") == "black_borders":
        copied_video = create_black_borders_fixture(job_dir)
    elif case.get("fixture") == "scene_changes":
        copied_video = create_scene_changes_fixture(job_dir)
    elif case.get("fixture") == "video_with_audio":
        copied_video = create_video_with_audio_fixture(job_dir)
    elif case.get("fixture") == "noisy_dialogue":
        copied_video = create_noisy_dialogue_fixture(job_dir)
    elif case.get("fixture") == "speech_caption":
        copied_video = create_speech_caption_fixture(job_dir)
    else:
        copied_video = copy_asset(video_path, job_dir, "video.mp4")
    if case.get("strip_audio"):
        no_audio_video = job_dir / f"no_audio_{copied_video.name}"
        server.run_command([
            "ffmpeg", "-y",
            "-i", str(copied_video),
            "-an",
            "-c:v", "copy",
            str(no_audio_video),
        ])
        copied_video = no_audio_video

    case_audio_path = audio_path if case.get("use_audio", True) else None
    copied_audio = copy_asset(case_audio_path, job_dir, "audio.mp3") if case_audio_path else None
    job = {
        "job_id": job_id,
        "status": "uploaded",
        "video_path": str(copied_video.resolve()),
        "audio_path": str(copied_audio.resolve()) if copied_audio else None,
        "video_name": copied_video.name,
        "audio_name": copied_audio.name if copied_audio else None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": case["command"],
    }

    result = {
        "id": case_id,
        "job_id": job_id,
        "command": case["command"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "unknown",
    }

    try:
        plan = server.align_plan_with_command(
            server.normalize_plan(json.loads(json.dumps(case["plan"]))),
            case["command"],
        )
        result["plan"] = plan
        output_path = Path(server.execute_pipeline(job, plan))
        result["warnings"] = job.get("warnings", [])
        if "ocr_redactions" in job:
            result["ocr_redactions"] = job["ocr_redactions"]
        if "face_privacy_blur" in job:
            result["face_privacy_blur"] = job["face_privacy_blur"]
        if "black_segments_removed" in job:
            result["black_segments_removed"] = job["black_segments_removed"]
        if "freeze_segments_removed" in job:
            result["freeze_segments_removed"] = job["freeze_segments_removed"]
        if "dedupe_frames" in job:
            result["dedupe_frames"] = job["dedupe_frames"]
        if "beat_cuts" in job:
            result["beat_cuts"] = job["beat_cuts"]
        if "scene_montage" in job:
            result["scene_montage"] = job["scene_montage"]
        if "energy_montage" in job:
            result["energy_montage"] = job["energy_montage"]
        if "crop_borders" in job:
            result["crop_borders"] = job["crop_borders"]
        if "boomerang" in job:
            result["boomerang"] = job["boomerang"]
        if "blur_background" in job:
            result["blur_background"] = job["blur_background"]
        if "chroma_key" in job:
            result["chroma_key"] = job["chroma_key"]
        if "film_damage" in job:
            result["film_damage"] = job["film_damage"]
        if "auto_captions" in job:
            result["auto_captions"] = job["auto_captions"]
        if "uploaded_audio_action" in job:
            result["uploaded_audio_action"] = job["uploaded_audio_action"]
        result["output_path"] = str(output_path)
        result["output_size_bytes"] = output_path.stat().st_size
        probe = ffprobe(output_path)
        result["ffprobe"] = probe
        duration = server.ffprobe_duration(output_path)
        result["duration_seconds"] = duration
        errors = validate_output(case.get("expect", {}), probe)
        errors.extend(validate_duration(case, duration))
        errors.extend(validate_expected_pixel(case, output_path, job_dir))
        plan_text = json.dumps(plan, sort_keys=True)
        for substring in case.get("expect_plan_substrings", []):
            if substring not in plan_text:
                errors.append(f"expected plan substring missing: {substring}")
        for substring in case.get("forbid_plan_substrings", []):
            if substring in plan_text:
                errors.append(f"forbidden plan substring present: {substring}")
        expected_warning = case.get("expect_warning")
        if expected_warning and not any(expected_warning in warning for warning in job.get("warnings", [])):
            errors.append(f"missing expected warning: {expected_warning}")
        forbidden_warning = case.get("forbid_warning")
        if forbidden_warning and any(forbidden_warning in warning for warning in job.get("warnings", [])):
            errors.append(f"unexpected warning containing: {forbidden_warning}")
        min_ocr_redactions = case.get("expect_ocr_redactions_min")
        if min_ocr_redactions is not None and len(job.get("ocr_redactions", [])) < min_ocr_redactions:
            errors.append(
                f"expected at least {min_ocr_redactions} OCR redactions, got {len(job.get('ocr_redactions', []))}"
            )
        min_face_regions = case.get("expect_face_privacy_regions_min")
        if min_face_regions is not None:
            actual_regions = job.get("face_privacy_blur", {}).get("regions", [])
            if len(actual_regions) < min_face_regions:
                errors.append(
                    f"expected at least {min_face_regions} face privacy regions, got {len(actual_regions)}"
                )
        min_black_segments = case.get("expect_black_segments_min")
        if min_black_segments is not None and len(job.get("black_segments_removed", [])) < min_black_segments:
            errors.append(
                f"expected at least {min_black_segments} black segments, got {len(job.get('black_segments_removed', []))}"
            )
        min_freeze_segments = case.get("expect_freeze_segments_min")
        if min_freeze_segments is not None and len(job.get("freeze_segments_removed", [])) < min_freeze_segments:
            errors.append(
                f"expected at least {min_freeze_segments} freeze segments, got {len(job.get('freeze_segments_removed', []))}"
            )
        min_dedupe_removed = case.get("expect_dedupe_removed_min")
        if min_dedupe_removed is not None and job.get("dedupe_frames", {}).get("removed", 0) < min_dedupe_removed:
            errors.append(
                f"expected at least {min_dedupe_removed} duplicate frames removed, got {job.get('dedupe_frames')}"
            )
        min_beat_cuts = case.get("expect_beat_cuts_min")
        if min_beat_cuts is not None and len(job.get("beat_cuts", [])) < min_beat_cuts:
            errors.append(
                f"expected at least {min_beat_cuts} beat cuts, got {len(job.get('beat_cuts', []))}"
            )
        min_scene_montage = case.get("expect_scene_montage_min")
        if min_scene_montage is not None and len(job.get("scene_montage", [])) < min_scene_montage:
            errors.append(
                f"expected at least {min_scene_montage} scene montage segments, got {len(job.get('scene_montage', []))}"
            )
        min_energy_montage = case.get("expect_energy_montage_min")
        if min_energy_montage is not None and len(job.get("energy_montage", [])) < min_energy_montage:
            errors.append(
                f"expected at least {min_energy_montage} energy montage segments, got {len(job.get('energy_montage', []))}"
            )
        expected_crop = case.get("expect_crop_borders")
        if expected_crop is not None and job.get("crop_borders") != expected_crop:
            errors.append(f"expected crop_borders={expected_crop}, got {job.get('crop_borders')}")
        expected_boomerang = case.get("expect_boomerang")
        if expected_boomerang is not None:
            actual_boomerang = job.get("boomerang")
            for key, expected_value in expected_boomerang.items():
                if not isinstance(actual_boomerang, dict) or actual_boomerang.get(key) != expected_value:
                    errors.append(f"expected boomerang {key}={expected_value}, got {actual_boomerang}")
        expected_blur_background = case.get("expect_blur_background")
        if expected_blur_background is not None:
            actual_blur_background = job.get("blur_background")
            for key, expected_value in expected_blur_background.items():
                if not isinstance(actual_blur_background, dict) or actual_blur_background.get(key) != expected_value:
                    errors.append(f"expected blur_background {key}={expected_value}, got {actual_blur_background}")
        expected_chroma_key = case.get("expect_chroma_key")
        if expected_chroma_key is not None:
            actual_chroma_key = job.get("chroma_key")
            for key, expected_value in expected_chroma_key.items():
                if not isinstance(actual_chroma_key, dict) or actual_chroma_key.get(key) != expected_value:
                    errors.append(f"expected chroma_key {key}={expected_value}, got {actual_chroma_key}")
        expected_film_damage = case.get("expect_film_damage")
        if expected_film_damage is not None:
            actual_film_damage = job.get("film_damage")
            for key, expected_value in expected_film_damage.items():
                if not isinstance(actual_film_damage, dict) or actual_film_damage.get(key) != expected_value:
                    errors.append(f"expected film_damage {key}={expected_value}, got {actual_film_damage}")
        min_auto_captions = case.get("expect_auto_captions_min")
        if min_auto_captions is not None and len(job.get("auto_captions", {}).get("segments", [])) < min_auto_captions:
            errors.append(
                f"expected at least {min_auto_captions} auto caption segments, got {job.get('auto_captions')}"
            )
        expected_uploaded_audio_action = case.get("expect_uploaded_audio_action")
        if expected_uploaded_audio_action is not None:
            actual_action = (job.get("uploaded_audio_action") or {}).get("type")
            if actual_action != expected_uploaded_audio_action:
                errors.append(f"expected uploaded_audio_action={expected_uploaded_audio_action}, got {job.get('uploaded_audio_action')}")
        if errors:
            result["status"] = "error"
            result["error"] = "; ".join(errors)
        else:
            result["status"] = "complete"
        return result
    except Exception as exc:
        result["status"] = "error"
        result["error"] = str(exc)
        return result
    finally:
        result["elapsed_seconds"] = round(time.time() - started, 2)


def write_report(report, report_dir):
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = report_dir / f"executor_smoke_{timestamp}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (report_dir / "executor_smoke_latest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Run real executor smoke tests without calling NIM.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--only", action="append", choices=sorted(CASES))
    return parser.parse_args()


def main():
    args = parse_args()
    selected_ids = args.only or list(CASES)
    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "cases": [],
    }

    for case_id in selected_ids:
        print(f"\n[{case_id}] {CASES[case_id]['command']}", flush=True)
        result = run_case(case_id, CASES[case_id], args.video, args.audio)
        report["cases"].append(result)
        if result["status"] == "complete":
            print(f"  passed in {result['elapsed_seconds']}s", flush=True)
        else:
            print(f"  FAILED: {result.get('error')}", flush=True)

    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    report["failures"] = sum(1 for result in report["cases"] if result["status"] != "complete")
    path = write_report(report, args.report_dir)
    print(f"\nReport: {path}", flush=True)
    print(f"Passed: {len(report['cases']) - report['failures']}/{len(report['cases'])}", flush=True)
    raise SystemExit(1 if report["failures"] else 0)


if __name__ == "__main__":
    main()
