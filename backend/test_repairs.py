import tempfile
import unittest
from pathlib import Path
from unittest import mock

import editing_capabilities
import server


class CapabilityKnowledgeTests(unittest.TestCase):
    def test_ai_knowledge_loader_only_uses_approved_pdf_extracts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            knowledge_dir = Path(tmpdir)
            for name in editing_capabilities.APPROVED_AI_KNOWLEDGE_FILES:
                (knowledge_dir / name).write_text(f"approved {name}", encoding="utf-8")
            (knowledge_dir / "ffmpeg_mega_cookbook.txt").write_text("not approved", encoding="utf-8")
            (knowledge_dir / "automation_output.txt").write_text("not approved", encoding="utf-8")

            with mock.patch.object(editing_capabilities, "AI_KNOWLEDGE_DIR", knowledge_dir):
                editing_capabilities.load_ai_knowledge_files.cache_clear()
                try:
                    loaded = editing_capabilities.load_ai_knowledge_files()
                finally:
                    editing_capabilities.load_ai_knowledge_files.cache_clear()

        self.assertEqual(
            [name for name, _text in loaded],
            list(editing_capabilities.APPROVED_AI_KNOWLEDGE_FILES),
        )


class FilterRepairTests(unittest.TestCase):
    def test_red_hold_from_description(self):
        step = {
            "description": "Make everything black and white except red tones",
            "filter": "colorchannelmixer=rr=0.2126:rg=0.7152:rb=0.0722:gr=0.2126:gg=0.7152:gb=0.0722:br=0.2126:bg=0.7152:bb=0.0722",
        }

        repaired = server.normalize_filter_steps([step], "video")[0]["filter"]

        self.assertEqual(repaired, server.COLOR_HOLD_FILTERS["red"])

    def test_broken_channel_threshold_falls_back_to_red_hold(self):
        broken = (
            "colorchannelmixer=rr=0.2126:rg=0.7152:rb=0.0722,"
            "eq=contrast=1.05:saturation=0.75,"
            "min(255,255*(G>128))):b='max(0,min(255,255*(B>128))'"
        )

        repaired = server.repair_filter_string(broken)

        self.assertEqual(repaired, server.COLOR_HOLD_FILTERS["red"])

    def test_named_color_hold_from_description(self):
        step = {
            "description": "Make the footage grayscale except blue lights",
            "filter": "hue=s=0,colorchannelmixer=rr=0.3:rg=0.59:rb=0.11",
        }

        repaired = server.normalize_filter_steps([step], "video")[0]["filter"]

        self.assertEqual(repaired, server.COLOR_HOLD_FILTERS["blue"])

    def test_colorbalance_aliases_are_normalized(self):
        repaired = server.repair_filter_string("colorbalance=ss=-0.2:ms=0.1:hb=0.3")

        self.assertEqual(repaired, "colorbalance=rs=-0.2:rm=0.1:bh=0.3")

    def test_generated_description_fragment_is_removed_from_filter_chain(self):
        repaired = server.repair_filter_string("noise=alls=18:allf=t+u,'description=")

        self.assertEqual(repaired, "noise=alls=18:allf=t+u")

    def test_drawtext_punctuation_text_is_requoted_before_splitting(self):
        repaired = server.repair_filter_string(
            "drawtext=text='Dr. Rao: we're live,don't blink':fontcolor=white:fontsize=48:"
            "x=(w-text_w)/2:y=h-(2*text_h):enable='between(t,0,4)':"
            "alpha=1-if(gte(t,4),1,0.25*(4-t))"
        )

        self.assertIn('drawtext=text="Dr. Rao: we', repaired)
        self.assertIn("don't blink", repaired)
        self.assertIn(f":alpha={server.SAFE_DRAWTEXT_ALPHA}", repaired)
        self.assertEqual(len(server.split_filter_chain(repaired)), 1)

    def test_valid_geq_is_not_replaced_by_color_hold(self):
        repaired = server.repair_filter_string("geq=r='p(X,Y)':g='p(X,Y)':b='p(X,Y)'")

        self.assertEqual(repaired, "geq=r='p(X,Y)':g='p(X,Y)':b='p(X,Y)'")

    def test_geq_comparison_mask_becomes_center_privacy_blur(self):
        repaired = server.repair_filter_string(
            "geq=r='255*(X/W-0.5)*(X/W-0.5)+Y*(Y/H-0.5)*(Y/H-0.5)<0.25*0.25':"
            "g='255*(X/W-0.5)*(X/W-0.5)+Y*(Y/H-0.5)*(Y/H-0.5)<0.25*0.25':"
            "b='255*(X/W-0.5)*(X/W-0.5)+Y*(Y/H-0.5)*(Y/H-0.5)<0.25*0.25'"
        )

        self.assertIn("delogo=x=__PRIVACY_X__:y=__PRIVACY_Y__:w=__PRIVACY_W__:h=__PRIVACY_H__", repaired)
        self.assertIn("vignette=angle=PI/3", repaired)
        self.assertNotIn("<0.25", repaired)

    def test_drawtext_alpha_fragment_is_merged_into_drawtext(self):
        broken = (
            "drawtext=text='Hello World':fontcolor=white:fontsize=48:"
            "x=(w-text_w)/2:y=(h-text_h)/2:enable='between(t,0,3)',"
            "alpha='min(1,t/3)':y='h-(h-t*50)'"
        )

        repaired = server.repair_filter_string(broken)

        self.assertNotIn(",alpha=", repaired)
        self.assertIn(":alpha='min(1,t/3)'", repaired)
        self.assertIn(":y='h-(h-t*50)'", repaired)

    def test_malformed_drawtext_alpha_expression_gets_safe_fade(self):
        broken = (
            "drawtext=text='Hello World':fontcolor=white:fontsize=48:"
            "x=(w-text_w)/2:y=(h-text_h)/2:enable='between(t,0,3)':"
            "alpha='1+if(lt(t',1),t,1-if(gt(t,2),t-2,0))"
        )

        repaired = server.repair_filter_string(broken)

        self.assertNotIn("alpha='1+if", repaired)
        self.assertNotIn("1-if(gt", repaired)
        self.assertIn(f":alpha={server.SAFE_DRAWTEXT_ALPHA}", repaired)

    def test_nested_drawtext_if_alpha_gets_safe_fade(self):
        broken = (
            "drawtext=text='Hello World':fontcolor=white:fontsize=48:"
            "x=(w-text_w)/2:y=(h-text_h)/2:enable='between(t,0,3)':"
            "alpha='if(lt(t\\,3)\\,t/3\\,1)'"
        )

        repaired = server.repair_filter_string(broken)

        self.assertNotIn("if(lt", repaired)
        self.assertIn(f":alpha={server.SAFE_DRAWTEXT_ALPHA}", repaired)

    def test_drawtext_malformed_tail_options_are_merged(self):
        broken = (
            "drawtext=text='Hello World':fontcolor=white:fontsize=48:"
            "x=(w-text_w)/2:y=(h-text_h)/2:enable='between(t,0,3)':"
            "alpha='if(lt(t\\,3)\\,t/3\\,1)',0):y='h-(h-t*100)':enable='between(t,3,6)'"
        )

        repaired = server.repair_filter_string(broken)

        self.assertNotIn(",0):", repaired)
        self.assertIn(":y='h-(h-t*100)'", repaired)
        self.assertIn(":enable='between(t,3,6)'", repaired)

    def test_full_frame_white_drawbox_becomes_timed_flash(self):
        repaired = server.repair_filter_string(
            "rgbashift=rh=14:bh=-14,"
            "drawbox=x=0:y=0:w=iw:h=ih:color=white@0.9:t=fill"
        )

        self.assertIn("drawbox=x=0:y=0:w=iw:h=ih:color=white@0.14:t=fill", repaired)
        self.assertIn("enable='lt(mod(t,1.2),0.06)'", repaired)
        self.assertNotIn("white@0.9:t=fill", repaired)

    def test_enabled_full_frame_drawbox_is_softened_and_shortened(self):
        repaired = server.repair_filter_string(
            "drawbox=x=0:y=0:w=iw:h=ih:color=white@0.9:t=fill:"
            "enable='between(mod(t,2),0,1)'"
        )

        self.assertIn("color=white@0.45", repaired)
        self.assertIn("enable='lt(mod(t,2),0.08)'", repaired)
        self.assertNotIn("white@0.9", repaired)
        self.assertNotIn("between(mod(t,2),0,1)", repaired)

    def test_strobe_drawbox_has_short_flash_window(self):
        repaired = server.repair_filter_string(
            "drawbox=x=0:y=0:w=iw:h=ih:color=white@0.45:t=fill:"
            "enable='lt(mod(t,0.5),0.25)'"
        )

        self.assertIn("color=white@0.45", repaired)
        self.assertIn("enable='lt(mod(t,0.5),0.04)'", repaired)
        self.assertNotIn("0.25", repaired)

    def test_raw_mod_strobe_drawbox_becomes_short_flash_window(self):
        repaired = server.repair_filter_string(
            "drawbox=x=0:y=0:w=iw:h=ih:color=white@0.45:t=fill:"
            "enable='mod(t,0.5)'"
        )

        self.assertIn("enable='lt(mod(t,0.5),0.04)'", repaired)
        self.assertNotIn("enable='mod(t,0.5)'", repaired)

    def test_mod_comparison_strobe_drawbox_becomes_short_flash_window(self):
        repaired = server.repair_filter_string(
            "drawbox=x=0:y=0:w=iw:h=ih:color=white@0.45:t=fill:"
            "enable='mod(t,0.5)<0.05'"
        )

        self.assertIn("enable='lt(mod(t,0.5),0.04)'", repaired)
        self.assertNotIn("mod(t,0.5)<0.05", repaired)

    def test_strobe_command_period_overrides_model_guess(self):
        plan = {
            "intent": "Harsh strobe light flashes every half second",
            "video_filters": [
                {
                    "description": "Strobe flash every half second",
                    "filter": "drawbox=x=0:y=0:w=iw:h=ih:color=white@0.45:t=fill:enable='lt(mod(t,1),0.08)'",
                }
            ],
            "final_encode": server.default_final_encode(),
        }

        aligned = server.align_plan_with_command(plan, "add harsh strobe light flashes every half second")
        plan_text = str(aligned)

        self.assertIn("lt(mod(t,0.5),0.04)", plan_text)
        self.assertNotIn("lt(mod(t,1),0.08)", plan_text)

    def test_frei0r_comma_expression_does_not_leave_context_fragment(self):
        repaired = server.repair_filter_string(
            "frei0r=filter_name=glitch0r:filter_params=min(1,energy_curve)"
        )

        self.assertEqual(repaired, server.FREI0R_NATIVE_FALLBACKS["glitch0r"])
        self.assertNotIn("energy_curve", repaired)

    def test_energy_curve_filters_are_timed_on_energy_peaks(self):
        filters = [
            {
                "filter": server.FREI0R_NATIVE_FALLBACKS["glitch0r"],
                "requires_context": "energy_curve",
                "timing": "continuous",
            }
        ]
        context = {"energy_curve": [0.1, 0.9], "energy_curve_times": [1.0, 2.0]}

        with tempfile.TemporaryDirectory() as tmp:
            chain = server.video_filter_chain(filters, context, Path(tmp))

        self.assertIn("between(t,0.950,1.150)", chain)
        self.assertIn("between(t,1.950,2.150)", chain)
        self.assertNotIn("energy_curve", chain)

    def test_eq_brightness_if_expression_becomes_safe_timed_filter(self):
        repaired = server.repair_filter_string(
            "eq=brightness=if(between(t,beat_times[0]-0.05,beat_times[0]+0.05),1.2,1):"
            "contrast=1:saturation=1"
        )

        self.assertEqual(repaired, "eq=brightness=0.18:contrast=1.05:saturation=1.05")

    def test_video_noise_filter_in_audio_chain_becomes_audio_degradation(self):
        repaired = server.repair_audio_filter_string("noise=alls=10:allf=t+u")

        self.assertEqual(repaired, "acrusher=bits=8:mode=log:mix=0.18")

    def test_per_beat_zoompan_becomes_timed_crop_zoom(self):
        filters = [
            {
                "filter": "zoompan=z='min(zoom+0.001,1.1)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1920x1080",
                "requires_context": "beat_times",
                "timing": "per_beat",
            }
        ]
        context = {"beat_times": [1.0]}

        with tempfile.TemporaryDirectory() as tmp:
            chain = server.video_filter_chain(filters, context, Path(tmp))

        self.assertIn("crop=iw-48:ih-48", chain)
        self.assertIn(r"between(t\,0.950\,1.150)", chain)
        self.assertIn("scale=iw+48:ih+48", chain)
        self.assertNotIn("zoompan", chain)

    def test_privacy_blur_placeholders_use_input_dimensions(self):
        repaired = server.repair_dimension_references(
            "delogo=x=__PRIVACY_X__:y=__PRIVACY_Y__:w=__PRIVACY_W__:h=__PRIVACY_H__:show=0",
            (1280, 720),
        )

        self.assertEqual(repaired, "delogo=x=480:y=250:w=320:h=220:show=0")

    def test_ocr_redact_filter_chain_uses_timed_delogo_boxes(self):
        chain = server.ocr_redact_filter_chain([
            {"x": 10, "y": 20, "w": 120, "h": 40, "start": 0.0, "end": 1.0},
            {"x": 30, "y": 60, "w": 90, "h": 30, "start": 1.0, "end": 2.0},
        ])

        self.assertIn("delogo=x=10:y=20:w=120:h=40:show=0:enable='between(t,0.000,1.000)'", chain)
        self.assertIn("delogo=x=30:y=60:w=90:h=30:show=0:enable='between(t,1.000,2.000)'", chain)

    def test_ocr_detection_stays_inside_delogo_safe_frame_edges(self):
        detection = server.normalized_ocr_detection(
            {"x": 0, "y": 0, "w": 30, "h": 24, "text": "ID", "confidence": 88},
            100,
            80,
            0.0,
            12,
        )

        self.assertGreaterEqual(detection["x"], 1)
        self.assertGreaterEqual(detection["y"], 1)
        self.assertLessEqual(detection["x"] + detection["w"], 99)
        self.assertLessEqual(detection["y"] + detection["h"], 79)

    def test_blackdetect_segments_are_parsed(self):
        stderr = (
            "[blackdetect @ 0x1] black_start:1 black_end:2 black_duration:1\n"
            "[blackdetect @ 0x1] black_start:4.5 black_end:5.25 black_duration:0.75\n"
        )

        self.assertEqual(server.parse_blackdetect_segments(stderr, 6.0), [(1.0, 2.0), (4.5, 5.25)])

    def test_keep_segments_excluding_removed_ranges(self):
        self.assertEqual(
            server.keep_segments_excluding(6.0, [(1.0, 2.0), (4.0, 5.0)]),
            [(0.0, 1.0), (2.0, 4.0), (5.0, 6.0)],
        )

    def test_freezedetect_segments_are_parsed(self):
        stderr = (
            "[freezedetect @ 0x1] lavfi.freezedetect.freeze_start: 1\n"
            "[freezedetect @ 0x1] lavfi.freezedetect.freeze_duration: 1\n"
            "[freezedetect @ 0x1] lavfi.freezedetect.freeze_end: 2\n"
        )

        self.assertEqual(server.parse_freezedetect_segments(stderr, 4.0), [(1.0, 2.0)])

    def test_cropdetect_values_are_parsed_and_ranked(self):
        stderr = (
            "[Parsed_cropdetect_0] crop=1280:720:0:0\n"
            "[Parsed_cropdetect_0] crop=960:540:160:90\n"
            "[Parsed_cropdetect_0] crop=960:540:160:90\n"
        )

        crops = server.parse_cropdetect_crops(stderr)

        self.assertEqual(crops, [(1280, 720, 0, 0), (960, 540, 160, 90), (960, 540, 160, 90)])
        self.assertEqual(server.best_cropdetect_crop(crops, (1280, 720)), (960, 540, 160, 90))


class PlanAlignmentTests(unittest.TestCase):
    def test_trim_request_adds_special_and_strips_model_trim_filters(self):
        plan = {
            "intent": "Trim opening",
            "video_filters": [{"filter": "trim=start=2,setpts=PTS-STARTPTS,eq=contrast=1.1"}],
            "audio_filters": [{"filter": "atrim=start=2,asetpts=PTS-STARTPTS,loudnorm=I=-14:TP=-1.5:LRA=11"}],
            "final_encode": server.default_final_encode(),
        }

        aligned = server.align_plan_with_command(plan, "remove the first 2 seconds and add contrast")

        self.assertEqual(aligned["special"], [{"type": "trim", "params": {"start": 2.0}}])
        self.assertEqual(aligned["video_filters"][0]["filter"], "eq=contrast=1.1")
        self.assertEqual(aligned["audio_filters"][0]["filter"], "loudnorm=I=-14:TP=-1.5:LRA=11")

    def test_trim_range_request_is_parsed(self):
        special = server.trim_special_from_command("keep only from 00:01 to 00:03.5")

        self.assertEqual(special, {"type": "trim", "params": {"start": 1.0, "end": 3.5}})

    def test_remove_segment_request_is_parsed(self):
        special = server.trim_special_from_command("cut out from 1 second to 2.5 seconds")

        self.assertEqual(special, {"type": "remove_segment", "params": {"start": 1.0, "end": 2.5}})

    def test_no_audio_sync_removes_audio_dependent_work(self):
        plan = {
            "intent": "Visual stress test",
            "analysis": [{"tool": "librosa", "function": "beat_track", "store_as": "beat_times"}],
            "audio_filters": [{"description": "Bass boost", "filter": "equalizer=f=60:g=10"}],
            "special": [
                {"type": "speed_ramp", "params": {"slow_factor": 0.2, "fast_factor": 5.0}},
                {"type": "pitch_shift", "params": {"semitones": 2}},
            ],
            "video_filters": [
                {"filter": "noise=alls=10:allf=t+u"},
                {
                    "filter": "rgbashift=rh=12:bh=-12",
                    "requires_context": "onset_times",
                    "timing": "per_onset",
                },
            ],
            "final_encode": {
                "vcodec": "libx264",
                "crf": 22,
                "preset": "fast",
                "acodec": "aac",
                "audio_bitrate": "192k",
            },
        }

        aligned = server.align_plan_with_command(
            plan,
            "make intense visual effects without relying on audio synchronization",
        )

        self.assertNotIn("analysis", aligned)
        self.assertNotIn("audio_filters", aligned)
        self.assertEqual([step["type"] for step in aligned["special"]], ["speed_ramp"])
        self.assertEqual(aligned["video_filters"], [{"filter": "noise=alls=10:allf=t+u"}])

    def test_remove_audio_request_keeps_visual_analysis_and_drops_audio_filters(self):
        plan = {
            "intent": "Beat visuals but no final audio",
            "analysis": [{"tool": "librosa", "function": "beat_track", "store_as": "beat_times"}],
            "audio_filters": [{"description": "Bass boost", "filter": "equalizer=f=60:g=10"}],
            "special": [
                {"type": "silence_remove", "params": {"threshold_db": -35}},
                {"type": "pitch_shift", "params": {"semitones": 3}},
            ],
            "video_filters": [{"filter": "eq=contrast=1.2"}],
            "final_encode": server.default_final_encode(),
        }

        aligned = server.align_plan_with_command(
            plan,
            "make the clip completely silent but shake the frame on every beat",
        )

        self.assertIn("analysis", aligned)
        self.assertNotIn("audio_filters", aligned)
        self.assertEqual(aligned["special"], [{"type": "remove_audio", "params": {}}])

    def test_remove_audio_detector_does_not_confuse_silence_removal(self):
        self.assertFalse(server.remove_audio_requested("remove all silences and normalize the audio"))
        self.assertTrue(server.remove_audio_requested("remove the audio and add contrast"))

    def test_heuristic_plan_emits_remove_audio_special(self):
        plan = server.heuristic_plan("make the clip completely silent and add cinematic contrast")

        self.assertEqual(plan["special"], [{"type": "remove_audio", "params": {}}])
        self.assertNotIn("audio_filters", plan)

    def test_replace_uploaded_audio_request_adds_special(self):
        plan = {
            "intent": "Use uploaded audio",
            "video_filters": [{"filter": "eq=contrast=1.05"}],
            "final_encode": server.default_final_encode(),
        }

        aligned = server.align_plan_with_command(plan, "replace the original audio with the uploaded music track")

        self.assertIn({"type": "replace_audio", "params": {}}, aligned["special"])

    def test_background_music_request_mixes_uploaded_audio(self):
        plan = {
            "intent": "Add background music",
            "video_filters": [{"filter": "eq=contrast=1.05"}],
            "special": [{"type": "replace_audio", "params": {}}],
            "final_encode": server.default_final_encode(),
        }

        aligned = server.align_plan_with_command(
            plan,
            "add the uploaded audio as background music under the dialogue and duck it behind the voice",
        )

        self.assertEqual(
            aligned["special"],
            [{"type": "mix_uploaded_audio", "params": {"original_volume": 1.0, "music_volume": 0.28, "duck": True}}],
        )

    def test_uploaded_audio_beat_sync_uses_uploaded_audio_without_mix(self):
        plan = {
            "intent": "Beat sync visuals",
            "analysis": [{"tool": "librosa", "function": "beat_track", "store_as": "beat_times"}],
            "video_filters": [{"filter": "eq=contrast=1.05"}],
            "final_encode": server.default_final_encode(),
        }

        aligned = server.align_plan_with_command(plan, "shake on every beat of the uploaded audio")

        self.assertIn({"type": "replace_audio", "params": {}}, aligned["special"])

    def test_cut_to_beat_request_adds_beat_cut_special_and_analysis(self):
        plan = {
            "intent": "Cut to the beat",
            "video_filters": [{"filter": "eq=contrast=1.05"}],
            "final_encode": server.default_final_encode(),
        }

        aligned = server.align_plan_with_command(plan, "make hard jump cuts on every beat of the uploaded audio")

        self.assertIn({"tool": "librosa", "function": "beat_track", "store_as": "beat_times"}, aligned["analysis"])
        self.assertIn({"type": "beat_cut", "params": {"context": "beat_times", "slice_duration": 0.35, "max_cuts": 24}}, aligned["special"])

    def test_heuristic_plan_emits_beat_cut_special(self):
        plan = server.heuristic_plan("cut to the beat of the music")

        self.assertIn({"tool": "librosa", "function": "beat_track", "store_as": "beat_times"}, plan["analysis"])
        self.assertIn({"type": "beat_cut", "params": {"context": "beat_times", "slice_duration": 0.35, "max_cuts": 24}}, plan["special"])

    def test_highlight_reel_request_adds_scene_montage_special(self):
        plan = {
            "intent": "Highlight reel",
            "video_filters": [{"filter": "eq=contrast=1.05"}],
            "final_encode": server.default_final_encode(),
        }

        aligned = server.align_plan_with_command(plan, "create a fast highlight reel from the best moments")

        self.assertIn({"type": "scene_montage", "params": {"threshold": 0.28, "slice_duration": 1.2, "max_segments": 12}}, aligned["special"])

    def test_heuristic_plan_emits_scene_montage_special(self):
        plan = server.heuristic_plan("make a quick montage from every scene")

        self.assertIn({"type": "scene_montage", "params": {"threshold": 0.28, "slice_duration": 1.2, "max_segments": 12}}, plan["special"])

    def test_high_energy_request_adds_energy_montage_special_and_analysis(self):
        plan = {
            "intent": "Hype reel",
            "video_filters": [{"filter": "eq=contrast=1.05"}],
            "final_encode": server.default_final_encode(),
        }

        aligned = server.align_plan_with_command(plan, "create a hype reel from the loudest moments in the music")

        self.assertIn({"tool": "librosa", "function": "rms_energy", "store_as": "energy_curve"}, aligned["analysis"])
        self.assertIn({"type": "energy_montage", "params": {"context": "energy_curve_times", "slice_duration": 1.0, "max_segments": 12}}, aligned["special"])

    def test_heuristic_plan_emits_energy_montage_special(self):
        plan = server.heuristic_plan("make a high energy montage from the loudest parts of the audio")

        self.assertIn({"tool": "librosa", "function": "rms_energy", "store_as": "energy_curve"}, plan["analysis"])
        self.assertIn({"type": "energy_montage", "params": {"context": "energy_curve_times", "slice_duration": 1.0, "max_segments": 12}}, plan["special"])

    def test_energy_reactive_effect_request_adds_rms_analysis_and_timed_filters(self):
        plan = {
            "intent": "Energy effects",
            "final_encode": server.default_final_encode(),
        }

        aligned = server.align_plan_with_command(
            plan,
            "glitch harder as the music gets louder and flash on the loudest moments",
        )

        self.assertIn({"tool": "librosa", "function": "rms_energy", "store_as": "energy_curve"}, aligned["analysis"])
        filters = aligned["video_filters"]
        self.assertTrue(all(step.get("requires_context") == "energy_curve" for step in filters))
        self.assertTrue(any("rgbashift" in step["filter"] for step in filters))
        self.assertTrue(any("drawbox" in step["filter"] for step in filters))
        self.assertNotIn("special", aligned)

    def test_heuristic_plan_emits_energy_reactive_glitch_and_flash(self):
        plan = server.heuristic_plan(
            "glitch harder as the music gets louder and flash on the loudest moments"
        )

        self.assertIn({"tool": "librosa", "function": "rms_energy", "store_as": "energy_curve"}, plan["analysis"])
        filters = plan["video_filters"]
        self.assertTrue(any("rgbashift" in step["filter"] for step in filters))
        self.assertTrue(any("drawbox" in step["filter"] for step in filters))
        self.assertTrue(all(step.get("requires_context") == "energy_curve" for step in filters))

    def test_energy_reactive_alignment_does_not_duplicate_heuristic_filters(self):
        command = "glitch harder as the music gets louder and flash on the loudest moments"
        plan = server.align_plan_with_command(server.heuristic_plan(command), command)

        descriptions = [step["description"] for step in plan["video_filters"]]
        self.assertEqual(descriptions.count("Glitch bursts on loudest audio moments"), 1)
        self.assertEqual(descriptions.count("White flash on loudest audio moments"), 1)

    def test_boomerang_request_adds_special_and_removes_full_reverse(self):
        plan = {
            "intent": "Ping-pong loop",
            "video_filters": [{"filter": "reverse,eq=contrast=1.1"}],
            "special": [{"type": "reverse", "params": {}}],
            "final_encode": server.default_final_encode(),
        }

        aligned = server.align_plan_with_command(plan, "make this a boomerang ping-pong loop")

        self.assertEqual(aligned["video_filters"], [{"filter": "eq=contrast=1.1"}])
        self.assertEqual(aligned["special"], [{"type": "boomerang", "params": {"loops": 1, "mute_reversed_audio": True}}])

    def test_heuristic_plan_emits_boomerang_special(self):
        plan = server.heuristic_plan("play forward then reverse as a boomerang loop")

        self.assertIn({"type": "boomerang", "params": {"loops": 1, "mute_reversed_audio": True}}, plan["special"])
        self.assertNotIn({"type": "reverse", "params": {}}, plan["special"])

    def test_heuristic_plan_emits_uploaded_audio_mix_special(self):
        plan = server.heuristic_plan("add background music from the uploaded audio under dialogue")

        self.assertIn(
            {"type": "mix_uploaded_audio", "params": {"original_volume": 1.0, "music_volume": 0.28, "duck": True}},
            plan["special"],
        )

    def test_license_plate_request_becomes_ocr_redact_without_center_privacy_blur(self):
        plan = {
            "intent": "Redact license plates",
            "video_filters": [{"filter": "delogo=x=480:y=250:w=320:h=220"}],
            "final_encode": server.default_final_encode(),
        }

        aligned = server.align_plan_with_command(plan, "blur all license plates for privacy")

        self.assertEqual(aligned["special"], [{"type": "ocr_redact", "params": {"sample_fps": 1.0, "confidence": 45}}])
        self.assertNotIn("video_filters", aligned)

    def test_face_and_license_request_keeps_face_privacy_blur_and_ocr_redact(self):
        plan = {
            "intent": "Blur faces and plates",
            "video_filters": [{"filter": "eq=contrast=1.05"}],
            "final_encode": server.default_final_encode(),
        }

        aligned = server.align_plan_with_command(plan, "blur faces and license plates for privacy")

        self.assertEqual(aligned["special"], [
            {"type": "face_privacy_blur", "params": {"target": "faces", "layout": "group"}},
            {"type": "ocr_redact", "params": {"sample_fps": 1.0, "confidence": 45}},
        ])
        self.assertEqual(aligned["video_filters"], [{"filter": "eq=contrast=1.05"}])

    def test_heuristic_plan_emits_face_privacy_special(self):
        plan = server.heuristic_plan("blur all faces for privacy")

        self.assertIn(
            {"type": "face_privacy_blur", "params": {"target": "faces", "layout": "group"}},
            plan["special"],
        )
        self.assertNotIn("__PRIVACY_X__", str(plan.get("video_filters", [])))

    def test_face_privacy_regions_are_dimension_based(self):
        regions = server.face_privacy_regions(1280, 720, {"target": "faces", "layout": "group"})

        self.assertEqual(len(regions), 3)
        for region in regions:
            self.assertGreater(region["w"], 0)
            self.assertGreater(region["h"], 0)
            self.assertLessEqual(region["x"] + region["w"], 1280)
            self.assertLessEqual(region["y"] + region["h"], 720)

    def test_face_privacy_filter_chain_supports_timed_opencv_detections(self):
        chain = server.face_privacy_filter_chain([
            {"x": 10, "y": 20, "w": 100, "h": 120, "start": 1.0, "end": 1.8}
        ])

        self.assertEqual(
            chain,
            "delogo=x=10:y=20:w=100:h=120:show=0:enable='between(t,1.000,1.800)'",
        )

    def test_face_privacy_uses_opencv_detections_when_available(self):
        detections = [{"x": 10, "y": 20, "w": 100, "h": 120, "start": 0.0, "end": 1.0}]
        job = {}

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "face_privacy.mp4"
            with (
                mock.patch.object(server, "ffprobe_video_dimensions", return_value=(1280, 720)),
                mock.patch.object(server, "detect_face_privacy_regions_with_opencv", return_value={
                    "mode": "opencv_cascade",
                    "detections": detections,
                    "sampled_frames": 4,
                }),
                mock.patch.object(server, "next_media_path", return_value=output_path),
                mock.patch.object(server, "run_video_filter_step"),
            ):
                result = server.apply_face_privacy_blur(
                    Path(tmp) / "input.mp4",
                    Path(tmp),
                    {"target": "faces", "layout": "group"},
                    job,
                )

        self.assertEqual(result, output_path)
        self.assertEqual(job["face_privacy_blur"]["mode"], "opencv_cascade")
        self.assertEqual(job["face_privacy_blur"]["regions"], detections)
        self.assertEqual(job["face_privacy_blur"]["opencv_detection"]["detected_regions"], 1)
        self.assertNotIn("warnings", job)

    def test_face_privacy_falls_back_when_opencv_detects_no_faces(self):
        job = {}

        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "face_privacy.mp4"
            with (
                mock.patch.object(server, "ffprobe_video_dimensions", return_value=(1280, 720)),
                mock.patch.object(server, "detect_face_privacy_regions_with_opencv", return_value={
                    "mode": "opencv_cascade",
                    "detections": [],
                    "sampled_frames": 4,
                }),
                mock.patch.object(server, "next_media_path", return_value=output_path),
                mock.patch.object(server, "run_video_filter_step"),
            ):
                result = server.apply_face_privacy_blur(
                    Path(tmp) / "input.mp4",
                    Path(tmp),
                    {"target": "faces", "layout": "group"},
                    job,
                )

        self.assertEqual(result, output_path)
        self.assertEqual(job["face_privacy_blur"]["mode"], "safe_regions_no_detection")
        self.assertEqual(len(job["face_privacy_blur"]["regions"]), 3)
        self.assertTrue(any("safe regions" in warning for warning in job["warnings"]))

    def test_heuristic_plan_emits_ocr_redact_special(self):
        plan = server.heuristic_plan("hide any visible screen text")

        self.assertEqual(plan["special"], [{"type": "ocr_redact", "params": {"sample_fps": 1.0, "confidence": 45}}])

    def test_black_screen_request_becomes_black_remove_special(self):
        plan = {
            "intent": "Remove black sections",
            "video_filters": [{"filter": "eq=contrast=1.05"}],
            "final_encode": server.default_final_encode(),
        }

        aligned = server.align_plan_with_command(plan, "remove all black screens and blank frames")

        self.assertIn(
            {
                "type": "black_remove",
                "params": {
                    "min_black_duration": 0.5,
                    "pixel_threshold": 0.1,
                    "picture_threshold": 0.98,
                },
            },
            aligned["special"],
        )

    def test_black_and_white_color_request_is_not_black_remove(self):
        self.assertFalse(server.black_remove_requested("make it black and white except red"))

    def test_heuristic_plan_emits_black_remove_special(self):
        plan = server.heuristic_plan("cut all blank screens from the clip")

        self.assertIn(
            {
                "type": "black_remove",
                "params": {
                    "min_black_duration": 0.5,
                    "pixel_threshold": 0.1,
                    "picture_threshold": 0.98,
                },
            },
            plan["special"],
        )

    def test_frozen_frames_request_becomes_freeze_remove_special(self):
        plan = {
            "intent": "Remove frozen sections",
            "video_filters": [{"filter": "eq=contrast=1.05"}],
            "final_encode": server.default_final_encode(),
        }

        aligned = server.align_plan_with_command(plan, "remove all frozen frames and stuck sections")

        self.assertIn({"type": "freeze_remove", "params": {"noise_db": -60, "min_duration": 0.5}}, aligned["special"])

    def test_add_freeze_frame_effect_is_not_freeze_remove(self):
        self.assertFalse(server.freeze_remove_requested("add freeze frame effect at the end"))

    def test_heuristic_plan_emits_freeze_remove_special(self):
        plan = server.heuristic_plan("cut out stuck frames from this video")

        self.assertIn({"type": "freeze_remove", "params": {"noise_db": -60, "min_duration": 0.5}}, plan["special"])

    def test_duplicate_frame_request_becomes_dedupe_special(self):
        plan = {
            "intent": "Remove duplicate frames",
            "video_filters": [{"filter": "eq=contrast=1.05"}],
            "final_encode": server.default_final_encode(),
        }

        aligned = server.align_plan_with_command(plan, "remove all duplicate and repeated frames")

        self.assertIn({"type": "dedupe_frames", "params": {"hi": 768, "lo": 320, "frac": 0.33, "max": 12}}, aligned["special"])

    def test_heuristic_plan_emits_dedupe_frames_special(self):
        plan = server.heuristic_plan("drop duplicate video frames")

        self.assertIn({"type": "dedupe_frames", "params": {"hi": 768, "lo": 320, "frac": 0.33, "max": 12}}, plan["special"])

    def test_remove_black_bars_becomes_crop_borders_special(self):
        plan = {
            "intent": "Remove letterbox bars",
            "video_filters": [{"filter": "crop=1920:804:0:138"}],
            "final_encode": server.default_final_encode(),
        }

        aligned = server.align_plan_with_command(plan, "remove the black bars around the video")

        self.assertIn({"type": "crop_borders", "params": {"limit": 24, "round": 2, "max_frames": 120}}, aligned["special"])

    def test_remove_letterbox_does_not_request_letterbox_output(self):
        plan = {
            "intent": "Remove letterbox bars",
            "special": [{"type": "crop_borders", "params": {"limit": 24, "round": 2, "max_frames": 120}}],
            "final_encode": server.default_final_encode(),
        }
        job = {"command": "remove the letterbox bars"}

        self.assertFalse(server.output_aspect_requested(job["command"]))
        self.assertIsNone(server.requested_output_format(job, plan))

    def test_heuristic_plan_emits_crop_borders_special(self):
        plan = server.heuristic_plan("crop out the pillarbox side bars")

        self.assertIn({"type": "crop_borders", "params": {"limit": 24, "round": 2, "max_frames": 120}}, plan["special"])

    def test_picture_in_picture_request_becomes_special_and_strips_multistream(self):
        plan = {
            "intent": "Picture in picture",
            "video_filters": [
                {
                    "filter": (
                        "eq=contrast=1.1,"
                        "split=2[v0][v1];[v1]scale=iw/2:ih/2[v1];"
                        "[v0][v1]overlay=W-w-5:H-h-5"
                    )
                }
            ],
            "final_encode": server.default_final_encode(),
        }

        aligned = server.align_plan_with_command(
            plan,
            "create a picture in picture duplicate of the video in the top right corner",
        )

        self.assertEqual(aligned["special"], [
            {"type": "picture_in_picture", "params": {"position": "top_right", "scale": 0.32}}
        ])
        self.assertEqual(aligned["video_filters"], [{"filter": "eq=contrast=1.1"}])

    def test_split_screen_mirror_request_becomes_special_and_strips_layout(self):
        plan = {
            "intent": "Split screen mirrored duplicate",
            "video_filters": [
                {"filter": "split=2"},
                {"filter": "[0:v]pad=iw*2:ih[left];[1:v]hflip[right];[left][right]hstack=inputs=2"},
                {"filter": "drawbox=x=iw/2-1:y=0:w=2:h=ih:color=white@1:t=fill"},
                {"filter": "eq=contrast=1.05"},
            ],
            "final_encode": server.default_final_encode(),
        }

        aligned = server.align_plan_with_command(
            plan,
            "create a split screen mirrored duplicate, left side normal and right side flipped",
        )

        self.assertEqual(aligned["special"], [
            {"type": "split_screen_mirror", "params": {"divider_color": "white"}}
        ])
        self.assertEqual(aligned["video_filters"], [{"filter": "eq=contrast=1.05"}])

    def test_vertical_blurred_background_request_becomes_special(self):
        plan = {
            "intent": "Vertical edit with blurred sides",
            "video_filters": [
                {"filter": "scale=-1:1080:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:color=black"},
                {"filter": "split=2[bg][fg];[bg]gblur=sigma=30[bg];[bg][fg]overlay=(W-w)/2:(H-h)/2"},
                {"filter": "eq=contrast=1.05"},
            ],
            "final_encode": server.default_final_encode(),
        }

        aligned = server.align_plan_with_command(
            plan,
            "make it vertical for reels with a blurred background instead of cropping the subject",
        )

        self.assertEqual(aligned["special"], [
            {"type": "blur_background", "params": {"width": 1080, "height": 1920, "sigma": 28}}
        ])
        self.assertEqual(aligned["video_filters"], [{"filter": "eq=contrast=1.05"}])
        self.assertIsNone(server.requested_output_format({"command": "vertical blurred background"}, aligned))

    def test_heuristic_plan_emits_blurred_background_special(self):
        plan = server.heuristic_plan("make this a TikTok with blurred sides")

        self.assertIn(
            {"type": "blur_background", "params": {"width": 1080, "height": 1920, "sigma": 28}},
            plan["special"],
        )

    def test_speed_request_becomes_factor_special_and_strips_generated_filters(self):
        plan = {
            "intent": "Speed up with natural pitch",
            "video_filters": [{"filter": "setpts=0.8*PTS,eq=contrast=1.05"}],
            "audio_filters": [{"filter": "rubberband=tempo=1.25:pitch=1.0,loudnorm=I=-14:TP=-1.5:LRA=11"}],
            "special": [{"type": "speed_ramp", "params": {"slow_factor": 0.75, "fast_factor": 1.25}}],
            "final_encode": server.default_final_encode(),
        }

        aligned = server.align_plan_with_command(
            plan,
            "make the whole clip 25 percent faster but keep audio pitch natural",
        )

        self.assertEqual(aligned["special"], [{"type": "speed_ramp", "params": {"factor": 1.25}}])
        self.assertEqual(aligned["video_filters"], [{"filter": "eq=contrast=1.05"}])
        self.assertEqual(aligned["audio_filters"], [{"filter": "loudnorm=I=-14:TP=-1.5:LRA=11"}])

    def test_speed_factor_parser_handles_common_phrases(self):
        self.assertEqual(
            server.speed_special_from_command("make it 25 percent faster"),
            {"type": "speed_ramp", "params": {"factor": 1.25}},
        )
        self.assertEqual(
            server.speed_special_from_command("slow it down by 50 percent"),
            {"type": "speed_ramp", "params": {"factor": 0.5}},
        )
        self.assertEqual(
            server.speed_special_from_command("play at half speed"),
            {"type": "speed_ramp", "params": {"factor": 0.5}},
        )

    def test_pitch_request_becomes_special_and_strips_generated_rubberband(self):
        plan = {
            "intent": "Cave echo and lower pitch",
            "audio_filters": [
                {"filter": "aecho=0.8:0.4:500:0.4"},
                {"filter": "rubberband=tempo=1.0:pitch=0.75"},
            ],
            "final_encode": server.default_final_encode(),
        }

        aligned = server.align_plan_with_command(
            plan,
            "make the audio sound like a cave echo and lower the pitch three semitones",
        )

        self.assertEqual(aligned["special"], [{"type": "pitch_shift", "params": {"semitones": -3.0}}])
        self.assertEqual(aligned["audio_filters"], [{"filter": "aecho=0.8:0.4:500:0.4"}])

    def test_pitch_parser_handles_natural_pitch_and_words(self):
        self.assertEqual(
            server.pitch_special_from_command("lower the pitch three semitones"),
            {"type": "pitch_shift", "params": {"semitones": -3.0}},
        )
        self.assertEqual(
            server.pitch_special_from_command("raise pitch by 2 semitones"),
            {"type": "pitch_shift", "params": {"semitones": 2.0}},
        )
        self.assertIsNone(server.pitch_special_from_command("speed up but keep audio pitch natural"))

    def test_pitch_ratio_from_semitones(self):
        self.assertAlmostEqual(server.pitch_ratio_from_semitones(-12), 0.5)
        self.assertAlmostEqual(server.pitch_ratio_from_semitones(0), 1.0)
        self.assertAlmostEqual(server.pitch_ratio_from_semitones(12), 2.0)

    def test_pitch_shift_uses_ffmpeg_rubberband_filter_fast_path(self):
        commands = []
        job = {}

        def fake_run_command(command, **_kwargs):
            commands.append(command)

        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = Path(tmpdir)
            input_path = job_dir / "input.mp4"
            input_path.write_bytes(b"video")
            output_path = job_dir / "pitch_shift.mp4"
            with (
                mock.patch.object(server, "ffprobe_has_audio", return_value=True),
                mock.patch.object(server, "next_media_path", return_value=output_path),
                mock.patch.object(server, "run_command", side_effect=fake_run_command),
            ):
                result = server.apply_pitch_shift(input_path, job_dir, {"semitones": -3}, job)

        self.assertEqual(result, output_path)
        self.assertEqual(len(commands), 1)
        self.assertIn("-af", commands[0])
        self.assertIn("rubberband=pitch=0.84089642:pitchq=quality", commands[0])
        self.assertEqual(job["pitch_shift"]["mode"], "ffmpeg_rubberband_filter")

    def test_pitch_shift_falls_back_to_cli_when_ffmpeg_filter_fails(self):
        commands = []
        job = {}

        def fake_run_command(command, **_kwargs):
            commands.append(command)
            if len(commands) == 1:
                raise RuntimeError("ffmpeg rubberband unavailable")

        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = Path(tmpdir)
            input_path = job_dir / "input.mp4"
            input_path.write_bytes(b"video")
            output_path = job_dir / "pitch_shift.mp4"
            with (
                mock.patch.object(server, "ffprobe_has_audio", return_value=True),
                mock.patch.object(server, "next_media_path", return_value=output_path),
                mock.patch.object(server, "run_command", side_effect=fake_run_command),
            ):
                result = server.apply_pitch_shift(input_path, job_dir, {"semitones": 2}, job)

        self.assertEqual(result, output_path)
        self.assertEqual(len(commands), 4)
        self.assertEqual(commands[1][0:4], ["ffmpeg", "-y", "-i", str(input_path)])
        self.assertEqual(commands[2][0], "rubberband")
        self.assertEqual(commands[3][0:4], ["ffmpeg", "-y", "-i", str(input_path)])
        self.assertEqual(job["pitch_shift"]["mode"], "rubberband_cli_fallback")
        self.assertTrue(any("retrying rubberband CLI" in warning for warning in job["warnings"]))

    def test_end_reverse_replaces_full_clip_reverse_filter(self):
        plan = {
            "intent": "End with reverse motion",
            "video_filters": [
                {
                    "filter": "reverse,zoompan=z='min(zoom+0.001,2.0)':x='0':y='0':d=1:s=1920x1080"
                }
            ],
            "special": [{"type": "speed_ramp", "params": {"slow_factor": 0.5, "fast_factor": 2}}],
            "final_encode": {
                "vcodec": "libx264",
                "crf": 22,
                "preset": "fast",
                "acodec": "aac",
                "audio_bitrate": "192k",
            },
        }

        aligned = server.align_plan_with_command(plan, "then end with a dramatic reverse-motion sequence")

        self.assertNotIn("reverse", aligned["video_filters"][0]["filter"])
        self.assertIn("zoompan", aligned["video_filters"][0]["filter"])
        self.assertEqual([step["type"] for step in aligned["special"]], ["speed_ramp", "end_reverse"])

    def test_text_rollout_removes_full_frame_overlay_and_adds_moving_text(self):
        plan = {
            "intent": "Fade text and roll out",
            "video_filters": [
                {
                    "filter": (
                        "drawtext=text='Hello World':fontcolor=white:fontsize=48:"
                        "x=(w-text_w)/2:y=(h-text_h)/2:enable='between(t,0,3)',"
                        "drawbox=x=0:y=0:w=iw:h=ih:color=white@0.9:t=fill:enable='between(t,3,6)'"
                    )
                }
            ],
            "final_encode": {
                "vcodec": "libx264",
                "crf": 22,
                "preset": "fast",
                "acodec": "aac",
                "audio_bitrate": "192k",
            },
        }

        aligned = server.align_plan_with_command(
            plan,
            "fade in the text Hello World and let the text disappear with a rolling out animation",
        )

        filters = [step["filter"] for step in aligned["video_filters"]]
        self.assertEqual(len(filters), 2)
        self.assertNotIn("drawbox", ",".join(filters))
        self.assertIn("(t-3)*220", filters[1])
        self.assertIn("Hello World", filters[1])

    def test_text_rollout_replaces_self_referential_y_motion(self):
        plan = {
            "intent": "Fade text and roll out",
            "video_filters": [
                {
                    "filter": (
                        "drawtext=text='Hello World':fontcolor=white:fontsize=48:"
                        "x=(w-text_w)/2:y=(h-text_h)/2:enable='between(t,0,3)'"
                    )
                },
                {
                    "filter": (
                        "drawtext=text='Hello World':fontcolor=white:fontsize=48:"
                        "x=(w-text_w)/2:y=(h-text_h)/2+y*(t-3):enable='between(t,3,6)'"
                    )
                },
            ],
            "final_encode": {
                "vcodec": "libx264",
                "crf": 22,
                "preset": "fast",
                "acodec": "aac",
                "audio_bitrate": "192k",
            },
        }

        aligned = server.align_plan_with_command(
            plan,
            "fade in the text Hello World and let the text disappear with a rolling out animation",
        )

        filters = [step["filter"] for step in aligned["video_filters"]]
        self.assertEqual(len(filters), 2)
        self.assertNotIn("y*(t-3)", ",".join(filters))
        self.assertIn("x='(w-text_w)/2+(t-3)*220'", filters[1])

    def test_text_rollout_adds_missing_outro_animation(self):
        plan = {
            "intent": "Fade text and roll out",
            "video_filters": [
                {
                    "filter": (
                        "drawtext=text='Hello World':fontcolor=white:fontsize=48:"
                        "x=(w-text_w)/2:y=(h-text_h)/2:enable='between(t,0,3)'"
                    )
                }
            ],
            "final_encode": {
                "vcodec": "libx264",
                "crf": 22,
                "preset": "fast",
                "acodec": "aac",
                "audio_bitrate": "192k",
            },
        }

        aligned = server.align_plan_with_command(
            plan,
            "fade in the text Hello World and let the text disappear with a rolling out animation",
        )

        filters = [step["filter"] for step in aligned["video_filters"]]
        self.assertEqual(len(filters), 2)
        self.assertIn("(t-3)*220", filters[1])
        self.assertIn("between(t,3,6)", filters[1])

    def test_social_aspect_request_strips_model_scale_pad_chain(self):
        plan = {
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
            "final_encode": {
                "vcodec": "libx264",
                "crf": 22,
                "preset": "fast",
                "acodec": "aac",
                "audio_bitrate": "192k",
            },
        }

        aligned = server.align_plan_with_command(
            plan,
            "export this as Instagram portrait 4:5 with warm color grade",
        )

        filters = ",".join(step["filter"] for step in aligned["video_filters"])
        self.assertIn("colorbalance", filters)
        self.assertNotIn("scale=-1:1080", filters)
        self.assertNotIn("pad=1080:1350", filters)

    def test_square_aspect_request_strips_expression_crop_scale_chain(self):
        plan = {
            "intent": "Square export with blue hold",
            "video_filters": [
                {"filter": server.COLOR_HOLD_FILTERS["blue"]},
                {
                    "filter": "crop=iw/2:ih/2:(iw-iw/2)/2:(ih-ih/2)/2,scale=-1:1",
                },
            ],
            "final_encode": {
                "vcodec": "libx264",
                "crf": 22,
                "preset": "fast",
                "acodec": "aac",
                "audio_bitrate": "192k",
            },
        }

        aligned = server.align_plan_with_command(
            plan,
            "make everything black and white except blue lights and turn it into a square 1:1 post",
        )

        filters = ",".join(step["filter"] for step in aligned["video_filters"])
        self.assertIn("colorhold=color=blue", filters)
        self.assertNotIn("crop=iw/2", filters)
        self.assertNotIn("scale=-1:1", filters)

    def test_square_aspect_request_strips_iw_square_crop_scale_chain(self):
        plan = {
            "intent": "Square export with blue hold",
            "video_filters": [
                {"filter": server.COLOR_HOLD_FILTERS["blue"]},
                {
                    "filter": "crop=iw:iw:0:0,scale=iw:iw",
                },
            ],
            "final_encode": {
                "vcodec": "libx264",
                "crf": 22,
                "preset": "fast",
                "acodec": "aac",
                "audio_bitrate": "192k",
            },
        }

        aligned = server.align_plan_with_command(
            plan,
            "make everything black and white except blue lights and turn it into a square 1:1 post",
        )

        filters = ",".join(step["filter"] for step in aligned["video_filters"])
        self.assertIn("colorhold=color=blue", filters)
        self.assertNotIn("crop=iw:iw", filters)
        self.assertNotIn("scale=iw:iw", filters)

    def test_privacy_blur_request_overrides_unstable_generated_mask(self):
        plan = {
            "intent": "Blur center and darken outside",
            "video_filters": [
                {
                    "filter": (
                        "noise=alls=18:allf=t+u,"
                        "geq=r='255*(1-p(X,Y))':g='255*(1-p(X,Y))':b='255*(1-p(X,Y))','description="
                    )
                }
            ],
            "final_encode": {
                "vcodec": "libx264",
                "crf": 22,
                "preset": "fast",
                "acodec": "aac",
                "audio_bitrate": "192k",
            },
        }

        aligned = server.align_plan_with_command(
            plan,
            "blur the center like a privacy blur and darken everything outside it",
        )

        filters = [step["filter"] for step in aligned["video_filters"]]
        self.assertEqual(len(filters), 1)
        self.assertIn("__PRIVACY_X__", filters[0])
        self.assertNotIn("description", filters[0])
        self.assertNotIn("geq", filters[0])


class FinalEncodeTests(unittest.TestCase):
    def test_requested_output_dimensions_handles_common_resolution_language(self):
        self.assertEqual(
            server.requested_output_dimensions("export this as sharp 4k UHD"),
            {"width": 3840, "height": 2160},
        )
        self.assertEqual(
            server.requested_output_dimensions("make it vertical 4k for reels"),
            {"width": 2160, "height": 3840},
        )
        self.assertEqual(
            server.requested_output_dimensions("make this a square 1080p social post"),
            {"width": 1080, "height": 1080},
        )

    def test_alignment_sets_final_encode_dimensions_for_4k_export(self):
        plan = {
            "intent": "Preserve edit and export",
            "video_filters": [{"filter": "eq=contrast=1.05"}],
            "final_encode": server.default_final_encode(),
        }

        aligned = server.align_plan_with_command(
            plan,
            "export this as sharp 4k UHD while preserving the edit",
        )

        self.assertEqual(aligned["final_encode"]["width"], 3840)
        self.assertEqual(aligned["final_encode"]["height"], 2160)
        self.assertEqual(aligned["video_filters"], [{"filter": "eq=contrast=1.05"}])

    def test_final_encode_settings_are_sanitized(self):
        job = {}

        settings = server.normalized_final_encode_settings(
            {
                "vcodec": "fake-video-codec",
                "crf": "not-a-number",
                "preset": "impossible",
                "acodec": "fake-audio-codec",
                "audio_bitrate": "nope",
                "width": "1279",
                "height": "719",
            },
            job,
        )

        self.assertEqual(settings["vcodec"], "libx264")
        self.assertEqual(settings["crf"], 22)
        self.assertEqual(settings["preset"], "fast")
        self.assertEqual(settings["acodec"], "aac")
        self.assertEqual(settings["audio_bitrate"], "192k")
        self.assertEqual(settings["width"], 1278)
        self.assertEqual(settings["height"], 718)
        self.assertTrue(any("final encode settings normalized" in warning for warning in job["warnings"]))

    def test_final_encode_dimension_requires_valid_pair(self):
        job = {}

        settings = server.normalized_final_encode_settings(
            {"width": "vertical", "height": "1920"},
            job,
        )

        self.assertNotIn("width", settings)
        self.assertNotIn("height", settings)
        self.assertTrue(any("dimensions" in warning for warning in job["warnings"]))

    def test_final_streams_are_copy_compatible_for_default_h264_aac(self):
        with mock.patch.object(server, "ffprobe_streams", return_value=[
            {"codec_type": "video", "codec_name": "h264"},
            {"codec_type": "audio", "codec_name": "aac"},
        ]):
            self.assertTrue(
                server.final_streams_are_copy_compatible(
                    Path("input.mp4"),
                    server.default_final_encode(),
                )
            )

    def test_final_stream_copy_is_disabled_when_resize_is_requested(self):
        settings = server.default_final_encode()
        settings.update({"width": 3840, "height": 2160})
        with mock.patch.object(server, "ffprobe_streams") as streams:
            self.assertFalse(server.final_streams_are_copy_compatible(Path("input.mp4"), settings))

        streams.assert_not_called()

    def test_final_encode_uses_stream_copy_for_compatible_default_output(self):
        commands = []
        job = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = Path(tmpdir)
            input_path = job_dir / "input.mp4"
            input_path.write_bytes(b"video")
            with (
                mock.patch.object(server, "ffprobe_streams", return_value=[
                    {"codec_type": "video", "codec_name": "h264"},
                    {"codec_type": "audio", "codec_name": "aac"},
                ]),
                mock.patch.object(server, "run_command", side_effect=lambda command: commands.append(command)),
            ):
                result = server.final_encode(input_path, job_dir, server.default_final_encode(), job)

        self.assertEqual(result.name, "linguist_output.mp4")
        self.assertEqual(len(commands), 1)
        self.assertIn("-c", commands[0])
        self.assertIn("copy", commands[0])
        self.assertEqual(job["final_encode_execution"], {"mode": "stream_copy"})

    def test_final_encode_falls_back_to_encode_when_stream_copy_fails(self):
        commands = []
        job = {}

        def fake_run_command(command):
            commands.append(command)
            if len(commands) == 1:
                raise RuntimeError("copy failed")

        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = Path(tmpdir)
            input_path = job_dir / "input.mp4"
            input_path.write_bytes(b"video")
            with (
                mock.patch.object(server, "ffprobe_streams", return_value=[
                    {"codec_type": "video", "codec_name": "h264"},
                    {"codec_type": "audio", "codec_name": "aac"},
                ]),
                mock.patch.object(server, "run_command", side_effect=fake_run_command),
            ):
                result = server.final_encode(input_path, job_dir, server.default_final_encode(), job)

        self.assertEqual(result.name, "linguist_output.mp4")
        self.assertEqual(len(commands), 2)
        self.assertIn("copy", commands[0])
        self.assertIn("libx264", commands[1])
        self.assertEqual(job["final_encode_execution"], {"mode": "encode"})
        self.assertTrue(any("final stream copy failed" in warning for warning in job["warnings"]))


class VideoExecutionTests(unittest.TestCase):
    def test_apply_video_filters_combines_filters_into_one_ffmpeg_pass(self):
        calls = []
        job = {}

        def fake_run_video_filter_step(current, output_path, chain):
            calls.append((current, output_path, chain))

        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = Path(tmpdir)
            input_path = job_dir / "input.mp4"
            input_path.write_bytes(b"video")
            output_path = job_dir / "video_filters.mp4"
            with (
                mock.patch.object(server, "ffprobe_duration", return_value=6.0),
                mock.patch.object(server, "ffprobe_video_dimensions", return_value=(1280, 720)),
                mock.patch.object(server, "next_media_path", return_value=output_path),
                mock.patch.object(server, "run_video_filter_step", side_effect=fake_run_video_filter_step),
            ):
                result = server.apply_video_filters(
                    input_path,
                    job_dir,
                    [
                        {"filter": "eq=contrast=1.1:saturation=1.2"},
                        {"filter": "vignette=angle=PI/3"},
                    ],
                    {},
                    job,
                )

        self.assertEqual(result, output_path)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][2], "eq=contrast=1.1:saturation=1.2,vignette=angle=PI/3")
        self.assertEqual(job["video_filter_execution"], {"mode": "combined", "filter_count": 2})

    def test_apply_video_filters_falls_back_to_step_by_step_when_combined_chain_fails(self):
        calls = []
        job = {}

        def fake_run_video_filter_step(current, output_path, chain):
            calls.append((current, output_path, chain))
            if len(calls) == 1:
                raise RuntimeError("combined chain rejected")

        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = Path(tmpdir)
            input_path = job_dir / "input.mp4"
            input_path.write_bytes(b"video")
            paths = [
                job_dir / "combined.mp4",
                job_dir / "step1.mp4",
                job_dir / "step2.mp4",
            ]
            with (
                mock.patch.object(server, "ffprobe_duration", return_value=6.0),
                mock.patch.object(server, "ffprobe_video_dimensions", return_value=(1280, 720)),
                mock.patch.object(server, "next_media_path", side_effect=paths),
                mock.patch.object(server, "run_video_filter_step", side_effect=fake_run_video_filter_step),
            ):
                result = server.apply_video_filters(
                    input_path,
                    job_dir,
                    [
                        {"filter": "eq=contrast=1.1:saturation=1.2"},
                        {"filter": "vignette=angle=PI/3"},
                    ],
                    {},
                    job,
                )

        self.assertEqual(result, paths[-1])
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[0][2], "eq=contrast=1.1:saturation=1.2,vignette=angle=PI/3")
        self.assertEqual(calls[1][2], "eq=contrast=1.1:saturation=1.2")
        self.assertEqual(calls[2][2], "vignette=angle=PI/3")
        self.assertEqual(job["video_filter_execution"], {"mode": "step_by_step", "filter_count": 2})
        self.assertTrue(any("combined video filter chain failed" in warning for warning in job["warnings"]))


class AudioExecutionTests(unittest.TestCase):
    def test_apply_audio_filters_combines_filters_into_one_ffmpeg_pass(self):
        commands = []
        job = {}

        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = Path(tmpdir)
            input_path = job_dir / "input.mp4"
            input_path.write_bytes(b"video")
            output_path = job_dir / "audio_filters.mp4"
            with (
                mock.patch.object(server, "ffprobe_has_audio", return_value=True),
                mock.patch.object(server, "next_media_path", return_value=output_path),
                mock.patch.object(server, "run_command", side_effect=lambda command: commands.append(command)),
            ):
                result = server.apply_audio_filters(
                    input_path,
                    job_dir,
                    [
                        {"filter": "highpass=f=300"},
                        {"filter": "lowpass=f=3400"},
                        {"filter": "loudnorm=I=-14:TP=-1.5:LRA=11"},
                    ],
                    job=job,
                )

        self.assertEqual(result, output_path)
        self.assertEqual(len(commands), 1)
        self.assertIn("-af", commands[0])
        self.assertIn("highpass=f=300,lowpass=f=3400,loudnorm=I=-14:TP=-1.5:LRA=11", commands[0])
        self.assertEqual(job["audio_filter_execution"], {"mode": "combined", "filter_count": 3})

    def test_apply_audio_filters_falls_back_to_step_by_step_when_combined_chain_fails(self):
        commands = []
        job = {}

        def fake_run_command(command):
            commands.append(command)
            if len(commands) == 1:
                raise RuntimeError("combined chain rejected")

        with tempfile.TemporaryDirectory() as tmpdir:
            job_dir = Path(tmpdir)
            input_path = job_dir / "input.mp4"
            input_path.write_bytes(b"video")
            paths = [
                job_dir / "combined.mp4",
                job_dir / "step1.mp4",
                job_dir / "step2.mp4",
            ]
            with (
                mock.patch.object(server, "ffprobe_has_audio", return_value=True),
                mock.patch.object(server, "next_media_path", side_effect=paths),
                mock.patch.object(server, "run_command", side_effect=fake_run_command),
            ):
                result = server.apply_audio_filters(
                    input_path,
                    job_dir,
                    [
                        {"filter": "highpass=f=300"},
                        {"filter": "lowpass=f=3400"},
                    ],
                    job=job,
                )

        self.assertEqual(result, paths[-1])
        self.assertEqual(len(commands), 3)
        self.assertIn("highpass=f=300,lowpass=f=3400", commands[0])
        self.assertIn("highpass=f=300", commands[1])
        self.assertIn("lowpass=f=3400", commands[2])
        self.assertEqual(job["audio_filter_execution"], {"mode": "step_by_step", "filter_count": 2})
        self.assertTrue(any("combined audio filter chain failed" in warning for warning in job["warnings"]))


class AnalysisFallbackTests(unittest.TestCase):
    def test_run_analysis_uses_synthetic_times_when_audio_is_missing(self):
        plan = {
            "analysis": [{"tool": "librosa", "function": "beat_track", "store_as": "beat_times"}],
            "final_encode": server.default_final_encode(),
        }
        job = {}

        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch.object(server, "source_audio_for_analysis", side_effect=RuntimeError("no audio")),
                mock.patch.object(server, "ffprobe_duration", return_value=2.0),
            ):
                context = server.run_analysis(plan, job, Path(tmp), Path(tmp) / "video.mp4")

        self.assertEqual(context["beat_times"], [0.5, 1.0, 1.5, 2.0])
        self.assertTrue(any("audio analysis fallback used synthetic timing" in warning for warning in job["warnings"]))


class ExecutionManifestTests(unittest.TestCase):
    def test_manifest_tracks_step_progress_by_phase(self):
        internal_plan = {
            "plan_id": "plan123",
            "steps": [
                {
                    "id": "analysis_1_beat_track",
                    "type": "analysis",
                    "worker": "AudioAnalysisWorker",
                    "depends_on": ["asset:audio_or_video"],
                    "confidence": 0.88,
                    "confidence_band": "execute",
                    "legacy_ref": {"section": "analysis", "index": 0},
                },
                {
                    "id": "video_1_shake",
                    "type": "video_filter",
                    "worker": "VideoFilterWorker",
                    "depends_on": ["analysis_1_beat_track"],
                    "confidence": 0.76,
                    "confidence_band": "execute_with_guardrails",
                    "legacy_ref": {"section": "video_filters", "index": 0},
                },
            ],
        }

        manifest = server.create_execution_manifest(internal_plan)
        self.assertEqual(manifest["progress"]["percent"], 0)
        self.assertEqual(manifest["steps"][0]["legacy_ref"]["section"], "analysis")

        manifest = server.mark_manifest_step_status(
            manifest,
            "running",
            step_type="analysis",
            legacy_section="analysis",
        )
        self.assertEqual(manifest["progress"]["active_step_id"], "analysis_1_beat_track")

        manifest = server.mark_manifest_step_status(
            manifest,
            "complete",
            step_type="analysis",
            legacy_section="analysis",
        )
        self.assertEqual(manifest["progress"]["complete_steps"], 1)
        self.assertEqual(manifest["progress"]["percent"], 50)
        self.assertEqual(manifest["steps"][1]["status"], "queued")


class ProductionPlanValidationTests(unittest.TestCase):
    def test_unsupported_special_operation_blocks_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "input.mp4"
            video_path.write_bytes(b"placeholder")
            plan = {
                "intent": "Invented special operation",
                "special": [{"type": "teleport_subject", "params": {}}],
                "final_encode": server.default_final_encode(),
            }
            _plan, internal_plan = server.prepare_production_plan(
                "teleport the subject across the frame",
                plan,
                {"video_path": str(video_path), "video_name": "input.mp4"},
                {"executor": {"ffmpeg_ready": True}},
            )

        validation = internal_plan["validation"]
        self.assertFalse(server.validation_allows_execution(internal_plan))
        self.assertTrue(any(issue["code"] == "unsupported_special_type" for issue in validation["issues"]))

    def test_unsupported_analysis_function_blocks_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "input.mp4"
            video_path.write_bytes(b"placeholder")
            plan = {
                "intent": "Invented analysis operation",
                "analysis": [{"tool": "librosa", "function": "emotion_track", "store_as": "emotion_times"}],
                "video_filters": [
                    {
                        "description": "Pulse from unsupported context",
                        "filter": "eq=brightness=0.2",
                        "requires_context": "emotion_times",
                        "timing": "per_beat",
                    }
                ],
                "final_encode": server.default_final_encode(),
            }
            _plan, internal_plan = server.prepare_production_plan(
                "pulse whenever the video feels emotional",
                plan,
                {"video_path": str(video_path), "video_name": "input.mp4"},
                {"executor": {"ffmpeg_ready": True}},
            )

        validation = internal_plan["validation"]
        self.assertFalse(server.validation_allows_execution(internal_plan))
        self.assertTrue(any(issue["code"] == "unsupported_analysis_function" for issue in validation["issues"]))

    def test_production_plan_normalizes_final_encode_before_execution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "input.mp4"
            video_path.write_bytes(b"placeholder")
            plan = {
                "intent": "Bad export settings from model",
                "video_filters": [{"filter": "eq=contrast=1.08"}],
                "final_encode": {
                    "vcodec": "h264",
                    "crf": "99",
                    "preset": "absurd",
                    "acodec": "wavpack",
                    "audio_bitrate": "9000k",
                    "width": 1921,
                    "height": 1081,
                },
            }
            repaired_plan, internal_plan = server.prepare_production_plan(
                "make it clearer and export safely",
                plan,
                {"video_path": str(video_path), "video_name": "input.mp4"},
                {"executor": {"ffmpeg_ready": True}},
            )

        self.assertTrue(server.validation_allows_execution(internal_plan))
        self.assertEqual(
            repaired_plan["final_encode"],
            {
                "vcodec": "libx264",
                "crf": 35,
                "preset": "fast",
                "acodec": "aac",
                "audio_bitrate": "512k",
                "width": 1920,
                "height": 1080,
            },
        )
        self.assertTrue(any("normalized final_encode" in fix for fix in internal_plan["validation"]["fixes"]))


class FallbackPlanningTests(unittest.TestCase):
    def setUp(self):
        server.clear_plan_cache()

    def test_heuristic_plan_handles_beat_shake_when_ai_is_unavailable(self):
        with mock.patch.object(server, "call_nim", side_effect=RuntimeError("offline")):
            plan = server.build_plan("shake the frame violently on every beat")

        self.assertEqual(plan["analysis"][0]["function"], "beat_track")
        self.assertIn("crop=iw-60", plan["video_filters"][0]["filter"])
        self.assertEqual(plan["video_filters"][0]["timing"], "per_beat")

    def test_heuristic_plan_preserves_beat_zoom_request(self):
        plan = server.heuristic_plan(
            "make the video pulse brighter and zoom slightly on every beat of the uploaded audio"
        )

        filters = [step["filter"] for step in plan["video_filters"]]
        self.assertTrue(any("eq=brightness" in value for value in filters))
        self.assertTrue(any("zoompan" in value for value in filters))
        self.assertTrue(all(step.get("timing") == "per_beat" for step in plan["video_filters"]))

    def test_heuristic_visual_stress_without_audio_sync_has_no_analysis(self):
        plan = server.align_plan_with_command(
            server.heuristic_plan(
                "create a dramatic final sequence: start with a cinematic teal orange grade, "
                "add subtle film grain, add a white flash every second, make the final 1.5 "
                "seconds reverse back like a rewind ending, but do it without relying on "
                "audio synchronization"
            ),
            "create a dramatic final sequence: start with a cinematic teal orange grade, "
            "add subtle film grain, add a white flash every second, make the final 1.5 "
            "seconds reverse back like a rewind ending, but do it without relying on "
            "audio synchronization",
        )

        self.assertNotIn("analysis", plan)
        self.assertNotIn("audio_filters", plan)
        self.assertTrue(all("requires_context" not in step for step in plan.get("video_filters", [])))
        self.assertIn({"type": "end_reverse", "params": {"duration": 1.5}}, plan["special"])

    def test_heuristic_plan_uses_real_speed_factor(self):
        plan = server.heuristic_plan("make the whole clip 25 percent faster but keep audio pitch natural")

        self.assertEqual(plan["special"], [{"type": "speed_ramp", "params": {"factor": 1.25}}])

    def test_heuristic_plan_treats_punctuated_slow_as_speed_change(self):
        plan = server.heuristic_plan("make it feel like a memory fading, warm, dreamlike, slow, with reverb")

        self.assertIn({"type": "speed_ramp", "params": {"factor": 0.75}}, plan["special"])

    def test_heuristic_dream_plan_uses_single_visual_pass(self):
        plan = server.heuristic_plan("make it feel like a memory fading, warm, dreamlike, slow, with reverb")

        self.assertEqual(len(plan["video_filters"]), 1)
        self.assertIn("colorbalance=", plan["video_filters"][0]["filter"])
        self.assertIn("gblur=sigma=0.9", plan["video_filters"][0]["filter"])
        self.assertIn("vignette=", plan["video_filters"][0]["filter"])

    def test_heuristic_plan_sets_4k_export_dimensions(self):
        plan = server.heuristic_plan("export this as sharp 4k UHD while preserving the edit")

        self.assertEqual(plan["final_encode"]["width"], 3840)
        self.assertEqual(plan["final_encode"]["height"], 2160)

    def test_speed_parser_does_not_confuse_slow_fade_with_clip_speed(self):
        self.assertIsNone(server.speed_special_from_command("add a slow fade to black"))

    def test_speed_ramp_uses_explicit_fast_intermediate_encode(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "input.mp4"
            output_path = Path(tmp) / "speed.mp4"
            captured = []

            with (
                mock.patch.object(server, "next_media_path", return_value=output_path),
                mock.patch.object(server, "ffprobe_has_audio", return_value=True),
                mock.patch.object(server, "run_command", side_effect=lambda command: captured.append(command)),
            ):
                result = server.apply_special_step(
                    input_path,
                    Path(tmp),
                    {"type": "speed_ramp", "params": {"factor": 0.75}},
                )

        self.assertEqual(result, output_path)
        command = captured[0]
        self.assertIn("-preset", command)
        self.assertIn("fast", command)
        self.assertIn("-c:a", command)
        self.assertIn("aac", command)

    def test_heuristic_plan_uses_precise_pitch_special(self):
        plan = server.heuristic_plan("make the audio sound like a cave echo and lower the pitch three semitones")

        self.assertIn({"type": "pitch_shift", "params": {"semitones": -3.0}}, plan["special"])
        self.assertTrue(any(step["filter"].startswith("aecho=") for step in plan["audio_filters"]))

    def test_heuristic_old_telephone_request_does_not_add_visual_film_grain(self):
        plan = server.heuristic_plan("make the audio sound like an old telephone call and normalize speech volume")

        video_filters = ",".join(step["filter"] for step in plan.get("video_filters", []))
        audio_filters = ",".join(step["filter"] for step in plan.get("audio_filters", []))
        self.assertNotIn("noise=alls", video_filters)
        self.assertIn("highpass=f=300", audio_filters)
        self.assertIn("lowpass=f=3400", audio_filters)

    def test_heuristic_old_footage_request_can_add_visual_grain(self):
        plan = server.heuristic_plan("make this old footage look grainy")

        video_filters = ",".join(step["filter"] for step in plan.get("video_filters", []))
        self.assertIn("noise=alls=18", video_filters)

    def test_build_plan_records_heuristic_fallback_on_job(self):
        job = {}

        with mock.patch.object(server, "call_nim", side_effect=RuntimeError("offline")):
            plan = server.build_plan("shake the frame violently on every beat", job)

        self.assertEqual(job["planner"], "heuristic")
        self.assertTrue(any("NIM planning failed" in warning for warning in job["warnings"]))
        self.assertEqual(plan["analysis"][0]["function"], "beat_track")

    def test_build_plan_records_nim_source_on_job(self):
        job = {}
        nim_plan = {
            "intent": "Subtle contrast edit",
            "video_filters": [{"filter": "eq=contrast=1.08:saturation=1.05"}],
            "final_encode": server.default_final_encode(),
        }

        with mock.patch.object(server, "call_nim", return_value=nim_plan):
            plan = server.build_plan("make it clearer", job)

        self.assertEqual(job["planner"], "nim")
        self.assertEqual(plan["intent"], "Subtle contrast edit")

    def test_build_plan_uses_cache_for_repeated_job_prompt(self):
        nim_plan = {
            "intent": "Cached contrast edit",
            "video_filters": [{"filter": "eq=contrast=1.08:saturation=1.05"}],
            "final_encode": server.default_final_encode(),
        }

        with (
            mock.patch.object(server, "runtime_capability_prompt", return_value="runtime"),
            mock.patch.object(server, "call_nim", return_value=nim_plan) as call_nim,
        ):
            first_job = {}
            second_job = {}
            first = server.build_plan("make it clearer", first_job)
            second = server.build_plan("  Make it   clearer  ", second_job)

        self.assertEqual(call_nim.call_count, 1)
        self.assertEqual(first["intent"], "Cached contrast edit")
        self.assertEqual(second["intent"], "Cached contrast edit")
        self.assertEqual(first_job["planner_cache"], {"status": "miss"})
        self.assertEqual(second_job["planner"], "nim")
        self.assertEqual(second_job["planner_cache"]["status"], "hit")

    def test_heuristic_plan_cache_uses_short_ttl(self):
        plan = {
            "intent": "Fallback edit",
            "video_filters": [{"filter": "eq=contrast=1.08:saturation=1.05"}],
            "final_encode": server.default_final_encode(),
        }

        with (
            mock.patch.object(server, "PLAN_CACHE_SECONDS", 900),
            mock.patch.object(server, "FALLBACK_PLAN_CACHE_SECONDS", 5),
            mock.patch.object(server.time, "time", return_value=100),
        ):
            server.store_cached_plan("fallback-key", plan, "heuristic")

        with (
            mock.patch.object(server, "PLAN_CACHE_SECONDS", 900),
            mock.patch.object(server.time, "time", return_value=104),
        ):
            cached = server.get_cached_plan("fallback-key")
        self.assertEqual(cached["source"], "heuristic")

        with (
            mock.patch.object(server, "PLAN_CACHE_SECONDS", 900),
            mock.patch.object(server.time, "time", return_value=106),
        ):
            self.assertIsNone(server.get_cached_plan("fallback-key"))

    def test_nim_plan_cache_uses_standard_ttl(self):
        plan = {
            "intent": "NIM edit",
            "video_filters": [{"filter": "eq=contrast=1.08:saturation=1.05"}],
            "final_encode": server.default_final_encode(),
        }

        with (
            mock.patch.object(server, "PLAN_CACHE_SECONDS", 900),
            mock.patch.object(server, "FALLBACK_PLAN_CACHE_SECONDS", 5),
            mock.patch.object(server.time, "time", return_value=100),
        ):
            server.store_cached_plan("nim-key", plan, "nim")

        with (
            mock.patch.object(server, "PLAN_CACHE_SECONDS", 900),
            mock.patch.object(server.time, "time", return_value=106),
        ):
            cached = server.get_cached_plan("nim-key")

        self.assertEqual(cached["source"], "nim")
        self.assertEqual(cached["plan"]["intent"], "NIM edit")

    def test_planner_cache_key_changes_with_system_prompt(self):
        original_key = server.planner_cache_key("make it cinematic", "runtime")

        with mock.patch.object(server, "NIM_SYSTEM_PROMPT", server.NIM_SYSTEM_PROMPT + "\nversion: next"):
            changed_key = server.planner_cache_key("make it cinematic", "runtime")

        self.assertNotEqual(original_key, changed_key)

    def test_cached_plan_is_deep_copied_between_jobs(self):
        nim_plan = {
            "intent": "Cached mutable edit",
            "video_filters": [{"filter": "eq=contrast=1.08:saturation=1.05"}],
            "final_encode": server.default_final_encode(),
        }

        with (
            mock.patch.object(server, "runtime_capability_prompt", return_value="runtime"),
            mock.patch.object(server, "call_nim", return_value=nim_plan),
        ):
            first = server.build_plan("make it clearer", {})
            first["video_filters"][0]["filter"] = "eq=contrast=9"
            second = server.build_plan("make it clearer", {})

        self.assertEqual(second["video_filters"][0]["filter"], "eq=contrast=1.08:saturation=1.05")

    def test_heuristic_plan_handles_caption_punctuation(self):
        plan = server.heuristic_plan(
            "add a lower third caption that says Dr. Rao: we're live, don't blink"
        )

        filters = ",".join(step["filter"] for step in plan["video_filters"])
        self.assertIn('drawtext=text="Dr. Rao', filters)
        self.assertIn("don't blink", filters)
        self.assertEqual(len(server.split_filter_chain(filters)), len(plan["video_filters"]))

    def test_auto_captions_request_adds_special_and_strips_static_caption(self):
        plan = {
            "intent": "Generated captions from speech",
            "video_filters": [
                {"description": "Wrong static subtitle", "filter": "drawtext=text='speech':fontcolor=white:fontsize=48:x=20:y=20"},
                {"description": "Keep contrast", "filter": "eq=contrast=1.05"},
            ],
            "final_encode": server.default_final_encode(),
        }

        aligned = server.align_plan_with_command(
            plan,
            "generate captions from the speech and burn them at the bottom",
        )

        self.assertIn({"type": "auto_captions", "params": {"source": "speech", "language": "en", "style": "bottom_box"}}, aligned["special"])
        filters = ",".join(step["filter"] for step in aligned.get("video_filters", []))
        self.assertNotIn("drawtext", filters)
        self.assertIn("eq=contrast=1.05", filters)

    def test_auto_captions_request_does_not_override_explicit_caption_text(self):
        plan = server.heuristic_plan("add a lower third caption that says Launch now")

        self.assertNotIn("special", plan)
        self.assertTrue(any("drawtext=" in step["filter"] for step in plan["video_filters"]))

    def test_heuristic_plan_emits_auto_captions_special(self):
        plan = server.heuristic_plan("add subtitles from speech at the bottom")

        self.assertIn({"type": "auto_captions", "params": {"source": "speech", "language": "en", "style": "bottom_box"}}, plan["special"])

    def test_parse_asr_metadata_segments(self):
        metadata = "\n".join([
            "frame:32 pts:32768 pts_time:2.048",
            "lavfi.asr.text=want to read or five",
            "frame:70 pts:71680 pts_time:4.480",
            "lavfi.asr.text=next phrase",
        ])

        segments = server.parse_asr_metadata_segments(metadata)

        self.assertEqual(segments[0], {"start": 0.0, "end": 2.048, "text": "want to read or five"})
        self.assertEqual(segments[1], {"start": 2.048, "end": 4.48, "text": "next phrase"})

    def test_caption_drawtext_chain_uses_textfiles(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            chain = server.caption_drawtext_chain(
                [{"start": 0, "end": 2, "text": "hello from speech recognition"}],
                Path(temp_dir),
                {},
            )

            self.assertIn("drawtext=textfile=", chain)
            self.assertIn("enable='between(t,0.000,2.000)'", chain)
            self.assertNotIn("text=hello", chain)
            self.assertEqual(len(list(Path(temp_dir).glob("caption_*.txt"))), 1)

    def test_audio_cleanup_request_injects_real_denoise_chain(self):
        plan = {
            "intent": "Clean dialogue",
            "video_filters": [{"filter": "eq=contrast=1.05"}],
            "final_encode": server.default_final_encode(),
        }

        aligned = server.align_plan_with_command(
            plan,
            "remove background noise and make the dialogue clearer",
        )

        filters = ",".join(step["filter"] for step in aligned["audio_filters"])
        self.assertIn("afftdn=nr=18:nf=-35", filters)
        self.assertIn("speechnorm=e=4:c=2:r=0.0005:l=1", filters)
        self.assertIn("loudnorm=I=-16:TP=-1.5:LRA=9", filters)

    def test_audio_cleanup_partial_denoise_still_adds_speech_cleanup(self):
        plan = {
            "intent": "Clean dialogue",
            "audio_filters": [{"filter": "afftdn=nr=18:nf=-35"}],
            "final_encode": server.default_final_encode(),
        }

        aligned = server.align_plan_with_command(
            plan,
            "remove background noise and make the dialogue clearer",
        )

        filters = ",".join(step["filter"] for step in aligned["audio_filters"])
        self.assertIn("afftdn=nr=18:nf=-35", filters)
        self.assertIn("speechnorm=e=4:c=2:r=0.0005:l=1", filters)

    def test_speakernorm_typo_is_repaired_to_speechnorm(self):
        repaired = server.repair_audio_filter_string("speakernorm=e=4:c=2:r=0.0005:l=1")

        self.assertEqual(repaired, "speechnorm=e=4:c=2:r=0.0005:l=1")

    def test_audio_cleanup_does_not_trigger_for_visual_noise(self):
        plan = server.heuristic_plan("add film grain and visual noise like old 16mm")

        audio_filters = ",".join(step["filter"] for step in plan.get("audio_filters", []))
        self.assertNotIn("afftdn", audio_filters)

    def test_heuristic_plan_emits_audio_cleanup_filters(self):
        plan = server.heuristic_plan("clean up audio, reduce hiss, and enhance voice")

        filters = ",".join(step["filter"] for step in plan["audio_filters"])
        self.assertIn("afftdn", filters)
        self.assertIn("speechnorm", filters)

    def test_green_screen_request_becomes_chroma_key_special(self):
        plan = {
            "intent": "Remove keyed background",
            "video_filters": [
                {"filter": "chromakey=0x00ff00:0.18:0.05,format=yuv420p"},
                {"filter": "noise=alls=18:allf=t+u"},
            ],
            "final_encode": server.default_final_encode(),
        }

        aligned = server.align_plan_with_command(
            plan,
            "remove green screen background and replace it with black",
        )

        self.assertIn(
            {
                "type": "chroma_key",
                "params": {
                    "key_color": "green",
                    "replacement_color": "black",
                    "similarity": 0.2,
                    "blend": 0.08,
                },
            },
            aligned["special"],
        )
        self.assertNotIn("video_filters", aligned)

    def test_green_screen_request_preserves_requested_style_filters(self):
        plan = {
            "intent": "Remove keyed background and grade",
            "video_filters": [
                {"filter": "chromakey=0x00ff00:0.18:0.05"},
                {"filter": "eq=contrast=1.2:saturation=1.1"},
            ],
            "final_encode": server.default_final_encode(),
        }

        aligned = server.align_plan_with_command(
            plan,
            "remove green screen background and add cinematic contrast",
        )

        filters = ",".join(step["filter"] for step in aligned.get("video_filters", []))
        self.assertNotIn("chromakey", filters)
        self.assertIn("eq=contrast=1.2:saturation=1.1", filters)

    def test_heuristic_plan_emits_chroma_key_special(self):
        plan = server.heuristic_plan("key out the blue screen and replace it with white")

        self.assertIn(
            {
                "type": "chroma_key",
                "params": {
                    "key_color": "blue",
                    "replacement_color": "white",
                    "similarity": 0.2,
                    "blend": 0.08,
                },
            },
            plan["special"],
        )

    def test_chroma_key_params_handle_rough_spill(self):
        params = server.chroma_key_params("remove rough green screen spill and replace with gray background")

        self.assertEqual(params["replacement_color"], "gray")
        self.assertGreater(params["similarity"], 0.2)
        self.assertGreater(params["blend"], 0.08)

    def test_security_camera_request_adds_scanlines_timestamp_and_desaturation(self):
        plan = server.heuristic_plan(
            "make it look like security camera footage with timestamp text scanlines and low saturation"
        )

        filters = ",".join(step["filter"] for step in plan.get("video_filters", []))
        self.assertIn("drawgrid", filters)
        self.assertIn("drawtext", filters)
        self.assertIn("saturation=0.", filters)
        self.assertNotIn("text=\"scanlines\"", filters)

    def test_security_camera_alignment_repairs_generic_text_only_plan(self):
        plan = {
            "intent": "Security camera",
            "video_filters": [
                {"description": "Text overlay", "filter": "drawtext=text=\"scanlines\":fontcolor=white:fontsize=48:x=0:y=0"}
            ],
            "final_encode": server.default_final_encode(),
        }

        aligned = server.align_plan_with_command(
            plan,
            "make it look like security camera footage with timestamp text scanlines and low saturation",
        )

        filters = ",".join(step["filter"] for step in aligned.get("video_filters", []))
        self.assertIn("drawgrid", filters)
        self.assertIn("drawtext", filters)
        self.assertIn("saturation=0.", filters)
        self.assertNotIn("text=\"scanlines\"", filters)

    def test_old_film_request_becomes_film_damage_special(self):
        plan = {
            "intent": "Generated damaged film look",
            "video_filters": [
                {"description": "Generated film scratches and dust", "filter": "noise=alls=24:allf=t+u,drawbox=x=100:y=0:w=2:h=ih:color=white@0.2:t=fill"},
                {"description": "Keep cinematic contrast", "filter": "eq=contrast=1.12:saturation=1.02"},
            ],
            "final_encode": server.default_final_encode(),
        }

        aligned = server.align_plan_with_command(
            plan,
            "make it look like old damaged 16mm film with scratches dust flicker and gate weave",
        )

        self.assertIn("film_damage", [step["type"] for step in aligned["special"]])
        filters = ",".join(step["filter"] for step in aligned.get("video_filters", []))
        self.assertNotIn("noise=alls", filters)
        self.assertNotIn("drawbox", filters)
        self.assertIn("eq=contrast=1.12:saturation=1.02", filters)

    def test_heuristic_plan_emits_film_damage_special(self):
        plan = server.heuristic_plan("make it look like scratched dusty old 8mm film")

        self.assertIn("film_damage", [step["type"] for step in plan["special"]])
        filters = ",".join(step["filter"] for step in plan.get("video_filters", []))
        self.assertNotIn("noise=alls=18:allf=t+u", filters)

    def test_film_damage_params_scale_intensity(self):
        subtle = server.film_damage_params("add subtle old film scratches")
        heavy = server.film_damage_params("make it heavy damaged 16mm film")

        self.assertLess(subtle["intensity"], heavy["intensity"])
        self.assertLess(subtle["grain"], heavy["grain"])
        self.assertLess(subtle["scratch_opacity"], heavy["scratch_opacity"])

    def test_film_damage_filter_chain_contains_gate_weave_scratches_and_dust(self):
        chain = server.film_damage_filter_chain(
            {"intensity": 0.9, "grain": 40, "gate_weave": 12, "scratch_opacity": 0.28, "dust_opacity": 0.5},
            1280,
            720,
        )

        self.assertIn("crop=iw-", chain)
        self.assertIn("sin(7*t)", chain)
        self.assertIn("noise=alls=40:allf=t+u", chain)
        self.assertIn("drawbox=x='iw*0.16", chain)
        self.assertIn("drawbox=x='iw*0.30'", chain)

    def test_comic_halftone_request_adds_missing_outlines_and_strips_unrequested_lens_distortion(self):
        plan = {
            "intent": "Comic book halftone with thick outlines",
            "video_filters": [
                {"filter": "pixelize=width=10:height=10,hue=s=0.75"},
                {"filter": "lenscorrection=k1=-0.35:k2=0.08"},
            ],
            "final_encode": server.default_final_encode(),
        }

        aligned = server.align_plan_with_command(
            plan,
            "make it look like a comic book halftone with thick outlines",
        )

        filters = ",".join(step["filter"] for step in aligned.get("video_filters", []))
        self.assertIn("pixelize", filters)
        self.assertIn("edgedetect=mode=colormix", filters)
        self.assertNotIn("lenscorrection", filters)

    def test_heuristic_plan_emits_comic_halftone_with_edges(self):
        plan = server.heuristic_plan("make it look like a comic book halftone with thick outlines")

        filters = ",".join(step["filter"] for step in plan.get("video_filters", []))
        self.assertIn("pixelize", filters)
        self.assertIn("edgedetect=mode=colormix", filters)

    def test_underwater_request_strips_identity_geq_and_adds_bounded_wave_motion(self):
        plan = {
            "intent": "Simulate underwater look",
            "video_filters": [
                {"filter": "colorbalance=rs=-0.2:rm=-0.1:rh=0.05:bs=0.25:bm=0.2:bh=0.3"},
                {"filter": "geq=r='p(X,Y)':g='p(X,Y)':b='p(X,Y)'"},
                {"filter": "gblur=sigma=1.8"},
            ],
            "audio_filters": [{"filter": "lowpass=f=800"}],
            "final_encode": server.default_final_encode(),
        }

        aligned = server.align_plan_with_command(
            plan,
            "make it look underwater with wavy distortion and muffled audio",
        )

        video_filters = ",".join(step["filter"] for step in aligned.get("video_filters", []))
        audio_filters = ",".join(step["filter"] for step in aligned.get("audio_filters", []))
        self.assertNotIn("geq=r='p(X,Y)'", video_filters)
        self.assertIn("crop=iw-24:ih-24", video_filters)
        self.assertIn("vignette=angle=PI/3", video_filters)
        self.assertIn("lowpass=f=800", audio_filters)
        self.assertIn("aecho=0.8:0.4:300:0.4", audio_filters)

    def test_heuristic_underwater_plan_includes_wave_motion_and_muffled_audio(self):
        plan = server.heuristic_plan("make it look underwater with wavy distortion and muffled audio")

        video_filters = ",".join(step["filter"] for step in plan.get("video_filters", []))
        audio_filters = ",".join(step["filter"] for step in plan.get("audio_filters", []))
        self.assertIn("crop=iw-24:ih-24", video_filters)
        self.assertIn("colorbalance=rs=-0.2", video_filters)
        self.assertIn("lowpass=f=800", audio_filters)


if __name__ == "__main__":
    unittest.main()
