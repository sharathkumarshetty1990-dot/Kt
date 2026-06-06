import io
import unittest
from pathlib import Path
from unittest import mock

import server


ROOT = Path(__file__).resolve().parents[1]
VIDEO_PATH = ROOT / "test_assets" / "suite_video.mp4"
AUDIO_PATH = ROOT / "test_assets" / "suite_beats.mp3"


class ApiTests(unittest.TestCase):
    def setUp(self):
        server.clear_plan_cache()
        self.client = server.app.test_client()

    def test_health_reports_operational_fields(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "ok")
        self.assertIn("jobs_in_memory", payload)
        self.assertIn("jobs_persisted", payload)
        self.assertIn("upload_root", payload)
        self.assertIn("worker_count", payload)
        self.assertIn("command_timeout_seconds", payload)
        self.assertIn("plan_cache", payload)
        self.assertIn("entries", payload["plan_cache"])
        self.assertIn("capabilities", payload)
        self.assertIn("executor", payload["capabilities"])

    def test_health_refreshes_runtime_capabilities_when_requested(self):
        fake_capabilities = {
            "checked_at": "2026-05-30T00:00:00+00:00",
            "media_commands": {},
            "python_modules": {},
            "executor": {"ffmpeg_ready": True},
        }
        with mock.patch.object(server, "runtime_capabilities", return_value=fake_capabilities) as capabilities:
            response = self.client.get("/health?refresh=1")

        self.assertEqual(response.status_code, 200)
        capabilities.assert_called_once_with(force=True)
        self.assertEqual(response.get_json()["capabilities"], fake_capabilities)

    def test_build_plan_includes_runtime_capability_note(self):
        plan = {
            "intent": "Subtle contrast edit",
            "video_filters": [{"filter": "eq=contrast=1.08:saturation=1.05"}],
            "final_encode": server.default_final_encode(),
        }
        with (
            mock.patch.object(server, "runtime_capability_prompt", return_value="OpenCV/cv2 semantic tracking: not ready."),
            mock.patch.object(server, "call_nim", return_value=plan) as call_nim,
        ):
            result = server.build_plan("blur faces for privacy")

        self.assertEqual(result, plan)
        sent_prompt = call_nim.call_args.args[0]
        self.assertIn("Runtime-verified executor state", sent_prompt)
        self.assertIn("OpenCV/cv2 semantic tracking: not ready.", sent_prompt)

    def test_call_nim_respects_configured_attempt_count(self):
        with (
            mock.patch.object(server, "NIM_API_KEY", "test-key"),
            mock.patch.object(server, "NIM_MAX_ATTEMPTS", 2),
            mock.patch.object(server.urllib.request, "urlopen", side_effect=TimeoutError("slow")) as urlopen,
            mock.patch.object(server.time, "sleep") as sleep,
        ):
            with self.assertRaisesRegex(RuntimeError, "after 2 attempts"):
                server.call_nim("make it cinematic")

        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once()

    def test_first_output_line_skips_proot_warnings(self):
        output = (
            "proot warning: can't sanitize binding \"/proc/self/fd/1\": No such file or directory\n"
            "rubberband 4.0.0\n"
        )

        self.assertEqual(server.first_output_line(output), "rubberband 4.0.0")

    def test_run_command_applies_timeout_and_reports_command_label(self):
        with (
            mock.patch.object(server, "prepare_command", side_effect=lambda args: args),
            mock.patch.object(server.subprocess, "run", side_effect=server.subprocess.TimeoutExpired(
                cmd=["ffmpeg", "-i", "input.mp4"],
                timeout=7,
                stderr="still running",
            )) as run,
        ):
            with self.assertRaisesRegex(RuntimeError, "ffmpeg -i input.mp4 timed out after 7s"):
                server.run_command(["ffmpeg", "-i", "input.mp4"], timeout=7)

        self.assertEqual(run.call_args.kwargs["timeout"], 7)

    def test_run_command_compacts_nonzero_exit_errors(self):
        result = mock.Mock(returncode=1, stderr="x" * 900, stdout="")

        with (
            mock.patch.object(server, "prepare_command", side_effect=lambda args: args),
            mock.patch.object(server.subprocess, "run", return_value=result),
        ):
            with self.assertRaisesRegex(RuntimeError, "ffmpeg -bad failed:"):
                server.run_command(["ffmpeg", "-bad"])

    def test_upload_requires_video(self):
        response = self.client.post("/upload", data={}, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json(), {"error": "video file required"})

    def test_upload_with_audio_persists_job_metadata(self):
        data = {
            "video": (io.BytesIO(VIDEO_PATH.read_bytes()), "suite_video.mp4"),
            "audio": (io.BytesIO(AUDIO_PATH.read_bytes()), "suite_beats.mp3"),
        }

        response = self.client.post("/upload", data=data, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 200)
        upload_payload = response.get_json()
        self.assertEqual(upload_payload["status"], "uploaded")
        self.assertEqual(upload_payload["video_name"], "suite_video.mp4")
        self.assertEqual(upload_payload["audio_name"], "suite_beats.mp3")

        job_id = upload_payload["job_id"]
        job_response = self.client.get(f"/job/{job_id}")
        self.assertEqual(job_response.status_code, 200)
        job = job_response.get_json()
        self.assertEqual(job["job_id"], job_id)
        self.assertEqual(job["status"], "uploaded")
        self.assertEqual(job["video_name"], "suite_video.mp4")
        self.assertEqual(job["audio_name"], "suite_beats.mp3")
        self.assertTrue(Path(job["video_path"]).exists())
        self.assertTrue(Path(job["audio_path"]).exists())
        self.assertTrue((Path(job["video_path"]).parent / "job.json").exists())

    def test_upload_rejects_invalid_video_bytes(self):
        data = {
            "video": (io.BytesIO(b"not a video"), "fake.mp4"),
        }

        response = self.client.post("/upload", data=data, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json(), {"error": "valid video file required"})

    def test_upload_rejects_invalid_audio_bytes(self):
        data = {
            "video": (io.BytesIO(VIDEO_PATH.read_bytes()), "suite_video.mp4"),
            "audio": (io.BytesIO(b"not audio"), "fake.mp3"),
        }

        response = self.client.post("/upload", data=data, content_type="multipart/form-data")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json(), {"error": "valid audio file required"})

    def test_missing_job_returns_404(self):
        response = self.client.get("/job/not-a-real-job")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_json(), {"error": "job not found"})

    def test_loaded_processing_job_is_marked_interrupted(self):
        job = {
            "job_id": "stale",
            "status": "processing",
            "video_path": str(VIDEO_PATH),
        }

        sanitized = server.sanitize_job_state(job, from_disk=True)

        self.assertEqual(sanitized["status"], "error")
        self.assertIn("processing interrupted", sanitized["error"])
        self.assertIn("failed_at", sanitized)

    def test_command_records_planner_fallback_warning(self):
        data = {
            "video": (io.BytesIO(VIDEO_PATH.read_bytes()), "suite_video.mp4"),
        }
        upload_response = self.client.post("/upload", data=data, content_type="multipart/form-data")
        job_id = upload_response.get_json()["job_id"]

        with (
            mock.patch.object(server, "call_nim", side_effect=RuntimeError("offline")),
            mock.patch.object(server.job_executor, "submit"),
        ):
            response = self.client.post(
                "/command",
                json={"job_id": job_id, "command": "make it pulse and zoom on every beat"},
            )

        self.assertEqual(response.status_code, 200)
        job_response = self.client.get(f"/job/{job_id}")
        job = job_response.get_json()
        self.assertEqual(job["planner"], "heuristic")
        self.assertTrue(any("NIM planning failed" in warning for warning in job["warnings"]))
        self.assertEqual(job["status"], "processing")

    def test_command_clears_stale_warnings_for_new_run(self):
        data = {
            "video": (io.BytesIO(VIDEO_PATH.read_bytes()), "suite_video.mp4"),
        }
        upload_response = self.client.post("/upload", data=data, content_type="multipart/form-data")
        job_id = upload_response.get_json()["job_id"]
        job = server.get_job_record(job_id)
        job["warnings"] = ["old warning from previous command"]

        plan = {
            "intent": "Subtle contrast edit",
            "video_filters": [{"filter": "eq=contrast=1.08:saturation=1.05"}],
            "final_encode": server.default_final_encode(),
        }
        with (
            mock.patch.object(server, "call_nim", return_value=plan),
            mock.patch.object(server.job_executor, "submit"),
        ):
            response = self.client.post(
                "/command",
                json={"job_id": job_id, "command": "make it clearer"},
            )

        self.assertEqual(response.status_code, 200)
        updated_job = self.client.get(f"/job/{job_id}").get_json()
        self.assertNotIn("old warning from previous command", updated_job.get("warnings", []))


if __name__ == "__main__":
    unittest.main()
