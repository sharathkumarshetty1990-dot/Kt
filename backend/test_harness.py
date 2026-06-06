#!/usr/bin/env python3
import argparse
import json
import mimetypes
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = ROOT / "test_assets" / "suite_video.mp4"
DEFAULT_AUDIO = ROOT / "test_assets" / "suite_beats.mp3"
DEFAULT_CORPUS = Path(__file__).resolve().with_name("command_corpus.json")
DEFAULT_REPORT_DIR = Path(__file__).resolve().with_name("test_reports")


def post_multipart(url, files, timeout):
    boundary = f"----linguist-harness-{int(time.time() * 1000)}"
    chunks = []
    for field, path in files:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        chunks.append(f"--{boundary}\r\n".encode("utf-8"))
        chunks.append(
            f'Content-Disposition: form-data; name="{field}"; filename="{path.name}"\r\n'.encode("utf-8")
        )
        chunks.append(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
        chunks.append(path.read_bytes())
        chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode("utf-8"))

    request = urllib.request.Request(
        url,
        data=b"".join(chunks),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    return read_json(request, timeout)


def read_json(request_or_url, timeout):
    try:
        with urllib.request.urlopen(request_or_url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def post_json(url, payload, timeout):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return read_json(request, timeout)


def health(base_url, timeout):
    return read_json(f"{base_url}/health", timeout)


def upload(base_url, video_path, audio_path, timeout):
    files = [("video", video_path)]
    if audio_path:
        files.append(("audio", audio_path))
    return post_multipart(f"{base_url}/upload", files, timeout)


def wait_for_job(base_url, job_id, timeout_seconds, poll_interval):
    deadline = time.time() + timeout_seconds
    last_status = None
    while time.time() < deadline:
        job = read_json(f"{base_url}/job/{job_id}", 20)
        status = job.get("status")
        if status != last_status:
            print(f"  status: {status}", flush=True)
            last_status = status
        if status in {"complete", "error"}:
            return job
        time.sleep(poll_interval)
    raise TimeoutError(f"job {job_id} did not finish within {timeout_seconds}s")


def ffprobe(path):
    command = [
        "proot-distro", "login", "ubuntu", "--",
        "ffprobe", "-v", "error",
        "-show_entries", "stream=index,codec_type,codec_name,width,height:format=duration",
        "-of", "json", path,
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        return {"error": result.stderr.strip() or result.stdout.strip()}
    return json.loads(result.stdout)


def filter_cases(corpus, only, category, include_slow):
    selected = corpus
    if only:
        wanted = set(only)
        selected = [case for case in selected if case["id"] in wanted]
    if category:
        selected = [case for case in selected if case.get("category") == category]
    if not include_slow:
        selected = [case for case in selected if not case.get("slow")]
    return selected


def validate_expectations(case, probe):
    expected = case.get("expect") or {}
    streams = probe.get("streams") or []
    video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    errors = []
    if expected and not video_stream:
        return ["missing video stream in ffprobe output"]
    for key in ["width", "height"]:
        if key in expected and video_stream.get(key) != expected[key]:
            errors.append(f"expected {key}={expected[key]}, got {video_stream.get(key)}")

    duration = None
    try:
        duration = float((probe.get("format") or {}).get("duration"))
    except (TypeError, ValueError):
        duration = None
    minimum = case.get("expect_duration_min")
    maximum = case.get("expect_duration_max")
    if minimum is not None and (duration is None or duration < minimum):
        errors.append(f"expected duration >= {minimum}, got {duration}")
    if maximum is not None and (duration is None or duration > maximum):
        errors.append(f"expected duration <= {maximum}, got {duration}")
    return errors


def validate_plan_expectations(case, plan):
    expected = case.get("expect_plan") or {}
    if not expected:
        return []

    errors = []
    plan_text = json.dumps(plan, sort_keys=True)
    special_types = [
        step.get("type")
        for step in plan.get("special", [])
        if isinstance(step, dict)
    ]
    analysis_functions = [
        step.get("function")
        for step in plan.get("analysis", [])
        if isinstance(step, dict)
    ]

    for substring in expected.get("required_substrings", []):
        if substring not in plan_text:
            errors.append(f"plan missing substring: {substring}")

    for group in expected.get("required_any_substrings", []):
        if not any(substring in plan_text for substring in group):
            errors.append(f"plan missing one of substrings: {', '.join(group)}")

    for substring in expected.get("forbidden_substrings", []):
        if substring in plan_text:
            errors.append(f"plan contains forbidden substring: {substring}")

    for key in expected.get("required_keys", []):
        if key not in plan:
            errors.append(f"plan missing key: {key}")

    for key in expected.get("forbidden_keys", []):
        if plan.get(key):
            errors.append(f"plan contains forbidden key with work: {key}")

    for special_type in expected.get("required_special", []):
        if special_type not in special_types:
            errors.append(f"plan missing special: {special_type}")

    for special_type in expected.get("forbidden_special", []):
        if special_type in special_types:
            errors.append(f"plan contains forbidden special: {special_type}")

    for analysis_function in expected.get("required_analysis", []):
        if analysis_function not in analysis_functions:
            errors.append(f"plan missing analysis function: {analysis_function}")

    for analysis_function in expected.get("forbidden_analysis", []):
        if analysis_function in analysis_functions:
            errors.append(f"plan contains forbidden analysis function: {analysis_function}")

    return errors


def run_case(base_url, case, video_path, audio_path, args):
    started = time.time()
    print(f"\n[{case['id']}] {case['command']}", flush=True)
    result = {
        "id": case["id"],
        "category": case.get("category"),
        "command": case["command"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "unknown",
    }

    try:
        upload_response = upload(base_url, video_path, audio_path, args.request_timeout)
        job_id = upload_response["job_id"]
        result["job_id"] = job_id
        print(f"  job: {job_id}", flush=True)

        plan = post_json(
            f"{base_url}/command",
            {"job_id": job_id, "command": case["command"]},
            args.plan_timeout,
        )
        result["intent"] = plan.get("intent")
        result["plan"] = plan
        print(f"  intent: {plan.get('intent')}", flush=True)
        plan_errors = validate_plan_expectations(case, plan)
        if plan_errors:
            result["plan_expectation_errors"] = plan_errors
            print(f"  plan warning: {'; '.join(plan_errors)}", flush=True)

        job = wait_for_job(base_url, job_id, args.job_timeout, args.poll_interval)
        result["job"] = job
        result["status"] = job.get("status")
        result["processing_seconds"] = job.get("processing_seconds")

        if job.get("status") != "complete":
            errors = plan_errors + [job.get("error") or "job failed"]
            result["error"] = "; ".join(errors)
            return result

        probe = ffprobe(job["output_path"])
        result["ffprobe"] = probe
        expectation_errors = plan_errors + validate_expectations(case, probe)
        if expectation_errors:
            result["status"] = "error"
            result["error"] = "; ".join(expectation_errors)
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
    path = report_dir / f"harness_{timestamp}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    latest = report_dir / "latest.json"
    latest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Run Linguist backend end-to-end command tests.")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--only", action="append", help="Run a specific corpus id. Can be repeated.")
    parser.add_argument("--category", help="Run one category from the corpus.")
    parser.add_argument("--include-slow", action="store_true", help="Include slow stress cases.")
    parser.add_argument("--limit", type=int, help="Limit number of selected cases.")
    parser.add_argument("--request-timeout", type=int, default=60)
    parser.add_argument("--plan-timeout", type=int, default=360)
    parser.add_argument("--job-timeout", type=int, default=360)
    parser.add_argument("--poll-interval", type=float, default=1.5)
    return parser.parse_args()


def main():
    args = parse_args()
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    cases = filter_cases(corpus, args.only, args.category, args.include_slow)
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        print("No test cases selected.", file=sys.stderr)
        return 2

    report = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "case_count": len(cases),
        "include_slow": args.include_slow,
        "results": [],
    }

    try:
        report["health"] = health(args.base_url, args.request_timeout)
    except Exception as exc:
        report["health_error"] = str(exc)
        path = write_report(report, args.report_dir)
        print(f"Backend health check failed. Report: {path}", file=sys.stderr)
        return 2

    failures = 0
    for case in cases:
        result = run_case(args.base_url, case, args.video, args.audio, args)
        report["results"].append(result)
        if result["status"] != "complete":
            failures += 1
            print(f"  FAILED: {result.get('error')}", flush=True)
        else:
            print(f"  passed in {result.get('elapsed_seconds')}s", flush=True)

    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    report["failures"] = failures
    report_path = write_report(report, args.report_dir)

    print(f"\nReport: {report_path}")
    print(f"Passed: {len(cases) - failures}/{len(cases)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
