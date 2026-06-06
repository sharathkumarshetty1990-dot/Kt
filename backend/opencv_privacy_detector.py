#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import cv2


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, int(round(float(value)))))


def cascade_candidates(target):
    data_dir = Path(cv2.data.haarcascades)
    target = str(target or "faces").lower()
    if target in {"person", "people", "humans", "body", "bodies"}:
        names = [
            "haarcascade_upperbody.xml",
            "haarcascade_fullbody.xml",
            "haarcascade_lowerbody.xml",
        ]
    else:
        names = [
            "haarcascade_frontalface_default.xml",
            "haarcascade_frontalface_alt2.xml",
            "haarcascade_profileface.xml",
        ]
    return [data_dir / name for name in names if (data_dir / name).exists()]


def normalized_box(x, y, w, h, width, height, frame_time, padding):
    left = clamp(x - padding, 1, max(1, width - 3))
    top = clamp(y - padding, 1, max(1, height - 3))
    right = clamp(x + w + padding, left + 2, max(left + 2, width - 1))
    bottom = clamp(y + h + padding, top + 2, max(top + 2, height - 1))
    return {
        "x": left,
        "y": top,
        "w": max(2, right - left),
        "h": max(2, bottom - top),
        "start": round(max(0.0, frame_time - 0.45), 3),
        "end": round(frame_time + 0.8, 3),
        "confidence": 1.0,
        "source": "opencv_cascade",
    }


def detect(video_path, params):
    target = params.get("target", "faces")
    sample_fps = max(0.2, min(3.0, float(params.get("sample_fps", 1.0) or 1.0)))
    max_frames = max(1, min(80, int(params.get("max_frames", 24) or 24)))
    padding = max(4, min(96, int(params.get("padding", 24) or 24)))
    min_size_ratio = max(0.03, min(0.4, float(params.get("min_size_ratio", 0.08) or 0.08)))

    cascades = [cv2.CascadeClassifier(str(path)) for path in cascade_candidates(target)]
    cascades = [cascade for cascade in cascades if not cascade.empty()]
    if not cascades:
        return {"mode": "opencv_cascade", "detections": [], "sampled_frames": 0, "error": "no cascade files available"}

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return {"mode": "opencv_cascade", "detections": [], "sampled_frames": 0, "error": "video open failed"}

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    if duration <= 0:
        duration = max_frames / sample_fps

    detections = []
    sampled = 0
    frame_times = []
    current = 0.0
    while current <= duration + 1e-6 and len(frame_times) < max_frames:
        frame_times.append(current)
        current += 1.0 / sample_fps

    for frame_time in frame_times:
        capture.set(cv2.CAP_PROP_POS_MSEC, frame_time * 1000.0)
        ok, frame = capture.read()
        if not ok or frame is None:
            continue
        sampled += 1
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        min_side = max(16, int(min(width, height) * min_size_ratio))
        for cascade in cascades:
            boxes = cascade.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=4,
                minSize=(min_side, min_side),
            )
            for x, y, w, h in boxes:
                detections.append(normalized_box(x, y, w, h, width, height, frame_time, padding))

    capture.release()
    detections.sort(key=lambda item: (item["start"], item["x"], item["y"]))
    return {
        "mode": "opencv_cascade",
        "detections": detections[:80],
        "sampled_frames": sampled,
        "duration": round(duration, 3),
    }


def main():
    if len(sys.argv) < 3:
        raise SystemExit("usage: opencv_privacy_detector.py VIDEO_PATH JSON_PARAMS")
    params = json.loads(sys.argv[2])
    print(json.dumps(detect(sys.argv[1], params), separators=(",", ":")))


if __name__ == "__main__":
    main()
