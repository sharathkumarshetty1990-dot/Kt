# Linguist Video

Natural-language video editing SPA with a Flask backend, NIM planning, and real FFmpeg execution.

## Layout

- `frontend/index.html` - single-file vanilla SPA.
- `backend/server.py` - Flask API and media execution pipeline.
- `backend/command_corpus.json` - regression command corpus.
- `backend/test_harness.py` - end-to-end backend test runner.
- `backend/executor_smoke.py` - real FFmpeg executor smoke tests that do not call NIM.
- `test_assets/` - local media files used by the harness.

## Backend Configuration

Copy `backend/.env.example` to your shell environment or process manager. Do not commit real secrets.

Required:

```sh
export NIM_API_KEY="your-key"
```

Useful settings:

```sh
export LINGUIST_HOST=127.0.0.1
export LINGUIST_PORT=5000
export LINGUIST_ALLOWED_ORIGIN=http://127.0.0.1:8000
export LINGUIST_UPLOAD_ROOT=/tmp/linguist
export LINGUIST_MAX_UPLOAD_MB=2048
export LINGUIST_WORKERS=2
```

## Local Run

Backend:

```sh
cd Kt/backend
python3 server.py
```

Frontend:

```sh
cd Kt/frontend
python3 -m http.server 8000 --bind 127.0.0.1
```

Open:

```text
http://127.0.0.1:8000/
```

## Production Run

Use a process manager and WSGI server instead of Flask's dev server:

```sh
cd Kt/backend
gunicorn -w 2 -b 127.0.0.1:5000 wsgi:app
```

Keep FFmpeg, ffprobe, rubberband, vidstab, and the Ubuntu proot media runtime available on the host. For horizontal scaling, run workers behind a reverse proxy and point every worker at shared upload storage.

## Regression Harness

Run fast unit checks for planner-output repair logic:

```sh
cd Kt/backend
python3 -m unittest test_repairs.py test_api.py
```

Run the executor pipeline directly with real local media and no NIM call:

```sh
python3 Kt/backend/executor_smoke.py
```

Run the local backend gate:

```sh
Kt/backend/verify_backend.sh
```

Run a quick smoke set:

```sh
python3 Kt/backend/test_harness.py --only strobe --only vertical_tiktok
```

Run the non-slow corpus:

```sh
python3 Kt/backend/test_harness.py
```

Run everything, including slow 4K and stress tests:

```sh
python3 Kt/backend/test_harness.py --include-slow
```

Reports are written to `backend/test_reports/latest.json` and timestamped JSON files.

## Production Readiness Notes

- Job metadata is persisted as `job.json` beside each upload, so job state can be recovered after restart.
- Media files are stored under `LINGUIST_UPLOAD_ROOT`.
- The backend remains process-local for active execution threads. A queue such as Redis/RQ, Celery, or a durable task service is the next scaling step for multi-machine processing.
- The harness is the gate for new executor repairs: add failed prompts to `command_corpus.json`, patch the executor, rerun, and keep reports.
