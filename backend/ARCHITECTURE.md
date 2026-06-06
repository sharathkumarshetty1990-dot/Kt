# Linguist Backend Architecture

The backend is intentionally structured as a model-planner-validator-executor system.
The AI should describe the edit. The planner should convert that description into an executable contract. The validator should block impossible work before execution. The executor should only run operations declared in the shared architecture registry.

## Core Boundaries

1. `server.py`
   - Owns Flask routes, uploads, job state, persistence, subprocess execution, and concrete media transforms.
   - Should not define planner policy or operation support rules directly.

2. `ai_planner.py`
   - Converts model JSON into an internal production plan.
   - Repairs dependencies, assigns workers, validates assets and contexts, and blocks unsupported operations before jobs enter the queue.
   - Should not duplicate executor capability constants.

3. `llm_provider.py`
   - Owns provider transport for NVIDIA NIM chat-completion calls: URL, auth header, timeout, retry loop, JSON response unwrap, and provider metadata.
   - It must not decide planner fallback behavior or normalize edit plans. Those remain server/planner responsibilities.
   - Secret values such as API keys must never be exposed through metadata or `/health`.

4. `editing_architecture.py`
   - Single source of truth for supported analysis functions, special operation types, worker routing, filter-complex policy, and final encode normalization.
   - `ANALYSIS_FUNCTION_SPECS` and `SPECIAL_OPERATION_SPECS` are typed operation contracts. They describe support, worker routing, confidence, domains, context dependencies, uploaded-audio dependencies, and runtime capability requirements.
   - `CAPABILITY_SPECS` defines the names and validation severity for runtime dependencies such as `audio_analysis_ready`, `asr_ready`, `ocr_ready`, `pitch_shift_ready`, `stabilize_ready`, and `frei0r_ready`.
   - `FILTER_CAPABILITY_REQUIREMENTS` maps FFmpeg filter names such as `frei0r`, `rubberband`, and `vidstabtransform` to the capabilities they require.
   - `runtime_operation_contract()` derives per-operation readiness from runtime executor probes. It reports operations as `ready`, `degraded`, or `blocked`.
   - `runtime_operation_prompt_contract()` is the prompt-facing view of the same readiness contract. The model should not choose blocked operations.
   - `VALIDATION_ISSUE_SPECS` owns issue ownership, repairability, and repair hints. Server code should not hardcode lists of validation codes.
   - `PLANNER_FALLBACK_POLICY` owns fail-open/fail-closed behavior for model outages, failed repairs, and non-repairable validation failures.
   - `EXECUTION_FAILURE_POLICY` owns executor fail-closed behavior for special operations, direct filter phases, and final encode fallback.
   - `VALIDATION_POLICY` owns confidence bands, retry thresholds, and external dependency prefixes.
   - New operations should be registered here first as typed specs, then implemented in the executor dispatch table.
   - `architecture_summary()` exposes the active operation contract through `/health`.
   - `architecture_fingerprint()` is part of the planner cache key and model prompt. Architecture changes must invalidate cached plans.

5. `ai_orchestrator.py`
   - Owns execution manifests, progress state, result inspection packets, and repair packets.
   - It should describe what happened, not perform media edits.

6. `edit_effectiveness.py`
   - Owns job-level inspection for whether a plan produced evidence of applied editing work.
   - It combines execution metadata from special operations, video/audio filters, uploaded audio, output aspect enforcement, and final encode dimensions.
   - A valid output artifact is not enough; jobs with planned edit operations and no applied-edit evidence must fail instead of being reported as complete.

7. `media_runner.py`
   - Owns subprocess execution for FFmpeg, ffprobe, rubberband, and related media tooling.
   - Centralizes Termux/Ubuntu-proot command preparation, timeouts, compact error tails, structured command errors, and command execution stats.
   - Server code should call `run_command()` or `run_command_result()` wrappers, not raw `subprocess.run()`.

8. `job_lifecycle.py`
   - Owns job statuses, allowed transitions, terminal-state semantics, command-acceptance rules, and status history.
   - `/command` must reject jobs whose lifecycle state does not accept a new command.

9. `job_errors.py`
   - Owns the stable `job_error` object stored on failed or rejected jobs.
   - Terminal failures should keep the legacy `error` string for compatibility, but frontend and retry logic should prefer `job_error`.
   - Error objects include code, phase, retryability, user action, details, and creation time.

10. `job_store.py`
   - Owns in-memory job records, persisted `job.json` files, disk reload, startup recovery, and atomic command claims.
   - A command must claim a job into `planning` before model planning starts, so concurrent requests cannot schedule multiple edits for the same job.
   - This is the state boundary that can later be replaced with SQLite, Redis, or a queue-backed store without rewriting route and executor logic.

11. `job_queue.py`
   - Owns asynchronous execution queueing, bounded pending capacity, worker-pool metrics, and overload rejection.
   - `/command` must reject with `503` when the queue is saturated instead of submitting unbounded work.
   - This is the execution-queue boundary that can later be replaced with Celery, RQ, Dramatiq, or a remote worker service.

12. `editing_capabilities.py`
   - Adds runtime and environment knowledge to AI prompts.
   - It must keep capability context grounded in approved local knowledge files and runtime probes.
   - Runtime capability notes should describe readiness, but they do not decide whether a plan is allowed to execute. The validator does that from `editing_architecture.py`.

13. `plan_contract.py`
   - Owns the public JSON plan boundary before planner validation.
   - Normalizes model, cache, repair, and heuristic plans into one predictable root shape.
   - Fills safe defaults for missing intent, list sections, special params, filter timing, and final encode settings.
   - Exposes `public_plan_contract_fingerprint()` so planner cache entries are invalidated when the public plan contract changes.

14. `planner_cache.py`
   - Owns planner-cache storage, TTL handling, fallback-plan TTL, LRU eviction, cloning, and cache stats.
   - Cache keys must include the model, system prompt hash, runtime readiness context, architecture fingerprint, public plan contract fingerprint, special parameter contract fingerprint, and normalized user command.
   - Heuristic fallback plans must use a shorter TTL than model plans because they are lower confidence.

15. `special_params.py`
   - Owns type normalization and clamping for supported special-operation params.
   - Known special params are normalized before planner validation and before execution can see them.
   - Exposes `special_param_contract_fingerprint()` so planner cache entries are invalidated when param semantics change.

16. `intent_contract.py`
   - Owns conservative natural-language intent coverage checks.
   - It does not try to understand every possible request; it catches high-signal misses such as beat sync without beat logic, auto captions without `auto_captions`, silence removal without `silence_remove`, and privacy redaction without the matching special worker.
   - Missing intent coverage is a model-repairable validation error.

17. `runtime_cache.py`
   - Owns runtime capability cache TTL, freshness state, forced refresh handling, and cache diagnostics.
   - It does not probe tools itself; `server.py` still owns the concrete environment probes until those are extracted behind a runtime-probe service.

18. `upload_policy.py`
   - Owns upload filename normalization, allowed media extensions, safe storage names, and upload-policy diagnostics.
   - Original upload names remain job metadata for UI display; filesystem paths must use sanitized storage names.

## Planning Flow

1. `/command` receives a natural language command for an uploaded job.
2. `build_plan()` asks NIM for the public JSON edit plan, with heuristic fallback only when the model is unavailable.
3. The prompt includes the runtime capability note and the current architecture contract so the model plans against the real executor surface.
4. The prompt also includes the runtime operation readiness contract, so the model sees which registered operations are currently ready, degraded, or blocked on this machine.
5. Public JSON plans pass through `plan_contract.py` before validation, so weird JSON shapes are normalized before they reach the production planner.
6. Planner cache keys include the system prompt, runtime note, normalized command, architecture fingerprint, public plan contract fingerprint, and special parameter contract fingerprint. The runtime note includes the readiness contract, so readiness changes invalidate cached plans.
7. `align_plan_with_command()` repairs known model misunderstandings against the original command.
8. `/command` claims the job through `job_store.py`, moving it to `planning` before any model or planner work starts.
9. `prepare_production_plan()` normalizes the plan, repairs missing contexts, assigns workers, validates support, checks intent coverage, and builds the internal plan.
10. Unsupported analysis functions, unsupported special operations, missing video assets, missing analysis contexts, missing final encode requirements, and clear missing intent coverage block execution.
11. If validation fails for model-correctable reasons, `/command` performs one NIM repair attempt with the rejected plan and validation feedback.
12. If NIM is unavailable, deterministic heuristic planning may run only under `heuristic_guardrailed` fallback policy and must still pass normal validation.
13. If model repair fails or validation is not model-repairable, the system rejects the command instead of inventing a misleading edit.
14. Operation-specific runtime capability failures are produced generically from `editing_architecture.py`. High-risk missing capabilities block execution; best-effort capabilities produce warnings.
15. Direct `video_filters` and `audio_filters` are single-input `-vf`/`-af` chains. Known `filter_complex` or multi-stream filters must be represented by supported special operations or rejected by validation.
16. Valid plans create an execution manifest and enter the bounded worker queue.
17. Queue saturation returns `503` and marks already-accepted jobs as errored if capacity disappears between validation and submission.

## Execution Flow

1. `execute_job_async()` marks the manifest running.
2. `execute_pipeline()` runs phases in order: analysis, special transforms, video filters, aspect enforcement, audio filters, final encode.
3. Each phase updates the manifest through `ai_orchestrator.py`.
4. Required special operations fail closed if unsupported, unimplemented, or failed at runtime.
5. Video and audio filter phases may use safe per-filter fallbacks, but the phase fails if none of the planned filters can be applied.
6. Final encode may retry with safe defaults because it packages an already-created edit rather than replacing a requested creative operation.
7. Result inspection verifies that a non-empty output artifact exists and has positive duration.
8. Edit-effectiveness inspection verifies that planned editing work produced applied-edit evidence.
9. Failures persist a repair packet so the next architecture pass can target real failure causes.

## Operational Endpoints

1. `/live`
   - Cheap process liveness check.
   - Does not touch media, model providers, runtime probes, or job state.

2. `/ready`
   - Deployment readiness check.
   - Verifies upload-root writability, AI-provider configuration, architecture integrity, and queue availability.
   - Returns `503` when the backend should not receive user editing traffic.

3. `/health`
   - Rich diagnostic endpoint for debugging.
   - May include runtime capabilities, planner cache state, architecture contracts, lifecycle contracts, provider metadata, and executor implementation details.

## Architectural Rules

- A model-generated operation is not supported until it is in `editing_architecture.py` and implemented by `server.py`.
- `SPECIAL_OPERATION_SPECS` defines the product contract. `SPECIAL_EXECUTORS` defines concrete implementation. `/health.executor_implementation` must not show registered-but-unimplemented operations in a healthy architecture.
- `/command` must fail fast if `architecture_integrity` detects registry/executor drift. It is better to reject planning than claim an operation is supported without an implementation.
- Planner code should ask `editing_architecture.py` questions such as operation confidence, worker routing, and uploaded-audio requirements. Do not re-create operation-specific sets in `ai_planner.py`.
- Planner code should ask `editing_architecture.py` for required capabilities. Do not add local checks such as "if operation is pitch_shift then check rubberband" in the planner.
- Planner validation output must be enriched through `VALIDATION_ISSUE_SPECS` so repair loops can distinguish model-correctable plans from runtime or asset failures.
- Intent coverage validation must remain conservative. It should only block clear misses where the command explicitly asks for an operation family that the plan does not represent.
- Planner confidence bands and retry policies must come from `VALIDATION_POLICY`, not local numeric thresholds.
- Planner fallback behavior must come from `PLANNER_FALLBACK_POLICY`. Heuristic fallback is allowed only when NIM is unavailable and only after validation; failed repair and non-repairable validation fail closed.
- Provider transport belongs in `llm_provider.py`. Route code should not construct provider HTTP requests directly.
- Queueing belongs in `job_queue.py`. Route code should not submit directly to `ThreadPoolExecutor` or any future worker backend.
- Subprocess execution belongs in `media_runner.py`. Route and worker code should not call raw `subprocess.run()` directly.
- Executor failure behavior must come from `EXECUTION_FAILURE_POLICY`. Do not add local skip-and-continue behavior for required planned work.
- Executor code should dispatch through `SPECIAL_EXECUTORS`, not a growing branch chain.
- Model prompting and planner caching must include the active architecture fingerprint. Stale plans from an older operation contract are not acceptable.
- Planner caching must include the active public plan contract fingerprint. Stale plans from an older public JSON contract are not acceptable.
- Planner caching must include the active special parameter contract fingerprint. Stale plans from older special-param semantics are not acceptable.
- Planner cache behavior belongs in `planner_cache.py`. Route code should not own raw cache dicts, TTL expiry, or eviction policy.
- Validation should reject impossible plans before execution rather than silently skipping requested work.
- Executor phases should fail when required planned work cannot be applied. Silent unchanged exports are not acceptable production behavior.
- Workers must record execution metadata that proves applied work. Final output existence alone is not enough to complete a job.
- Runtime probes in `server.py` must publish every capability named by `CAPABILITY_SPECS`. If a capability is added to the registry, the backend should expose an executor readiness bit for it.
- `architecture_integrity()` must compare registry capability names against executor probe keys. `/command` should fail fast if a capability exists in the product contract but has no runtime probe.
- `architecture_registry_issues()` must report internal contract mistakes, including operation specs or filter mappings that reference undefined capabilities, invalid capability severities, and incomplete validation issue specs.
- Final encode settings must be normalized by `editing_architecture.py` before queueing and again defensively before export.
- Runtime capability context may influence the model, but the validator remains authoritative.
- Runtime operation readiness context is a planning hint and cache input; it is not an execution permission. The validator still gates every generated plan.
- Runtime capability cache behavior belongs in `runtime_cache.py`; tool probes should not be cached through ad hoc globals.
- Filter-complex-only filters in direct filter sections are validation errors, not warnings. Silent fallback from an impossible multi-stream request risks producing the wrong edit.
- Planner repair retries are one-shot and validation-driven. Repairability must come from `VALIDATION_ISSUE_SPECS`, not local server code.
- `/health` should show both runtime capabilities and the static architecture contract so production issues can distinguish "unsupported by design" from "supported but currently unavailable".
- `/live` and `/ready` should remain cheap enough for deployment probes. Keep expensive media tests out of readiness checks.
- Broad live prompt sweeps are not the primary architecture tool. Prefer contract hardening, focused probes, and failure-driven repairs.
- Job status writes must go through `transition_job_status()` so persisted jobs retain lifecycle history and invalid command races are blocked.
- Terminal job failures must include `job_error` from `job_errors.py`; do not add new bare error-string-only failure paths.
- Job persistence and command claiming must go through `job_store.py`. Do not mutate the global job map directly from routes or workers.
- Execution queue capacity must be bounded and visible through `/ready` and `/health`. Unbounded hidden work queues are not production behavior.
- Upload validation belongs in `upload_policy.py`. Routes should not write user-controlled filenames directly to disk.
