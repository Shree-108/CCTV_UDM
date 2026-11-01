# Implementation plan — AI microservice, S3 and Admin integration

Status: **proposal, waiting for approval.** Nothing below has been implemented yet.

## 1. Approach in one paragraph

Keep the existing application and its quick sampled pipeline exactly as they are, and add a thin
**service layer** next to it: an authenticated `/api/v1` API, an S3 client, a job worker that turns
"store + time window" into existing extraction/detection jobs, and an uploader that copies the results to S3.
The same code base runs in two modes: **local** (today's dashboard, unchanged, plus `/api/v1` for testing)
and **service** (on EC2: only `/health` and `/api/v1`, protected by an API key, dashboard routes off).
No new framework, no model change, no change to thresholds or class mapping.

```
Admin cron 1 ──POST /api/v1/analyze──► service/routes_v1.py ──► service_analyses (SQLite)
                                              │
                                              ▼
                                   service/worker.py (bounded queue, N workers, timeout)
                                              │ 1 list videos in input prefix (service/s3.py)
                                              │ 2 pick the videos that overlap [start, end)
                                              │ 3 window plan: sample_count instants across the window
                                              │ 4 per video: existing extractor (stream via presigned URL)
                                              │ 5 existing detect.run (person detection + uniform model)
                                              │ 6 existing business CSV + person-level CSV + manifest JSON
                                              │ 7 upload to output prefix, verify every object
                                              ▼
Admin cron 2 ◄── reads CSV + manifest from S3 ── status "completed" via GET /api/v1/analysis/{id}
```

## 2. Decisions that need your approval first

| # | Decision | Proposal |
|---|---|---|
| D1 | Commit the uncommitted 29-Oct changes (crop-size fix etc.) before starting | **Yes** — review and commit them as their own commit |
| D2 | API authentication | `X-API-Key` header, keys stored as SHA-256 hashes in env; plus a security group that only admits the Admin backend |
| D3 | Business CSV `Time Stamp` in service output | Keep the column; fill it with the **clock time** of the frame when the recording start time is known from S3 (it is), because the Admin dashboard reasons in clock hours. Local uploads keep video-relative time. Requires your approval (changes the column's meaning for service output). |
| D4 | Confidence format | Keep `92.0%` (current approved format). The brief's `0.92` is not adopted unless you approve. |
| D5 | Row unit | Keep one row per screenshot (current). Per-person detail goes in `person_results.csv` and the manifest. |
| D6 | How screenshots are spread over the hour | `sample_count` instants, one random instant in each equal part of the window (same rule as today, applied to the window); an instant with no footage is reported as "no footage", never replaced. Default 4, configurable 1–60 per request. |
| D7 | Read videos without full download | Open via short-lived presigned URL with OpenCV/FFmpeg (seek), fall back to download to a temp file. |
| D8 | Where MongoDB code lives | Not in this service. This repo provides the import contract + an example script for the Admin team. |

## 3. Files

### 3.1 New

| File | What it contains | Why |
|---|---|---|
| `backend/service/__init__.py` | package | |
| `backend/service/settings.py` | Service env settings (mode, API key hashes, bucket, region, allowed prefixes, workers, timeouts, temp dir) | One validated place for service config |
| `backend/service/auth.py` | `require_api_key` dependency (constant-time compare), request-ID middleware | D2 |
| `backend/service/schemas.py` | Pydantic models: `AnalyzeRequest`, `AnalysisStatus`, `AnalysisResults`, `ErrorResponse`, `HealthResponse` | Request/response validation, OpenAPI |
| `backend/service/routes_v1.py` | `GET /health`, `POST /api/v1/analyze`, `GET /api/v1/analysis/{id}`, `GET …/results`, `GET /api/v1/analyses`, `POST …/retry` | API_DESIGN.md |
| `backend/service/s3.py` | `S3Store`: list objects (paginated), filter videos, head, presigned GET, download to temp, upload with SHA-256 checksum + verify, error mapping (missing, access denied, network) | AWS integration |
| `backend/service/video_keys.py` | Parse camera / recording start time from an S3 key with a configurable pattern | Needs the Admin team's naming convention (§5) |
| `backend/service/window_plan.py` | Spread `sample_count` instants over [start, end) and map each to (video, frame) | D6, reuses `plan.py` helpers |
| `backend/service/worker.py` | `ServiceRunner`: bounded queue, `SERVICE_WORKERS` threads, per-analysis timeout, temp-dir cleanup, stage updates, retry | Background processing |
| `backend/service/outputs.py` | Writes `uniform_detection_report.csv` (six columns), `person_results.csv`, `analysis_metadata.json`; uploads them and the screenshots | Evidence mapping |
| `deploy/ai-service.service` | systemd unit (Uvicorn, restart policy, env file) | EC2 |
| `deploy/nginx-ai-service.conf` | TLS reverse proxy, body/time limits | EC2 |
| `deploy/iam-policy.json` | Least-privilege S3 policy | AWS |
| `deploy/ai-service.env.example` | Service env template (placeholders) | EC2 |
| `integrations/admin_examples/cron_trigger_analysis.py` | Example cron job 1 (HTTP only, no model) | Admin team |
| `integrations/admin_examples/cron_import_csv_to_mongodb.py` | Example cron job 2 (S3 CSV → MongoDB idempotent upsert) | Admin team |
| `requirements-service.txt` | `boto3` (+ pinned versions for EC2) | |
| `requirements-dev.txt` | `moto[s3]`, `pytest` | mocked S3 tests |
| `tests/service/test_*.py` | API, auth, S3 (moto), window plan, worker, outputs, duplicate + retry, cron contract | Testing |
| `docs/*.md` | The remaining documents listed in §6 | |

### 3.2 Modified (small, additive)

| File | Change |
|---|---|
| `backend/main.py` | Include `routes_v1`; request-ID middleware; in `SU_MODE=service` skip dashboard routers and the capture scheduler |
| `backend/core/config.py` | Read the new env vars |
| `backend/core/storage.py` | Add `S3Storage` implementing the existing `StorageBackend` interface |
| `backend/database/db.py` | Two new tables (DATABASE_DESIGN.md) — additive, with the existing automatic backup |
| `backend/extraction/extractor.py` | Accept a stream URL as the video source (OpenCV already can); keep path behaviour |
| `backend/extraction/runner.py` | A public method to create a job from a service plan (no API change) |
| `.env.example`, `README.md`, `CLAUDE.md` | Document service mode |

### 3.3 Reused unchanged

`plan.py`, `detect.py`, `screenshots.py`, `business_csv.py`, `reports.py`, `master_csv.py`, `classifier.py`,
`person.py`, `analysis/pipeline.py:Models`, `analysis/media.py:probe`, `model_config.py`, `settings.py`, all
frontend files.

## 4. Phases (mapped to the brief)

| Phase | Work | Done when |
|---|---|---|
| 2 | This plan + docs | You approve D1–D8 |
| 3 | FastAPI v1 API, auth, schemas, DB tables, worker with a **local folder** standing in for S3 | Tests pass with local fixtures |
| 4 | `S3Store` + `S3Storage`; moto tests; one **real** integration test against a test prefix you create | Real test listed, read and wrote objects and you saw them in the bucket |
| 5 | Outputs: six-column CSV, person CSV, manifest with stable IDs, upload + verify | Every CSV row maps to an uploaded screenshot key (test) |
| 6 | Frontend: integration spec for the Admin frontend; optional "service analyses" view in this dashboard | You decide if the local view is wanted |
| 7 | Admin cron examples + contracts | Admin team confirms the contract |
| 8 | Full test run, report actual numbers | Reported, nothing claimed that did not run |
| 9 | EC2 guide + deploy files; deployment itself only with your AWS access | Guide reviewed; deployment verified on the instance if you run it |

## 5. Needed from you / the Admin team

1. **S3 naming convention** of input videos (camera, recording start time, time zone, format) and 2–3 real
   example keys.
2. A **test prefix** in `yk-cctv-prediction` (e.g. `_ai_service_test/`) and new IAM credentials or an EC2 role
   with the policy in `deploy/iam-policy.json`. The keys shared earlier must be deactivated.
3. Number of stores / cameras per hour (sizing).
4. Admin backend language/framework and the existing MongoDB collections (to fit the import contract).
5. Decisions D1–D8.

## 6. Documents

Delivered now: PROJECT_AUDIT, EXISTING_ARCHITECTURE, IMPLEMENTATION_PLAN, API_DESIGN, DATABASE_DESIGN,
DEPLOYMENT_GUIDE. To be written with the implementation (they need the real code to be accurate):
TARGET_ARCHITECTURE, API_DOCUMENTATION (generated from OpenAPI + examples), AWS_S3_SETUP, IAM_PERMISSIONS,
EC2_DEPLOYMENT_GUIDE (expands DEPLOYMENT_GUIDE), CRON_JOB_INTEGRATION, MONGODB_INTEGRATION,
FRONTEND_INTEGRATION, TESTING_GUIDE, TROUBLESHOOTING, ENVIRONMENT_VARIABLES.

## 7. Risks

| Risk | Mitigation |
|---|---|
| The model is experimental; production numbers may mislead the Admin dashboard | Report model version in every result; label results "indicative"; retrain per DATA_COLLECTION_PLAN |
| MP4 files with the index at the end make remote seeking slow | Fall back to download; ask the recorder export for fast-start MP4 |
| Unknown S3 naming | Configurable pattern; refuse files whose time cannot be read rather than guess |
| Four to six frames cannot represent an hour | Documented everywhere as sampled analysis; configurable count |
| SQLite on one instance | Fine for one EC2 instance; one writer; back up the file; move to Postgres only if several instances are needed |
