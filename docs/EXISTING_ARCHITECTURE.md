# Existing architecture

How the application works today (2026-10-30), file by file. "Modify?" says whether the AI microservice
work needs to change the file; "Reuse?" says whether the microservice can call it as it is.

## 1. Big picture

```
Browser (Next.js dashboard, :3010)
   │  HTTP JSON / file downloads (no auth, CORS localhost)
   ▼
FastAPI app (backend/main.py, :8010)
   ├─ api/routes.py ............ upload, full-video analyses, alerts, settings, health
   ├─ api/routes_extraction.py . quick sampled jobs, screenshots, per-job CSV, master CSV, annotated video
   ├─ api/routes_results.py .... cross-video results + technical CSV exports
   ├─ api/routes_reports.py .... latest / monthly / historical business reports
   └─ api/routes_capture.py .... cameras + scheduled capture (local sources)
        │
        ├─ ExtractionRunner (1 worker thread) ── plan → extract → detect → CSVs → master CSV
        ├─ JobRunner (1 worker thread) ───────── full-video pipeline (tracker, alerts, annotated video)
        └─ CaptureService (scheduler thread) ── grab frame from local source → analyse
             │
             ├─ Models: PersonDetector (YOLO11n) + UniformDetector (YOLO11n-cls)
             ├─ SQLite: storage/sightline_uniform.db
             └─ Files:  storage/{uploads,extractions,analyses,captures,exports}
```

## 2. Backend start-up — `backend/main.py`

| | |
|---|---|
| Responsibility | Creates the FastAPI app; `AppState` opens the DB, marks interrupted jobs, loads Settings and Models, starts the two job runners and the capture scheduler; warms up the models; runs the master-CSV sync. |
| Classes / functions | `AppState`, `create_app(state=None, preload=True)` |
| Depends on | `core.config`, `core.settings`, `database.db`, `analysis.pipeline.Models`, `analysis.jobs.JobRunner`, `extraction.runner.ExtractionRunner`, `capture.service`, all routers |
| Modify? | **Yes** — add the `/api/v1` router, API-key dependency, request-ID middleware, service mode. |
| Reuse? | Yes (one app, two modes). |

## 3. Core — `backend/core/`

| File | Responsibility | Key items | Modify? | Reuse? |
|---|---|---|---|---|
| `config.py` | Paths and env: `ROOT`, `STORAGE`, `UPLOADS`, `DB_PATH`, `CORS_ORIGINS`, upload limits | constants | Yes — new env vars (API keys, S3, service mode) | Yes |
| `model_config.py` | Reads `config/uniform_config.yaml`; model path, SHA-256 identity, class map, `crop_size` | `load`, `uniform_model_path`, `model_identity`, `version_label` | No | Yes |
| `settings.py` | Runtime settings (defaults from YAML, overrides in `storage/settings.json`), validation | `Settings` | No (sample count already configurable) | Yes |
| `storage.py` | Storage interface; only `LocalStorage` (S3 is a documented stub) | `StorageBackend`, `LocalStorage`, `get_storage`, `rel` | **Yes** — add `S3Storage` | Yes |

## 4. Database — `backend/database/db.py`

| | |
|---|---|
| Responsibility | SQLite schema and all queries; additive migrations at start-up (backup before adding a column). |
| Tables | `media`, `analyses`, `persons`, `alerts`, `cameras`, `captures`, `extractions`, `frame_detection_runs` |
| Class | `DB` (`add_*`, `update_*`, lists, `interrupt_*`, `media_by_sha256`, `_backup`) |
| Modify? | **Yes** — add `service_analyses` and `service_analysis_videos` (see DATABASE_DESIGN.md). |
| Reuse? | Yes. |

## 5. Quick sampled pipeline — `backend/extraction/` (the one the microservice reuses)

| File | Responsibility | Key items | Modify? | Reuse? |
|---|---|---|---|---|
| `plan.py` | Which frames: `interval`, `random`, `random_segments` (one random frame per equal part, seeded) | `Sample`, `plan_segments`, `plan_random`, `describe`, `hhmmss`, `clock_time` | Small — a window plan across several videos | Yes |
| `extractor.py` | Seek to each planned frame, save PNG + `image_manifest.csv`, record failures | `extract`, `image_id`, `row_meta` | Small — accept a stream URL as well as a path | Yes |
| `runner.py` | Job validation, duplicate prevention, worker thread, detection runs, CSV writes, master append | `ExtractionRunner`, `validate_video`, `plan_for`, `read_manifest` | Small — expose a call used by the service worker | Yes |
| `detect.py` | Per-screenshot inference, `detection_results.csv` (one row per person), annotated image | `classify_frame`, `run`, `annotate`, `outcome_for`, `RESULT_FIELDS` | No | Yes |
| `screenshots.py` | One record per sampled screenshot + remark text | `screenshot_results`, `remark` | No | Yes |
| `business_csv.py` | Six-column business CSV per job; evidence links | `BUSINESS_FIELDS`, `business_row`, `records`, `write` | No (unless the CSV format decision changes it) | Yes |
| `master_csv.py` | Append-only cumulative CSV with ledger, lock, crash repair | `append`, `sync`, `status` | No | Yes |
| `reports.py` | `screenshot_summary.csv`, `uniform_compliance_details.csv` | `build`, `write_run_reports` | No | Yes |
| `reports_service.py` | Latest / monthly / historical reports from DB + files | `overview`, `month_report`, `history_records` | No | Yes |
| `results.py` | Cross-video results for the dashboard, hourly summary, failure rows | `collect`, `summarize`, `hourly` | No | Yes |

## 6. Model wrappers

| File | Responsibility | Key items | Modify? | Reuse? |
|---|---|---|---|---|
| `backend/detection/person.py` | YOLO person detection (+ ByteTrack for video), device selection | `PersonDetector`, `Person`, `select_device` | No | Yes |
| `backend/detection/grid.py` | Multi-camera DVR layouts and ROI polygons | `detect_layout`, `AnalysisArea`, `validate_area` | No | Yes (full frame) |
| `backend/uniform/classifier.py` | Upper-body crop, visibility checks, trained classifier / stripe baseline, decision thresholds | `UniformDetector`, `TrainedClassifier`, `upper_body_box`, `visibility_issue` | No | Yes |
| `backend/analysis/pipeline.py` | `Models` container (loads both models once) + full-video pipeline | `Models`, `analyze_image`, `analyze_video` | No | `Models` yes |

## 7. Full-video pipeline and other backend modules (kept, not used by the microservice)

| File | Responsibility |
|---|---|
| `analysis/jobs.py` | `JobRunner`: one background thread for full-video analyses |
| `analysis/runner.py` | `start_analysis` shared by upload and capture |
| `analysis/media.py` | Upload probe (duration, fps, resolution, preview, layout) — **reused** for video validation |
| `analysis/annotate.py`, `compliance.py`, `temporal.py` | Drawing, store-level rule, presence timeline |
| `tracking/memory.py` | Per-track vote window |
| `reports/writer.py` | JSON / CSV / TXT reports of a full-video analysis |
| `capture/*` | Scheduled capture from local folders / video files; `cctv_adapter.py` interface for a future NVR |
| `api/routes_capture.py` | Cameras and captures API |

## 8. API routers (67 operations, all under `/api`, no authentication)

| Router | Main endpoints |
|---|---|
| `routes.py` | `GET /api/health`, `POST /api/media`, `GET /api/media/{id}`, `POST/GET /api/analyses…`, alerts, settings, overview, analytics, search |
| `routes_extraction.py` | `POST/GET /api/extractions…`, `/frames`, `/uniform-compliance`, `/uniform_compliance.csv`, `/evidence/{image_id}`, `/annotated-video`, `/detect`, `GET /api/detection-runs/{id}…`, master CSV status/sync/download |
| `routes_results.py` | `GET /api/frame-results` + six technical CSV exports |
| `routes_reports.py` | `GET /api/reports/overview`, `/months/{YYYY-MM}`, `/months/{YYYY-MM}.csv`, `/records` |
| `routes_capture.py` | `/api/cameras…`, `/api/captures…`, `/api/capture/status` |

## 9. Frontend (Next.js, `app/` routes + `frontend/` code)

| Area | Files | Role |
|---|---|---|
| API client | `frontend/api/client.ts`, `endpoints.ts`, `types.ts` | `API_BASE` from `NEXT_PUBLIC_API_URL`; typed calls for every endpoint used |
| Shell | `frontend/components/shell/AppShell.tsx`, `nav.ts` | Sidebar (Video Analysis, Results Dashboard, Alerts, Settings, Help), command palette |
| Video Analysis | `features/analysis/VideoAnalysisPage.tsx` (quick mode via `ProcessPanel`), `AnalysisWorkspace.tsx` + `AnalysisView.tsx` (full-video mode) | Upload, configure, progress stages, results |
| Results Dashboard | `features/results/ResultsDashboard.tsx`, `ScreenshotEvidence.tsx` | Sections A–F, master/monthly CSV, evidence viewer, technical tables |
| Frame Extraction | `features/extraction/*` | Job details, interval sampling, detection runs |
| Others | `features/{alerts,settings,help,overview,analytics}` | Alerts, settings & model status, help |
| UI kit | `components/ui/*`, `hooks/useQuery.ts`, `stores/app.tsx`, `lib/format.ts` | Shared components, polling queries, toasts |

The frontend only talks to the backend through `frontend/api/*`; it never touches files or S3 directly.
**Modify?** Only if the dashboard should show service analyses (Phase 6). **Reuse?** Yes.

## 10. Training tools (`training/`, not part of the running service)

`bootstrap.py` (how v0.1.0/v0.2.0 were trained), `train.py`, `evaluate.py`, `prepare_dataset.py`,
`register_model.py` (activate / rollback), `candidate.py` (versioned candidates), `diagnose.py`,
`audit_dataset.py`, `label_review.py`, `DATA_COLLECTION_PLAN.md`.

## 11. Configuration and files

| Item | Purpose |
|---|---|
| `config/uniform_config.yaml` | Model path, version, SHA-256, class map, `crop_size`, thresholds, extraction defaults (4 screenshots) |
| `.env` / `.env.example` | Ports / CORS, upload limits, storage paths, device, scheduler switch |
| `storage/settings.json` | Per-installation overrides from the Settings page |
| `storage/` | All runtime data (git-ignored) |
| `models/` | Weights (tracked) |

## 12. Tests (`tests/`)

pytest with a temporary storage folder and database (`conftest.py`); the uniform model is pinned to a
missing file except in tests that use the `real_model` fixture. Covers sampling, extraction, detection,
CSV schemas, master CSV, reports, API, capture, model configuration and activation.
