# Project audit — Yatrikart CCTV Uniform Detection

Audit date: 2026-10-30. Repository: `C:\SHREE\YATRIKART\PROJECTS\YK_CCTV_U_D_M_main\YK_CCTV_U_D_M`,
branch `publish/cctv-udm` (published as `main` of https://github.com/Shree-108/CCTV_UDM at commit `03266dc`).
Everything below was checked in the source code, the database or by running it on this PC. Nothing in the
application was changed for this audit.

## 1. Summary

The project is a **working local application**: a FastAPI backend plus a Next.js (React) dashboard that
analyses uploaded CCTV videos, saves screenshots, classifies each person's shirt with an existing model and
writes CSV reports. It is **not yet** the system described in the target architecture:

| Target requirement | Today |
|---|---|
| Authenticated API for the Admin backend | **Missing.** No endpoint has authentication (CORS only). |
| Fetch videos from S3 by store + time range | **Missing.** Videos arrive by manual upload. `SU_S3_BUCKET` only changes a status badge. No `boto3` dependency. |
| Upload results to S3 | **Missing.** Everything is written to local disk (`storage/`). |
| One analysis = one store + one time window (several videos) | **Missing.** One job = one uploaded video. |
| Configurable screenshot count | **Exists.** `sample_count` (default 4, from `config/uniform_config.yaml` → Settings). |
| Existing model on sampled frames | **Exists** and is reusable as is (`backend/extraction/*`). |
| Six-column business CSV per analysis | **Exists** (`business_csv.py`), plus an append-only master CSV and monthly reports. |
| Stable IDs linking CSV row → screenshot → prediction | **Partly.** Job ID + image ID + run ID exist internally; the business CSV links by file name only. |
| Status / results / list / retry endpoints | **Partly.** Equivalent local endpoints exist under `/api/extractions` and `/api/detection-runs` (different shape). |
| Background processing, queue, duplicate protection | **Exists** for local jobs (one worker thread, duplicate-job checks). No timeouts. |
| MongoDB | **Not in this repository** (belongs to the Admin backend). |
| Admin backend / cron jobs | **Not in this repository.** |
| EC2 deployment (systemd, HTTPS, IAM) | **Missing.** Only local run instructions; no Dockerfile, no systemd unit. |

## 2. Stack (verified)

| Part | Technology | Version on this PC |
|---|---|---|
| Backend | Python + FastAPI + Uvicorn | Python 3.14.8, FastAPI 0.142.2, Uvicorn 0.54.0 |
| Computer vision | Ultralytics YOLO, PyTorch, OpenCV | ultralytics 8.4.173, torch 2.14.1 (CPU), opencv 5.0.0 |
| Person detector | YOLO11n (COCO class 0 "person"), ByteTrack for full-video mode | `models/person/yolo11n.pt` (5.6 MB) |
| Uniform classifier | YOLO11n-cls, 224 px, classes `{0: non_uniform, 1: uniform}` | `models/uniform_best.pt` (3.2 MB), v0.2.0 **experimental**, SHA-256 `5d785e8b…` |
| Database | SQLite (WAL) | `storage/sightline_uniform.db` |
| Storage | Local disk behind a small `StorageBackend` interface | `backend/core/storage.py` (`LocalStorage` only) |
| Frontend | Next.js 16.3 (React 19), TypeScript, Tailwind | Node 24 |
| Tests | pytest (backend), `tsc --noEmit` (frontend) | see §8 |

## 3. Model and inference (as implemented)

1. **Sampling** (`backend/extraction/plan.py`): `random_segments` splits the video's frames into
   `sample_count` equal parts and takes one random frame per part with a recorded seed (reproducible).
2. **Extraction** (`extractor.py`): seeks to each planned frame (does not decode the whole video), saves the
   original pixels as PNG.
3. **Detection** (`detect.py` → `classify_frame`): YOLO11n person detection (confidence ≥ 0.40, input 640),
   then for **each person separately** an upper-body crop (12–55 % of the box height, 10–90 % of the width).
4. **Visibility checks** (`backend/uniform/classifier.py`): too small (< 50 px), cut off, squat box
   (height/width < `min_box_aspect` 1.5), occluded, too blurred, low resolution → **UNCERTAIN** with a reason.
5. **Classification**: crop resized to `crop_size` 112×160 (see §6.1), YOLO-cls gives P(uniform).
   Classes are mapped **by name** through `class_map`, never by index.
6. **Decision**: P(uniform) ≥ 0.70 → UNIFORM; P(non_uniform) ≥ 0.60 → NON_UNIFORM; otherwise UNCERTAIN.
7. **Screenshot outcome**: UNIFORM_DETECTED (≥ 1 uniform), REVIEW_REQUIRED, NO_UNIFORM_DETECTED,
   PROCESSING_FAILED. Annotated copy: green / red / amber boxes.

Measured on this PC (Intel i3-1115G4, 2 cores / 4 threads, 7.8 GB RAM, no NVIDIA GPU):

| Step | Time / memory |
|---|---|
| Load + warm up both models | 10.6 s, ~340 MB RSS |
| One 1920×1080 screenshot (1–3 people) | 117–532 ms |
| Seek + decode one frame of a 14-minute video | ~350 ms |
| Peak memory during inference | ~372 MB |
| Real CCTV footage size | 0.9–1.4 GB per camera-hour (1080p 25 fps ≈ 0.88 GB/h) |

**Conclusion:** inference on 5–6 screenshots takes a few seconds on CPU. The cost of an hourly analysis is
**getting the video**, not running the model. A GPU is not needed for the sampled workflow.

## 4. Data flow and outputs (as implemented)

`POST /api/media` (upload) → `POST /api/extractions` (job, `run_detection: true`) → worker thread →
`storage/extractions/<job>/frames/*.png` + `image_manifest.csv` → detection run →
`storage/extractions/<job>/detections/<run>/` (`annotated/*.jpg`, `detection_results.csv` one row per person,
`screenshot_summary.csv`, `uniform_compliance_details.csv`, `uniform_compliance_job_<job>.csv`) →
`storage/exports/uniform_compliance_master.csv` (append-only, ledger-deduplicated).

Business CSV (fixed): `Store Name, Date, Time Stamp, img name, confidence score, remark (Example)`, one row
per screenshot. `Date` only when the recording start time is known; `Time Stamp` = position **in the video**
(HH:MM:SS); `confidence score` = P(uniform) as a percentage (`92.5%`, or `Person 1: 92.5%; Person 2: 41.0%`).

## 5. Security findings

| # | Finding | Severity | Notes |
|---|---|---|---|
| S1 | **No authentication or authorisation** on any of the 67 API operations | High | Fine on a single PC; must not be exposed on EC2 as is. |
| S2 | CORS allows `localhost:3010` only | OK | CORS is not access control; it only affects browsers. |
| S3 | No secrets in the repository or history | OK | Scanned tracked + untracked files and all commits; `.env` absent. The AWS keys shared elsewhere are **not** in the repo — still treat them as compromised and deactivate them. |
| S4 | File download endpoints | OK | Screenshot / annotated / CSV endpoints only serve names listed in the job's own manifest (no path traversal). Uploads are saved under random names. |
| S5 | Upload validation | OK | Extension whitelist, size limit (`MAX_VIDEO_UPLOAD_MB`, default 1024), OpenCV decode check, SHA-256 recorded. |
| S6 | Error messages | Medium | Some API errors include project-relative paths (`storage/...`); absolute paths are masked in the master CSV status. |
| S7 | Logging | Low | `logging.basicConfig` text logs; no request IDs; one `print()` in `backend/analysis/pipeline.py:407`. |
| S8 | Privacy | Info | `storage/`, `data/`, `runs/` contain images of real people and are git-ignored. The repository on GitHub is **public**. |

## 6. Architecture and code-quality findings

### 6.1 Uncommitted changes made after publishing (29 Oct 2026)

15 modified and 6 new files are not committed (not made in this audit):
`backend/uniform/classifier.py`, `backend/core/model_config.py`, `config/uniform_config.yaml`
(`crop_size: [112, 160]`, `min_box_aspect: 1.5`), `backend/api/routes_extraction.py` (screenshot evidence
endpoint), `backend/extraction/business_csv.py`, `reports_service.py`, frontend evidence viewer
(`ScreenshotEvidence.tsx`), training scripts (`candidate.py`, `diagnose.py`, `register_model.py`
activate/rollback) and three new test files.

The most important one fixes a **training/inference preprocessing mismatch**: `training/bootstrap.py`
resized every crop to 112×160 before training v0.1.0/v0.2.0, but inference did not. The config notes that the
light-blue shirt that scored P(uniform) 0.94 scores 0.03 when the crop matches training. These changes should
be reviewed and committed before the microservice work starts, so it builds on a known state.

### 6.2 Other findings

| # | Finding | Impact on the target |
|---|---|---|
| A1 | Two pipelines: quick sampled (`backend/extraction/`) and full-video (`backend/analysis/pipeline.py`). Both use the same `UniformDetector`. | Reuse the **quick sampled** pipeline for the microservice. Keep full-video local-only. |
| A2 | One job = one video. No concept of a store + time window spanning several files. | New "service analysis" layer needed on top of existing jobs. |
| A3 | Storage abstraction exists (`StorageBackend.put`) but only `LocalStorage`; most code writes local paths directly. | S3 should be added as **upload after local processing** (keeps the pipeline unchanged). |
| A4 | Business `Time Stamp` is video-relative; `Date` blank without a recording start time. | With S3 keys that carry the recording time, Date and clock time become known — but using clock time in `Time Stamp` changes its meaning → **needs your approval**. |
| A5 | Example CSV in the brief uses `0.92`; the implemented, approved format is `92.0%` with one row per screenshot. | **Needs a decision** before the Admin import is written. |
| A6 | Background worker: one thread, no per-job timeout; interrupted jobs are marked `interrupted` at start-up. | Add timeouts and a bounded queue for the service. |
| A7 | Hidden pages (Overview, Image Analysis, Frame Extraction, Analytics, History, Upload) still have routes. | Not dead code; keep. |
| A8 | `requirements.txt` uses ranges, not pins; developed on Python 3.14. | Pin versions and test on the EC2 Python (3.12 recommended). Install the CPU PyTorch wheel on EC2. |
| A9 | `package.json` names `pnpm@12.3.4` in `packageManager` but the lock file in use is npm's. | Harmless locally; document. |
| A10 | `AGENTS.md` is generated by `next dev`. | Leave as is. |
| A11 | No hardcoded absolute paths in code; all paths relative to the project root or env vars. | Good — works on EC2 unchanged. |
| A12 | No Dockerfile, systemd unit, reverse proxy or health-check configuration. | To be added (Phase 9). |
| A13 | The uniform model is experimental (trained on ~4 people). Known false "uniform" on light shirts; partly addressed by A-6.1. | Results must be presented as indicative until retrained and evaluated (`training/DATA_COLLECTION_PLAN.md`). |

### 6.3 Not present (by design or missing)

- MongoDB, Admin backend, cron jobs — belong to the company's Admin system; this repo will provide the
  contract and examples only.
- Live CCTV / NVR — only an interface (`backend/capture/cctv_adapter.py`).
- User login.

## 7. Dependencies to add for the target

| Package | Why | Where |
|---|---|---|
| `boto3` | S3 list / get / put / presigned URLs | runtime |
| `moto[s3]` | mocked S3 for unit tests | tests only |
| (none for MongoDB) | MongoDB is in the Admin backend; the example import script would need `pymongo` there | Admin side |

No new web framework is needed; FastAPI is already used.

## 8. Tests (actually executed)

See the test run recorded at the end of this audit (§9).

## 9. Test run on 2026-10-30

Run on the current working tree (including the uncommitted 29-Oct changes), this PC, Python 3.14.8:

```
.venv\Scripts\python -m pytest tests -q     ->  134 passed, 1 warning in 226.61 s (exit 0)
npx tsc --noEmit                            ->  exit 0 (no type errors)
```

The warning is a Starlette deprecation notice about `httpx` in the test client. No S3, MongoDB or EC2 tests
exist yet (those parts do not exist). ESLint is not configured, so no lint run was possible.

## 10. Questions for the project owner and the Admin team

1. **S3 input layout:** how are files named inside `…/{store}/input_data/`? One file per camera per hour?
   Does the name or metadata give the camera and the **recording start time** (time zone)? Container format
   (MP4 / CP Plus `.dav`)?
2. **Store identity:** is `cart_50` the store ID? Where does the store name come from — the request, or a
   mapping the service should keep?
3. **CSV format** (A4, A5): keep `92.0%` and video-relative `Time Stamp`, or switch to `0.92` and clock time
   for the service output? One row per screenshot (current) or one row per person (as the brief's example
   suggests)?
4. **Authentication** between Admin backend and the service: API key in a header (proposed), HMAC-signed
   requests, or private network only (VPC + security group)?
5. **Volume:** how many stores and cameras per hour? This decides the instance size and worker count.
6. **Admin stack:** language and framework of the Admin backend, and the existing MongoDB collections.
7. **Model:** keep v0.2.0 (experimental) in production, or wait for a retrained model?
