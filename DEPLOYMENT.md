# Deploying Sightline · Uniform Compliance on another PC

Everything the app needs is inside the project folder. No path depends on the PC it was built on: model paths are relative to the project root and are set in `config/uniform_config.yaml`.

## Required files

| File | In git? | Purpose |
|---|---|---|
| `models/uniform_best.pt` | **yes, once you commit it** (see *Model weights*) | the trained uniform classifier (YOLO-cls) |
| `config/uniform_config.yaml` | yes | model path, version, SHA-256, class map, thresholds, image size, device |
| `models/person/yolo11n.pt` | yes (tracked) | official COCO person detector. If it is missing it is downloaded to this path on first start, which needs internet |
| `.env` | **no** (per PC) | optional: ports / CORS / storage location |
| `storage/` | **no** (per PC) | uploads, results, SQLite database, per-PC Settings overrides. Created automatically |

## Steps

```bash
# 1. clone / copy the project
git clone https://github.com/Shree-108/CCTV_UDM.git && cd CCTV_UDM
git lfs pull                                  # only if the model is stored with Git LFS

# 2. install dependencies (Python 3.10+; Node 20+)
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt     # Linux/macOS: .venv/bin/python
npm install

# 3. place / restore the trained weights (skip if they came with the clone)
#    -> models/uniform_best.pt
.venv\Scripts\python -m training.register_model --check     # must print "MATCH - approved model"

# 4. configure environment (optional)
copy .env.example .env                        # edit CORS_ORIGINS / SU_DEVICE if needed
#    frontend: .env.local with NEXT_PUBLIC_API_URL=http://<server>:8010 if the browser is on another machine

# 5. start
.venv\Scripts\python -m uvicorn backend.main:app --host 127.0.0.1 --port 8010
npm run dev                                   # production: npm run build && npm run start
```

To check that it worked, open http://localhost:3010 and look at **Overview → Model status**:
- **Uniform classifier:** should be `Ready`.
- **Model version:** should be `v<version>` and the `path · sha256 …` hint should end in `✓ verified`.

If it says **MODEL FILE MISSING — restore models/uniform_best.pt**, the weights are not in place. If it says **WRONG MODEL FILE**, the file is not the approved one: its checksum differs from `config/uniform_config.yaml`.

The backend log prints the same information on every start, for example `Uniform model loaded: models/uniform_best.pt (yatrikart-uniform-cls v1.0.0 · sha256 …)`.

## Model weights: one file, one version, everywhere

1. **Approve a model once.**
   - Put the file at `models/uniform_best.pt`.
   - Set `model.version` in `config/uniform_config.yaml`.
   - Record its checksum: `python -m training.register_model --version 1.0.0`.
   - Commit both files.
2. **Every deployment** loads the file from that path. On start it compares the file's SHA-256 with the config and shows the version and checksum in the dashboard and in every report.
3. **Replace the model** only by repeating step 1 with a new version number. Do not keep copies such as `best_old.pt` / `best (1).pt`, because git history already keeps old versions.

### How the file travels with the project

A YOLO11n-cls classifier (the architecture `training/train.py` uses) is about **3 MB**. That is far below GitHub's limits (warning at 50 MB, rejected at 100 MB), so **commit it normally**:

```bash
git add models/uniform_best.pt config/uniform_config.yaml
git commit -m "Uniform model v1.0.0"
```

`.gitignore` ignores every other `*.pt` (training runs, scratch copies) and explicitly allows `models/uniform_best.pt` and `models/person/yolo11n.pt`.

If the file is **larger than ~50 MB** (for example a bigger backbone), use **Git LFS** instead of normal git:

```bash
git lfs install
git lfs track "models/uniform_best.pt"
git add .gitattributes models/uniform_best.pt
git commit -m "Track uniform model with Git LFS"
# deployment PC: git lfs install && git clone <repo-url>   (or `git lfs pull` in an existing clone)
```

If the weights must not be in the repository at all, store `uniform_best.pt` in one controlled place, such as a release asset or a company drive. Copy it to `models/uniform_best.pt` on each PC. The SHA-256 in the config still guarantees that every PC runs the identical file (`python -m training.register_model --check`).

## Inference settings

These are set in `config/uniform_config.yaml` → `inference:`.

| Key | Default | Meaning |
|---|---|---|
| `uniform_confidence` | 0.70 | P(uniform) needed for **COMPLIANT**; lower → **UNCERTAIN — PLEASE VERIFY** |
| `non_uniform_confidence` | 0.60 | P(non-uniform) needed for **NON-COMPLIANT**; lower → **UNCERTAIN** |
| `min_person_height_px` | 50 | smaller (distant) people → UNCERTAIN |
| `person_confidence` / `inference_size` | 0.40 / 640 | person detector |
| `device` | auto | `auto`, `cpu` or `cuda` (or `SU_DEVICE` in `.env`) |
| `overall_rule` | any_uniform | `any_uniform`: COMPLIANT if ≥ 1 person in uniform (staff presence). `all_uniform`: NON-COMPLIANT if anyone judged is not in uniform |
| `untrained_model_mode` | baseline | what to do if the weights are missing: `baseline` (stripe heuristic, labelled indicative) or `disabled` (all UNCERTAIN) |

The Settings page can override these per PC; overrides are saved in `storage/settings.json`. **Reset to defaults** goes back to the config values.

Some people can't be judged reliably: they are too small or far away, cut off, hidden behind someone else, blurred, or the classifier is not confident enough. These people are reported as **UNCERTAIN — PLEASE VERIFY** instead of being forced into compliant or non-compliant. The system classifies only the upper-body clothing crop. It never uses faces or identity.

## Scheduled capture (local test sources)

The backend can capture a frame from each registered camera on a fixed schedule and analyse it with the same pipeline as an uploaded image. Each result is stored with its **store ID, camera ID, capture time and model version**.

**Defaults:** `capture:` in `config/uniform_config.yaml`, editable under Settings → Advanced.

| Setting | Default | Meaning |
|---|---|---|
| `capture_interval_minutes` | 15 | Slots aligned to the window start: 09:00, 09:15, … |
| `capture_active_start` / `_end` | 09:00 / 21:00 | Local time of the PC; the end is exclusive, so the last slot is 20:45 (48 per day) |
| `default_store_id` | STORE-001 | Store/card ID given to new cameras |
| `capture_enabled` | true | Master switch for scheduled captures |
| `video_step_seconds` | 60 | Video sources: footage advanced per capture |

**Sources:** for now only **local** test sources can be used, either a folder of images or a video file. Live CCTV (RTSP/NVR) will be added once authorised access and the recorder details are confirmed.

**Guarantees:**
- One capture at a time per camera.
- Each slot is captured at most once per camera, including after a restart. A restart catches up the slot in progress but never back-fills older slots.
- The original frame is always kept, even if analysis fails.
- Every attempt is recorded with status `done`, `capture_failed` or `analysis_failed`.

To register a camera (the backend must be running):

```bash
curl -X POST http://127.0.0.1:8010/api/cameras -H "Content-Type: application/json" -d "{\"id\": \"CAM-01\", \"source_type\": \"video\", \"source\": \"D:/footage/store_interior.mp4\"}"
```

Other camera fields:
- `store_id`: defaults to `default_store_id`.
- `name`
- `area`: the camera tile and store ROI, same format as the analysis API. The default is the whole frame.
- `enabled`

API endpoints:

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/cameras/{id}/capture` | Capture now (manual) |
| GET | `/api/captures?camera_id=&store_id=&status=&start=&end=` | Capture history |
| GET | `/api/captures/{id}/image` | Original captured frame |
| GET | `/api/capture/status` | Schedule status and next slot |
| GET | `/api/analyses?source=capture&camera_id=CAM-01` | Analyses from captures |
| PUT / DELETE | `/api/cameras/{id}` | Change or remove a camera. Removing a camera keeps its history. |

To turn the background scheduler off, set `SU_CAPTURE_SCHEDULER=off` in `.env`. Manual captures still work.

## Frame extraction from uploaded videos

Open **Frame Extraction** in the sidebar (`/extract`), upload a video, choose how to sample it, and click **EXTRACT FRAMES**. No model is run; frames are saved exactly as decoded.

**Sampling strategies** (the strategy and its settings are recorded on the job and on every manifest row):
- **Random across video (default, `random_segments`):** the video is split into `sample_count` (default **4**) equal parts and one random frame is taken from each, with `random.Random(seed)`. The frames are always distinct and inside the video. If the video has fewer usable frames than requested, the job is refused with "too short" — fewer images are never returned silently. The selected timestamps and their status (saved / failed) are shown on the job and on the dashboard. The initial frame is off by default and, if switched on, is extra (not one of the 4).
- **Fixed interval:** scheduled samples at 1×, 2×, 3× … the interval, as long as that time is at or before the **last frame**. A 15:00 video at a 15-minute interval has **no** scheduled sample (the first would be at 15:00, after the last frame): 1 image with the initial frame, none without it — never four.
- **Random frames (`random`):** `sample_count` distinct frames drawn uniformly from frame 1 to the last frame with `random.Random(seed)`. The seed is recorded; the same video, count and seed always give the same frames. Leave the seed empty to have one generated (and recorded).
- **The initial frame** (00:00:00) is optional and recorded separately as `sample_type = initial`.
- **The page shows the exact count before you start.**
- Sampling does **not** guarantee that every person in the video is captured: only the sampled frames are ever analysed.

**Traceability fields (optional, never guessed):** `store_id`, `camera_id` and `recording_start_time` (ISO 8601 clock time of the video's first frame). Leave them empty when unknown — they stay blank in the CSVs and show as "unknown" on the dashboard. A real clock time (`recording_timestamp`) is only produced when `recording_start_time` was supplied; otherwise only the position inside the video (`video_timestamp_*`) is shown, and it is never presented as a clock time.

**Defaults** (Settings → Advanced; `extraction:` in `config/uniform_config.yaml`):
- sampling: `default_strategy: random_segments`, `default_sample_count: 4`
- interval (fixed-interval mode): 30 s (15 min is a preset)
- format: PNG (lossless); JPG (quality 95) is optional
- maximum frames per job: 5000

**Output** (`storage/extractions/<extraction_job_id>/`, a new folder per job, never overwritten):

| File | Contents |
|---|---|
| `frames/<video>_<media id>_<job>_<initial, sNNNN or rNNNN>_f<frame>_t<hhmmssmmm>.png` | The saved frames, unchanged |
| `image_manifest.csv` | Successfully saved frames only, written row by row (schema below) |
| `failures.csv` | Samples that could not be extracted (the job also records "Only N of M screenshots could be extracted"): `image_id, extraction_job_id, source_video, stage, sample_type, target_seconds, target_hhmmss, target_frame, error_category, error_message` (`decoder_error`, `no_frame`, `path_too_long`, `write_failed`) |

**Validation and safety:**
- **Rejected uploads:** wrong file type, unreadable video or missing codec, unknown duration, frame rate or resolution, and files over `MAX_VIDEO_UPLOAD_MB`. Invalid store / camera IDs (letters, digits, `.`, `_`, `-` only) and unparseable recording start times are refused, not corrected.
- **Duplicates:** only one active job per video. An identical finished job (same strategy, settings and metadata; for random sampling only with the same explicit seed) is shown instead of being repeated; use **Extract again** for a new run.
- **Stopping and restarts:** **Stop** cancels a running job and keeps exactly the frames already saved. Jobs still running when the server stops are marked `interrupted`.

**API:**

| Method | Endpoint |
|---|---|
| GET | `/api/media/{id}/extraction-plan?sampling_strategy=&interval_seconds=&sample_count=&random_seed=&include_initial=` |
| POST | `/api/extractions` `{media_id, sampling_strategy?, interval_seconds? \| sample_count? + random_seed?, include_initial?, store_id?, camera_id?, recording_start_time?, run_detection?, threshold?, new_run?}` — with only `media_id` it uses the Settings default (4 screenshots, random across video, no initial frame); with `interval_seconds` and no strategy it uses fixed interval |
| GET | `/api/extractions[/{id}]` |
| POST | `/api/extractions/{id}/cancel` |
| GET | `/api/extractions/{id}/frames` |
| GET | `/api/extractions/{id}/image_manifest.csv` |
| GET | `/api/extractions/{id}/failures.csv` |

### Uniform detection on extracted frames

There are two ways to run the **existing trained model** on an extraction's frames:
- Before extracting, turn on **Run uniform detection on the frames afterwards**.
- On a finished job, click **Run uniform detection** (no new upload needed).

**The model must be trained.** If `models/uniform_best.pt` is missing, unreadable or does not match the checksum in `config/uniform_config.yaml`, the run is refused with that reason (and listed in `processing_failures`). No substitute model is used, and in particular not the stripe-pattern baseline.

**How detection works:** the model is two-stage — the YOLO person detector (`models/person/yolo11n.pt`) finds people, then the YOLO-cls classifier (`models/uniform_best.pt`, classes `non_uniform` / `uniform`) scores each person's upper-body (shirt) crop. A *target-shirt detection* is a person whose shirt crop scores P(uniform) ≥ the threshold. The threshold defaults to Settings → Uniform confidence (**0.70**) and can be set per run; it is recorded on every row. The box is the shirt region of the person box.

**Outcome per image (`prediction`):**

| Outcome | When |
|---|---|
| `UNIFORM_DETECTED` | At least one target-shirt detection |
| `NO_UNIFORM_DETECTED` | None, and every person found could be judged (or nobody was found). This does **not** mean anyone is non-compliant. |
| `REVIEW_REQUIRED` | None, while at least one person could not be judged (too small, cut off, occluded, blurred or low confidence). The reason is in `review_reason`. |
| `PROCESSING_FAILED` | The image could not be read or inference raised an error. No prediction is made; prediction-specific fields are blank. Never recorded as `NO_UNIFORM_DETECTED`. |

`processing_status` (image level, separate from the outcome): `ok`, `annotation_failed` (inference ran but the annotated copy could not be saved) or `failed`.

**Output** (`storage/extractions/<extraction_job_id>/detections/<detection_run_id>/`, a new folder per run):
- **`annotated/`:** a copy of each frame with the boxes and scores drawn on. The original frames are never modified.
- **`detection_results.csv`:** schema below. `review_status` starts as `unreviewed`; human review is not implemented yet.

**Repeated requests:** the same extraction + model checksum + threshold returns the existing run; **Run again** (`new_run`) creates a new run with its own `detection_run_id` and folder.

**API:**

| Method | Endpoint |
|---|---|
| POST | `/api/extractions/{id}/detect` `{threshold?, new_run?}` |
| GET | `/api/extractions/{id}/detections` |
| GET | `/api/detection-runs/{run}` |
| GET | `/api/detection-runs/{run}/results` |
| GET | `/api/detection-runs/{run}/detection_results.csv` |
| POST | `/api/detection-runs/{run}/cancel` |

### CSV schemas and how they link

```
image_manifest.csv ──image_id──> detection_results.csv (1 image : 1..n rows)
       │ extraction_job_id               │ detection_run_id
       └──────────────> processing_failures.csv / hourly_summary.csv <──┘
```

**`image_manifest.csv`** — one row per successfully extracted image:
`image_id, extraction_job_id, source_video, store_id, camera_id, video_timestamp_seconds, video_timestamp_hhmmss, frame_number, sampling_strategy, sampling_interval_seconds, image_filename, original_image_path, image_width, image_height, extraction_status`, then `source_video_id, sample_type, random_seed, recording_start_time, recording_timestamp`.

**`detection_results.csv`** — one row per **person found** in an image (`detection_type` = `uniform_shirt`, `not_uniform` or `uncertain`), all with the same `image_id`; one row with `detection_count=0` when nobody was found; one row with `PROCESSING_FAILED` when the image failed:
`image_id, extraction_job_id, detection_run_id, source_video, store_id, camera_id, video_timestamp_seconds, video_timestamp_hhmmss, original_image_path, annotated_image_path, detection_count, prediction, confidence, bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax, model_version, confidence_threshold, processing_status, review_status`, then `detection_index, detection_type, person_confidence, person_count, recording_timestamp, review_reason`.
- `detection_count` = target-shirt detections in the image (the same on each of its rows); `person_count` = people found.
- `confidence` = the classifier's P(uniform) for that person (blank if the person could not be classified); `person_confidence` = the person detector's score.

**`screenshot_summary.csv`** — one row per successfully extracted screenshot (4 for a default job):
`image_id, extraction_job_id, source_video_filename, store_id, camera_id, recording_date, video_timestamp_seconds, video_timestamp_hhmmss, recording_timestamp, screenshot_filename, original_image_path, original_s3_key, extraction_status, number_of_people_detected, number_of_people_classified, number_of_uniform_compliant, number_of_uniform_non_compliant, number_of_uncertain_predictions, number_of_processing_failures, overall_image_status`, then `recording_date_status, detection_run_id, annotated_image_path, confidence_threshold, model_version, review_reason`.
- `overall_image_status` is the image outcome (`UNIFORM_DETECTED`, `NO_UNIFORM_DETECTED`, `REVIEW_REQUIRED`, `PROCESSING_FAILED`) or `NOT_ANALYSED` when no detection run processed the screenshot (people counts then blank).
- "compliant" / "non-compliant" mean the classifier's existing labels `uniform` (P(uniform) ≥ threshold, default 0.70) / `non_uniform` (P(non_uniform) ≥ 0.60) for the **shirt appearance** only; `uncertain` covers everything else. A failed screenshot has blank people counts and `number_of_processing_failures = 1` — it is never counted as non-compliant.

**`uniform_compliance_details.csv`** — one row per person found (no row for a screenshot without people; one `PROCESSING_FAILED` row for a screenshot that failed):
`image_id, extraction_job_id, detection_run_id, store_id, camera_id, recording_date, video_timestamp_hhmmss, screenshot_filename, person_index, person_confidence, uniform_prediction, uniform_confidence_score, confidence_threshold, compliance_status, review_status, annotated_image_path, annotated_s3_key, model_version, processing_status`, then `recording_date_status, video_timestamp_seconds, recording_timestamp, source_video_filename, original_image_path, bbox_*, review_reason`.
- `uniform_prediction`: `UNIFORM_DETECTED` / `NON_UNIFORM_DETECTED` / `UNCERTAIN`; `compliance_status` is the same, except `UNCERTAIN` → `REVIEW_REQUIRED` and failures → `PROCESSING_FAILED`.
- `uniform_confidence_score` = the classifier's P(uniform); `person_confidence` = the person detector's score.

For both: `recording_date` / `recording_timestamp` are filled only when the recording start time was supplied (otherwise blank, `recording_date_status = unknown`; the processing date is never used). `*_s3_key` columns are always blank (S3 is not implemented). A snapshot of both is written to the run folder when a detection run finishes (`detections/<run>/screenshot_summary.csv`, `uniform_compliance_details.csv`); for older runs they are generated on request. Downloads: per run `GET /api/detection-runs/{run}/screenshot_summary.csv` and `/uniform_compliance_details.csv`; combined (latest processed run per job, dashboard filters) `GET /api/frame-results/screenshot_summary.csv` and `/uniform_compliance_details.csv?source_video=&camera_id=&store_id=&date=`.

**`processing_failures.csv`** (export only, built from the stored files and runs):
`stage, extraction_job_id, detection_run_id, image_id, source_video, store_id, camera_id, video_timestamp_seconds, video_timestamp_hhmmss, target_frame, original_image_path, error_category, error_message, retry_status`.
- `stage`: `extraction_job`, `extraction`, `detection_run`, `inference`, `annotation`.
- `retry_status`: `retried_ok` / `retried_failed` when another job of the same video (or another run of the same extraction) handled the same frame, otherwise `not_retried`. Nothing is retried automatically.

**`hourly_summary.csv`** (export only): one row per job and hour that contains footage:
`source_video, extraction_job_id, detection_run_id, store_id, camera_id, hour_basis, hour_start, hour_end, expected_samples, successful_samples, failed_samples, analysed_images, processing_failed_images, uniform_images, no_uniform_images, review_required_images, target_shirt_detections, person_detections, summary_status`.
- `hour_basis` is `recording_time` (real clock hours) only when the recording start time is known, otherwise `video_relative` (`hour_start` like `video 01:00:00`).
- `summary_status`: `complete`, `complete_with_failures`, `extraction_incomplete`, `analysis_incomplete`, `not_analysed`, `in_progress`.

Jobs and runs (state, counts, threshold, model version and checksum, seed, store / camera / recording start) are stored in the existing SQLite database (`extractions`, `frame_detection_runs`); per-image rows live in the CSV files, and the combined exports are generated from them, so the exports always match the stored records. Older jobs' files (`manifest.csv`, older column names, `read_failed` / `inference_failed`) are converted when read.

## Results Dashboard

**Results Dashboard** (`/results`) is the one-page workflow: upload a video → choose the sampling → **START PROCESSING**. This extracts the screenshots, then runs the existing uniform model. Everything shown is counted from the stored records.

**Page contents:**
- **Progress:** video duration, plus progress for both stages, store / camera and the run ID.
- **Filters:** source video, camera, store, recording date (only for jobs with a known recording start), result, processing status.
- **Totals:** videos and jobs (incomplete jobs), images extracted, images analysed, `UNIFORM_DETECTED` / `NO_UNIFORM_DETECTED` / `REVIEW_REQUIRED` / `PROCESSING_FAILED` images, and target-shirt detections (labelled as detection records, not images or people).
- **Screenshot results table:** video, camera, video time, recorded-at (or "unknown"), result, target shirts, people, confidences, processing status. Click a row for the original and annotated screenshots and their metadata (image / job / run IDs, threshold, model version, file paths).
- **Videos and processing jobs**, **Hourly summary**, **Processing errors** (with retry status).
- **CSV exports:** `image_manifest`, `detection_results`, `processing_failures`, `hourly_summary` — all follow the filters.

**Counting rules** (also stated on the page):
- Image counts are **sampled screenshots, not unique people**; detections are not linked across screenshots.
- `NO_UNIFORM_DETECTED` is **not** a confirmed uniform violation.
- Each video uses its latest detection run that processed frames.

**Storage:**
- Everything is on local disk under `storage/`, behind a small storage interface (`backend/core/storage.py`, `LocalStorage`). The paths recorded in the CSVs are what that interface returns.
- **S3 is not implemented.** `SU_S3_BUCKET` in `.env` is only displayed; nothing is uploaded. A future S3 backend would implement the same `put()` / `status()` methods, use a private bucket and the standard AWS credential provider chain (never credentials in code or config), and record the real `s3://` URI. It must only be called working after an actual upload and read-back test.

**Live CCTV:** not configured. `backend/capture/cctv_adapter.py` defines the interface a real integration must implement (`fetch_clip(store_id, camera_id, start, end) -> VideoClip`) and `ingest_clip()`, which sends a supplied clip through the same pipeline as an upload (tested with a local file only). Needed before a real adapter can be written: the recorder / VMS vendor and model and written authorisation; the vendor's API documentation for exporting recorded footage (or RTSP playback); a read-only service account (kept outside the code); network access from this server to the recorder; the channel → store_id / camera_id mapping and the recorder's time zone; and a test window of footage to verify end to end. No endpoint is assumed and no camera retrieval is simulated.

**API:**
- `GET /api/frame-results?source_video=&camera_id=&store_id=&date=&outcome=&processing_status=`
- `GET /api/frame-results/image_manifest.csv?source_video=&camera_id=&store_id=&date=`
- `GET /api/frame-results/detection_results.csv?source_video=&camera_id=&store_id=&date=&outcome=&processing_status=`
- `GET /api/frame-results/processing_failures.csv?source_video=&camera_id=&store_id=`
- `GET /api/frame-results/hourly_summary.csv?source_video=&camera_id=&store_id=&date=`
