@AGENTS.md

# Yatrikart Uniform Compliance — development notes

Local CCTV analysis app: FastAPI backend (`backend/`, port 8010) + Next.js dashboard (`app/`, `frontend/`,
port 3010). Detects people, classifies each person's upper-body crop as uniform / non-uniform / uncertain,
and reports per screenshot in a fixed six-column business CSV. See README.md for setup.

## Commands

```bash
.venv\Scripts\python -m uvicorn backend.main:app --host 127.0.0.1 --port 8010   # API
npm run dev                                                                     # dashboard
.venv\Scripts\python -m pytest tests -q                                         # tests (temp storage)
npx tsc --noEmit                                                                # frontend type check
```

There is no JS test runner and no ESLint config; frontend behaviour is checked with `tsc` and in the browser.

## Two analysis pipelines (do not mix them up)

1. **Quick sampled analysis** — used by BOTH Video Analysis (default mode) and the Results Dashboard.
   `POST /api/media` → `POST /api/extractions` (`run_detection: true`) → `backend/extraction/runner.py`
   (one worker thread) → `plan.py` (`random_segments`: one random frame per quarter, seeded) →
   `extractor.py` (seeks to each planned frame; never decodes the whole video) → `detect.py`
   (`classify_frame` per screenshot: person detector + per-person classifier) → reports →
   `business_csv.write` (per-job CSV) → `master_csv.append` → run marked `done`.
   Frontend: `ProcessPanel` in `frontend/features/results/ResultsDashboard.tsx`, reused by
   `frontend/features/analysis/VideoAnalysisPage.tsx`.
2. **Full-video analysis** — `POST /api/analyses` → `backend/analysis/pipeline.py`: every analysed frame
   (5 fps by default), ByteTrack, temporal voting, alerts, annotated H.264 video. Video Analysis
   `?mode=full`; also started from the dashboard's "Annotated video" panel. Its results are NOT in the
   business CSVs.

## Key modules

| Module | Role |
|---|---|
| `backend/uniform/classifier.py` | `upper_body_box` (12–55 % height, 10–90 % width — shared with training), visibility checks, `TrainedClassifier` (maps classes BY NAME via `class_map`), decision thresholds |
| `backend/extraction/detect.py` | per-screenshot inference, `detection_results.csv` (one row per person, `person_reason` for uncertain people), annotated image (green / red / amber) |
| `backend/extraction/screenshots.py` | one record per sampled screenshot + the remark text; distinguishes pending / detection failed / not analysed |
| `backend/extraction/business_csv.py` | the six business columns (`BUSINESS_FIELDS`) |
| `backend/extraction/master_csv.py` | append-only master CSV + ledger (dedup key = job id + image id), file lock, crash repair |
| `backend/extraction/reports_service.py` | latest / monthly / historical reports, derived from DB + stored files |
| `backend/database/db.py` | SQLite schema; columns added later are additive migrations (backup before adding) |
| `backend/core/model_config.py`, `config/uniform_config.yaml` | model path, version, SHA-256, class map, thresholds |

## Rules that must hold

- **Business CSV schema is fixed:** `Store Name, Date, Time Stamp, img name, confidence score, remark (Example)`,
  one row per screenshot. Never add, rename or reorder columns. Person-level detail goes in
  `detection_results.csv` / the database.
- **Never invent data:** blank Date when the recording time is unknown; no score when no prediction was made;
  a failed or pending analysis is never reported as non-uniform.
- **Confidence score** in the CSV is the classifier's P(uniform) per person, not the person detector's.
- **The master CSV is append-only.** Do not rewrite earlier rows; reports must stay consistent with it.
- **Do not overwrite `models/uniform_best.pt`** or retrain without being asked. The model is experimental
  (v0.2.0); its known weakness is false "uniform" on light-coloured shirts. See `training/DATA_COLLECTION_PLAN.md`.
- **Training and inference crops must match** (`upper_body_box`); change both together or neither.
- **Privacy:** `storage/`, `data/` and `runs/` contain images of real people and are git-ignored. Never commit them.
- **No fake integrations:** S3 is not implemented and live CCTV is not connected; status badges must say so.

## Tests

`tests/conftest.py` points `SU_STORAGE_DIR` / `SU_DB_PATH` at a temp folder and pins the uniform model to a
missing file; tests that need the real weights use the `real_model` fixture (`tests/test_frame_detection.py`).
Most pipeline tests replace `detect.classify_frame` with deterministic outputs. The app session is shared, so
tests that read the master CSV patch `master_csv.EXPORTS` to their own folder.
