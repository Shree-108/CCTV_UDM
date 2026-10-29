# Yatrikart Uniform Compliance (CCTV)

Checks CCTV video from a store for staff wearing the approved **black-and-white vertically striped uniform
shirt**, and reports the results per screenshot, per video, per month and cumulatively in CSV files.

- **Backend:** Python, FastAPI, Ultralytics YOLO (person detection + uniform classification), SQLite.
- **Frontend:** Next.js dashboard.
- **Runs locally** on one PC. Nothing is uploaded to any outside service.

> **Status: prototype.** The uniform classifier (`models/uniform_best.pt`, v0.2.0) is marked
> **experimental**. It was trained on very few people and approves some non-uniform shirts (for example
> light-blue or plain white) as uniform. Treat its results as indicative until it has been retrained on
> real labelled CCTV data — see [training/DATA_COLLECTION_PLAN.md](training/DATA_COLLECTION_PLAN.md).

## What it does

**Video Analysis** (`/analysis/video`) — two modes:

- **Quick sampled analysis (default).** Upload a video, enter the store and (if known) the recording
  start time, click *Start analysis*. Four random frames are taken, one from each quarter of the video
  (the random seed is saved, so it is reproducible). Every person in each frame is detected and their
  upper-body crop is classified as uniform, non-uniform or review-required. Only those four frames are
  checked, not the whole video.
- **Full-video annotated analysis.** Every analysed frame with a person tracker, an annotated output
  video, alerts and a report. Much slower on long videos.

**Results Dashboard** (`/results`):

- **A** summary cards, **B** the latest completed analysis with its four screenshots,
  **C** that analysis's CSV, **D** monthly reports with a monthly CSV,
  **E** all historical rows with the cumulative master CSV, **F** video history and annotated video.
- Annotated screenshots: green box = uniform, red = non-uniform, amber = review required, with the
  score next to each person.

**Business CSV** — always exactly these six columns, one row per screenshot:

```
Store Name, Date, Time Stamp, img name, confidence score, remark (Example)
```

`Date` is filled only when the recording start time was entered; it is never guessed. Several people in
one screenshot share one row (`Person 1: 92.5%; Person 2: 41.0%`). Every finished video is appended once
to `storage/exports/uniform_compliance_master.csv`.

Also included: image analysis, alerts, scheduled capture from **local** test sources, frame extraction,
and dataset tools for training.

**Not included:** live CCTV / RTSP / NVR connections (only an interface for a future integration), cloud
storage (S3 is not implemented; everything stays on local disk), user login or access control.

## Prerequisites

| | Version | Notes |
|---|---|---|
| Python | 3.10 or newer | developed and tested with 3.14 |
| Node.js | 20 or newer | developed with 24 |
| Git | any recent | |
| GPU | optional | runs on CPU; set `SU_DEVICE=cuda` to use an NVIDIA GPU |
| Disk | ~3 GB | mostly PyTorch and `node_modules` |

`ffmpeg` is installed automatically with the Python packages (`imageio-ffmpeg`).

## Install

```bash
git clone https://github.com/Shree-108/CCTV_UDM.git
cd CCTV_UDM

python -m venv .venv
# Windows
.venv\Scripts\python -m pip install -r requirements.txt
# Linux / macOS
.venv/bin/python -m pip install -r requirements.txt

npm install
```

Use **npm** (the repository has `package-lock.json`). `package.json` also names pnpm in its
`packageManager` field; if Corepack is enabled and complains, run `npm install` with Corepack disabled.

### Model files

Both models are in the repository:

| File | Size | What it is |
|---|---|---|
| `models/person/yolo11n.pt` | 5.5 MB | Ultralytics YOLO11n person detector (COCO) |
| `models/uniform_best.pt` | 3.1 MB | uniform classifier, YOLO11n-cls, classes `non_uniform` / `uniform` |

The uniform model's SHA-256 is recorded in `config/uniform_config.yaml`. To confirm your copy is the
approved file:

```bash
.venv\Scripts\python -m training.register_model --check
```

If `models/uniform_best.pt` is missing, the app still starts but falls back to a simple stripe-pattern
check and says **MODEL FILE MISSING** everywhere a result appears.

## Configure (optional)

Everything works with the defaults. To change them:

- **Backend:** copy `.env.example` to `.env` (ports / CORS, upload limits, storage folder, device).
- **Frontend:** create `.env.local` with `NEXT_PUBLIC_API_URL=http://<server>:8010` if the browser is on
  another machine.
- **Model and thresholds:** `config/uniform_config.yaml` (P(uniform) ≥ 0.70 = uniform,
  P(non-uniform) ≥ 0.60 = non-uniform, otherwise review). The Settings page overrides these per PC.

## Run

Two terminals, both in the project folder:

```bash
# terminal 1 — API on http://localhost:8010
.venv\Scripts\python -m uvicorn backend.main:app --host 127.0.0.1 --port 8010

# terminal 2 — dashboard on http://localhost:3010
npm run dev
```

Open http://localhost:3010/analysis/video to analyse a video, then http://localhost:3010/results.
The backend loads both models at start-up (a few seconds); until then the dashboard shows
*Backend not reachable*.

For a production build: `npm run build && npm run start`.

## Where results are written

All created automatically under `storage/` (never committed):

| Path | Contents |
|---|---|
| `storage/sightline_uniform.db` | SQLite database of uploads, jobs and analyses |
| `storage/uploads/` | uploaded videos and images, unchanged |
| `storage/extractions/<job>/frames/` | the sampled screenshots (original pixels) |
| `storage/extractions/<job>/detections/<run>/` | annotated screenshots, per-person results, the job's CSVs |
| `storage/exports/uniform_compliance_master.csv` | cumulative business CSV (append-only) |
| `storage/analyses/<id>/` | full-video analysis output (annotated video, reports) |

Back up `storage/` to keep the history. Deleting it starts from an empty database.

## Test

```bash
.venv\Scripts\python -m pytest tests -q
npx tsc --noEmit
```

The tests create a temporary storage folder and database, so they never touch your real data. A few
tests load `models/uniform_best.pt`; everything else uses stand-in model outputs. The full suite takes a
few minutes on CPU.

## Dataset and training tools

| Command | What it does |
|---|---|
| `python -m training.audit_dataset --roots <folders>` | read-only audit: counts, unreadable files, duplicates, leakage between splits |
| `python -m training.label_review` | review list for existing labels + person crops from analysed screenshots to label |
| `python -m training.prepare_dataset extract --source <video>` | one upper-body crop per tracked person, for labelling |
| `python -m training.bootstrap` / `training.train` / `training.evaluate` | training and evaluation |
| `python -m training.register_model` | record a new model's version and SHA-256 in the config |

Training data (`data/`) and outputs (`runs/`) are not in the repository: they contain images of real people.

## Troubleshooting

| Problem | Fix |
|---|---|
| Dashboard says *Backend not reachable* | Start the API (terminal 1) and wait for `Application startup complete`; refresh |
| Port 3010 or 8010 already in use | Stop the other process, or change the port and update `CORS_ORIGINS` / `NEXT_PUBLIC_API_URL` |
| *MODEL FILE MISSING* | `models/uniform_best.pt` is not in place; restore it from the repository |
| *WRONG MODEL FILE* | the file's checksum differs from `config/uniform_config.yaml`; restore the approved file |
| *This video could not be opened* | unsupported codec; re-export as MP4 (H.264) |
| Video too short | quick analysis needs at least four distinct frames |
| Slow analysis | the full-video mode is slow on CPU; use the quick mode, or a GPU (`SU_DEVICE=cuda`) |

## Project layout

```
app/                      Next.js routes (pages)
frontend/                 dashboard UI: api client, shell, features/{analysis,results,extraction,...}
backend/main.py           FastAPI app and start-up
backend/api/              REST endpoints (media, analyses, extractions, results, reports, capture)
backend/extraction/       quick sampled pipeline: sampling, extraction, detection, CSVs, master CSV, reports
backend/analysis/         full-video pipeline: tracking, alerts, annotated video
backend/uniform/          upper-body crop, visibility checks, uniform classifier
backend/detection/        person detector, DVR grid layouts
backend/core/             configuration, settings, storage
backend/database/db.py    SQLite schema and queries
config/uniform_config.yaml  model path, version, checksum, class map, thresholds
models/                   model weights
training/                 dataset tools, training, evaluation, data collection plan
tests/                    pytest suite (tests/media: small test clip and images)
```

More detail: [DEPLOYMENT.md](DEPLOYMENT.md) (moving to another PC, settings, CSV schemas) and
[CLAUDE.md](CLAUDE.md) (architecture and development notes).

## Third-party code and licences

- [Ultralytics](https://github.com/ultralytics/ultralytics) (YOLO) is licensed under **AGPL-3.0**; this
  affects how a project using it may be distributed.
- `tests/media/` is built from Ultralytics sample images (with a synthetic striped patch added).
- This repository does not yet include its own licence file.
