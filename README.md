# Sightline · Uniform Compliance

This app analyses **uploaded** CCTV images and videos. It reports whether the people inside the selected store area are wearing the approved **black-and-white vertically striped uniform shirt**.

The frontend reuses the Sightline dashboard (Next.js), which was originally built for litter detection. All litter-detection logic has been replaced with a new person- and uniform-detection pipeline (FastAPI + Ultralytics YOLO + ByteTrack).

> **Out of scope for this version:** live CCTV, RTSP, IP cameras, ONVIF, DVR/NVR connections and face or identity recognition. The system checks clothing only and never identifies who a person is.

## Run

```bash
pip install -r requirements.txt
npm install                     # node_modules is already present in this copy

# terminal 1 — API on :8010
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8010
# terminal 2 — dashboard on :3010
npm run dev
```

Open http://localhost:3010.

**Trained model:** the app loads `models/uniform_best.pt`, as configured in `config/uniform_config.yaml`. All paths are relative to the project root. Moving the project to another PC is covered in **[DEPLOYMENT.md](DEPLOYMENT.md)**: required files, model transfer (git / Git LFS), checksum verification and settings.

The ports are 3010/8010, so this app can run next to the original litter app (3000/8000). To change them, set `NEXT_PUBLIC_API_URL` in `.env.local` and `CORS_ORIGINS` in `.env`.

```bash
python -m pytest tests          # 19 tests: logic + end-to-end API with real inference
```

## Workflow

1. **Overview.** Shows the latest result, KPIs, recent analyses, model status and alerts.
2. **Image Analysis: automatic.** Drop an image. Once the upload completes, the uniform check starts on its own, with no Analyze button. The page shows *Analyzing uniform…* and then the result.
   - **Single-camera image, or a saved camera/ROI preset:** the result appears with no click.
   - **Multi-camera DVR image with no saved camera:** click the store-interior tile (or draw a Custom area) and the check starts immediately.
   - **Changing the area:** use *Change camera / ROI* on the result to adjust the camera or ROI and re-analyse.
   - **Video Analysis:** drop a file. The page shows the file name, size, resolution, duration and frame rate. Videos keep the steps below.
3. **Select store interior camera.**
   - The DVR layout (single, 2×2, 1×2, 2×1, 2×3, 3×3, 4×4) is auto-detected from the tile seams.
   - Click the correct tile, or choose **Custom** and drag a rectangle, then **Confirm selection**.
   - Only that tile is sent to the model. The other grids are never analysed.
4. **Store ROI (optional).**
   - **Define**, **Edit** (drag the corners), **Clear** or **Save** a polygon inside the tile.
   - Only people whose feet fall inside it count.
   - *Save ROI* pre-selects the same camera and ROI for the next upload.
5. **Start analysis.**
   - Videos show live progress: the annotated frame, %, frame count, people, uniformed, non-uniform, current status and FPS.
6. **Result.**
   - The annotated video or image, the final status, per-person table, video analytics, alerts with evidence.
   - Downloads: annotated video, annotated image, TXT report, JSON report and CSV.
7. **Alerts / Analytics / History.** All values come from the SQLite database of real analyses. Nothing is sample data.

| Status | Rule |
|---|---|
| 🟢 COMPLIANT | ≥ 1 person inside the store ROI is confidently in the approved uniform |
| 🔴 NON-COMPLIANT | People are inside, their clothing was judged, and nobody is in uniform |
| 🟡 UNCERTAIN — PLEASE VERIFY | People are inside but there isn't enough visual evidence |
| ⚪ NO PERSON DETECTED | Nobody is inside the store ROI |

This is the default `overall_rule: any_uniform` (staff presence). With `all_uniform` in `config/uniform_config.yaml`, the result is NON-COMPLIANT as soon as any judged person is not in uniform. Each person is also listed individually with their own verdict and confidence (for example *Person 2 → NON-COMPLIANT → 91%*).

## Pipeline

```
upload ─► grid crop (selected camera only) ─► YOLO person detection ─► ByteTrack IDs (video)
       ─► store ROI filter (feet point) ─► visibility checks (small / cut-off / occluded / blurred → UNCERTAIN)
       ─► UPPER-BODY crop ─► uniform classifier ─► confidence check (below threshold → UNCERTAIN)
       ─► temporal voting per tracked person ─► presence timeline (occlusion-tolerant)
       ─► compliance decision ─► alerts + evidence frames ─► annotated media + reports
```

- **Only the torso is classified.** Shelves, posters, packaging and background patterns can never count as a uniform.
- **Confirmation needs several frames.** In a video, both UNIFORM and NON-UNIFORM require several votes (*Temporal confirmation frames*). Until then the person is shown as `CHECKING` or `UNIFORM? n/3`, and the store status as **MONITORING**. No verdict is ever based on a single frame.
- **Short disappearances don't break presence.** If a confirmed uniform disappears briefly (occlusion, walking behind a shelf), presence continues for *Occlusion tolerance* seconds.
- **Alerts:**
  - **NO UNIFORM** fires after *No-uniform alert after* seconds with people inside and no confirmed uniform. It is raised once per episode, with an annotated evidence frame. It also fires when the whole video ends as NO UNIFORM.
  - **LOW CONFIDENCE** fires only for a person who stays uncertain across many frames.

```
backend/
  main.py                 FastAPI app (AppState: DB, settings, models, job runner)
  api/routes.py           REST API (media, analyses, alerts, analytics, settings, health)
  detection/person.py     YOLO person detection + ByteTrack
  detection/grid.py       DVR layouts, auto-detection, selected area, polygon ROI
  uniform/classifier.py   upper-body crop, visibility checks, trained classifier / stripe baseline
  tracking/memory.py      per-track vote window with hysteresis, whole-video verdict
  analysis/pipeline.py    image + video analysis, alerts, evidence, H.264 output
  analysis/jobs.py        background video worker (cancellable)
  analysis/compliance.py  status rule    analysis/temporal.py  presence timeline
  analysis/annotate.py    burned-in boxes / labels / ROI / status bar
  reports/writer.py       JSON / CSV / TXT reports
  database/db.py          SQLite (media, analyses, persons, alerts, analytics)
frontend/                 reused Sightline shell + UI kit; features/{overview,analysis,alerts,analytics,settings,help}
training/                 prepare_dataset.py · train.py · evaluate.py · register_model.py · dataset.yaml
config/uniform_config.yaml  model path / version / sha256 / class map / thresholds / image size / device
models/uniform_best.pt    the trained uniform classifier (tracked in git; see DEPLOYMENT.md)
models/person/            yolo11n.pt (tracked; downloaded to this path if missing)
models/uniform/eval/      report.json from training/evaluate.py, shown in Settings
storage/                  uploads, analyses/<id>/ (annotated media, evidence, reports), database
```

The camera layer is modular: `analysis/media.py` produces frames from uploaded files. A live source can be added there later, without touching detection, tracking or compliance.

## ⚠️ When the trained uniform model is missing

Until `models/uniform_best.pt` exists, the UI shows **MODEL FILE MISSING — restore models/uniform_best.pt** everywhere a result appears. An older copy at `models/uniform/best.pt` is still picked up, with a warning to move it. The analysis then uses a **stripe-pattern baseline**: a hand-built check for regular, vertical, black-and-white stripes on the torso.
- **What it rejects:** plain, horizontal-striped, checked and coloured shirts in tests.
- **What it can't do:** it is **not reliable on real, distant CCTV**, and it cannot tell your shirt from a customer's similar striped shirt.
- **Turning it off:** in *Settings → Advanced → When the uniform model is not trained → disabled*, everyone is reported as UNCERTAIN instead.

```bash
python -m training.prepare_dataset extract --source path/to/cctv_footage_or_reference_photos --every 10
# sort data/unlabeled/*.jpg into data/uniform, data/non_uniform, data/uncertain
python -m training.train          # group-aware split, low-res/blur augmentation, YOLO-cls
python -m training.evaluate       # accuracy, precision, recall, F1, confusion matrix, false-uniform rate
python -m training.register_model --version 1.1.0   # record the new weights' SHA-256 + version in the config
```

The reference photos are a starting point only. Collect hundreds of crops per class from **your own cameras**:
- **Uniform:** different people, distances, lighting and angles, plus partial occlusion.
- **Non-uniform, including hard negatives:** plain black and white shirts, checks, horizontal stripes, *other* vertical stripes, prints, T-shirts, jackets and hoodies.

Rotation augmentation is deliberately off, because a rotated vertical stripe becomes a horizontal one. Prioritise a low **false-uniform rate**: a customer mistaken for staff makes the compliance result meaningless.

## Test media

`tests/media/` contains a synthetic 2×2 DVR clip and frame. Camera 1 has one person with a painted vertical-striped shirt. Camera 4 shows the same scene without it. These files are used by the automated tests.
