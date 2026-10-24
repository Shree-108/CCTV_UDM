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
git clone <repo-url> && cd YK_CCTV_U_D_M
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
