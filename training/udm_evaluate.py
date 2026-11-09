"""Evaluate the production model and v0.3.0 candidates on the FROZEN TEST VIDEOS of data/udm_v1 - per viewpoint.

    python -m training.udm_evaluate                                     # production + every v0.3.0 candidate
    python -m training.udm_evaluate --candidates models/uniform/versions/<folder> ...
    python -m training.udm_evaluate --split val                          # the validation videos instead
    python -m training.udm_evaluate --dataset striped_v1                 # objective "ANY black-and-white stripes"
    python -m training.udm_evaluate --dataset striped_v1 --split probe   # rotated unseen test people (SYNTHETIC)

Every model makes the decision it would make in the application, with the production thresholds (Settings):
  production v0.2.0  the existing pipeline on the person box: torso band (12-55 % of the box), visibility gates
                     (too small, cut off, seated / leaning, occluded, blurred) and its classifier
  candidates         the garment crop (backend/uniform/garment.py: shoulders -> hips from body keypoints), letterboxed
  P(uniform) >= uniform_confidence -> uniform, P(non_uniform) >= non_uniform_confidence -> non_uniform,
  otherwise uncertain (manual review). Positive class = REQUIRED UNIFORM; an uncertain answer is never an approval.

Reported per model: precision, recall, F1, false-positive rate, false-negative rate, review rate, confusion matrix
(actual x uniform / non_uniform / uncertain) - overall and separately per viewpoint, pose, video (store / camera /
lighting) and distance (person height in pixels); plus every false positive and false negative with its image.
Writes runs/udm_eval/<time>/report.html, report.json, results.csv and evaluation.json into each candidate folder.
Nothing is trained and no model or configuration is changed.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.core.model_config import INFERENCE, rel, sha256_of, uniform_model_path  # noqa: E402
from backend.core.settings import Settings  # noqa: E402
from backend.detection.person import Person  # noqa: E402
from backend.uniform.classifier import TorchvisionClassifier, TrainedClassifier, UniformDetector  # noqa: E402

DS = ROOT / "data" / "udm_v1"
VERSIONS = ROOT / "models" / "uniform" / "versions"
OUT = ROOT / "runs" / "udm_eval"
PREDS = ("uniform", "non_uniform", "uncertain")


def decide(probs: dict, uc: float, nc: float) -> str:
    if probs.get("uniform", 0.0) >= uc:
        return "uniform"
    if probs.get("non_uniform", 0.0) >= nc:
        return "non_uniform"
    return "uncertain"


def load_candidate(folder: Path):
    card = json.loads((folder / "model_card.json").read_text(encoding="utf-8"))
    w = str(folder / "uniform_best.pt")
    clf = TorchvisionClassifier(w, "cpu") if card.get("backend") == "torchvision" else TrainedClassifier(w, "cpu")
    clf.crop_size = None                                   # garment crops are already 224 x 224 letterboxed
    return card, clf


def metrics(rows: list[dict], key: str) -> dict:
    tp = sum(r["actual"] == "uniform" and r[key] == "uniform" for r in rows)
    fp = sum(r["actual"] != "uniform" and r[key] == "uniform" for r in rows)
    fn = sum(r["actual"] == "uniform" and r[key] != "uniform" for r in rows)
    tn = len(rows) - tp - fp - fn
    p = tp / (tp + fp) if tp + fp else None
    rc = tp / (tp + fn) if tp + fn else None
    rd = lambda v: None if v is None else round(v, 3)  # noqa: E731
    return {"n": len(rows), "uniform": sum(r["actual"] == "uniform" for r in rows),
            "non_uniform": sum(r["actual"] == "non_uniform" for r in rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": rd(p), "recall": rd(rc), "f1": rd(2 * p * rc / (p + rc)) if p and rc else (0.0 if tp == 0 and (fp or fn) else None),
            "false_positive_rate": rd(fp / (fp + tn) if fp + tn else None),
            "false_negative_rate": rd(fn / (fn + tp) if fn + tp else None),
            "review_rate": rd(sum(r[key] == "uncertain" for r in rows) / len(rows)) if rows else None,
            "confusion": {a: {q: sum(r["actual"] == a and r[key] == q for r in rows) for q in PREDS}
                          for a in ("uniform", "non_uniform")}}


def distance(h: int) -> str:
    return "far (<250 px)" if h < 250 else "middle (250-500 px)" if h < 500 else "near (>=500 px)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", nargs="*", type=Path)
    ap.add_argument("--split", default="test", choices=("test", "val", "probe"))
    ap.add_argument("--dataset", default="udm_v1", choices=("udm_v1", "striped_v1"))
    a = ap.parse_args()
    striped = a.dataset == "striped_v1"
    if a.split == "probe" and not striped:
        raise SystemExit("--split probe exists for --dataset striped_v1 only")
    st = Settings().get()
    uc, nc = float(st["uniform_confidence"]), float(st["non_uniform_confidence"])
    pattern = "striped-v0.4.0-*" if striped else "specific-uniform-v0.3.0-*"
    cands = a.candidates if a.candidates is not None else sorted(p for p in VERSIONS.glob(pattern) if (p / "model_card.json").exists())
    sdir = ROOT / "data" / "striped_v1"
    if striped:          # labels striped / non_striped = app labels uniform / non_uniform (config class_map)
        manifest = [r | {"label": {"striped": "uniform", "non_striped": "non_uniform"}[r["label"]], "video_id": r["video"],
                         "crop_source": "", "_path": sdir / r["file"]}
                    for r in csv.DictReader(open(sdir / "manifest.csv", encoding="utf-8")) if r["split"] == a.split]
    else:
        manifest = [r | {"pattern": "vertical" if r["label"] == "uniform" else "none", "synthetic": "0",
                         "_path": DS / "sets" / r["file"]}
                    for r in csv.DictReader(open(DS / "sets" / "manifest.csv", encoding="utf-8")) if r["split"] == a.split]
    harvest = {Path(r["crop"]).name: r for r in csv.DictReader(open(DS / "harvest.csv", encoding="utf-8"))}
    out = OUT / datetime.now().strftime("%Y%m%d-%H%M%S")
    (out / "img").mkdir(parents=True)

    rows = []
    for m in manifest:
        name = Path(m["file"]).name
        real = m["synthetic"] in ("0", 0)
        h = harvest.get(name) if real else None
        rows.append({"file": name, "actual": m["label"], "view": m["view"] or "?", "pose": m["pose"] or "?",
                     "pattern": m["pattern"], "video": m["video_id"], "distance": distance(int(m["person_h"])),
                     "crop_source": m["crop_source"], "_garment": m["_path"], "_person": DS / h["crop"] if h else None, "_h": h})

    # ---- production v0.2.0: the existing pipeline on the person box
    prod_path = uniform_model_path()
    ud = UniformDetector(prod_path, "cpu", "disabled")
    models = {"production": f"{rel(prod_path)} v0.2.0 (torso band + gates) sha256 {sha256_of(prod_path)[:12]}"}
    if any(r["_h"] is None for r in rows):                # synthetic probe: no person box -> production cannot run
        models.pop("production")
    for r in rows if "production" in models else []:
        img = cv2.imread(str(r["_person"]))
        h = r["_h"]
        x1, y1, x2, y2 = (int(h[k]) for k in ("x1", "y1", "x2", "y2"))
        ox, oy = max(0, int(x1 - 0.05 * (x2 - x1))), max(0, int(y1 - 0.05 * (y2 - y1)))
        p = Person(1, (x1 - ox, y1 - oy, x2 - ox, y2 - oy), 0.9)
        ud.classify(img, [p], uc, nc, int(INFERENCE["min_person_height_px"]), occluders=[p.box])
        r["production"], r["production_p"] = p.label, round(p.extra.get("uniform_prob", -1), 3)
        r["production_why"] = p.reason

    # ---- candidates: garment crops
    cards = {}
    for i, folder in enumerate(cands):
        key = f"cand{i + 1}"
        card, clf = load_candidate(folder)
        cards[key] = card | {"folder": rel(folder)}
        models[key] = f"{rel(folder)} ({card.get('backend')} {card.get('arch')})"
        imgs = [cv2.imread(str(r["_garment"])) for r in rows]
        probs = [x for j in range(0, len(imgs), 32) for x in clf.predict(imgs[j:j + 32])]
        for r, pr in zip(rows, probs):
            r[key], r[f"{key}_p"] = decide(pr, uc, nc), round(pr.get("uniform", 0.0), 3)

    keys = list(models)
    groups = {"overall": {"all": rows}}
    for g in ("pattern", "view", "pose", "video", "distance"):
        d = defaultdict(list)
        for r in rows:
            d[r[g]].append(r)
        groups[g] = dict(sorted(d.items()))
    report = {"created": datetime.now().isoformat(timespec="seconds"), "split": a.split, "thresholds": {"uniform": uc, "non_uniform": nc},
              "dataset": {"name": a.dataset, "labels": "data/udm_v1/labels.csv (assistant visual review)"
                          + (" mapped by training/striped_dataset.py" if striped else ""), "splits": "data/udm_v1/splits.yaml",
                          "synthetic": a.split == "probe"},
              "models": models, "candidate_cards": cards,
              "metrics": {g: {k: {name: metrics(rs, name) for name in keys} for k, rs in d.items()} for g, d in groups.items()}}
    for r in rows:
        for k in keys:
            if r[k] != r["actual"] and (r[k] == "uniform" or r["actual"] == "uniform"):
                shutil.copy2(r["_garment"], out / "img" / r["file"])
                break
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fields = ["file", "actual", "pattern", "view", "pose", "video", "distance", "crop_source"] + [f for k in keys for f in (k, f"{k}_p")] + ["production_why"]
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (out / "report.html").write_text(_html(report, rows, keys), encoding="utf-8")
    for key, card in cards.items():
        (ROOT / card["folder"] / f"evaluation_{a.dataset}_{a.split}.json").write_text(json.dumps(
            {"report": rel(out / "report.json"), "metrics": {g: {k: v[key] for k, v in d.items()} for g, d in report["metrics"].items()}},
            indent=2), encoding="utf-8")

    for g, d in report["metrics"].items():
        print(f"\n== by {g}")
        for k, per in d.items():
            line = " | ".join(f"{m}: P {v['precision']} R {v['recall']} F1 {v['f1']} FPR {v['false_positive_rate']} "
                              f"rev {v['review_rate']}" for m, v in per.items())
            first = next(iter(per.values()))
            print(f"  {k[:30]:30s} (n={first['n']:3d}, pos={first['uniform']:3d})  {line}")
    print("\nmodels:", json.dumps(models, indent=1))
    print(f"report: {rel(out / 'report.html')}")


def _html(report: dict, rows: list[dict], keys: list[str]) -> str:
    e = html.escape
    def table(d: dict) -> str:
        hd = "".join(f"<th>{e(k)}<br>P / R / F1</th><th>FPR / FNR</th><th>review</th>" for k in keys)
        body = ""
        for name, per in d.items():
            first = per[keys[0]]
            body += f"<tr><td>{e(str(name))}</td><td>{first['n']} ({first['uniform']} u / {first['non_uniform']} n)</td>"
            for k in keys:
                m = per[k]
                body += (f"<td>{m['precision']} / {m['recall']} / {m['f1']}</td><td>{m['false_positive_rate']} / "
                         f"{m['false_negative_rate']}</td><td>{m['review_rate']}</td>")
            body += "</tr>"
        return f"<table><tr><th>group</th><th>n</th>{hd}</tr>{body}</table>"
    sections = "".join(f"<h2>By {e(g)}</h2>{table(d)}" for g, d in report["metrics"].items())
    cm = ""
    for k in keys:
        c = report["metrics"]["overall"]["all"][k]["confusion"]
        cm += (f"<h3>{e(k)}: {e(report['models'][k])}</h3><table><tr><th>actual \\ predicted</th>" + "".join(f"<th>{q}</th>" for q in PREDS)
               + "</tr>" + "".join(f"<tr><td>{a}</td>" + "".join(f"<td>{c[a][q]}</td>" for q in PREDS) + "</tr>" for a in c) + "</table>")
    err = ""
    for k in keys:
        bad = [r for r in rows if r[k] != r["actual"] and (r[k] == "uniform" or r["actual"] == "uniform")]
        err += f"<h3>{e(k)}: {sum(r[k] == 'uniform' for r in bad)} false positives, {sum(r['actual'] == 'uniform' for r in bad)} missed uniforms</h3><div>"
        for r in bad:
            err += (f"<figure><img src='img/{e(r['file'])}'><figcaption>{'FALSE POSITIVE' if r[k] == 'uniform' else 'missed: ' + r[k]}"
                    f"<br>{e(r['view'])} · {e(r['pose'])} · P(u) {r[f'{k}_p']}</figcaption></figure>")
        err += "</div>"
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>UDM v0.3.0 evaluation</title><style>
body{{font:13px system-ui,sans-serif;margin:16px;color:#1b1f24}} table{{border-collapse:collapse;margin:6px 0 14px}}
td,th{{border:1px solid #d0d7de;padding:3px 7px;text-align:left}} figure{{display:inline-block;margin:4px;width:140px;font-size:11px}}
figure img{{width:140px;height:140px}}</style></head><body><h1>Uniform classifier evaluation - {e(report['split'])} split</h1>
<p>Dataset {e(report['dataset']['name'])}{' - SYNTHETIC PROBE (rotated unseen test people), not a substitute for real footage' if report['dataset']['synthetic'] else ''}.
Frozen test videos: stores / staff never used in training. Positive class = "uniform" in the app's terms (for striped_v1: black-and-white
striped clothing). Thresholds uniform ≥ {report['thresholds']['uniform']},
non-uniform ≥ {report['thresholds']['non_uniform']}, otherwise review (never an approval). Labels: assistant visual review -
to be verified by a person. Created {e(report['created'])}.</p>
<h2>Models</h2><ul>{''.join(f'<li><b>{e(k)}</b>: {e(v)}</li>' for k, v in report['models'].items())}</ul>
<h2>Confusion matrices</h2>{cm}{sections}<h2>False positives and missed uniforms</h2>{err}</body></html>"""


if __name__ == "__main__":
    main()
