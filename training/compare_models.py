"""Test the production model and candidate models on labelled images they were NOT trained on - side by side.

    python -m training.compare_models                                         # production on data/validation/*
    python -m training.compare_models --candidate models/candidates/<folder>  # production vs a candidate
    python -m training.compare_models --data data/validation/store50_nov --candidate <folder> --candidate <folder2>

Test data: folders with uniform/ and non_uniform/ sub-folders of upper-body crops (the shirt region, as production
crops it) - normally data/validation/<subset>/ filled by `python -m training.apply_labels --to test`. Videos: cut
the crops of an unseen video with `python -m training.prepare_dataset extract --source video.mp4 --out <folder>`,
sort them into <folder>/uniform and <folder>/non_uniform (or label them via label_review + apply_labels), then
pass --data <folder>.

Every model uses the production decision rule and thresholds (Settings): P(uniform) >= uniform_confidence ->
uniform, P(non_uniform) >= non_uniform_confidence -> non_uniform, otherwise uncertain (review).

Writes runs/model_tests/<time>/ (git-ignored: the images show real people - keep them local):
  report.html   per image: evidence image, actual label, each model's prediction, P(uniform), correct / wrong;
                per model: precision, recall, F1, false-positive and false-negative rate, review rate
  results.csv   the same per image;  report.json  all numbers
Images whose video / session also appears in the training data are flagged: a score on them is not a test on
unseen footage. Nothing is trained, and no model or configuration is changed.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.core.model_config import rel, sha256_of, uniform_model_path  # noqa: E402
from backend.core.settings import Settings  # noqa: E402
from backend.uniform.classifier import TrainedClassifier  # noqa: E402
from training.apply_labels import CLASSES, IMG_EXT, sessions_in  # noqa: E402
from training.candidate import read_card  # noqa: E402
from training.evaluate import decide  # noqa: E402

OUT = ROOT / "runs" / "model_tests"


def load_model(path: Path) -> tuple[str, TrainedClassifier]:
    """(display name, classifier). A candidate folder (models/candidates/<x>/) uses its own crop shape."""
    weights = path / "uniform_best.pt" if path.is_dir() else path
    if not weights.is_file():
        raise SystemExit(f"No weights at {weights}")
    clf = TrainedClassifier(str(weights), "cpu")
    card = read_card(weights.parent)
    if card:
        cs = card.get("preprocessing", {}).get("crop_size")
        clf.crop_size = tuple(cs) if cs else None
    return f"{rel(weights)} ({sha256_of(weights)[:12]})", clf


def test_sets(data: list[Path] | None) -> list[Path]:
    if data:
        return data
    root = ROOT / "data" / "validation"
    found = [d for d in sorted(root.iterdir()) if d.is_dir() and any((d / c).is_dir() for c in CLASSES)] if root.exists() else []
    if not found:
        raise SystemExit("No test data: data/validation/ has no <subset>/uniform|non_uniform folders yet. Label unseen "
                         "footage with training.label_review + training.apply_labels --to test (or pass --data).")
    return found


def binary(rows: list[dict], key: str) -> dict:
    """Positive class = REQUIRED UNIFORM. A 'review' (uncertain) prediction is not a uniform approval."""
    tp = sum(r["actual"] == "uniform" and r[key] == "uniform" for r in rows)
    fp = sum(r["actual"] != "uniform" and r[key] == "uniform" for r in rows)
    fn = sum(r["actual"] == "uniform" and r[key] != "uniform" for r in rows)
    tn = len(rows) - tp - fp - fn
    p = tp / (tp + fp) if tp + fp else None
    rc = tp / (tp + fn) if tp + fn else None
    f1 = 2 * p * rc / (p + rc) if p and rc else 0.0
    rd = lambda v: None if v is None else round(v, 4)  # noqa: E731
    return {"n": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": rd(p), "recall": rd(rc), "f1": rd(f1),
            "false_positive_rate": rd(fp / (fp + tn) if fp + tn else None),
            "false_negative_rate": rd(fn / (fn + tp) if fn + tp else None),
            "accuracy": rd(sum(r[key] == r["actual"] for r in rows) / len(rows) if rows else None),
            "review_rate": rd(sum(r[key] == "uncertain" for r in rows) / len(rows) if rows else None),
            "violations_wrongly_reported": sum(r["actual"] == "uniform" and r[key] == "non_uniform" for r in rows)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, action="append", help="folder with uniform/ and non_uniform/ (repeatable)")
    ap.add_argument("--candidate", type=Path, action="append", default=[], help="candidate folder or .pt (repeatable)")
    ap.add_argument("--no-production", action="store_true", help="test only the candidates")
    a = ap.parse_args()

    st = Settings().get()
    uc, nc = float(st["uniform_confidence"]), float(st["non_uniform_confidence"])
    models = ([("production", *load_model(uniform_model_path()))] if not a.no_production else []) + \
             [(f"candidate{i + 1}", *load_model(p)) for i, p in enumerate(a.candidate)]
    if not models:
        raise SystemExit("No model to test")
    train_sessions = sessions_in([ROOT / "data" / c for c in CLASSES])
    out = OUT / datetime.now().strftime("%Y%m%d-%H%M%S")
    (out / "evidence").mkdir(parents=True)

    rows: list[dict] = []
    for ds in test_sets(a.data):
        for cls in CLASSES:
            for p in sorted((ds / cls).glob("*")) if (ds / cls).is_dir() else []:
                if p.suffix.lower() not in IMG_EXT or (img := cv2.imread(str(p))) is None:
                    continue
                ev = out / "evidence" / f"{len(rows):05d}{p.suffix.lower()}"
                shutil.copy2(p, ev)
                rows.append({"set": ds.name, "file": rel(p), "actual": cls, "evidence": f"evidence/{ev.name}",
                             "seen_in_training": "__" in p.stem and p.stem.split("__")[0] in train_sessions, "_img": img})
    if not rows:
        raise SystemExit("The test folders contain no images")
    for key, _, clf in models:
        imgs = [r["_img"] for r in rows]
        probs = [x for i in range(0, len(imgs), 64) for x in clf.predict(imgs[i:i + 64])]
        for r, pr in zip(rows, probs):
            r[key] = decide(pr, uc, nc)
            r[f"{key}_p_uniform"] = round(pr.get("uniform", 0.0), 4)
            r[f"{key}_correct"] = r[key] == r["actual"]

    unseen = [r for r in rows if not r["seen_in_training"]]
    report = {"created": datetime.now().isoformat(timespec="seconds"), "thresholds": {"uniform": uc, "non_uniform": nc},
              "models": {k: name for k, name, _ in models}, "images": len(rows), "unseen_images": len(unseen),
              "sets": sorted({r["set"] for r in rows}),
              "metrics_all": {k: binary(rows, k) for k, _, _ in models},
              "metrics_unseen_only": {k: binary(unseen, k) for k, _, _ in models},
              "per_set": {s: {k: binary([r for r in rows if r["set"] == s], k) for k, _, _ in models}
                          for s in sorted({r["set"] for r in rows})}}
    (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    fields = ["set", "file", "actual", "seen_in_training", "evidence"] + [
        f for k, _, _ in models for f in (k, f"{k}_p_uniform", f"{k}_correct")]
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (out / "report.html").write_text(_html(report, rows, models), encoding="utf-8")

    for scope in ("metrics_unseen_only", "metrics_all"):
        print(f"\n{scope.replace('_', ' ')}:")
        for k, name, _ in models:
            m = report[scope][k]
            print(f"  {k:11s} n={m['n']:4d}  precision {m['precision']}  recall {m['recall']}  F1 {m['f1']}  "
                  f"FPR {m['false_positive_rate']}  FNR {m['false_negative_rate']}  review {m['review_rate']}   {name}")
    if len(unseen) < len(rows):
        print(f"\nWARNING: {len(rows) - len(unseen)} of {len(rows)} images come from videos that are also in the training "
              "data - only 'unseen only' measures performance on new footage.")
    print(f"\nreport: {rel(out / 'report.html')}")


def _html(report: dict, rows: list[dict], models: list) -> str:
    keys = [k for k, _, _ in models]
    e = html.escape
    head = "".join(f"<th>{e(k)}</th>" for k in keys)
    msum = ""
    for scope in ("metrics_unseen_only", "metrics_all"):
        msum += f"<h2>{e(scope.replace('_', ' '))}</h2><table><tr><th>model</th><th>n</th><th>precision</th><th>recall</th>" \
                "<th>F1</th><th>false-positive rate</th><th>false-negative rate</th><th>review rate</th>" \
                "<th>uniformed staff reported as violation</th></tr>"
        for k, name, _ in models:
            m = report[scope][k]
            msum += (f"<tr><td>{e(k)}<br><small>{e(name)}</small></td><td>{m['n']}</td><td>{m['precision']}</td>"
                     f"<td>{m['recall']}</td><td>{m['f1']}</td><td>{m['false_positive_rate']}</td>"
                     f"<td>{m['false_negative_rate']}</td><td>{m['review_rate']}</td><td>{m['violations_wrongly_reported']}</td></tr>")
        msum += "</table>"
    body = ""
    for r in rows:
        cells = "".join(
            f"<td class='{'ok' if r[f'{k}_correct'] else 'bad'}'>{e(r[k])}<br><small>P(uniform) {r[f'{k}_p_uniform']:.2f}"
            f" · {'correct' if r[f'{k}_correct'] else 'WRONG'}</small></td>" for k in keys)
        body += (f"<tr><td><a href='{e(r['evidence'])}'><img src='{e(r['evidence'])}' loading='lazy'></a></td>"
                 f"<td>{e(r['set'])}<br><small>{e(r['file'])}</small>{'<br><b>seen in training</b>' if r['seen_in_training'] else ''}"
                 f"</td><td><b>{e(r['actual'])}</b></td>{cells}</tr>")
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Uniform model test</title><style>
body{{font:13px system-ui,sans-serif;margin:16px;color:#1b1f24}} table{{border-collapse:collapse;margin-bottom:16px}}
td,th{{border:1px solid #d0d7de;padding:4px 8px;vertical-align:top;text-align:left}} img{{height:110px}}
.ok{{background:#e6f4ea}} .bad{{background:#fde7e9}} small{{color:#57606a}}</style></head><body>
<h1>Uniform classifier test</h1><p>{report['images']} labelled images ({report['unseen_images']} from footage not in the training
data) · thresholds: uniform ≥ {report['thresholds']['uniform']}, non-uniform ≥ {report['thresholds']['non_uniform']}, otherwise
review · positive class = required uniform · created {e(report['created'])}</p>{msum}
<h2>Per image</h2><table><tr><th>evidence</th><th>set / file</th><th>actual</th>{head}</tr>{body}</table></body></html>"""


if __name__ == "__main__":
    main()
