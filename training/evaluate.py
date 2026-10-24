"""Evaluate the uniform classifier with the SAME decision rule as production
(UNIFORM_CONFIDENCE / NON_UNIFORM_CONFIDENCE -> uniform / non_uniform / uncertain).

    python -m training.evaluate                       # trained model on data/cls/val
    python -m training.evaluate --baseline            # stripe heuristic, for comparison
    python -m training.evaluate --data some/folder    # any folder of <class>/*.jpg

Also evaluates each scenario subset in data/validation/<subset>/<true_class>/ e.g.
    data/validation/similar_striped_shirt/non_uniform/
    data/validation/partial_uniform/uncertain/ (or uniform/)
    data/validation/low_res_cctv/uniform/
Reports precision / recall / F1 / accuracy / confusion matrix, uniform
false-positive rate, and a threshold sweep that recommends UNIFORM_CONFIDENCE
for the target precision (false positives make compliance unreliable).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.core.config import UNIFORM_EVAL  # noqa: E402
from backend.core.model_config import uniform_model_path  # noqa: E402
from backend.core.settings import Settings  # noqa: E402
from backend.uniform.classifier import CLASSES, StripePatternBaseline, TrainedClassifier  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_folder(folder: Path) -> tuple[list[np.ndarray], list[str], list[str]]:
    imgs, labels, names = [], [], []
    for cls in CLASSES:
        d = folder / cls
        if not d.exists():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in IMG_EXT and (img := cv2.imread(str(p))) is not None:
                imgs.append(img); labels.append(cls); names.append(str(p.relative_to(folder)))
    return imgs, labels, names


def decide(probs: dict, uc: float, nc: float) -> str:
    pu = probs.get("uniform", 0.0)
    pn = probs.get("non_uniform", 0.0)
    if pu >= uc:
        return "uniform"
    if pn >= nc:
        return "non_uniform"
    return "uncertain"


def metrics(y_true: list[str], y_pred: list[str]) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=list(CLASSES), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(CLASSES))
    n_neg = sum(t != "uniform" for t in y_true)
    fp = sum(t != "uniform" and q == "uniform" for t, q in zip(y_true, y_pred))
    return {
        "n": len(y_true),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "per_class": {c: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                          "f1": round(float(f[i]), 4), "support": int(s[i])} for i, c in enumerate(CLASSES)},
        "confusion_matrix": {"labels": list(CLASSES), "rows_true_cols_pred": cm.tolist()},
        "uniform_false_positives": fp,
        "uniform_false_positive_rate": round(fp / n_neg, 4) if n_neg else None,
    }


def threshold_sweep(y_true, probs, target: float) -> dict:
    rows, rec = [], None
    for th in np.round(np.arange(0.40, 0.99, 0.05), 2):
        pred_u = [pr.get("uniform", 0) >= th for pr in probs]
        tp = sum(pu and t == "uniform" for pu, t in zip(pred_u, y_true))
        fp = sum(pu and t != "uniform" for pu, t in zip(pred_u, y_true))
        fn = sum((not pu) and t == "uniform" for pu, t in zip(pred_u, y_true))
        prec = tp / (tp + fp) if tp + fp else 1.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        rows.append({"threshold": float(th), "precision": round(prec, 4), "recall": round(recall, 4), "fp": fp})
        if rec is None and prec >= target and tp > 0:
            rec = float(th)
    return {"target_precision": target, "recommended_UNIFORM_CONFIDENCE": rec, "sweep": rows}


def save_cm_plot(cm: list[list[int]], title: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), CLASSES, rotation=30); ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=10)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i][j], ha="center", va="center")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def evaluate(model_path: str | None = None, data: str | None = None, baseline: bool = False) -> dict:
    import yaml
    with open(ROOT / "training" / "dataset.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    st = Settings().get()
    uc, nc = float(st["uniform_confidence"]), float(st["non_uniform_confidence"])
    model_path = model_path or str(uniform_model_path())
    if baseline or not Path(model_path).exists():
        if not baseline:
            print(f"No model at {model_path}; evaluating the stripe baseline instead.")
        clf, tag = StripePatternBaseline(), "baseline"
    else:
        clf, tag = TrainedClassifier(model_path, "cpu"), "trained"

    # the trained model's report is shown in Settings -> Model evaluation
    out_dir = UNIFORM_EVAL.parent if tag == "trained" else UNIFORM_EVAL.parent.parent / "eval_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)
    sets = {"validation_split": ROOT / (data or f"{cfg['output_dir']}/val")}
    sub_root = ROOT / cfg["evaluation"]["subsets_dir"]
    if sub_root.exists():
        for d in sorted(sub_root.iterdir()):
            if d.is_dir() and any((d / c).exists() for c in CLASSES):
                sets[d.name] = d

    report = {"model": tag, "UNIFORM_CONFIDENCE": uc, "NON_UNIFORM_CONFIDENCE": nc, "sets": {}}
    for name, folder in sets.items():
        imgs, y_true, files = load_folder(folder)
        if not imgs:
            print(f"[{name}] no images in {folder} - skipped")
            continue
        probs = []
        for i in range(0, len(imgs), 64):
            probs += clf.predict(imgs[i:i + 64])
        y_pred = [decide(p, uc, nc) for p in probs]
        m = metrics(y_true, y_pred)
        m["threshold"] = threshold_sweep(y_true, probs, cfg["evaluation"]["target_uniform_precision"])
        m["false_positive_files"] = [f for f, t, q in zip(files, y_true, y_pred) if t != "uniform" and q == "uniform"][:50]
        report["sets"][name] = m
        save_cm_plot(m["confusion_matrix"]["rows_true_cols_pred"], f"{tag}: {name}", out_dir / f"cm_{name}.png")
        _print(name, m)

    with open(out_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"\nReport + confusion matrices saved in {out_dir}")
    return report


def _print(name: str, m: dict) -> None:
    print(f"\n=== {name}  (n={m['n']}) ===")
    print(f"Accuracy: {m['accuracy']:.3f}")
    for c, v in m["per_class"].items():
        print(f"  {c:12s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} (n={v['support']})")
    print("Confusion matrix (rows=true, cols=pred):", *CLASSES)
    for c, row in zip(CLASSES, m["confusion_matrix"]["rows_true_cols_pred"]):
        print(f"  {c:12s} {row}")
    fpr = m["uniform_false_positive_rate"]
    print(f"Uniform FALSE POSITIVES: {m['uniform_false_positives']}"
          + (f" (rate {fpr:.3f})" if fpr is not None else ""))
    rec = m["threshold"]["recommended_UNIFORM_CONFIDENCE"]
    print(f"Recommended UNIFORM_CONFIDENCE for precision >= {m['threshold']['target_precision']}: {rec}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--data", help="folder with <class>/ subfolders (default: data/cls/val)")
    ap.add_argument("--baseline", action="store_true")
    a = ap.parse_args()
    evaluate(a.model, a.data, a.baseline)
