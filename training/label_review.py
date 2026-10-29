"""Human-review lists for the uniform labels. READ-ONLY: never moves, renames, deletes or relabels anything.

    python -m training.label_review            # both outputs, into runs/label_review/<timestamp>/

1. label_review.csv  every labelled crop in data/{uniform,non_uniform,uncertain} with the current model's
   P(uniform) and review flags. A flag means "a person should look", NOT that the label is wrong: the model
   was trained on these same crops (so agreement proves nothing) and it is known to over-predict uniform.
2. cctv_label_queue.csv + crops/  the shirt region of every person the detector found in the analysed CCTV
   screenshots (storage/extractions/*), with the model's prediction, ready to be labelled by a person.
   Real store footage of many different people is what the model lacks - especially NON-uniform shirts.

Reviewer columns (fill in by hand): reviewer_label = uniform | non_uniform | uncertain | exclude, and
reviewer_notes. Only crops with a reviewer_label should ever be copied into a training set, grouped by
source video / person so the same person never ends up in both training and test data.
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.audit_dataset import dhash  # noqa: E402

LABELS = ("uniform", "non_uniform", "uncertain")
REVIEW_FIELDS = ["reviewer_label", "reviewer_notes"]


def load_model(path: Path):
    from ultralytics import YOLO
    m = YOLO(str(path), task="classify")
    uniform_index = next(i for i, n in m.names.items() if n == "uniform")       # by NAME, never by position
    return lambda img: float(m.predict(img, imgsz=224, verbose=False)[0].probs.data[uniform_index])


def label_review(p_uniform, out: Path, near_bits: int = 6) -> int:
    rows = []
    for label in LABELS:
        for p in sorted((ROOT / "data" / label).glob("*")):
            img = cv2.imread(str(p)) if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"} else None
            if img is None:
                continue
            rows.append({"path": p.relative_to(ROOT).as_posix(), "current_label": label,
                         "source_group": p.stem.split("__")[0] if "__" in p.stem else "",
                         "width": img.shape[1], "height": img.shape[0], "_h": dhash(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)),
                         "model_p_uniform": round(p_uniform(img), 4)})
    for r in rows:
        flags = []
        pu = r["model_p_uniform"]
        if r["current_label"] == "uniform" and pu < 0.40:
            flags.append("model strongly disagrees (P(uniform) < 0.40)")
        if r["current_label"] == "non_uniform" and pu >= 0.70:
            flags.append("model calls it uniform (P(uniform) >= 0.70)")
        if min(r["width"], r["height"]) < 48:
            flags.append("very small crop")
        twins = sum(1 for o in rows if o is not r and bin(o["_h"] ^ r["_h"]).count("1") <= near_bits)
        if twins:
            flags.append(f"{twins} near-duplicate(s)")
        r["flags"] = "; ".join(flags)
        r["needs_review"] = "yes" if any(not f.endswith("near-duplicate(s)") for f in flags) else "no"
    fields = ["path", "current_label", "source_group", "width", "height", "model_p_uniform", "needs_review", "flags"] + REVIEW_FIELDS
    with open(out / "label_review.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r["needs_review"] != "yes", r["current_label"], r["path"])))
    return sum(r["needs_review"] == "yes" for r in rows)


def cctv_queue(out: Path) -> int:
    crops_dir = out / "crops"
    crops_dir.mkdir()
    rows, seen = [], set()
    for res in sorted((ROOT / "storage" / "extractions").glob("*/detections/*/detection_results.csv")):
        with open(res, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if not r.get("detection_type") or r.get("bbox_xmin") in ("", None):
                    continue
                src = ROOT / r["original_image_path"]
                key = (src.name, r["bbox_xmin"], r["bbox_ymin"], r["bbox_xmax"], r["bbox_ymax"])
                if key in seen or not src.exists():          # the same screenshot analysed by several runs
                    continue
                seen.add(key)
                img = cv2.imread(str(src))
                if img is None:
                    continue
                x1, y1, x2, y2 = (int(float(r[k])) for k in ("bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"))
                crop = img[max(0, y1):y2, max(0, x1):x2]
                if crop.size == 0:
                    continue
                name = f"{src.stem}__p{r['detection_index']}.jpg"
                cv2.imwrite(str(crops_dir / name), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
                rows.append({"crop": f"crops/{name}", "source_video": r["source_video"], "screenshot": src.name,
                             "video_time": r["video_timestamp_hhmmss"], "person": r["detection_index"],
                             "bbox": f"{x1},{y1},{x2},{y2}", "model_result": r["detection_type"],
                             "model_p_uniform": r["confidence"], "why_not_scored": r.get("person_reason", ""),
                             "model_version": r["model_version"]})
    fields = ["crop", "source_video", "screenshot", "video_time", "person", "bbox", "model_result", "model_p_uniform",
              "why_not_scored", "model_version"] + REVIEW_FIELDS
    with open(out / "cctv_label_queue.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/uniform_best.pt")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else ROOT / "runs" / "label_review" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    flagged = label_review(load_model(ROOT / a.model), out)
    queued = cctv_queue(out)
    print(f"label_review.csv: {flagged} labelled crops flagged for review")
    print(f"cctv_label_queue.csv: {queued} person crops from analysed CCTV screenshots to label")
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
