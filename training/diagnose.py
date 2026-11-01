"""Developer diagnostics: what the uniform classifier saw and decided for every person in a screenshot.

    python -m training.diagnose --job ext-20261026-113607-ec812f      # every saved screenshot of a job
    python -m training.diagnose --images a.png b.jpg                  # any image files
    python -m training.diagnose --job <id> --only uniform,uncertain    # panels only for these labels

Runs the SAME code as the app (PersonDetector.detect + UniformDetector.classify with the configured
thresholds, crop shape and model) - nothing is re-implemented. Writes runs/diagnostics/<timestamp>/:
  diagnostics.csv   one row per person: box, torso crop box, visibility issue, raw classifier output,
                    P(uniform), final label, confidence, thresholds, crop shape, model version
  panels/*.png      per person: the frame with this person highlighted | person crop | torso crop |
                    the exact classifier input
runs/ is git-ignored and not served by the API: these images show real people - keep them local.
"""
from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.core.model_config import DEVICE_TO_GPU, INFERENCE, MODEL, PERSON_MODEL, uniform_model_path, version_label  # noqa: E402
from backend.detection.person import PersonDetector, select_device  # noqa: E402
from backend.uniform.classifier import UniformDetector, upper_body_box, upper_body_crop  # noqa: E402

FIELDS = ["image", "person", "person_box", "person_confidence", "torso_box", "visibility_issue",
          "raw_output", "p_uniform", "label", "confidence", "uniform_threshold", "non_uniform_threshold",
          "crop_size", "model_version", "panel"]
COLOR = {"uniform": (80, 200, 60), "non_uniform": (60, 60, 230), "uncertain": (0, 190, 255)}
H = 260


def _fit(img: np.ndarray | None, h: int = H) -> np.ndarray:
    if img is None or img.size == 0:
        return np.full((h, h // 2, 3), 40, np.uint8)
    return cv2.resize(img, (max(1, round(img.shape[1] * h / img.shape[0])), h), interpolation=cv2.INTER_AREA)


def panel(frame: np.ndarray, p, clf_input: np.ndarray | None, text: str) -> np.ndarray:
    view = frame.copy()
    x1, y1, x2, y2 = map(int, p.box)
    cv2.rectangle(view, (x1, y1), (x2, y2), COLOR.get(p.label, (200, 200, 200)), 3)
    tx1, ty1, tx2, ty2 = upper_body_box(p.box)
    cv2.rectangle(view, (tx1, ty1), (tx2, ty2), (255, 255, 255), 1)
    person = frame[max(0, y1):y2, max(0, x1):x2]
    parts = [_fit(view), _fit(person), _fit(upper_body_crop(frame, p.box)), _fit(clf_input)]
    out = np.hstack([np.pad(x, ((0, 0), (0, 6), (0, 0))) for x in parts])
    bar = np.zeros((26, out.shape[1], 3), np.uint8)
    cv2.putText(bar, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, out])


def diagnose(images: list[Path], out: Path, only: set[str] | None = None) -> list[dict]:
    device = select_device(DEVICE_TO_GPU[INFERENCE["device"]])
    det = PersonDetector(PERSON_MODEL, device)
    uni = UniformDetector(uniform_model_path(), device, INFERENCE.get("untrained_model_mode", "baseline"))
    uc, nc = INFERENCE["uniform_confidence"], INFERENCE["non_uniform_confidence"]
    crop_size = MODEL.get("crop_size")
    version = version_label().split(" · ")[0] if uni.trained else uni.model_version
    (out / "panels").mkdir(parents=True, exist_ok=True)
    rows = []
    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            rows.append({"image": str(img_path), "visibility_issue": "image could not be read"})
            continue
        people = det.detect(frame, INFERENCE["person_confidence"], INFERENCE["inference_size"])
        uni.classify(frame, people, uc, nc, INFERENCE["min_person_height_px"], occluders=[p.box for p in people])
        for i, p in enumerate(people, 1):
            probs = p.extra.get("probs")
            crop = upper_body_crop(frame, p.box) if probs is not None else None
            clf_input = cv2.resize(crop, crop_size, interpolation=cv2.INTER_AREA) if crop is not None and crop_size else crop
            name = f"{img_path.stem}__p{i}.png"
            pu = p.extra.get("uniform_prob")
            text = (f"person {i}: {p.label} {p.conf:.2f} | P(uniform) {'-' if pu is None else f'{pu:.3f}'} | "
                    f"thr {uc}/{nc} | {p.reason or 'judged'} | {version}")
            if only is None or p.label in only:
                cv2.imwrite(str(out / "panels" / name), panel(frame, p, clf_input, text))
            else:
                name = ""
            rows.append({"image": str(img_path), "person": i, "person_box": [round(v) for v in p.box],
                         "person_confidence": round(p.det_conf, 3), "torso_box": list(upper_body_box(p.box)),
                         "visibility_issue": p.reason if probs is None else "",
                         "raw_output": {k: round(v, 4) for k, v in probs.items()} if probs else "",
                         "p_uniform": "" if pu is None else round(pu, 4), "label": p.label,
                         "confidence": round(p.conf, 4), "uniform_threshold": uc, "non_uniform_threshold": nc,
                         "crop_size": list(crop_size) if crop_size else "as-is", "model_version": version,
                         "panel": f"panels/{name}" if name else ""})
    with open(out / "diagnostics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    return rows


def job_images(job_id: str) -> list[Path]:
    from backend.extraction.runner import EXTRACTIONS
    frames = EXTRACTIONS / job_id / "frames"
    if not frames.is_dir():
        raise SystemExit(f"No saved screenshots for job {job_id}")
    return sorted(frames.iterdir())


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--job", nargs="+", help="extraction job id(s)")
    src.add_argument("--images", nargs="+", type=Path)
    ap.add_argument("--only", help="comma-separated labels to draw panels for (default: all)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    images = [p for j in args.job for p in job_images(j)] if args.job else args.images
    out = args.out or ROOT / "runs" / "diagnostics" / datetime.now().strftime("%Y%m%d-%H%M%S")
    rows = diagnose(images, out, set(args.only.split(",")) if args.only else None)
    counts: dict[str, int] = {}
    for r in rows:
        counts[r.get("label", "unreadable")] = counts.get(r.get("label", "unreadable"), 0) + 1
    print(f"{len(images)} images, {len(rows)} people: {counts}\nWritten to {out}")


if __name__ == "__main__":
    main()
