"""Run the EXISTING uniform model on extracted frames.

The project's trained model is two-stage (config/uniform_config.yaml):
  1. YOLO person detector (models/person/yolo11n.pt) finds people;
  2. YOLO-cls uniform classifier (models/uniform_best.pt) judges each person's upper-body
     (shirt) crop as uniform / non_uniform, or UNCERTAIN (too small, cut off, occluded,
     blurred, or below the thresholds).
A "target-shirt detection" here is a person whose shirt crop the classifier scores as uniform
with P(uniform) >= the configured threshold. Its bounding box is that shirt (torso) region
of the person box - it is derived from the person detection, not a separately trained shirt box.

detection_results.csv: one row per PERSON the detector found (detection_type uniform_shirt |
not_uniform | uncertain), all linked to the same image_id; one row with detection_count 0 when
nobody was found; one row with prediction PROCESSING_FAILED (prediction-specific fields blank)
when the image could not be read or inference failed. `detection_count` is the number of
target-shirt detections in the image, `person_count` the number of people found.

Outcome per image (`prediction`, the same on every row of the image)
  UNIFORM_DETECTED     >= 1 target-shirt detection passes the threshold
  NO_UNIFORM_DETECTED  none passes, and every person found could be judged (or nobody was found).
                       This does NOT establish that anyone is non-compliant.
  REVIEW_REQUIRED      none passes while at least one person could not be judged
                       (insufficient visual evidence)
  PROCESSING_FAILED    the image could not be read or inference raised an error - no prediction made

`processing_status` is separate from the outcome: ok | annotation_failed | failed.

Confidence is the classifier's softmax score for "uniform" - not a calibrated probability that
the person is compliant. Detections are counted per image; nothing is linked across frames and
the number of shirts is not a number of unique people.

Originals are never modified; annotated copies go to the run's own folder.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from backend.core.config import ROOT
from backend.core.storage import StorageBackend, get_storage
from backend.extraction.plan import hhmmss
from backend.uniform.classifier import upper_body_box

UNIFORM_DETECTED, NO_UNIFORM_DETECTED, REVIEW_REQUIRED = "UNIFORM_DETECTED", "NO_UNIFORM_DETECTED", "REVIEW_REQUIRED"
PROCESSING_FAILED = "PROCESSING_FAILED"
OUTCOMES = (UNIFORM_DETECTED, NO_UNIFORM_DETECTED, REVIEW_REQUIRED, PROCESSING_FAILED)
RESULT_FIELDS = ["image_id", "extraction_job_id", "detection_run_id", "source_video", "store_id", "camera_id",
                 "video_timestamp_seconds", "video_timestamp_hhmmss", "original_image_path", "annotated_image_path",
                 "detection_count", "prediction", "confidence", "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax",
                 "model_version", "confidence_threshold", "processing_status", "review_status",
                 # supporting fields (after the required ones)
                 "detection_index", "detection_type", "person_confidence", "person_count", "recording_timestamp",
                 "review_reason",
                 # why THIS person is uncertain (too small / far, upper body cut off, occluded by another person,
                 # too blurred, low resolution, low confidence); "" for people who were judged
                 "person_reason"]
# processing_status (image level, separate from the outcome):
#   ok                 inference ran and the annotated copy was saved
#   annotation_failed  inference ran, but the annotated copy could not be saved
#   failed             the original could not be read, or inference raised an error (no prediction is made)
PROCESSING_STATUSES = ("ok", "annotation_failed", "failed")
DETECTION_TYPE = {"uniform": "uniform_shirt", "non_uniform": "not_uniform", "uncertain": "uncertain"}
_LEGACY_FAILED = ("read_failed", "inference_failed")          # written by runs before the schema change
GREEN, AMBER, GREY, RED = (153, 211, 52), (36, 191, 251), (170, 170, 170), (113, 113, 248)


def normalize(row: dict, run: dict, meta: dict | None = None) -> dict:
    """A detection_results.csv row in the current schema - also for files written by older runs.
    `meta` = extractor.row_meta(extraction) supplies store / camera when the file lacks them."""
    if "capture_timestamp" in row and "video_timestamp_hhmmss" not in row:
        hms = row.pop("capture_timestamp")
        h, m, s = hms.split(":")
        row["video_timestamp_hhmmss"] = hms
        row.setdefault("timestamp_seconds", str(round(int(h) * 3600 + int(m) * 60 + float(s), 3)))
    for old, new in (("timestamp_seconds", "video_timestamp_seconds"), ("timestamp_hhmmss", "video_timestamp_hhmmss")):
        if old in row and new not in row:
            row[new] = row.pop(old)
    if row.get("processing_status") in _LEGACY_FAILED:
        row["processing_status"] = "failed"
    if row.get("processing_status") == "failed":
        row["prediction"] = PROCESSING_FAILED
    if "detection_type" not in row:                               # older runs wrote target shirts only
        row["detection_type"] = "uniform_shirt" if row.get("confidence") else ""
    defaults = {"extraction_job_id": run["extraction_id"], "detection_run_id": run["id"],
                "confidence_threshold": run["threshold"], "store_id": (meta or {}).get("store_id", ""),
                "camera_id": (meta or {}).get("camera_id", ""), "detection_index": "", "person_confidence": "",
                "person_count": "", "recording_timestamp": "", "person_reason": ""}
    for k, v in defaults.items():
        if row.get(k) in (None, ""):
            row[k] = v
    return row


def _abs(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def _shirt_box(box, w: int, h: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = upper_body_box(box)
    return max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2)


def _label(img, text: str, x: int, y: int, color, scale: float) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    y0 = max(0, y - th - 8)
    cv2.rectangle(img, (x, y0), (x + tw + 8, y0 + th + 8), color, -1)
    cv2.putText(img, text, (x + 4, y0 + th + 3), cv2.FONT_HERSHEY_SIMPLEX, scale, (15, 12, 10), 1, cv2.LINE_AA)


def annotate(img: np.ndarray, people: list, outcome: str, threshold: float, model_label: str) -> np.ndarray:
    """A COPY of the frame with the model's boxes and scores; the original is not touched."""
    out = img.copy()
    h, w = out.shape[:2]
    scale = max(0.4, min(0.8, w / 1600))
    thick = max(2, w // 500)
    for p in people:
        x1, y1, x2, y2 = (int(v) for v in p.box)
        if p.label == "uniform":
            sx1, sy1, sx2, sy2 = _shirt_box(p.box, w, h)
            cv2.rectangle(out, (sx1, sy1), (sx2, sy2), GREEN, thick)
            _label(out, f"uniform_shirt {p.conf:.2f}", sx1, sy1, GREEN, scale)
        elif p.label == "uncertain":
            cv2.rectangle(out, (x1, y1), (x2, y2), AMBER, 1)
            _label(out, f"uncertain ({p.reason or 'low confidence'})", x1, y1, AMBER, scale * 0.85)
        elif p.label == "non_uniform":         # confidently NOT the uniform shirt (P(non_uniform) >= its threshold)
            cv2.rectangle(out, (x1, y1), (x2, y2), RED, thick)
            _label(out, f"non_uniform {p.conf:.2f}", x1, y1, RED, scale * 0.85)
        else:
            cv2.rectangle(out, (x1, y1), (x2, y2), GREY, 1)
            _label(out, f"no target shirt ({p.extra.get('uniform_prob', 0):.2f})", x1, y1, GREY, scale * 0.85)
    color = {UNIFORM_DETECTED: GREEN, NO_UNIFORM_DETECTED: GREY, REVIEW_REQUIRED: AMBER}[outcome]
    n = {k: sum(p.label == k for p in people) for k in ("uniform", "non_uniform", "uncertain")}
    # OpenCV's Hershey fonts draw ASCII only (no "·" or em dash)
    text = {UNIFORM_DETECTED: f"APPROVED UNIFORM DETECTED ({n['uniform']})",
            NO_UNIFORM_DETECTED: "NO APPROVED UNIFORM DETECTED", REVIEW_REQUIRED: "REVIEW REQUIRED"}[outcome]
    model_ascii = model_label.encode("ascii", "replace").decode().replace("?", "-")
    banner = (f"{text}  |  {n['uniform']} uniform, {n['non_uniform']} non-uniform, {n['uncertain']} review"
              f"  |  threshold {threshold:.2f}  |  {model_ascii}")
    cv2.rectangle(out, (0, 0), (w, int(28 * max(scale, 0.5) + 8)), (20, 20, 20), -1)
    cv2.putText(out, banner, (8, int(22 * max(scale, 0.5))), cv2.FONT_HERSHEY_SIMPLEX, max(scale, 0.45), color, 1, cv2.LINE_AA)
    return out


def classify_frame(img: np.ndarray, models, cfg: dict, threshold: float) -> list:
    """Existing pipeline pieces, whole frame: person detection -> shirt crop classification."""
    with models.image_lock:
        people = models.image_detector.detect(img, cfg["person_confidence"], cfg["inference_size"])
        models.uniform.classify(img, people, threshold, cfg["non_uniform_confidence"], cfg["min_person_height_px"],
                                occluders=[p.box for p in people])
    return people


def outcome_for(people: list) -> tuple[str, str]:
    dets = [p for p in people if p.label == "uniform"]
    if dets:
        return UNIFORM_DETECTED, ""
    unclear = [p for p in people if p.label == "uncertain"]
    if unclear:
        reasons = sorted({p.reason or "low confidence" for p in unclear})
        return REVIEW_REQUIRED, f"{len(unclear)} person(s) could not be judged: {'; '.join(reasons)}"
    return NO_UNIFORM_DETECTED, ""


def _person_order(p) -> tuple:
    """Target shirts first (highest score first), then the other people left to right."""
    return (p.label != "uniform", -p.conf if p.label == "uniform" else 0, p.box[0])


def run(frames: list[dict], out_dir: Path, models, cfg: dict, threshold: float, model_version: str,
        on_progress: Callable[[dict], None] | None = None, should_stop: Callable[[], bool] = lambda: False,
        run_id: str = "", storage: StorageBackend | None = None) -> dict:
    """`frames` = image manifest rows of an extraction (current schema). Writes detection_results.csv
    row by row."""
    storage = storage or get_storage()
    ann_dir = out_dir / "annotated"
    ann_dir.mkdir(parents=True, exist_ok=False)                  # a run folder is never reused
    model_label = model_version.split(" · ")[0]
    stats = {"processed": 0, "succeeded": 0, "failed": 0, "uniform_images": 0, "no_uniform_images": 0,
             "review_images": 0, "detections": 0, "people": 0, "stopped": False}
    with open(out_dir / "detection_results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        w.writeheader(); f.flush()
        for fr in frames:
            if should_stop():
                stats["stopped"] = True
                break
            base = {"image_id": fr["image_id"], "extraction_job_id": fr["extraction_job_id"], "detection_run_id": run_id,
                    "source_video": fr["source_video"], "store_id": fr.get("store_id", ""),
                    "camera_id": fr.get("camera_id", ""), "video_timestamp_seconds": fr["video_timestamp_seconds"],
                    "video_timestamp_hhmmss": fr["video_timestamp_hhmmss"] or hhmmss(float(fr["video_timestamp_seconds"])),
                    "original_image_path": fr["original_image_path"], "annotated_image_path": "",
                    "detection_count": "", "prediction": PROCESSING_FAILED, "confidence": "", "bbox_xmin": "",
                    "bbox_ymin": "", "bbox_xmax": "", "bbox_ymax": "", "model_version": model_version,
                    "confidence_threshold": threshold, "processing_status": "failed", "review_status": "unreviewed",
                    "detection_index": "", "detection_type": "", "person_confidence": "", "person_count": "",
                    "recording_timestamp": fr.get("recording_timestamp", ""), "review_reason": "", "person_reason": ""}
            img = cv2.imread(str(_abs(fr["original_image_path"])))
            if img is None:
                rows = [base | {"review_reason": "original image could not be read"}]
                stats["failed"] += 1
            else:
                try:
                    people = classify_frame(img, models, cfg, threshold)
                except Exception as e:  # noqa: BLE001 - one bad image must not stop the run
                    rows = [base | {"review_reason": f"inference error: {e}"}]
                    stats["failed"] += 1
                else:
                    stats["succeeded"] += 1
                    outcome, reason = outcome_for(people)
                    stats[{UNIFORM_DETECTED: "uniform_images", NO_UNIFORM_DETECTED: "no_uniform_images",
                           REVIEW_REQUIRED: "review_images"}[outcome]] += 1
                    ann_path = ann_dir / f"{Path(fr['image_filename']).stem}_annotated.jpg"
                    try:
                        ok = cv2.imwrite(str(ann_path), annotate(img, people, outcome, threshold, model_label),
                                         [cv2.IMWRITE_JPEG_QUALITY, 90])
                    except cv2.error:
                        ok = False
                    n_dets = sum(p.label == "uniform" for p in people)
                    common = base | {"annotated_image_path": storage.put(ann_path, "annotated") if ok else "",
                                     "detection_count": n_dets, "prediction": outcome, "person_count": len(people),
                                     "processing_status": "ok" if ok else "annotation_failed",
                                     "review_reason": reason if ok else "; ".join(
                                         x for x in (reason, "annotated copy could not be saved") if x)}
                    if people:
                        h, wdt = img.shape[:2]
                        rows = []
                        for i, p in enumerate(sorted(people, key=_person_order), 1):
                            x1, y1, x2, y2 = _shirt_box(p.box, wdt, h)
                            pu = p.extra.get("uniform_prob")
                            dc = getattr(p, "det_conf", None)
                            rows.append(common | {
                                "detection_index": i, "detection_type": DETECTION_TYPE.get(p.label, p.label),
                                "person_reason": (p.reason or "low confidence") if p.label == "uncertain" else "",
                                "confidence": round(pu, 4) if pu is not None else "",
                                "person_confidence": round(dc, 4) if dc is not None else "",
                                "bbox_xmin": x1, "bbox_ymin": y1, "bbox_xmax": x2, "bbox_ymax": y2})
                        stats["detections"] += n_dets
                        stats["people"] += len(people)
                    else:
                        rows = [common]
            w.writerows(rows); f.flush()
            stats["processed"] += 1
            if on_progress:
                on_progress(stats)
    return stats
