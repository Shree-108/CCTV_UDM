"""Harvest PERSON crops (the full detected box) from CCTV videos, for labelling a multi-view uniform dataset.

    python -m training.harvest_people                         # every distinct uploaded video in storage/
    python -m training.harvest_people --videos a.mp4 b.mp4    # specific files
    python -m training.harvest_people --every 2 --out data/udm_v1

The production person detector (models/person/yolo11n.pt, configured confidence) runs on frames sampled every
`--every` seconds (videos longer than 5 minutes: every 3 x that), decoded strictly in order - no seeking (NVR
exports return wrong frames when seeking). Each person box (+5 % margin) is saved WHOLE - the garment crop is
decided later (backend/uniform/classifier.py: garment_box), so the same harvest serves any crop policy.
Near-identical crops of the same video (perceptual hash) are skipped, so a static person is not saved 100 times.

Output (data/ is git-ignored - these are images of real people):
  <out>/crops/<video id>/<video id>__f<frame>__p<n>.jpg   file name prefix = the video = the leakage group
  <out>/harvest.csv   crop, video_id, source_video, frame, seconds, box, frame size, box aspect, det confidence
Nothing is labelled here: labels are added by a person (training/udm_dataset.py).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sqlite3
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.core.config import DB_PATH  # noqa: E402
from backend.core.model_config import INFERENCE, PERSON_MODEL  # noqa: E402
from backend.detection.person import PersonDetector  # noqa: E402

FIELDS = ["crop", "video_id", "source_video", "frame", "seconds", "x1", "y1", "x2", "y2", "frame_w", "frame_h",
          "aspect", "det_conf"]


def dhash(img: np.ndarray) -> int:
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def stored_videos() -> list[tuple[str, Path]]:
    """(name, path) of every distinct uploaded video (by content hash; synthetic test clips skipped)."""
    c = sqlite3.connect(DB_PATH)
    seen, out = set(), []
    for name, path, sha in c.execute("SELECT filename, path, sha256 FROM media WHERE kind='video' ORDER BY created_at"):
        p = Path(path)
        if not p.exists() or "SYNTHETIC" in name.upper():
            continue
        key = sha or hashlib.sha256(p.read_bytes()[: 2 ** 22]).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append((name, p))
    return out


def video_id(name: str, path: Path) -> str:
    stem = "".join(ch if ch.isalnum() else "-" for ch in Path(name).stem).strip("-")[:40]
    return f"{stem}-{hashlib.sha1(str(path).encode()).hexdigest()[:6]}"


def harvest(name: str, path: Path, out: Path, every: float, det: PersonDetector, w: csv.DictWriter) -> int:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    step = max(1, round(fps * (every * 3 if n_frames / fps > 300 else every)))
    vid = video_id(name, path)
    d = out / "crops" / vid
    d.mkdir(parents=True, exist_ok=True)
    hashes: list[int] = []
    n = kept = 0
    while cap.grab():
        if n % step == 0:
            ok, frame = cap.retrieve()
            if ok and frame is not None:
                H, W = frame.shape[:2]
                people = det.detect(frame, float(INFERENCE["person_confidence"]), int(INFERENCE["inference_size"]))
                for i, p in enumerate(people, 1):
                    x1, y1, x2, y2 = p.box
                    if y2 - y1 < 80:
                        continue
                    mx, my = 0.05 * (x2 - x1), 0.05 * (y2 - y1)
                    a, b = max(0, int(x1 - mx)), max(0, int(y1 - my))
                    c_, e = min(W, int(x2 + mx)), min(H, int(y2 + my))
                    crop = frame[b:e, a:c_]
                    h = dhash(crop)
                    if any(bin(h ^ o).count("1") <= 4 for o in hashes[-400:]):
                        continue                                  # near-identical to a crop already kept
                    hashes.append(h)
                    fn = d / f"{vid}__f{n:07d}__p{i}.jpg"
                    cv2.imwrite(str(fn), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    w.writerow({"crop": fn.relative_to(out).as_posix(), "video_id": vid, "source_video": name, "frame": n,
                                "seconds": round(n / fps, 2), "x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                                "frame_w": W, "frame_h": H, "aspect": round((y2 - y1) / max(1, x2 - x1), 3),
                                "det_conf": round(p.det_conf, 3)})
                    kept += 1
        n += 1
    cap.release()
    print(f"  {name}: {n} frames decoded, {kept} person crops kept", flush=True)
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="*", type=Path)
    ap.add_argument("--every", type=float, default=2.0, help="seconds between analysed frames")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "udm_v1")
    a = ap.parse_args()
    if (a.out / "harvest.csv").exists():
        raise SystemExit(f"{a.out} already has a harvest.csv - choose a new --out (nothing is overwritten)")
    a.out.mkdir(parents=True, exist_ok=True)
    det = PersonDetector(PERSON_MODEL, "cpu")
    vids = [(p.name, p) for p in a.videos] if a.videos else stored_videos()
    total = 0
    with open(a.out / "harvest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for name, p in vids:
            total += harvest(name, p, a.out, a.every, det, w)
            f.flush()
    print(f"{total} crops from {len(vids)} videos -> {a.out}")


if __name__ == "__main__":
    main()
