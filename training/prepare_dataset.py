"""Dataset preparation for the uniform classifier.

1) extract : run the production person detector/tracker on CCTV videos or images
             and save UPPER-BODY crops (exactly as production sees them) into
             data/unlabeled/ for manual sorting into data/uniform, data/non_uniform,
             data/uncertain.
       python -m training.prepare_dataset extract --source path/to/cctv.mp4 --every 10

2) build   : group-aware train/val split into data/cls/, plus low-resolution /
             blur / JPEG / lighting augmented copies for the train split.
       python -m training.prepare_dataset build
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.core.config import PERSON_MODEL, TRACKER_CFG  # noqa: E402
from backend.core.settings import Settings  # noqa: E402
from backend.detection.person import select_device  # noqa: E402
from backend.uniform.classifier import upper_body_crop, visibility_issue  # noqa: E402

S = Settings().get()

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".ts", ".mpg"}


def load_cfg() -> dict:
    with open(ROOT / "training" / "dataset.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- extract
def extract(source: Path, out: Path, every: int, min_height: int, conf: float) -> None:
    from backend.detection.person import PersonDetector
    det = PersonDetector(PERSON_MODEL, select_device(S["gpu_acceleration"]), TRACKER_CFG)
    out.mkdir(parents=True, exist_ok=True)
    files = [source] if source.is_file() else sorted(p for p in source.rglob("*") if p.suffix.lower() in IMG_EXT | VID_EXT)
    saved = skipped = 0
    for f in files:
        if f.suffix.lower() in IMG_EXT:
            frame = cv2.imread(str(f))
            if frame is None:
                continue
            people = det.detect(frame, conf, int(S["inference_size"]))
            s, k = _save_crops(frame, people, out, f.stem, 0, min_height)
            saved, skipped = saved + s, skipped + k
            if not people:
                print(f"  no person found in {f.name}")
            continue
        det.reset()
        cap = cv2.VideoCapture(str(f))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % every == 0:
                people = det.track(frame, conf, int(S["inference_size"]))
                s, k = _save_crops(frame, people, out, f.stem, idx, min_height)
                saved, skipped = saved + s, skipped + k
            idx += 1
        cap.release()
        print(f"  {f.name}: done")
    print(f"Saved {saved} crops to {out} ({skipped} too small/occluded -> saved under _uncertain_candidates)")
    print("Now sort them into data/uniform, data/non_uniform and data/uncertain.")


def _save_crops(frame, people, out: Path, clip: str, idx: int, min_height: int) -> tuple[int, int]:
    saved = skipped = 0
    boxes = [p.box for p in people]
    for p in people:
        crop = upper_body_crop(frame, p.box)
        if crop is None:
            continue
        issue = visibility_issue(p.box, [b for b in boxes if b is not p.box], frame.shape, min_height)
        dest = out / "_uncertain_candidates" if issue else out
        dest.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest / f"{clip}__t{p.track_id}__f{idx}.jpg"), crop)
        saved += 1
        skipped += bool(issue)
    return saved, skipped


# --------------------------------------------------------------------------- augment
def degrade(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Simulate distant / cheap CCTV: downscale, blur, noise, JPEG, lighting."""
    h, w = img.shape[:2]
    f = rng.uniform(0.25, 0.6)
    small = cv2.resize(img, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
    out = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    r = rng.random()
    if r < 0.35:  # vertical motion blur (walking)
        k = rng.choice([3, 5, 7])
        kernel = np.zeros((k, k), np.float32); kernel[:, k // 2] = 1.0 / k
        out = cv2.filter2D(out, -1, kernel)
    elif r < 0.6:
        out = cv2.GaussianBlur(out, (3, 3), 0)
    gain, bias = rng.uniform(0.5, 1.3), rng.uniform(-30, 30)
    out = np.clip(out.astype(np.float32) * gain + bias, 0, 255)
    out = np.clip(out + np.random.default_rng(rng.randint(0, 10**6)).normal(0, rng.uniform(0, 8), out.shape), 0, 255)
    ok, buf = cv2.imencode(".jpg", out.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, rng.randint(25, 70)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# --------------------------------------------------------------------------- build
def group_key(p: Path) -> str:
    parts = p.stem.split("__")
    return "__".join(parts[:2]) if len(parts) >= 2 else p.stem


def build(cfg: dict) -> Path:
    rng = random.Random(cfg.get("seed", 42))
    out = ROOT / cfg["output_dir"]
    if out.exists():
        shutil.rmtree(out)
    counts: dict[str, dict[str, int]] = {}
    for cls, src in cfg["sources"].items():
        src_dir = ROOT / src
        files = sorted(p for p in src_dir.rglob("*") if p.suffix.lower() in IMG_EXT) if src_dir.exists() else []
        groups: dict[str, list[Path]] = defaultdict(list)
        for p in files:
            groups[group_key(p)].append(p)
        keys = sorted(groups)
        rng.shuffle(keys)
        n_val = max(1, round(len(keys) * cfg["val_split"])) if len(keys) > 1 else 0
        split = {k: ("val" if i < n_val else "train") for i, k in enumerate(keys)}
        counts[cls] = {"train": 0, "val": 0}
        for k, paths in groups.items():
            for p in paths:
                dest = out / split[k] / cls
                dest.mkdir(parents=True, exist_ok=True)
                img = cv2.imread(str(p))
                if img is None:
                    continue
                name = f"{p.parent.name}_{p.stem}"
                cv2.imwrite(str(dest / f"{name}.jpg"), img)
                counts[cls][split[k]] += 1
                aug = cfg.get("lowres_augment", {})
                if split[k] == "train" and aug.get("enabled"):
                    for j in range(int(aug.get("copies_per_image", 1))):
                        cv2.imwrite(str(dest / f"{name}_lr{j}.jpg"), degrade(img, rng))
                        counts[cls]["train"] += 1
        # every class folder must exist for Ultralytics
        for s in ("train", "val"):
            (out / s / cls).mkdir(parents=True, exist_ok=True)
    print("Dataset built in", out)
    for cls, c in counts.items():
        print(f"  {cls:12s} train={c['train']:5d}  val={c['val']:5d}")
    if any(c["train"] == 0 for c in counts.values()):
        print("WARNING: a class has no training images. Add crops before training.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="save upper-body crops from CCTV footage")
    e.add_argument("--source", required=True, type=Path)
    e.add_argument("--out", default=ROOT / "data" / "unlabeled", type=Path)
    e.add_argument("--every", type=int, default=10, help="use every Nth video frame")
    e.add_argument("--min-height", type=int, default=int(S["min_person_height_px"]))
    e.add_argument("--conf", type=float, default=float(S["person_confidence"]))
    sub.add_parser("build", help="split labelled crops into data/cls")
    a = ap.parse_args()
    if a.cmd == "extract":
        extract(a.source, a.out, a.every, a.min_height, a.conf)
    else:
        build(load_cfg())


if __name__ == "__main__":
    main()
