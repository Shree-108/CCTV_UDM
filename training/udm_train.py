"""Train the v0.3.0 garment-crop uniform classifiers on data/udm_v1 (YOLO-cls, EfficientNet-B0, MobileNetV3).

    python -m training.udm_train                                   # all three
    python -m training.udm_train --archs mobilenet_v3_large --epochs 20

Data (every architecture gets exactly the same images):
  train  data/udm_v1/sets/train (labelled CCTV garment crops of the train VIDEOS)
         + the earlier labelled crops data/uniform, data/non_uniform (store A - the same store as the train videos)
         + generated look-alike shirts as non_uniform (training/bootstrap.py: horizontal / coloured / wide / thin
           stripes, checks, plain black / white / colour) - TRAIN ONLY, never in validation or test
  val    data/udm_v1/sets/val - real footage only; used for early stopping / checkpoint selection
  test   data/udm_v1/sets/test is NOT read here (frozen; training/udm_evaluate.py)

Output: models/uniform/versions/v0.3.0-<arch>-<time>/uniform_best.pt + model_card.json (backend, crop_mode garment,
input size, dataset version, training configuration). The production model (models/uniform_best.pt) and
config/uniform_config.yaml are never changed; activation is a separate, approved step.
CPU: torch uses all cores; expect minutes per epoch for EfficientNet-B0.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.core.model_config import sha256_of  # noqa: E402
from backend.uniform.garment import prepare  # noqa: E402

DS = ROOT / "data" / "udm_v1"
VERSIONS = ROOT / "models" / "uniform" / "versions"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSES = ("non_uniform", "uniform")


def dataset_version() -> str:
    h = hashlib.sha256()
    for f in ("labels.csv", "splits.yaml", "sets/manifest.csv"):
        h.update((DS / f).read_bytes())
    return f"udm_v1-{h.hexdigest()[:10]}"


def assemble(work: Path, synth_per_kind: int, seed: int) -> dict:
    """train/ and val/ folders for all trainers. Returns the image counts."""
    from training.bootstrap import SYNTH, synth_negative
    counts = {}
    for split in ("train", "val"):
        for cls in CLASSES:
            d = work / split / cls
            d.mkdir(parents=True)
            src = DS / "sets" / split / cls
            for p in sorted(src.glob("*.jpg")) if src.exists() else []:
                shutil.copy2(p, d / p.name)
            counts[f"{split}/{cls}/cctv"] = len(list(d.glob("*")))
    for cls in CLASSES:                                         # earlier labelled crops (store A) - train only
        n = 0
        for p in sorted((ROOT / "data" / cls).glob("*")):
            if p.suffix.lower() in IMG_EXT and (im := cv2.imread(str(p))) is not None:
                cv2.imwrite(str(work / "train" / cls / f"old__{p.stem}.jpg"), prepare(im), [cv2.IMWRITE_JPEG_QUALITY, 95])
                n += 1
        counts[f"train/{cls}/earlier_crops"] = n
    rng = random.Random(seed)
    n = 0
    for kind in SYNTH:                                          # generated look-alikes - train only
        for i in range(synth_per_kind):
            cv2.imwrite(str(work / "train" / "non_uniform" / f"synth__{kind}__{i:03d}.jpg"), prepare(synth_negative(kind, rng)))
            n += 1
    counts["train/non_uniform/generated_lookalikes"] = n
    return counts


def train_yolo(work: Path, epochs: int, seed: int, out: Path, fixed: bool = False) -> Path:
    """fixed: train all `epochs` and keep the LAST weights - for when the validation set is too small / easy to
    choose a checkpoint (every epoch scores 100 %: early stopping would keep epoch 1)."""
    from ultralytics import YOLO
    model = YOLO("yolo11n-cls.pt")
    model.train(data=str(work), device="cpu", imgsz=224, epochs=epochs, batch=32, patience=epochs if fixed else 8, seed=seed, deterministic=True,
                optimizer="AdamW", lr0=5e-4, cos_lr=True, fliplr=0.5, flipud=0.0, degrees=0.0, hsv_h=0.0, hsv_s=0.3, hsv_v=0.4,
                erasing=0.3, scale=0.2, auto_augment=None, plots=False, workers=0, verbose=False,
                project=str(work.parent / "yolo"), name="run", exist_ok=True)
    out.mkdir(parents=True, exist_ok=False)
    dst = out / "uniform_best.pt"
    shutil.copy2(Path(model.trainer.last if fixed else model.trainer.best), dst)
    return dst


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archs", nargs="+", default=["yolo_cls", "mobilenet_v3_large", "efficientnet_b0"])
    ap.add_argument("--epochs", type=int, default=None, help="default: yolo 30, mobilenet 20, efficientnet 15")
    ap.add_argument("--synth-per-kind", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fixed-schedule", action="store_true", help="YOLO: all epochs, keep the last weights (no early stopping)")
    a = ap.parse_args()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    work = ROOT / "runs" / "udm_train" / stamp / "data"
    counts = assemble(work, a.synth_per_kind, a.seed)
    version = dataset_version()
    print(f"dataset {version}: {json.dumps(counts)}", flush=True)
    for arch in a.archs:
        out = VERSIONS / f"v0.3.0-{arch}-{stamp}"
        cfg = {"dataset_version": version, "train_counts": counts, "crop_mode": "garment", "input": "224x224 letterboxed",
               "seed": a.seed, "splits": "data/udm_v1/splits.yaml (video-level; test frozen)", "created": stamp,
               "status": "candidate - not approved for production"}
        print(f"\n=== {arch} -> {out.relative_to(ROOT).as_posix()}", flush=True)
        if arch == "yolo_cls":
            ep = a.epochs or 30
            w = train_yolo(work, ep, a.seed, out, a.fixed_schedule)
            card = cfg | {"backend": "yolo_cls", "arch": "yolo11n-cls (ImageNet-pretrained)", "classes": list(CLASSES),
                          "train_config": {"epochs": ep, "batch": 32, "optimizer": "AdamW", "lr0": 5e-4, "cos_lr": True,
                                           "patience": None if a.fixed_schedule else 8,
                                           "weights": "last epoch (fixed schedule)" if a.fixed_schedule else "best validation epoch",
                                           "imgsz": 224, "augment": "fliplr 0.5, hsv, scale 0.2, erasing 0.3; "
                                           "no rotation / vertical flip / RandAugment (would turn stripes)"},
                          "sha256": sha256_of(w)}
            (out / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
        else:
            from training.train_torchvision import train as tv_train
            ep = a.epochs or (20 if arch.startswith("mobilenet") else 15)
            tv_train(work, arch, ep, a.seed, out, batch=32, lr=3e-4, crop_size=None, input_size=[224, 224],
                     patience=ep + 1 if a.fixed_schedule else 5, keep="last" if a.fixed_schedule else "best",
                     info=cfg | {"crop_size": None, "preprocessing": {"crop_size": None}})
        print(f"saved {out.relative_to(ROOT).as_posix()} (production model unchanged)", flush=True)


if __name__ == "__main__":
    main()
