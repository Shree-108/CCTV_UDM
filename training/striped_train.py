"""Train "person wearing ANY black-and-white striped clothing" classifiers (striped_v1): YOLO-cls, MobileNetV3,
EfficientNet-B0 - same data, same fixed schedule, garment crops.

    python -m training.striped_train                       # all three
    python -m training.striped_train --archs mobilenet_v3_large --epochs 15

Data (classes non_striped / striped):
  train  data/striped_v1/sets/train (real CCTV garment crops of the train videos + re-textured shirts + "black-and-white
         object beside a plain shirt" negatives) + the earlier crops data/uniform (black-and-white vertical stripes ->
         striped) and data/non_uniform (-> non_striped) + generated flat patterns: black-and-white horizontal / wide
         vertical stripes -> striped; coloured stripes, plain colours, prints -> non_striped. Generated = TRAIN ONLY.
  val    data/striped_v1/sets/val (real only)
Augmentation: ANY rotation and flips (the stripe direction does not matter for this objective - this is what turns
the real vertical-stripe footage into horizontal / diagonal stripes and lying / bending people), light, blur,
low resolution, occlusion. Fixed schedule, last weights (the validation videos are too easy to pick a checkpoint).
Output models/uniform/versions/striped-v0.4.0-<arch>-<time>/ - candidates only, never activated here.
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

DS = ROOT / "data" / "striped_v1"
VERSIONS = ROOT / "models" / "uniform" / "versions"
CLASSES = ("non_striped", "striped")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GEN = {"horizontal_bw": "striped", "wide_vertical_bw": "striped", "coloured_vertical": "non_striped",
       "plain_dark": "non_striped", "plain_light": "non_striped", "plain_colour": "non_striped", "print": "non_striped"}


def dataset_version() -> str:
    return "striped_v1-" + hashlib.sha256((DS / "manifest.csv").read_bytes()).hexdigest()[:10]


def assemble(work: Path, per_kind: int, seed: int) -> dict:
    from training.bootstrap import synth_negative
    counts = {}
    for split in ("train", "val"):
        for cls in CLASSES:
            d = work / split / cls
            d.mkdir(parents=True)
            for p in sorted((DS / "sets" / split / cls).glob("*.jpg")):
                shutil.copy2(p, d / p.name)
            counts[f"{split}/{cls}/striped_v1"] = len(list(d.glob("*")))
    for src, cls in (("uniform", "striped"), ("non_uniform", "non_striped")):
        n = 0
        for p in sorted((ROOT / "data" / src).glob("*")):
            if p.suffix.lower() in IMG_EXT and (im := cv2.imread(str(p))) is not None:
                cv2.imwrite(str(work / "train" / cls / f"old__{p.stem}.jpg"), prepare(im), [cv2.IMWRITE_JPEG_QUALITY, 95])
                n += 1
        counts[f"train/{cls}/earlier_crops"] = n
    rng = random.Random(seed)
    for kind, cls in GEN.items():
        for i in range(per_kind):
            cv2.imwrite(str(work / "train" / cls / f"gen__{kind}__{i:03d}.jpg"), prepare(synth_negative(kind, rng)))
        counts[f"train/{cls}/generated_{kind}"] = per_kind
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archs", nargs="+", default=["yolo_cls", "mobilenet_v3_large", "efficientnet_b0"])
    ap.add_argument("--epochs", type=int, default=None, help="default: yolo 25, mobilenet 15, efficientnet 12")
    ap.add_argument("--per-kind", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    work = ROOT / "runs" / "striped_train" / stamp / "data"
    counts = assemble(work, a.per_kind, a.seed)
    version = dataset_version()
    print(f"dataset {version}: {json.dumps(counts)}", flush=True)
    for arch in a.archs:
        out = VERSIONS / f"striped-v0.4.0-{arch}-{stamp}"
        base = {"objective": "person wearing ANY black-and-white striped clothing", "dataset_version": version,
                "train_counts": counts, "crop_mode": "garment", "input": "224x224 letterboxed", "seed": a.seed,
                "splits": "data/udm_v1/splits.yaml (video-level; test frozen)", "created": stamp,
                "class_map_note": "striped -> app label 'uniform', non_striped -> 'non_uniform' (config class_map)",
                "status": "candidate - not approved for production"}
        print(f"\n=== {arch} -> {out.relative_to(ROOT).as_posix()}", flush=True)
        if arch == "yolo_cls":
            from ultralytics import YOLO
            ep = a.epochs or 25
            model = YOLO("yolo11n-cls.pt")
            model.train(data=str(work), device="cpu", imgsz=224, epochs=ep, batch=32, patience=ep, seed=a.seed,
                        deterministic=True, optimizer="AdamW", lr0=5e-4, cos_lr=True, fliplr=0.5, flipud=0.5, degrees=180.0,
                        hsv_h=0.0, hsv_s=0.3, hsv_v=0.4, erasing=0.3, scale=0.2, auto_augment=None, plots=False, workers=0,
                        verbose=False, project=str(work.parent / "yolo"), name="run", exist_ok=True)
            out.mkdir(parents=True, exist_ok=False)
            w = out / "uniform_best.pt"
            shutil.copy2(Path(model.trainer.last), w)
            card = base | {"backend": "yolo_cls", "arch": "yolo11n-cls (ImageNet-pretrained)", "classes": list(CLASSES),
                           "train_config": {"epochs": ep, "batch": 32, "optimizer": "AdamW", "lr0": 5e-4, "cos_lr": True,
                                            "weights": "last epoch (fixed schedule)", "augment": "any rotation, flips, hsv, "
                                            "scale 0.2, erasing 0.3"}, "sha256": sha256_of(w)}
            (out / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
        else:
            from training.train_torchvision import train as tv_train
            ep = a.epochs or (15 if arch.startswith("mobilenet") else 12)
            tv_train(work, arch, ep, a.seed, out, batch=32, lr=3e-4, crop_size=None, input_size=[224, 224],
                     patience=ep + 1, keep="last", classes=list(CLASSES), any_rotation=True,
                     info=base | {"preprocessing": {"crop_size": None}, "weights": "last epoch (fixed schedule)"})
        print(f"saved {out.relative_to(ROOT).as_posix()} (production model unchanged)", flush=True)


if __name__ == "__main__":
    main()
