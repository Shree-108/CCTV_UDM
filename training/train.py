"""Train the uniform classifier (Ultralytics YOLO-cls) - separate from production.

    python -m training.train                 # build split + train + evaluate
    python -m training.train --skip-build    # reuse existing data/cls

The best weights are copied to models/uniform/best.pt, which the
dashboard picks up automatically on next start.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.core.config import UNIFORM_MODEL  # noqa: E402
from backend.detection.person import select_device  # noqa: E402
from training.prepare_dataset import build, load_cfg  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    cfg = load_cfg()
    data_dir = ROOT / cfg["output_dir"]
    if not args.skip_build or not data_dir.exists():
        build(cfg)

    from ultralytics import YOLO
    t = dict(cfg["train"])
    base = t.pop("base_model")
    if args.epochs:
        t["epochs"] = args.epochs
    model = YOLO(base)
    model.train(data=str(data_dir), device=select_device(args.device),
                project=str(ROOT / "runs" / "uniform_cls"), name="train", exist_ok=True, **t)

    best = Path(model.trainer.best)
    dest = UNIFORM_MODEL
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, dest)
    print(f"Saved production model -> {dest}")

    from training.evaluate import evaluate
    evaluate(str(dest))


if __name__ == "__main__":
    main()
