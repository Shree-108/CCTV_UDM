"""Train the uniform classifier (Ultralytics YOLO-cls) - separate from production.

    python -m training.train                 # build split + train + evaluate
    python -m training.train --skip-build    # reuse existing data/cls

The best weights are saved as a NEW candidate (training/candidate.py: models/candidates/<name>-<time>/)
and evaluated there. The production model is never touched; activate a reviewed candidate with
    python -m training.register_model --activate models/candidates/<folder> --version X.Y.Z
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.detection.person import select_device  # noqa: E402
from training.candidate import save_candidate  # noqa: E402
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

    folder = save_candidate(Path(model.trainer.best), "uniform-cls",
                            {"trainer": "training/train.py", "dataset": str(data_dir.relative_to(ROOT)),
                             "train_config": {"base_model": base, **t}})
    print(f"Saved candidate -> {folder} (production model unchanged)")

    from training.evaluate import evaluate
    evaluate(str(folder / "uniform_best.pt"), out_dir=folder / "eval")
    print("Review the evaluation, then: python -m training.register_model --activate", folder.relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
