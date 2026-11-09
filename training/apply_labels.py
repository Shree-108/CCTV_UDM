"""Copy reviewed CCTV crops into the training set or into a frozen TEST set (two classes: uniform / non_uniform).

    python -m training.label_review                                        # 1. make the queue (crops + CSV)
    #   2. open runs/label_review/<time>/cctv_label_queue.csv, fill reviewer_label = uniform | non_uniform |
    #      uncertain | exclude  (look at the crop; label the SHIRT, not the person)
    python -m training.apply_labels --queue runs/label_review/<time>/cctv_label_queue.csv --to train
    python -m training.apply_labels --queue <csv> --to test --subset store50_nov       # unseen-footage test set
    python -m training.apply_labels --queue <csv> --to train --dry-run                 # show what would happen

Rules
  * Only rows with reviewer_label uniform / non_uniform are copied; uncertain / exclude / empty are skipped.
  * Files are COPIED - the queue, data/ and every existing file stay as they are; nothing is moved, renamed,
    relabelled or deleted, and an existing file is never overwritten.
  * The copy is named  <session>__all__<crop name>.jpg  where session = the source video. Training splits and
    the leave-one-out evaluation group by that prefix, so one video never ends up in both train and validation.
  * Leakage guard: a session that is already in the TEST set (data/validation/*) cannot be added to training,
    and a session already in training (data/uniform, data/non_uniform) cannot be added to a test set.
  * Hard negatives (other striped shirts, other collared button-down shirts) are simply non_uniform.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLASSES = ("uniform", "non_uniform")
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRAIN_ROOT = ROOT / "data"
TEST_ROOT = ROOT / "data" / "validation"


def session_of(source_video: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "-", Path(source_video).stem).strip("-")[:60] or "unknown"


def sessions_in(folders: list[Path]) -> set[str]:
    out = set()
    for d in folders:
        if d.exists():
            out |= {p.stem.split("__")[0] for p in d.rglob("*") if p.suffix.lower() in IMG_EXT and "__" in p.stem}
    return out


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def apply(queue: Path, to: str, subset: str | None, dry_run: bool = False) -> Counter:
    if to == "test" and not (subset and re.fullmatch(r"[A-Za-z0-9_-]{1,60}", subset)):
        raise SystemExit("--to test needs --subset NAME (letters, digits, '_' or '-'), e.g. store50_nov")
    with open(queue, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or "reviewer_label" not in rows[0]:
        raise SystemExit(f"{queue} has no reviewer_label column (make it with python -m training.label_review)")
    train_sessions = sessions_in([TRAIN_ROOT / c for c in CLASSES])
    test_sessions = sessions_in([TEST_ROOT])
    stats: Counter = Counter()
    for r in rows:
        label = (r.get("reviewer_label") or "").strip().lower()
        if label not in CLASSES:
            stats[f"skipped ({label or 'not labelled'})"] += 1
            continue
        src = (queue.parent / r["crop"]).resolve()
        if not src.is_file():
            stats["missing crop file"] += 1
            continue
        session = session_of(r.get("source_video") or "")
        if to == "train" and session in test_sessions:
            raise SystemExit(f"Video {r.get('source_video')!r} is already in the TEST set (data/validation): adding it to "
                             "training would leak test data. Label it as test, or use another video for training.")
        if to == "test" and session in train_sessions:
            raise SystemExit(f"Video {r.get('source_video')!r} is already in the TRAINING data: a test set must contain "
                             "footage the model was not trained on.")
        dest_dir = (TRAIN_ROOT / label) if to == "train" else (TEST_ROOT / subset / label)
        dest = dest_dir / f"{session}__all__{src.stem}{src.suffix.lower()}"
        if dest.exists():
            stats["already there (same file)" if _sha(dest) == _sha(src) else "NAME CLASH - not overwritten"] += 1
            continue
        shown = dest_dir.relative_to(TRAIN_ROOT.parent).as_posix()      # data/uniform, data/validation/<subset>/...
        stats[f"{'would copy' if dry_run else 'copied'} -> {shown}"] += 1
        if not dry_run:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue", type=Path, required=True, help="cctv_label_queue.csv with reviewer_label filled in")
    ap.add_argument("--to", choices=("train", "test"), required=True)
    ap.add_argument("--subset", help="test set name (data/validation/<subset>/)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    for k, n in sorted(apply(a.queue, a.to, a.subset, a.dry_run).items()):
        print(f"  {n:5d}  {k}")


if __name__ == "__main__":
    main()
