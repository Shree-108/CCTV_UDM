"""Labelled multi-view uniform dataset (v1): labelling sheets, video-level splits, garment crops.

    python -m training.udm_dataset sheets      # numbered contact sheets of the harvested people -> label them
    python -m training.udm_dataset build       # labels.csv + splits.yaml -> train / val / test garment crops
    python -m training.udm_dataset summary     # counts per split, class, view and pose

Files in data/udm_v1/ (git-ignored: real people):
  harvest.csv     every harvested person crop (training/harvest_people.py)
  labels.csv      one row per crop: label (uniform | non_uniform | exclude), view (front | back | side_left |
                  side_right | three_quarter), pose (standing | walking | sitting | bending | leaning), notes,
                  labelled_by. `exclude` = the shirt cannot be judged (hidden, too small, too blurred, not a person)
                  - excluded crops are never used.
  splits.yaml     which VIDEO goes to train / val / test. A video is never split: near-identical frames of one
                  person must not appear on both sides. The test videos are frozen once chosen.
  sheets/         contact sheets used for labelling
  sets/<split>/<class>/<crop>.jpg   garment crops (backend/uniform/garment.py), 224 x 224 letterboxed
  sets/manifest.csv                 every built crop with split, class, view, pose, crop source (pose | fallback)

Two classes: uniform (1 = REQUIRED_UNIFORM) and non_uniform (0 = NOT_REQUIRED_UNIFORM). Hard negatives (other
striped / collared / plain black or white shirts) are non_uniform. No augmented or generated image is ever put in
val or test.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.uniform.garment import PoseModel, garment_region, prepare  # noqa: E402

DS = ROOT / "data" / "udm_v1"
CLASSES = ("uniform", "non_uniform")
LABELS = CLASSES + ("exclude",)
VIEWS = ("front", "back", "side_left", "side_right", "three_quarter")
POSES = ("standing", "walking", "sitting", "bending", "leaning")
LABEL_FIELDS = ["crop", "video_id", "label", "view", "pose", "notes", "labelled_by"]


def harvest_rows() -> list[dict]:
    with open(DS / "harvest.csv", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_labels() -> dict[str, dict]:
    p = DS / "labels.csv"
    if not p.exists():
        return {}
    with open(p, newline="", encoding="utf-8") as f:
        return {r["crop"]: r for r in csv.DictReader(f)}


def sheets(per_sheet: int = 40, height: int = 210) -> None:
    """Numbered thumbnails per video (number = row in harvest.csv) for labelling."""
    rows = harvest_rows()
    out = DS / "sheets"
    out.mkdir(exist_ok=True)
    by_video = defaultdict(list)
    for i, r in enumerate(rows):
        by_video[r["video_id"]].append((i, r))
    for vid, items in by_video.items():
        for s in range(0, len(items), per_sheet):
            tiles = []
            for i, r in items[s:s + per_sheet]:
                im = cv2.imread(str(DS / r["crop"]))
                im = cv2.resize(im, (max(1, int(im.shape[1] * height / im.shape[0])), height))
                im = cv2.copyMakeBorder(im, 22, 3, 3, 3, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                cv2.putText(im, str(i), (3, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 2)
                tiles.append(im)
            rows_img, row, width = [], [], 0
            for t in tiles:
                if width + t.shape[1] > 1500 and row:
                    rows_img.append(row); row, width = [], 0
                row.append(t); width += t.shape[1]
            rows_img.append(row)
            W = max(sum(t.shape[1] for t in r) for r in rows_img)
            img = np.vstack([np.hstack(r + [np.full((r[0].shape[0], W - sum(t.shape[1] for t in r), 3), 255, np.uint8)])
                             for r in rows_img])
            cv2.imwrite(str(out / f"{vid}__{s // per_sheet:02d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"sheets for {len(by_video)} videos in {out}")


def build() -> None:
    labels = read_labels()
    splits = yaml.safe_load((DS / "splits.yaml").read_text(encoding="utf-8"))
    split_of = {v: s for s in ("train", "val", "test") for v in splits.get(s, [])}
    dup = [v for v in split_of if sum(v in splits.get(s, []) for s in ("train", "val", "test")) > 1]
    if dup:
        raise SystemExit(f"videos in more than one split: {dup}")
    sets = DS / "sets"
    if sets.exists():
        raise SystemExit(f"{sets} exists - builds are never overwritten; move it away first")
    pose = PoseModel()
    man = []
    for r in harvest_rows():
        lab = labels.get(r["crop"])
        if not lab or lab["label"] not in CLASSES:
            continue
        split = split_of.get(r["video_id"])
        if split is None:
            raise SystemExit(f"video {r['video_id']} has labelled crops but no split in splits.yaml")
        person = cv2.imread(str(DS / r["crop"]))
        x1, y1, x2, y2, src = garment_region(person, pose)
        d = sets / split / lab["label"]
        d.mkdir(parents=True, exist_ok=True)
        name = Path(r["crop"]).name
        cv2.imwrite(str(d / name), prepare(person[y1:y2, x1:x2]), [cv2.IMWRITE_JPEG_QUALITY, 95])
        man.append({"file": f"{split}/{lab['label']}/{name}", "split": split, "label": lab["label"], "video_id": r["video_id"],
                    "view": lab.get("view", ""), "pose": lab.get("pose", ""), "crop_source": src,
                    "person_h": int(r["y2"]) - int(r["y1"]), "aspect": r["aspect"]})
    with open(sets / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(man[0]))
        w.writeheader()
        w.writerows(man)
    summary()


def summary() -> None:
    p = DS / "sets" / "manifest.csv"
    rows = list(csv.DictReader(open(p, encoding="utf-8"))) if p.exists() else []
    if not rows:
        lab = read_labels()
        print("labels:", Counter(r["label"] for r in lab.values()))
        return
    for split in ("train", "val", "test"):
        rs = [r for r in rows if r["split"] == split]
        print(f"{split:5s} n={len(rs):4d}  {dict(Counter(r['label'] for r in rs))}  videos={len({r['video_id'] for r in rs})}"
              f"  pose-crop {sum(r['crop_source'] == 'pose' for r in rs)}/{len(rs)}")
        if split == "test":
            for k in ("view", "pose"):
                print(f"      by {k}: {dict(Counter((r[k] or '?', r['label']) for r in rs))}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("sheets", "build", "summary"))
    a = ap.parse_args()
    {"sheets": sheets, "build": build, "summary": summary}[a.cmd]()


if __name__ == "__main__":
    main()
