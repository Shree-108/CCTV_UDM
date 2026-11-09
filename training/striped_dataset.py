"""Dataset for the objective "a PERSON wearing ANY black-and-white striped clothing" (striped_v1).

    python -m training.striped_dataset build      # relabel data/udm_v1 + synthetic training data + synthetic probe
    python -m training.striped_dataset summary

Classes: striped (STRIPED_CLOTHING) | non_striped (NON_STRIPED_CLOTHING); UNCERTAIN = not judged (visibility gates)
or excluded from the data. Built from the labelled CCTV people of data/udm_v1 (same garment crops, same VIDEO-level
splits, so the frozen test stays two stores whose people are never in training):

  REAL (data/udm_v1/labels.csv, mapped):
    uniform (black-and-white vertical stripes)          -> striped, pattern "vertical"
    non_uniform                                         -> non_striped (pattern: plain / coloured check / jacket ...)
    except the crops listed in EXCLUDE below            -> excluded (ambiguous black-and-white check, or the garment crop
                                                           caught a NEIGHBOUR's stripes - its label would be wrong)
  SYNTHETIC - TRAIN ONLY (never validation / test):
    the shirts of real TRAIN-video people re-textured with black-and-white stripes: horizontal, vertical, diagonal,
    crossed, irregular, different widths -> striped; the same shirts with COLOURED stripes / checks -> non_striped
    (the shirt area is found by colour around the chest; the real folds and shading are kept)
  PROBE (separate, synthetic): the same re-texturing on TEST-video people (unseen), per pattern - to see whether the
    model reacts to directions that do not exist in the real footage. It is NOT a substitute for real test footage.

Output data/striped_v1/ (git-ignored): sets/{train,val,test}/{striped,non_striped}/, probe/{striped,non_striped}/,
manifest.csv (file, split, label, pattern, view, pose, video, synthetic).
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SRC = ROOT / "data" / "udm_v1"
OUT = ROOT / "data" / "striped_v1"
# reviewed on 2026-11-05: the grey-and-white plaid shirt (black-and-white "crossed stripes"? ambiguous) and crops whose
# garment box caught the neighbouring staff member's striped shirt (top-down POS camera)
EXCLUDE = {
    "20261029-20261029183009-20261029183120-1-969b80__f0000040__p2.jpg",  # neighbour's stripes in the crop
    "20261029-20261029183009-20261029183120-1-969b80__f0000120__p2.jpg",  # neighbour's stripes in the crop
    "20261029-20261029183009-20261029183120-1-969b80__f0000640__p2.jpg",  # neighbour's stripes in the crop
    "20261029-20261029183009-20261029183120-1-969b80__f0001080__p2.jpg",  # neighbour's stripes in the crop
    "20261029-20261029183009-20261029183120-1-969b80__f0001200__p4.jpg",  # neighbour's stripes in the crop
    "20261029-20261029183009-20261029183120-1-969b80__f0001280__p2.jpg",  # neighbour's stripes in the crop
    "20261029-20261029183009-20261029183120-1-969b80__f0000240__p3.jpg",  # grey-and-white plaid: ambiguous
    "20261029-20261029183009-20261029183120-1-969b80__f0001200__p2.jpg",  # grey-and-white plaid: ambiguous
    "20261029-20261029183009-20261029183120-1-969b80__f0001240__p2.jpg",  # grey-and-white plaid: ambiguous
    "20261029-20261029183009-20261029183120-1-969b80__f0001400__p2.jpg",  # grey-and-white plaid: ambiguous
}
B, W = 25, 235                                            # black / white values of generated stripes
COLOURS = [((40, 40, 200), (230, 230, 230)), ((160, 70, 20), (230, 230, 230)), ((40, 140, 40), (200, 230, 240)),
           ((30, 90, 200), (40, 200, 230)), ((120, 40, 120), (220, 220, 220)), ((40, 40, 120), (230, 200, 160))]


def shirt_mask(img: np.ndarray) -> np.ndarray:
    """Pixels of the shirt in a letterboxed garment crop: colour close to the chest's, away from the grey padding."""
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    core = lab[int(h * 0.35):int(h * 0.7), int(w * 0.35):int(w * 0.65)].reshape(-1, 3)
    ref = np.median(core, axis=0)
    dist = np.linalg.norm(lab - ref, axis=2)
    m = (dist < 18).astype(np.uint8)
    body = np.zeros_like(m)                               # the shirt is on the body: a central ellipse, not the shelves
    cv2.ellipse(body, (w // 2, int(h * 0.55)), (int(w * 0.42), int(h * 0.5)), 0, 0, 360, 1, -1)
    m &= body
    pad = (np.abs(img.astype(int) - 114).sum(axis=2) < 6)
    m[pad] = 0
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab_, stats, _ = cv2.connectedComponentsWithStats(m)
    if n > 1:                                              # keep the largest region (the shirt)
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lab_ == k).astype(np.uint8)
    return m


def pattern_image(h: int, w: int, kind: str, rng: random.Random, colours=None) -> np.ndarray:
    period = rng.randint(7, 22)
    duty = rng.uniform(0.35, 0.6)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    def stripes(angle):
        t = np.deg2rad(angle)
        u = xx * np.cos(t) + yy * np.sin(t)
        if kind == "irregular":                          # varying widths and a wavy course
            u = u + rng.uniform(3, 8) * np.sin(yy / rng.uniform(15, 35)) + rng.uniform(2, 5) * np.sin(xx / 11.0)
        return ((u % period) / period) < duty
    if kind == "horizontal":
        on = stripes(90 + rng.uniform(-6, 6))
    elif kind == "vertical":
        on = stripes(rng.uniform(-6, 6))
    elif kind == "diagonal":
        on = stripes(rng.choice([1, -1]) * rng.uniform(30, 60))
    elif kind == "crossed":
        on = stripes(rng.uniform(-5, 5)) ^ stripes(90 + rng.uniform(-5, 5)) if rng.random() < 0.5 else \
            stripes(rng.uniform(-5, 5)) | stripes(90 + rng.uniform(-5, 5))
    else:                                                  # irregular
        on = stripes(rng.uniform(-90, 90))
    dark, light = colours if colours else ((B, B, B), (W, W, W))
    out = np.empty((h, w, 3), np.float32)
    out[on], out[~on] = dark, light
    return out


def retexture(img: np.ndarray, mask: np.ndarray, kind: str, rng: random.Random, colours=None) -> np.ndarray:
    h, w = img.shape[:2]
    pat = pattern_image(h, w, kind, rng, colours)
    lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    shade = cv2.GaussianBlur(lum, (0, 0), 3)
    inside = shade[mask > 0]
    shade = np.clip(shade / max(1.0, np.percentile(inside, 75) if inside.size else 1.0), 0.45, 1.15)[..., None]
    soft = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.2)[..., None]
    out = img.astype(np.float32) * (1 - soft) + pat * shade * soft
    return np.clip(out, 0, 255).astype(np.uint8)


def plain_shirt(img: np.ndarray) -> np.ndarray | None:
    """Mask when the crop shows a large, plain, LIGHT shirt that stands out from the background - re-texturing a
    check, a dark jacket or a shirt the colour of the shelves would put stripes on the background."""
    m = shirt_mask(img)
    content = (np.abs(img.astype(int) - 114).sum(axis=2) >= 6)
    fill = m.sum() / max(1, content.sum())
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    if fill < 0.25 or not m.any():
        return None
    inside, outside = lab[..., 0][m > 0], lab[..., 0][(m == 0) & content]
    texture = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_32F)[m > 0].std()
    if np.median(inside) < 125 or (outside.size and abs(float(np.median(inside)) - float(np.median(outside))) < 20) or texture > 30:
        return None
    return m


def stripes_beside(img: np.ndarray, mask: np.ndarray, rng: random.Random) -> np.ndarray | None:
    """NEGATIVE: a plain-shirted person with black-and-white stripes on something NEXT to them (shelf, box, sign)."""
    h, w = img.shape[:2]
    content = (np.abs(img.astype(int) - 114).sum(axis=2) >= 6).astype(np.uint8)
    free = content & (cv2.dilate(mask, np.ones((15, 15), np.uint8)) == 0)
    cols = np.where(free.sum(axis=0) > h * 0.25)[0]
    if len(cols) < w * 0.12:
        return None
    side = cols[cols < w // 2] if rng.random() < 0.5 and (cols < w // 2).sum() > w * 0.1 else cols[cols >= w // 2]
    if len(side) < w * 0.08:
        return None
    region = np.zeros_like(mask)
    region[:, side.min():side.max() + 1] = 1
    region &= free
    return retexture(img, region, rng.choice(("horizontal", "vertical", "diagonal")), rng)


def build(per_crop: int, seed: int) -> None:
    rng = random.Random(seed)
    if OUT.exists():
        raise SystemExit(f"{OUT} exists - builds are never overwritten; move it away first")
    labels = {Path(r["crop"]).name: r for r in csv.DictReader(open(SRC / "labels.csv", encoding="utf-8"))}
    man_src = list(csv.DictReader(open(SRC / "sets" / "manifest.csv", encoding="utf-8")))
    rows = []
    for r in man_src:
        name = Path(r["file"]).name
        if name in EXCLUDE:
            continue
        label = "striped" if r["label"] == "uniform" else "non_striped"
        d = OUT / "sets" / r["split"] / label
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SRC / "sets" / r["file"], d / name)
        rows.append({"file": f"sets/{r['split']}/{label}/{name}", "split": r["split"], "label": label,
                     "pattern": "vertical" if label == "striped" else "none", "view": r["view"], "pose": r["pose"],
                     "video": r["video_id"], "synthetic": 0, "person_h": r["person_h"]})
    kinds = ("horizontal", "vertical", "diagonal", "crossed", "irregular")
    for split, dest in (("train", "sets/train"),):
        n_pos = n_neg = 0
        for r in [x for x in man_src if x["split"] == split and x["label"] == "non_uniform"]:
            name = Path(r["file"]).name
            if name in EXCLUDE:
                continue
            img = cv2.imread(str(SRC / "sets" / r["file"]))
            m = plain_shirt(img)
            if m is None:
                continue
            for k in range(per_crop):
                kind = kinds[(n_pos + k) % len(kinds)]
                pos = retexture(img, m, kind, rng)
                neg = retexture(img, m, rng.choice(("horizontal", "vertical", "diagonal", "crossed")), rng, rng.choice(COLOURS))
                near = stripes_beside(img, m, rng)
                items = [("striped", pos, kind), ("non_striped", neg, "coloured")] +                         ([("non_striped", near, "bw_object_beside")] if near is not None else [])
                for lab, im, pat in items:
                    d = OUT / dest / lab
                    d.mkdir(parents=True, exist_ok=True)
                    fn = f"synth-{pat}-{k}__{name}"
                    cv2.imwrite(str(d / fn), im, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    rows.append({"file": f"{dest}/{lab}/{fn}", "split": split if dest != "probe" else "probe", "label": lab,
                                 "pattern": pat, "view": r["view"], "pose": r["pose"], "video": r["video_id"], "synthetic": 1,
                                 "person_h": r["person_h"]})
                n_pos += 1
                n_neg += 1
        print(f"{split}: {n_pos} synthetic striped + {n_neg} synthetic coloured-stripe negatives -> {dest}")
    # PROBE: the real, unseen TEST people rotated (lying down / bent over / stripes running in another direction)
    for r in [x for x in rows if x["split"] == "test" and x["synthetic"] == 0]:
        img = cv2.imread(str(OUT / r["file"]))
        for ang in (45, 90, 135):
            m = cv2.getRotationMatrix2D((112, 112), ang, 1.0)
            rot = cv2.warpAffine(img, m, (224, 224), borderValue=(114, 114, 114))
            d = OUT / "probe" / r["label"]
            d.mkdir(parents=True, exist_ok=True)
            fn = f"rot{ang}__{Path(r['file']).name}"
            cv2.imwrite(str(d / fn), rot, [cv2.IMWRITE_JPEG_QUALITY, 95])
            rows.append(r | {"file": f"probe/{r['label']}/{fn}", "split": "probe", "synthetic": 1,
                             "pattern": (f"rotated {ang} deg" if r["label"] == "striped" else "none"), "pose": f"rotated {ang}"})
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary()


def summary() -> None:
    rows = list(csv.DictReader(open(OUT / "manifest.csv", encoding="utf-8")))
    for split in ("train", "val", "test", "probe"):
        rs = [r for r in rows if r["split"] == split]
        print(f"{split:5s} n={len(rs):4d} real={sum(r['synthetic'] == '0' for r in rs):4d}  "
              f"{dict(Counter((r['label'], r['pattern']) for r in rs))}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("build", "summary"))
    ap.add_argument("--per-crop", type=int, default=1, help="synthetic variants per plain-shirt crop")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    build(a.per_crop, a.seed) if a.cmd == "build" else summary()


if __name__ == "__main__":
    main()
