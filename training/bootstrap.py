"""Bootstrap uniform classifier from a FEW real photos (experimental, not production).

    python -m training.bootstrap              # leave-one-person-out evaluation + final model
    python -m training.bootstrap --skip-cv    # final model only

Use this only while too few real CCTV crops exist for training/train.py. It:
  * takes the real upper-body crops in data/uniform and data/non_uniform
    (file name "<person>__t<track>__f<frame>.jpg"; the <person> part is the group),
  * makes CCTV-style variations of each crop (distance, blur, low light, angle,
    occlusion, JPEG) - variations of the SAME person never cross train/test,
  * adds procedurally generated look-alike shirts as NON-UNIFORM (horizontal,
    coloured, wide and thin stripes, checks, plain black / white / colour),
  * evaluates honestly: for every uniform person, a model trained WITHOUT that
    person (and without one held-out group of every negative source) is tested on
    them - so the scores measure "a person the model has never seen",
  * trains the final model on everything -> models/uniform_best.pt and writes the
    evaluation to models/uniform/eval/report.json (shown in Settings).

The decision rule is the production one: P(uniform) >= uniform_confidence ->
COMPLIANT, P(non_uniform) >= non_uniform_confidence -> NON-COMPLIANT, else UNCERTAIN.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.core.model_config import INFERENCE, MODEL  # noqa: E402
from training.candidate import save_candidate  # noqa: E402
from training.evaluate import decide, metrics  # noqa: E402
from training.prepare_dataset import degrade, group_key, load_cfg  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WORK = ROOT / "runs" / "bootstrap"
SIZE = MODEL["crop_size"] or (112, 160)   # w, h of training crops; inference resizes to the same (model.crop_size)


# --------------------------------------------------------------------------- variations
def vary(img: np.ndarray, rng: random.Random, kind: str = "random") -> np.ndarray:
    """One realistic variation of a real crop. `kind` forces a test scenario."""
    h, w = img.shape[:2]
    out = img.copy()
    if kind in ("random", "angle"):          # framing jitter / slight turn (perspective)
        s = rng.uniform(0.82, 1.0) if kind == "random" else 0.95
        cw, ch = int(w * s), int(h * s)
        x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
        out = out[y0:y0 + ch, x0:x0 + cw]
        h2, w2 = out.shape[:2]
        k = rng.uniform(0.04, 0.12) if kind == "random" else 0.18
        src = np.float32([[0, 0], [w2, 0], [w2, h2], [0, h2]])
        if rng.random() < 0.5:
            dst = np.float32([[w2 * k, h2 * k / 2], [w2, 0], [w2, h2], [w2 * k, h2 - h2 * k / 2]])
        else:
            dst = np.float32([[0, 0], [w2 * (1 - k), h2 * k / 2], [w2 * (1 - k), h2 - h2 * k / 2], [0, h2]])
        out = cv2.warpPerspective(out, cv2.getPerspectiveTransform(src, dst), (w2, h2), borderMode=cv2.BORDER_REPLICATE)
        if rng.random() < 0.5:
            out = cv2.flip(out, 1)
    if kind in ("random", "low_light"):
        g = rng.uniform(0.35, 1.25) if kind == "random" else 0.3
        out = np.clip(out.astype(np.float32) * g + (rng.uniform(-20, 20) if kind == "random" else -10), 0, 255).astype(np.uint8)
        if kind == "low_light":
            out = np.clip(out + np.random.default_rng(rng.randint(0, 10**6)).normal(0, 6, out.shape), 0, 255).astype(np.uint8)
    if kind == "occluded" or (kind == "random" and rng.random() < 0.3):
        hh, ww = out.shape[:2]
        f = 0.35 if kind == "occluded" else rng.uniform(0.1, 0.3)
        color = [rng.randint(20, 200)] * 3
        if rng.random() < 0.5:   # counter / table in front (bottom band)
            out[int(hh * (1 - f)):, :] = color
        else:                    # shelf / person at one side
            x = int(ww * f)
            if rng.random() < 0.5:
                out[:, :x] = color
            else:
                out[:, ww - x:] = color
    if kind == "far":                         # tiny + upscaled: stripes barely resolved
        hh, ww = out.shape[:2]
        small = cv2.resize(out, (max(8, ww // 5), max(8, hh // 5)), interpolation=cv2.INTER_AREA)
        out = cv2.resize(small, (ww, hh), interpolation=cv2.INTER_LINEAR)
    if kind == "cctv" or (kind == "random" and rng.random() < 0.5):
        out = degrade(out, rng)
    return cv2.resize(out, SIZE, interpolation=cv2.INTER_AREA)


SCENARIOS = ("original", "cctv", "far", "angle", "low_light", "occluded")


def scenario(img: np.ndarray, kind: str, seed: int) -> np.ndarray:
    if kind == "original":
        return cv2.resize(img, SIZE, interpolation=cv2.INTER_AREA)
    return vary(img, random.Random(seed), kind)


# --------------------------------------------------------------------------- synthetic negatives
SYNTH = ("horizontal_bw", "coloured_vertical", "wide_vertical_bw", "pinstripe", "check", "plain_dark",
         "plain_light", "plain_colour", "print")


def _shirt(pattern: np.ndarray, rng: random.Random) -> np.ndarray:
    """Put a pattern into a torso-like crop: background at the sides, folds, a placket line."""
    w, h = SIZE
    bg = np.full((h, w, 3), [rng.randint(10, 230) for _ in range(3)], np.uint8)
    mask = np.zeros((h, w), np.uint8)
    m = rng.randint(0, 12)
    cv2.fillPoly(mask, [np.array([[m + 6, 0], [w - m - 6, 0], [w - m, h], [m, h]], np.int32)], 255)
    shade = np.tile(np.linspace(rng.uniform(0.7, 1.0), rng.uniform(0.7, 1.0), w, dtype=np.float32), (h, 1))
    shade *= 1 - 0.15 * np.abs(np.sin(np.linspace(0, rng.uniform(2, 6), h)))[:, None]   # folds
    shirt = np.clip(pattern.astype(np.float32) * shade[..., None], 0, 255).astype(np.uint8)
    out = np.where(mask[..., None] > 0, shirt, bg)
    if rng.random() < 0.6:   # button placket
        cv2.line(out, (w // 2, 0), (w // 2, h), [rng.randint(0, 255)] * 3, 1)
    return out


def synth_negative(kind: str, rng: random.Random) -> np.ndarray:
    w, h = SIZE
    W, H = w * 2, h * 2
    c1, c2 = np.array([rng.randint(0, 50)] * 3), np.array([rng.randint(200, 255)] * 3)
    colour = np.array([rng.randint(0, 255) for _ in range(3)])
    x, y = np.meshgrid(np.arange(W), np.arange(H))
    if kind == "horizontal_bw":
        p = (y // rng.randint(5, 12)) % 2
        img = np.where(p[..., None] == 1, c1, c2)
    elif kind == "coloured_vertical":
        p = (x // rng.randint(5, 12)) % 2
        img = np.where(p[..., None] == 1, colour, c2 if rng.random() < 0.5 else c1)
    elif kind == "wide_vertical_bw":
        p = (x // rng.randint(28, 50)) % 2
        img = np.where(p[..., None] == 1, c1, c2)
    elif kind == "pinstripe":     # mostly one colour with thin faint lines
        base = c2 if rng.random() < 0.5 else colour
        img = np.where(((x % rng.randint(14, 24)) < 2)[..., None], base * 0.7, base)
    elif kind == "check":
        a, b = rng.randint(8, 24), rng.randint(8, 24)
        p = ((x // a) + (y // b)) % 2
        img = np.where(p[..., None] == 1, c1 if rng.random() < 0.5 else colour, c2)
    elif kind == "plain_dark":
        img = np.broadcast_to(c1, (H, W, 3))
    elif kind == "plain_light":
        img = np.broadcast_to(c2, (H, W, 3))
    elif kind == "plain_colour":
        img = np.broadcast_to(colour, (H, W, 3))
    else:                          # "print": random blobs / logo-like shapes
        img = np.full((H, W, 3), colour, np.uint8)
        for _ in range(rng.randint(3, 12)):
            cv2.circle(img, (rng.randint(0, W), rng.randint(0, H)), rng.randint(6, 40),
                       [rng.randint(0, 255) for _ in range(3)], -1)
    img = cv2.resize(np.ascontiguousarray(img).astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA)
    out = _shirt(img, rng)
    return degrade(out, rng) if rng.random() < 0.5 else out


# --------------------------------------------------------------------------- data
def load_real() -> dict[str, dict[str, list[np.ndarray]]]:
    """{class: {group: [crops]}} from data/uniform and data/non_uniform."""
    out: dict[str, dict[str, list[np.ndarray]]] = {}
    for cls in ("uniform", "non_uniform"):
        groups: dict[str, list[np.ndarray]] = defaultdict(list)
        for p in sorted((ROOT / "data" / cls).glob("*")):
            if p.suffix.lower() in IMG_EXT and (im := cv2.imread(str(p))) is not None:
                groups[group_key(p).split("__")[0]].append(im)
        out[cls] = dict(groups)
    return out


def dataset_version(real: dict) -> str:
    h = hashlib.sha256()
    for cls in sorted(real):
        for g in sorted(real[cls]):
            for im in real[cls][g]:
                h.update(cls.encode()); h.update(g.encode()); h.update(im.tobytes())
    n_u, n_n = len(real["uniform"]), len(real["non_uniform"])
    return f"bootstrap-{h.hexdigest()[:8]} ({n_u} uniform people, {n_n} real non-uniform groups, {len(SYNTH)} synthetic patterns)"


def write_split(root: Path, train: dict[str, list[np.ndarray]], val: dict[str, list[np.ndarray]]) -> dict:
    if root.exists():
        shutil.rmtree(root)
    counts = {}
    for split, data in (("train", train), ("val", val)):
        for cls, imgs in data.items():
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i, im in enumerate(imgs):
                cv2.imwrite(str(d / f"{i:05d}.jpg"), im)
            counts[f"{split}/{cls}"] = len(imgs)
    return counts


def build_train(real, held_u: set[str], held_n: set[str], held_synth: set[str], rng: random.Random,
                per_pos: int, per_neg: int, per_synth: int) -> tuple[dict, dict]:
    """Training + validation sets from everything NOT held out. Validation = 15% of the
    variations (checkpoint selection only - the honest score is the held-out test)."""
    items: dict[str, list[np.ndarray]] = {"uniform": [], "non_uniform": []}
    # balance per PERSON / group, not per frame: a person filmed for 100 CCTV frames must not
    # outweigh a person with one photo (the model would learn that one person, not the shirt)
    for cls, budget, held in (("uniform", per_pos, held_u), ("non_uniform", per_neg, held_n)):
        for g, crops in real[cls].items():
            if g in held:
                continue
            order = list(range(len(crops)))
            rng.shuffle(order)
            items[cls] += [vary(crops[order[i % len(order)]], rng) for i in range(max(budget, min(len(crops), 2 * budget)))]
    for k in SYNTH:
        if k not in held_synth:
            items["non_uniform"] += [synth_negative(k, rng) for _ in range(per_synth)]
    train, val = {}, {}
    for cls, imgs in items.items():
        rng.shuffle(imgs)
        n_val = max(1, int(len(imgs) * 0.15))
        val[cls], train[cls] = imgs[:n_val], imgs[n_val:]
    return train, val


def build_test(real, held_u, held_n, held_synth, seed: int) -> list[tuple[np.ndarray, str, str]]:
    """(crop, true label, scenario) for the held-out groups only."""
    out = []

    def spread(crops: list, n: int = 8) -> list:   # evenly spaced frames, not 100 near-duplicates
        return crops if len(crops) <= n else [crops[int(i)] for i in np.linspace(0, len(crops) - 1, n)]

    for g in held_u:
        for i, im in enumerate(spread(real["uniform"][g])):
            out += [(scenario(im, s, seed + i), "uniform", f"uniform/{s}") for s in SCENARIOS]
    for g in held_n:
        for i, im in enumerate(spread(real["non_uniform"][g])):
            out += [(scenario(im, s, seed + i), "non_uniform", f"normal_clothes/{s}") for s in SCENARIOS]
    rng = random.Random(seed)
    for k in held_synth:
        out += [(synth_negative(k, rng), "non_uniform", f"lookalike/{k}") for _ in range(6)]
    return out


# --------------------------------------------------------------------------- train / test
def train(data_dir: Path, name: str, epochs: int, seed: int) -> Path:
    from ultralytics import YOLO
    t = dict(load_cfg()["train"])
    base = t.pop("base_model")
    t.update(epochs=epochs, batch=32, patience=max(8, epochs // 3), seed=seed, deterministic=True, plots=False)
    model = YOLO(base)
    model.train(data=str(data_dir), device="cpu", project=str(WORK / "runs"), name=name, exist_ok=True,
                verbose=False, workers=0, **t)
    return Path(model.trainer.best)


def predict(weights: Path, crops: list[np.ndarray]) -> list[dict[str, float]]:
    from backend.uniform.classifier import TrainedClassifier
    clf = TrainedClassifier(str(weights), "cpu")
    return [p for i in range(0, len(crops), 32) for p in clf.predict(crops[i:i + 32])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--skip-cv", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    real = load_real()
    people = sorted(real["uniform"])
    negs = sorted(real["non_uniform"])
    if len(people) < 2 or not negs:
        raise SystemExit("Need crops of >= 2 different uniform people in data/uniform and >= 1 in data/non_uniform")
    uc, nc = INFERENCE["uniform_confidence"], INFERENCE["non_uniform_confidence"]
    print(f"uniform people: {people}\nreal non-uniform groups: {negs}")

    report = {"model": "trained (bootstrap)", "UNIFORM_CONFIDENCE": uc, "NON_UNIFORM_CONFIDENCE": nc,
              "dataset_version": dataset_version(real), "protocol": "leave-one-uniform-person-out", "sets": {},
              "folds": []}
    if not args.skip_cv:
        y_true, y_pred, scen = [], [], []
        for k, person in enumerate(people):
            rng = random.Random(args.seed + k)
            held_n = {negs[k % len(negs)]}
            held_s = {SYNTH[(2 * k) % len(SYNTH)], SYNTH[(2 * k + 1) % len(SYNTH)]}
            tr, va = build_train(real, {person}, held_n, held_s, rng, 60, 30, 25)
            counts = write_split(WORK / f"fold{k}", tr, va)
            w = train(WORK / f"fold{k}", f"fold{k}", args.epochs, args.seed)
            test = build_test(real, {person}, held_n, held_s, args.seed + 100 * k)
            probs = predict(w, [t[0] for t in test])
            preds = [decide(p, uc, nc) for p in probs]
            y_true += [t[1] for t in test]; y_pred += preds; scen += [t[2] for t in test]
            fm = metrics([t[1] for t in test], preds)
            report["folds"].append({"held_out_uniform_person": person, "held_out_negatives": sorted(held_n | held_s),
                                    "counts": counts | {"test": len(test)}, "metrics": fm,
                                    "cases": [{"scenario": s, "true": t, "pred": p, "p_uniform": round(pr.get("uniform", 0), 3)}
                                              for (_, t, s), p, pr in zip(test, preds, probs)]})
            print(f"fold {k} (held out {person}): acc {fm['accuracy']}  "
                  f"uniform P/R {fm['per_class']['uniform']['precision']}/{fm['per_class']['uniform']['recall']}")
        report["sets"]["held_out_people (all folds)"] = metrics(y_true, y_pred)
        by = defaultdict(lambda: [0, 0])
        for t, p, s in zip(y_true, y_pred, scen):
            by[s][0] += t == p; by[s][1] += 1
        report["scenarios"] = {s: f"{a}/{n} correct" for s, (a, n) in sorted(by.items())}

    # ---- final model: everything
    rng = random.Random(args.seed)
    tr, va = build_train(real, set(), set(), set(), rng, 60, 30, 25)
    report["final_counts"] = write_split(WORK / "final", tr, va)
    best = train(WORK / "final", "final", args.epochs, args.seed)
    report["train_params"] = {"base_model": load_cfg()["train"]["base_model"], "epochs": args.epochs,
                              "imgsz": load_cfg()["train"]["imgsz"], "seed": args.seed}
    # a NEW candidate - the production model and its evaluation report are never overwritten
    folder = save_candidate(best, "uniform-cls-bootstrap", {"trainer": "training/bootstrap.py",
                                                            "dataset_version": report["dataset_version"],
                                                            "train_config": report["train_params"]})
    report["model_path"] = (folder / "uniform_best.pt").relative_to(ROOT).as_posix()
    (folder / "eval").mkdir()
    (folder / "eval" / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Saved candidate {folder.relative_to(ROOT).as_posix()} (production model unchanged). Review eval/report.json, "
          f"then: python -m training.register_model --activate {folder.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
