"""Fine-tune an EfficientNet-B0 / MobileNetV3 uniform classifier (Phase B candidate backend).

    python -m training.train_torchvision --data data/cls --arch efficientnet_b0 --out models/uniform/versions/v0.3.0-efficientnet_b0

`--data` is a split folder like the YOLO-cls one: <data>/train/<class>/*.jpg and <data>/val/<class>/*.jpg (class
folders uniform / non_uniform). Crops are resized to model.crop_size (exactly as inference does), then to the
network input. ImageNet-pretrained weights are downloaded once by torchvision.

The checkpoint with the best VALIDATION accuracy (ties: lower validation loss) is written to <out>/uniform_best.pt
in the torch_backbones format, with a model card. <out> must not exist: a candidate is never overwritten, and the
production model (models/uniform_best.pt) is never touched. Using a candidate: set model.backend: torchvision and
model.path in config/uniform_config.yaml - only after its evaluation has been reviewed and approved.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.core.model_config import MODEL, sha256_of  # noqa: E402
from backend.uniform import torch_backbones as tb  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSES = ["non_uniform", "uniform"]          # output order; mapped by name via class_map


def _load(folder: Path, size=None, classes=None) -> tuple[list[np.ndarray], list[int]]:
    """`size` = (w, h) every image is resized to first (the torso-band models' crop_size), None = as stored
    (garment crops are already 224 x 224 letterboxed)."""
    xs, ys = [], []
    for ci, cls in enumerate(classes or CLASSES):
        for p in sorted((folder / cls).glob("*")):
            if p.suffix.lower() in IMG_EXT and (im := cv2.imread(str(p))) is not None:
                xs.append(cv2.resize(im, size, interpolation=cv2.INTER_AREA) if size else im)
                ys.append(ci)
    return xs, ys


def _augment(im: np.ndarray, rng: random.Random, any_rotation: bool = False) -> np.ndarray:
    """Realistic CCTV variation - training only. Never a large rotation or a vertical flip: turned vertical stripes
    become horizontal stripes, which are a NEGATIVE. Every change keeps the garment recognisable."""
    h, w = im.shape[:2]
    if rng.random() < 0.5:                                    # left / right view of the same shirt
        im = cv2.flip(im, 1)
    if any_rotation:                                          # objective "ANY black-and-white stripes": the direction
        if rng.random() < 0.5:                                # does not matter -> any angle (lying, bending, horizontal)
            im = cv2.flip(im, 0)
        m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(0, 360), rng.uniform(0.9, 1.1))
        im = cv2.warpAffine(im, m, (w, h), borderValue=(114, 114, 114))
    elif rng.random() < 0.5:                                  # small tilt (camera angle / lean): <= 8 degrees
        m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-8, 8), rng.uniform(0.9, 1.1))
        im = cv2.warpAffine(im, m, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.4:                                    # perspective (top-down / side cameras)
        k = rng.uniform(0.03, 0.10)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[w * rng.uniform(0, k), h * rng.uniform(0, k)], [w * (1 - rng.uniform(0, k)), h * rng.uniform(0, k)],
                          [w * (1 - rng.uniform(0, k)), h * (1 - rng.uniform(0, k))], [w * rng.uniform(0, k), h * (1 - rng.uniform(0, k))]])
        im = cv2.warpPerspective(im, cv2.getPerspectiveTransform(src, dst), (w, h), borderMode=cv2.BORDER_REPLICATE)
    if rng.random() < 0.4:                                    # crop that keeps >= 80 % of the garment
        s = rng.uniform(0.8, 1.0)
        cw, ch = int(w * s), int(h * s)
        x0, y0 = rng.randint(0, w - cw), rng.randint(0, h - ch)
        im = cv2.resize(im[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_LINEAR)
    f = im.astype(np.float32)                                 # exposure / brightness / contrast
    f = (f - f.mean()) * rng.uniform(0.7, 1.3) + f.mean() * rng.uniform(0.6, 1.3) + rng.uniform(-20, 20)
    im = np.clip(f, 0, 255).astype(np.uint8)
    if rng.random() < 0.25:                                   # mild motion blur (walking / turning)
        k = rng.choice([3, 5])
        ker = np.zeros((k, k), np.float32)
        if rng.random() < 0.5:
            ker[k // 2, :] = 1.0 / k
        else:
            ker[:, k // 2] = 1.0 / k
        im = cv2.filter2D(im, -1, ker)
    if rng.random() < 0.35:                                   # distant / low-resolution CCTV + compression
        s = rng.uniform(0.3, 0.7)
        im = cv2.resize(cv2.resize(im, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA), (w, h))
        ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, rng.randint(30, 75)])
        im = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if rng.random() < 0.3:                                    # partial occlusion (counter / shelf / another person)
        im = im.copy()
        fr = rng.uniform(0.1, 0.3)
        side = rng.randint(0, 3)
        c = rng.randint(20, 200)
        if side == 0:
            im[int(h * (1 - fr)):, :] = c
        elif side == 1:
            im[:, : int(w * fr)] = c
        elif side == 2:
            im[:, int(w * (1 - fr)):] = c
        else:
            im[: int(h * fr * 0.6), :] = c
    return im


def train(data: Path, arch: str, epochs: int, seed: int, out: Path, batch: int = 32, lr: float = 3e-4,
          pretrained: bool = True, info: dict | None = None, crop_size="model", input_size: list[int] | None = None,
          patience: int = 5, keep: str = "best", classes: list[str] | None = None, any_rotation: bool = False) -> Path:
    """crop_size: "model" = model.crop_size of the config (torso-band crops), None = images as stored (garment crops).
    input_size: [h, w] network input (default torch_backbones.DEFAULT_INPUT). Early stopping after `patience`
    epochs without a better validation result; the cosine schedule decays the learning rate.
    keep: "best" = the best validation epoch, "last" = the final epoch (fixed schedule - for a validation set too
    small / easy to choose a checkpoint)."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    rng = random.Random(seed)
    size = MODEL["crop_size"] if crop_size == "model" else crop_size
    classes = list(classes or CLASSES)
    xtr, ytr = _load(data / "train", size, classes)
    xva, yva = _load(data / "val", size, classes)
    if not xtr or not xva or len(set(ytr)) < 2:
        raise SystemExit(f"{data} needs train/ and val/ crops of both classes ({', '.join(classes)})")
    out.mkdir(parents=True, exist_ok=False)            # never overwrite a candidate
    torch.set_num_threads(max(1, torch.get_num_threads()))
    model = tb.build(arch, len(classes), pretrained=pretrained)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    # class-balanced loss: the classes are not the same size
    counts = np.bincount(ytr, minlength=len(classes)).astype(np.float32)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (len(classes) * np.maximum(counts, 1))))
    inp, mean, std = list(input_size or tb.DEFAULT_INPUT), tb.IMAGENET_MEAN, tb.IMAGENET_STD
    xv = tb.to_tensor(xva, inp, mean, std)
    yv = torch.tensor(yva)
    best = (-1.0, float("inf"))
    best_state, history = None, []
    since = 0                                  # epochs without a better validation result
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        order = list(range(len(xtr)))
        rng.shuffle(order)
        total = 0.0
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            x = tb.to_tensor([_augment(xtr[j], rng, any_rotation) for j in idx], inp, mean, std)
            y = torch.tensor([ytr[j] for j in idx])
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        sched.step()
        model.eval()
        with torch.inference_mode():
            logits = torch.cat([model(xv[i:i + 64]) for i in range(0, len(xv), 64)])
            vloss = float(loss_fn(logits, yv))
            vacc = float((logits.argmax(1) == yv).float().mean())
        history.append({"epoch": ep + 1, "train_loss": round(total / len(order), 4), "val_loss": round(vloss, 4),
                        "val_acc": round(vacc, 4), "seconds": round(time.time() - t0, 1)})
        print(f"  {arch} epoch {ep + 1}/{epochs}: train loss {history[-1]['train_loss']}  val loss {vloss:.4f}  "
              f"val acc {vacc:.4f}  ({history[-1]['seconds']} s)", flush=True)
        if (vacc, -vloss) > (best[0], -best[1]):
            best = (vacc, vloss)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            since = 0
        else:
            since += 1
            if since >= patience:
                print(f"  early stop after epoch {ep + 1}", flush=True)
                break
    if keep == "last":
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    weights = out / "uniform_best.pt"
    torch.save({"format": tb.FORMAT, "arch": arch, "classes": classes, "input_size": inp, "mean": mean, "std": std,
                "state_dict": best_state}, weights)
    card = {"backend": "torchvision", "arch": arch, "classes": classes, "any_rotation_augment": any_rotation, "crop_size": list(size or []),
            "input_size": inp, "pretrained": "ImageNet" if pretrained else None, "epochs": epochs, "seed": seed,
            "batch": batch, "lr": lr, "best_val_acc": best[0], "best_val_loss": best[1], "history": history, "weights_kept": keep,
            "train_counts": {c: int(n) for c, n in zip(classes, counts)}, "data": str(data),
            "sha256": sha256_of(weights), "status": "candidate - not approved for production"} | (info or {})
    (out / "model_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    return weights


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--arch", choices=tb.ARCHS, default="efficientnet_b0")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    w = train(a.data, a.arch, a.epochs, a.seed, a.out)
    print(f"Saved candidate {w} (production model unchanged)")


if __name__ == "__main__":
    main()
