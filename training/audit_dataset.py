"""Read-only audit of the uniform-classifier image data. NEVER moves, renames, deletes or relabels files.

    python -m training.audit_dataset                                   # data/ + runs/bootstrap/final
    python -m training.audit_dataset --roots data some/other/dataset   # any folders
    python -m training.audit_dataset --near 6                          # near-duplicate distance (bits of 64)

Layout understood: <root>/[<split>/]<class>/<file>  with split in train|val|valid|test and class in
uniform|non_uniform|uncertain (anything else is reported as class "unlabelled:<folder>").
Source group = the part of the file name before "__" ("cctvstaff__t3__f120.jpg" -> "cctvstaff"); files
without it (e.g. generated "00012.jpg") are linked to an original by perceptual hash when possible.

Writes runs/audit/<timestamp>/: report.md (summary), report.json (all numbers), manifest.csv (one row
per file: path, split, class, group, sha256, dhash, size, blur, brightness, readable, duplicate links).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSES = {"uniform", "non_uniform", "uncertain"}
SPLITS = {"train": "train", "val": "val", "valid": "val", "test": "test"}


def dhash(gray: np.ndarray) -> int:
    """64-bit difference hash: robust to resizing, mild blur/brightness and JPEG re-compression."""
    s = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (s[:, 1:] > s[:, :-1]).flatten()
    return int(sum(1 << i for i, b in enumerate(bits) if b))


def describe(root: Path, p: Path) -> dict:
    rel = p.relative_to(root).parts
    split = next((SPLITS[x] for x in rel[:-1] if x in SPLITS), "unsplit")
    folder = rel[-2] if len(rel) > 1 else ""
    label = folder if folder in CLASSES else f"unlabelled:{folder or '.'}"
    raw = p.read_bytes()
    rec = {"root": root.as_posix(), "path": p.as_posix(), "split": split, "class": label,
           "group": p.stem.split("__")[0] if "__" in p.stem else "", "ext": p.suffix.lower(), "bytes": len(raw),
           "sha256": hashlib.sha256(raw).hexdigest(), "readable": False, "width": None, "height": None,
           "blur": None, "brightness": None, "dhash": None}
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        return rec
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rec.update(readable=True, height=int(img.shape[0]), width=int(img.shape[1]),
               blur=round(float(cv2.Laplacian(g, cv2.CV_64F).var()), 1), brightness=round(float(g.mean()), 1),
               dhash=dhash(g))
    return rec


def hamming_pairs(items: list[dict], limit: int) -> list[tuple[int, int, int]]:
    """All index pairs (i < j) whose dHash differs in <= limit bits (vectorised, chunked)."""
    idx = [i for i, r in enumerate(items) if r["dhash"] is not None]
    if not idx:
        return []
    h = np.array([items[i]["dhash"] for i in idx], dtype=np.uint64)
    bits = np.unpackbits(h.view(np.uint8).reshape(-1, 8), axis=1).astype(np.uint8)       # n x 64
    out = []
    for a in range(0, len(idx), 128):
        d = (bits[a:a + 128, None, :] != bits[None, :, :]).sum(axis=2)                    # chunk x n
        for i, j in zip(*np.nonzero(d <= limit)):
            gi, gj = a + i, j
            if gi < gj:
                out.append((idx[gi], idx[gj], int(d[i, j])))
    return out


def pct(values: list[float]) -> dict:
    if not values:
        return {}
    v = np.array(values, dtype=float)
    return {k: round(float(np.percentile(v, q)), 1) for k, q in (("min", 0), ("p10", 10), ("median", 50), ("p90", 90), ("max", 100))}


def audit(roots: list[Path], near: int) -> tuple[dict, list[dict]]:
    items = []
    for root in roots:
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_EXT:
                items.append(describe(root, p))
    # exact duplicates
    by_sha = defaultdict(list)
    for i, r in enumerate(items):
        by_sha[r["sha256"]].append(i)
    exact = [g for g in by_sha.values() if len(g) > 1]
    # near duplicates (+ link generated files to a named original)
    pairs = hamming_pairs(items, near)
    near_of = defaultdict(set)
    for i, j, _ in pairs:
        near_of[i].add(j)
        near_of[j].add(i)
    for i, r in enumerate(items):
        if not r["group"]:
            named = Counter(items[j]["group"] for j in near_of[i] if items[j]["group"])
            r["linked_group"] = named.most_common(1)[0][0] if named else ""
        else:
            r["linked_group"] = r["group"]
        r["exact_duplicates"] = len(by_sha[r["sha256"]]) - 1
        r["near_duplicates"] = len(near_of[i])
    # leakage across splits of the same root
    leaks = {"exact": 0, "near": 0, "groups": {}, "examples": []}
    for i, j, d in pairs:
        a, b = items[i], items[j]
        if a["root"] == b["root"] and a["split"] != b["split"] and "unsplit" not in (a["split"], b["split"]):
            leaks["exact" if a["sha256"] == b["sha256"] else "near"] += 1
            if len(leaks["examples"]) < 15:
                leaks["examples"].append({"a": a["path"], "b": b["path"], "bits": d})
    for root in {r["root"] for r in items}:
        split_groups = defaultdict(set)
        for r in items:
            if r["root"] == root and r["linked_group"] and r["split"] != "unsplit":
                split_groups[r["linked_group"]].add(r["split"])
        crossing = {g: sorted(s) for g, s in split_groups.items() if len(s) > 1}
        if crossing:
            leaks["groups"][root] = crossing
    readable = [r for r in items if r["readable"]]
    per = defaultdict(Counter)
    groups = defaultdict(Counter)
    for r in items:
        per[f"{r['root']}::{r['split']}"][r["class"]] += 1
        groups[r["class"]][r["linked_group"] or "(unknown)"] += 1
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"), "roots": [r.as_posix() for r in roots],
        "near_duplicate_bits": near, "total_files": len(items), "readable": len(readable),
        "unreadable": [r["path"] for r in items if not r["readable"]],
        "formats": dict(Counter(r["ext"] for r in items)),
        "counts": {k: dict(v) for k, v in per.items()},
        "source_groups_per_class": {k: dict(v.most_common()) for k, v in groups.items()},
        "unlabelled_files": sum(r["class"].startswith("unlabelled") for r in items),
        "exact_duplicate_sets": len(exact), "files_in_exact_duplicate_sets": sum(len(g) for g in exact),
        "near_duplicate_pairs": len(pairs),
        "files_with_a_near_duplicate": sum(1 for i in range(len(items)) if near_of[i]),
        "leakage_across_splits": leaks,
        "quality": {"width": pct([r["width"] for r in readable]), "height": pct([r["height"] for r in readable]),
                    "aspect_w_over_h": pct([r["width"] / r["height"] for r in readable]),
                    "blur_laplacian_var": pct([r["blur"] for r in readable]),
                    "brightness_0_255": pct([r["brightness"] for r in readable])},
    }
    return report, items


def markdown(r: dict) -> str:
    L = [f"# Dataset audit ({r['created_at']})", "", f"Roots: {', '.join(r['roots'])}", "",
         f"- Files: {r['total_files']} ({r['readable']} readable, {len(r['unreadable'])} unreadable)",
         f"- Formats: {r['formats']}", f"- Files without a class folder: {r['unlabelled_files']}",
         f"- Exact duplicate sets: {r['exact_duplicate_sets']} ({r['files_in_exact_duplicate_sets']} files)",
         f"- Near-duplicate pairs (dHash <= {r['near_duplicate_bits']} bits): {r['near_duplicate_pairs']}; "
         f"files with at least one near-duplicate: {r['files_with_a_near_duplicate']}", "", "## Counts per root/split and class", ""]
    for k, v in r["counts"].items():
        L.append(f"- {k}: {v}")
    L += ["", "## Source groups per class (people / clips the images come from)", ""]
    for k, v in r["source_groups_per_class"].items():
        L.append(f"- {k}: {len(v)} groups — {v}")
    lk = r["leakage_across_splits"]
    L += ["", "## Leakage across splits", "", f"- Exact duplicates across splits: {lk['exact']}",
          f"- Near-duplicates across splits: {lk['near']}"]
    for root, g in lk["groups"].items():
        L.append(f"- {root}: source groups present in more than one split: {g}")
    L += ["", "## Image size and quality", ""] + [f"- {k}: {v}" for k, v in r["quality"].items()]
    if r["unreadable"]:
        L += ["", "## Unreadable files", ""] + [f"- {p}" for p in r["unreadable"]]
    return "\n".join(L) + "\n"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", default=["data", "runs/bootstrap/final"])
    ap.add_argument("--near", type=int, default=6, help="max differing dHash bits for a near-duplicate (default 6)")
    ap.add_argument("--out", default=None, help="output folder (default runs/audit/<timestamp>)")
    a = ap.parse_args(argv)
    roots = [(ROOT / x).resolve() if not Path(x).is_absolute() else Path(x) for x in a.roots]
    missing = [str(r) for r in roots if not r.is_dir()]
    if missing:
        sys.exit(f"Not a folder: {', '.join(missing)}")
    report, items = audit(roots, a.near)
    out = Path(a.out) if a.out else ROOT / "runs" / "audit" / datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (out / "report.md").write_text(markdown(report), encoding="utf-8")
    fields = ["path", "root", "split", "class", "group", "linked_group", "ext", "bytes", "sha256", "dhash", "readable",
              "width", "height", "blur", "brightness", "exact_duplicates", "near_duplicates"]
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows({**r, "dhash": f"{r['dhash']:016x}" if r["dhash"] is not None else ""} for r in items)
    print(markdown(report))
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
