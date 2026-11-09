"""Register the approved uniform weights in config/uniform_config.yaml.

    python -m training.register_model                    # record SHA-256 of models/uniform_best.pt
    python -m training.register_model --version 1.1.0    # also set model.version
    python -m training.register_model --check            # verify only (exit 1 on mismatch / missing)
    python -m training.register_model --activate models/candidates/<folder> --version 0.3.0
                                       # switch to a reviewed candidate (path, checksum, crop shape)
    python -m training.register_model --rollback         # restore the config saved before the last activation

Once the checksum is recorded, every deployment verifies it runs the exact same
model file, and the dashboard shows the version + checksum. Does NOT train.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.core.model_config import CONFIG_PATH, MODEL, rel, sha256_of, uniform_model_path  # noqa: E402


def _set(text: str, key: str, value: str, raw: bool = False) -> str:
    """Replace `  key: ...` inside the `model:` block, keeping comments. `raw` = write YAML as-is (lists, null)."""
    block = re.search(r"(?ms)^model:\n(.*?)(?=^\S|\Z)", text)
    if not block:
        raise SystemExit(f"{CONFIG_PATH.name}: no model: section")
    new = value if raw else f'"{value}"'
    body, n = re.subn(rf'(?m)^(  {key}:)[^\n#]*?(\s*#[^\n]*)?$', lambda m: f"{m.group(1)} {new}{m.group(2) or ''}",
                      block.group(1), count=1)
    if not n:
        raise SystemExit(f"{CONFIG_PATH.name}: model.{key} not found")
    return text[:block.start(1)] + body + text[block.end(1):]


BACKUPS = CONFIG_PATH.parent / "backups"


def activate(folder: Path, version: str | None, status: str | None) -> None:
    """Point the config at a candidate. The current config is backed up first; no model file is moved."""
    from training.candidate import read_card
    folder = folder if folder.is_absolute() else ROOT / folder
    card, weights = read_card(folder), folder / "uniform_best.pt"
    if not card or not weights.exists():
        raise SystemExit(f"{folder}: not a candidate folder (needs uniform_best.pt + model_card.json)")
    if not weights.resolve().is_relative_to(ROOT.resolve()):          # the config holds project-relative paths only
        raise SystemExit(f"{folder}: a candidate must be inside the project (models/candidates/...)")
    digest = sha256_of(weights)
    if digest != card["sha256"]:
        raise SystemExit(f"{weights.name} does not match its model card checksum - not activated")
    BACKUPS.mkdir(parents=True, exist_ok=True)
    backup = BACKUPS / f"{CONFIG_PATH.stem}.{datetime.now():%Y%m%d-%H%M%S-%f}.yaml"
    shutil.copy2(CONFIG_PATH, backup)
    cs = card.get("preprocessing", {}).get("crop_size")
    text = CONFIG_PATH.read_text(encoding="utf-8")
    text = _set(text, "path", rel(weights))
    text = _set(text, "sha256", digest)
    text = _set(text, "crop_size", f"[{cs[0]}, {cs[1]}]" if cs else "null", raw=True)
    text = _set(text, "backend", card.get("backend") or "yolo_cls")           # yolo_cls | torchvision
    text = _set(text, "crop_mode", card.get("crop_mode") or "torso_band")     # the crop the weights were trained on
    text = _set(text, "dataset_version", str(card.get("dataset_version") or card.get("dataset") or ""))
    text = _set(text, "version", version or folder.name)
    text = _set(text, "status", status or "candidate")
    CONFIG_PATH.write_text(text, encoding="utf-8")
    print(f"Active model: {rel(weights)} (sha256 {digest[:12]}). Previous config saved as {rel(backup)}.\n"
          "Restart the API to load it. Undo: python -m training.register_model --rollback")


def rollback() -> None:
    backups = sorted(BACKUPS.glob(f"{CONFIG_PATH.stem}.*.yaml")) if BACKUPS.exists() else []
    if not backups:
        raise SystemExit("No saved config to roll back to")
    shutil.copy2(backups[-1], CONFIG_PATH)
    backups[-1].rename(backups[-1].with_suffix(".restored"))          # kept, but never restored twice
    print(f"Restored {rel(CONFIG_PATH)} from {rel(backups[-1])}. Restart the API to load the previous model.")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version")
    ap.add_argument("--status", help="e.g. experimental | production")
    ap.add_argument("--dataset-version")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--activate", type=Path, help="candidate folder (with model_card.json) to make active")
    ap.add_argument("--rollback", action="store_true", help="restore the config saved by the last --activate")
    args = ap.parse_args(argv)
    if args.rollback:
        return rollback()
    if args.activate:
        return activate(args.activate, args.version, args.status)

    p = uniform_model_path()
    if not p.exists():
        raise SystemExit(f"Uniform model missing: {rel(p)} - restore the trained weights there first.")
    digest = sha256_of(p)
    print(f"model    {rel(p)}  ({p.stat().st_size / 2 ** 20:.1f} MB)")
    print(f"sha256   {digest}")
    if args.check:
        if not MODEL["sha256"]:
            raise SystemExit("No sha256 registered in the config yet.")
        ok = digest == MODEL["sha256"]
        print("MATCH - approved model" if ok else f"MISMATCH - expected {MODEL['sha256']}")
        raise SystemExit(0 if ok else 1)

    text = CONFIG_PATH.read_text(encoding="utf-8")
    text = _set(text, "sha256", digest)
    if args.version:
        text = _set(text, "version", args.version)
    if args.status is not None:
        text = _set(text, "status", args.status)
    if args.dataset_version is not None:
        text = _set(text, "dataset_version", args.dataset_version)
    CONFIG_PATH.write_text(text, encoding="utf-8")
    print(f"Registered in {rel(CONFIG_PATH)}" + (f" as version {args.version}" if args.version else ""))


if __name__ == "__main__":
    main()
