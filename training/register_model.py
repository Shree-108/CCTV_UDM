"""Register the approved uniform weights in config/uniform_config.yaml.

    python -m training.register_model                    # record SHA-256 of models/uniform_best.pt
    python -m training.register_model --version 1.1.0    # also set model.version
    python -m training.register_model --check            # verify only (exit 1 on mismatch / missing)

Once the checksum is recorded, every deployment verifies it runs the exact same
model file, and the dashboard shows the version + checksum. Does NOT train.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.core.model_config import CONFIG_PATH, MODEL, rel, sha256_of, uniform_model_path  # noqa: E402


def _set(text: str, key: str, value: str) -> str:
    """Replace `  key: ...` inside the `model:` block, keeping comments."""
    block = re.search(r"(?ms)^model:\n(.*?)(?=^\S|\Z)", text)
    if not block:
        raise SystemExit(f"{CONFIG_PATH.name}: no model: section")
    body, n = re.subn(rf'(?m)^(  {key}:)[^\n#]*?(\s*#[^\n]*)?$', rf'\g<1> "{value}"\g<2>', block.group(1), count=1)
    if not n:
        raise SystemExit(f"{CONFIG_PATH.name}: model.{key} not found")
    return text[:block.start(1)] + body + text[block.end(1):]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--version")
    ap.add_argument("--status", help="e.g. experimental | production")
    ap.add_argument("--dataset-version")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

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
