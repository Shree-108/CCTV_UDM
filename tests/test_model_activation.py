"""Candidate models are saved beside - never over - the active model, and can be activated and rolled back.
Also: the training data builder uses the production crop shape, and the diagnostics tool runs the app's code."""
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from backend.core import model_config as mc
from tests.test_model_config import dummy_cls  # noqa: F401  (fixture re-used)
from training import candidate, register_model

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "models" / "uniform_best.pt"


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """A copy of the real config + a private candidates folder; the real files are never written."""
    cfg = tmp_path / "config" / "uniform_config.yaml"
    cfg.parent.mkdir()
    shutil.copy2(ROOT / "config" / "uniform_config.yaml", cfg)
    monkeypatch.setattr(register_model, "CONFIG_PATH", cfg)
    monkeypatch.setattr(register_model, "BACKUPS", cfg.parent / "backups")
    cands = ROOT / "runs" / "test_candidates" / tmp_path.name               # inside the project (git-ignored runs/)
    monkeypatch.setattr(candidate, "CANDIDATES", cands)
    yield cfg
    shutil.rmtree(cands, ignore_errors=True)


def model_section(cfg: Path) -> dict:
    return yaml.safe_load(cfg.read_text(encoding="utf-8"))["model"]


def test_candidate_is_saved_beside_the_active_model_and_can_be_rolled_back(sandbox, dummy_cls):  # noqa: F811
    before_prod = mc.sha256_of(PRODUCTION) if PRODUCTION.exists() else None
    original = sandbox.read_text(encoding="utf-8")
    folder = candidate.save_candidate(dummy_cls, "uniform-cls-test", {"dataset_version": "test-set-1"})
    card = candidate.read_card(folder)
    assert card["sha256"] == mc.sha256_of(folder / "uniform_best.pt") and card["status"] == "candidate"
    assert card["classes"] == {"0": "non_uniform", "1": "uncertain", "2": "uniform"}       # JSON keys
    assert card["preprocessing"]["crop_size"] == list(mc.MODEL["crop_size"])

    register_model.main(["--activate", str(folder), "--version", "0.9.0-test"])
    m = model_section(sandbox)
    assert m["path"] == (folder / "uniform_best.pt").relative_to(ROOT).as_posix()    # project-relative
    assert (m["sha256"], m["version"], m["status"], m["crop_size"]) == (card["sha256"], "0.9.0-test", "candidate", [112, 160])
    assert m["dataset_version"] == "test-set-1"
    assert "# [width, height] every torso crop" in sandbox.read_text(encoding="utf-8")   # comments kept

    register_model.main(["--rollback"])
    assert sandbox.read_text(encoding="utf-8") == original                 # exactly the previous config
    with pytest.raises(SystemExit, match="No saved config"):
        register_model.main(["--rollback"])                               # a backup is restored only once
    if before_prod:
        assert mc.sha256_of(PRODUCTION) == before_prod                     # the production weights were never touched


def test_candidate_outside_the_project_is_refused(sandbox, dummy_cls, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(candidate, "CANDIDATES", tmp_path / "elsewhere")
    folder = candidate.save_candidate(dummy_cls, "uniform-cls-test", {})
    with pytest.raises(SystemExit, match="inside the project"):
        register_model.main(["--activate", str(folder)])


def test_tampered_candidate_is_not_activated(sandbox, dummy_cls):  # noqa: F811
    original = sandbox.read_text(encoding="utf-8")
    folder = candidate.save_candidate(dummy_cls, "uniform-cls-test", {})
    with open(folder / "uniform_best.pt", "ab") as f:
        f.write(b"tampered")
    with pytest.raises(SystemExit, match="checksum"):
        register_model.main(["--activate", str(folder)])
    assert sandbox.read_text(encoding="utf-8") == original


def test_training_crops_are_built_at_the_inference_shape(tmp_path):
    from training.prepare_dataset import build
    for cls in ("uniform", "non_uniform"):
        for g in ("a", "b"):
            d = tmp_path / "src" / cls
            d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{g}__t1__f0.jpg"), np.full((241, 226, 3), 120, np.uint8))
    out = build({"sources": {c: str(tmp_path / "src" / c) for c in ("uniform", "non_uniform")},
                 "output_dir": str(tmp_path / "cls"), "val_split": 0.5, "seed": 1,
                 "lowres_augment": {"enabled": True, "copies_per_image": 1}})
    shapes = {cv2.imread(str(p)).shape for p in out.rglob("*.jpg")}
    assert shapes == {(160, 112, 3)}                                       # = model.crop_size, as inference resizes


def test_diagnostics_use_the_app_pipeline(tmp_path):
    from training import diagnose
    cap = cv2.VideoCapture(str(ROOT / "tests" / "media" / "dvr_grid.mp4"))
    ok, frame = cap.read()
    cap.release()
    assert ok
    img = tmp_path / "frame.png"
    cv2.imwrite(str(img), frame)
    rows = diagnose.diagnose([img, tmp_path / "missing.png"], tmp_path / "out")
    with open(tmp_path / "out" / "diagnostics.csv", newline="", encoding="utf-8") as f:
        written = list(csv.DictReader(f))
    assert list(written[0]) == diagnose.FIELDS and len(written) == len(rows)
    assert written[-1]["visibility_issue"] == "image could not be read"
    for r in written[:-1]:                                                # every person: a decision + its evidence
        assert r["label"] in ("uniform", "non_uniform", "uncertain") and (tmp_path / "out" / r["panel"]).exists()
        assert r["raw_output"] or r["visibility_issue"]
