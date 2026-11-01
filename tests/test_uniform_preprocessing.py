"""The classifier sees torso crops at the shape its weights were trained on (model.crop_size).

2026-10-29 audit: training/bootstrap.py resized every training crop to 112x160, inference passed crops
as-is. A light-blue patterned shirt scored P(uniform) 0.94 as-is and 0.03 at the training shape."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.core import model_config as mc
from backend.uniform import classifier
from tests.test_model_config import dummy_cls  # noqa: F401  (fixture re-used)

ROOT = Path(__file__).resolve().parents[1]
REAL_WEIGHTS = ROOT / "models" / "uniform_best.pt"


def shapes_seen(clf, crops) -> list[tuple]:
    seen = []
    real = clf.model.predict
    clf.model.predict = lambda batch, **kw: (seen.extend(c.shape for c in batch), real(batch, **kw))[1]
    clf.predict(crops)
    return seen


def test_crops_reach_the_model_at_the_training_shape(dummy_cls, monkeypatch):  # noqa: F811
    monkeypatch.setitem(classifier.MODEL, "crop_size", (112, 160))
    clf = classifier.TrainedClassifier(str(dummy_cls), "cpu")
    crops = [np.full((241, 226, 3), 90, np.uint8), np.full((80, 200, 3), 200, np.uint8)]
    assert shapes_seen(clf, crops) == [(160, 112, 3), (160, 112, 3)]
    assert [c.shape for c in crops] == [(241, 226, 3), (80, 200, 3)]          # the caller's crops are untouched


def test_rollback_passes_crops_as_is(dummy_cls, monkeypatch):  # noqa: F811
    monkeypatch.setitem(classifier.MODEL, "crop_size", None)
    clf = classifier.TrainedClassifier(str(dummy_cls), "cpu")
    assert shapes_seen(clf, [np.full((241, 226, 3), 90, np.uint8)]) == [(241, 226, 3)]


def test_config_rejects_a_malformed_crop_size(tmp_path, monkeypatch):
    raw = (ROOT / "config" / "uniform_config.yaml").read_text(encoding="utf-8")
    bad = tmp_path / "bad.yaml"
    bad.write_text(raw.replace("crop_size: [112, 160]", "crop_size: [112]"), encoding="utf-8")
    monkeypatch.setattr(mc, "CONFIG_PATH", bad)
    with pytest.raises(ValueError, match="crop_size"):
        mc.load()


def test_training_and_inference_use_the_same_crop_shape():
    from training import bootstrap
    assert mc.MODEL["crop_size"] == (112, 160) and tuple(bootstrap.SIZE) == mc.MODEL["crop_size"]


def test_results_record_the_preprocessing_in_the_model_label(monkeypatch):
    if not REAL_WEIGHTS.exists():
        pytest.skip("trained weights not on this machine")
    monkeypatch.setattr(mc, "uniform_model_path", lambda: REAL_WEIGHTS)       # conftest pins a missing model
    ident = mc.model_identity()
    assert "crop 112x160" in mc.version_label(ident).split(" · ")[0]          # the part stored with every result


@pytest.fixture(scope="module")
def real_clf():
    if not REAL_WEIGHTS.exists():
        pytest.skip("trained weights not on this machine")
    return classifier.TrainedClassifier(str(REAL_WEIGHTS), "cpu")


def crops(pattern: str) -> list[np.ndarray]:
    files = sorted((ROOT / "data").glob(pattern))
    if not files:
        pytest.skip(f"data/{pattern} not on this machine (data/ is git-ignored)")
    return [cv2.imread(str(f)) for f in files]


def p_uniform(clf, imgs) -> np.ndarray:
    return np.array([d["uniform"] for i in range(0, len(imgs), 32) for d in clf.predict(imgs[i:i + 32])])


def test_patterned_light_shirt_is_not_approved_and_striped_uniform_is(real_clf, monkeypatch):
    """Regression guard on the two real CCTV people of data/: the black-and-white striped staff member
    (data/uniform) and the light-blue patterned shirt (data/non_uniform). These are TRAINING images, so
    this checks the preprocessing, not how well the model generalises - there is no independent test set."""
    staff, customer = crops("uniform/cctvstaff__*"), crops("non_uniform/cctvcustomer__*")
    uc = mc.INFERENCE["uniform_confidence"]
    assert (p_uniform(real_clf, staff) >= uc).all()
    assert (p_uniform(real_clf, customer) < uc).all()
    monkeypatch.setattr(real_clf, "crop_size", None)                           # the pre-fix behaviour
    assert (p_uniform(real_clf, customer) >= uc).sum() > 0                     # ...approved some of them


def test_squat_person_boxes_go_to_review(monkeypatch):
    """Seated / leaning / cut-off people: the torso band would cover the head, so no shirt verdict."""
    frame = (1080, 1920, 3)
    assert mc.INFERENCE["min_box_aspect"] == 1.5
    assert classifier.visibility_issue((500, 300, 1100, 1070), [], frame, 50).startswith("seated")   # h/w 1.28
    assert classifier.visibility_issue((500, 100, 800, 1000), [], frame, 50) is None                 # h/w 3.0, standing
    monkeypatch.setattr(classifier, "MIN_BOX_ASPECT", 0.0)                                          # 0 = off
    assert classifier.visibility_issue((500, 300, 1100, 1070), [], frame, 50) is None
