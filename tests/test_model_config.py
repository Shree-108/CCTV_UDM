"""Model portability: config-relative paths, class mapping, version/checksum, overall rule."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from backend.analysis import compliance
from backend.core import model_config as mc

ROOT = Path(__file__).resolve().parents[1]


def test_config_paths_are_project_relative():
    raw = yaml.safe_load((ROOT / "config" / "uniform_config.yaml").read_text(encoding="utf-8"))
    for p in [raw["model"]["path"], raw["person_detector"]["path"], *raw["model"]["legacy_paths"]]:
        assert not Path(p).is_absolute() and ":" not in p, f"PC-specific path in config: {p}"
    assert mc._project_path(raw["model"]["path"]) == ROOT / raw["model"]["path"]
    assert mc.PERSON_MODEL == ROOT / raw["person_detector"]["path"]
    assert mc.rel(mc._project_path(raw["model"]["path"])) == raw["model"]["path"]


def test_missing_model_is_reported_with_expected_path():
    if mc.uniform_model_path().exists():
        pytest.skip("trained weights present on this machine")
    ident = mc.model_identity()
    assert ident["present"] is False and ident["expected_path"] == mc.rel(mc.UNIFORM_MODEL)
    assert ident["expected_path"] in mc.version_label(ident)


def test_all_uniform_rule():
    d = lambda labels: compliance.decide(labels, "all_uniform")  # noqa: E731
    assert d(["uniform", "non_uniform"]) == compliance.NO_UNIFORM
    assert d(["uniform", "uncertain"]) == compliance.UNCERTAIN
    assert d(["uniform", "uniform"]) == compliance.COMPLIANT
    assert d([]) == compliance.NO_PERSON


@pytest.fixture(scope="module")
def dummy_cls(tmp_path_factory):
    """An UNTRAINED 3-class YOLO-cls file - only exercises loading / mapping, never deployed."""
    from ultralytics import YOLO
    from ultralytics.nn.tasks import ClassificationModel
    m = YOLO("yolo11n-cls.yaml")
    m.model = ClassificationModel("yolo11n-cls.yaml", nc=3, verbose=False)
    m.model.names = {0: "non_uniform", 1: "uncertain", 2: "uniform"}
    p = tmp_path_factory.mktemp("w") / "dummy_cls.pt"
    m.save(str(p))
    return p


def test_trained_classifier_maps_classes(dummy_cls):
    from backend.uniform.classifier import TrainedClassifier
    clf = TrainedClassifier(str(dummy_cls), "cpu")
    out = clf.predict([np.full((120, 90, 3), 128, np.uint8)])[0]
    assert set(out) == {"uniform", "non_uniform", "uncertain"}
    assert abs(sum(out.values()) - 1.0) < 1e-3


def test_trained_classifier_rejects_unmapped_classes(dummy_cls, monkeypatch):
    from backend.uniform import classifier
    monkeypatch.setitem(classifier.MODEL, "class_map", {"uniform": "uniform"})
    with pytest.raises(ValueError, match="class_map"):
        classifier.TrainedClassifier(str(dummy_cls), "cpu")


def test_checksum_mismatch_is_flagged(dummy_cls, monkeypatch):
    from backend.uniform import classifier
    monkeypatch.setattr(mc, "uniform_model_path", lambda: dummy_cls)
    monkeypatch.setattr(classifier, "model_identity", mc.model_identity)
    monkeypatch.setitem(mc.MODEL, "sha256", "0" * 64)
    det = classifier.UniformDetector(dummy_cls, "cpu")
    assert det.trained and det.status["state"] == "mismatch"
    monkeypatch.setitem(mc.MODEL, "sha256", mc.sha256_of(dummy_cls))
    det = classifier.UniformDetector(dummy_cls, "cpu")
    assert det.status["state"] == "ready" and det.status["model"]["verified"] is True
    assert mc.MODEL["version"] in det.model_version
