"""Garment crop (backend/uniform/garment.py) and crop_mode "garment" in the classifier - with a stand-in pose
model and classifier (no weights needed). The production default (torso_band) is unchanged."""
import numpy as np
import pytest

from backend.core.model_config import MODEL
from backend.detection.person import Person
from backend.uniform import classifier, garment


class FakePose:
    """Keypoints of a person facing the camera / seated: shoulders, elbows, hips."""

    def __init__(self, kp):
        self.kp = kp

    def keypoints(self, person):
        return self.kp


def skeleton(sh_y, hip_y, x_l=40, x_r=120, hips=True):
    kp = np.zeros((17, 3), np.float32)
    kp[garment.L_SH] = (x_r, sh_y, 0.9)
    kp[garment.R_SH] = (x_l, sh_y, 0.9)
    if hips:
        kp[garment.L_HIP] = (x_r - 10, hip_y, 0.9)
        kp[garment.R_HIP] = (x_l + 10, hip_y, 0.9)
    return kp


def test_prepare_keeps_aspect_ratio_without_stretching():
    crop = np.full((100, 40, 3), 255, np.uint8)
    out = garment.prepare(crop)
    assert out.shape == (224, 224, 3)
    cols = np.where(out[112, :, 0] == 255)[0]
    assert 85 <= len(cols) <= 95                            # 40 x 100 -> ~90 x 224: stripes are not widened


def test_shirt_from_shoulders_to_hips_includes_the_collar_not_the_head():
    person = np.zeros((400, 160, 3), np.uint8)
    x1, y1, x2, y2, src = garment.garment_region(person, FakePose(skeleton(100, 250)))
    assert src == "pose"
    assert 60 < y1 < 100 and 250 <= y2 <= 270                # just above the shoulders (collar) down to the hips
    assert x1 < 40 and x2 > 120                               # a little wider than the shoulders (sleeves)


def test_seated_behind_a_counter_hips_hidden():
    person = np.zeros((300, 200, 3), np.uint8)               # box cut at the waist: head + shoulders + chest
    _, y1, _, y2, src = garment.garment_region(person, FakePose(skeleton(110, 0, hips=False)))
    assert src == "pose" and y1 > 70 and y2 > 200            # the shirt below the shoulders - not the head band


def test_no_keypoints_falls_back_to_the_box_shape():
    assert garment.garment_region(np.zeros((600, 200, 3), np.uint8), FakePose(None))[4] == "fallback"
    tall = garment.garment_region(np.zeros((600, 200, 3), np.uint8), None)
    squat = garment.garment_region(np.zeros((300, 220, 3), np.uint8), None)
    assert tall[1] < 0.15 * 600 and squat[1] > 0.2 * 300     # standing: upper body; cut at the waist: lower part


class FakeBackend:
    name, device = "trained", "cpu"

    def __init__(self):
        self.seen = []

    def predict(self, crops):
        self.seen += [c.shape for c in crops]
        return [{"uniform": 0.95, "non_uniform": 0.05} for _ in crops]


@pytest.fixture()
def garment_detector(monkeypatch):
    monkeypatch.setitem(MODEL, "crop_mode", "garment")
    det = classifier.UniformDetector.__new__(classifier.UniformDetector)
    det.backend, det._pose = FakeBackend(), FakePose(skeleton(60, 170))
    return det


def test_garment_mode_classifies_a_seated_person(garment_detector):
    frame = np.random.default_rng(0).integers(0, 255, (400, 400, 3), dtype=np.uint8)
    seated = Person(1, (100, 100, 300, 330), 0.9)             # h/w 1.15: torso_band mode calls this "seated / leaning"
    garment_detector.classify(frame, [seated], 0.70, 0.60, 50, occluders=[seated.box])
    assert seated.label == "uniform" and garment_detector.backend.seen == [(224, 224, 3)]


def test_garment_mode_still_refuses_tiny_or_occluded_people(garment_detector):
    frame = np.random.default_rng(1).integers(0, 255, (600, 600, 3), dtype=np.uint8)
    tiny = Person(1, (10, 10, 30, 40), 0.9)
    back = Person(2, (200, 100, 300, 400), 0.9)
    front = Person(3, (190, 150, 310, 500), 0.9)              # stands in front (lower feet) and covers the torso
    garment_detector.classify(frame, [tiny, back], 0.70, 0.60, 50, occluders=[tiny.box, back.box, front.box])
    assert (tiny.label, tiny.reason) == ("uncertain", "too small / far")
    assert (back.label, back.reason) == ("uncertain", "occluded by another person")


def test_production_default_is_still_the_torso_band():
    assert MODEL["crop_mode"] == "torso_band" and MODEL["backend"] == "yolo_cls"
