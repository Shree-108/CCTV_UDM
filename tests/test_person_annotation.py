"""Each detected person is classified on their OWN upper-body crop and drawn with their OWN result:
green = confirmed uniform, red = confirmed non-uniform, amber = uncertain. No position rules."""
import numpy as np
import pytest

from backend.extraction import detect
from backend.extraction.detect import AMBER, GREEN, RED, annotate
from backend.uniform.classifier import UniformDetector, upper_body_box
from tests.test_compliance_reports import Person
from tests.test_frame_detection import real_model  # noqa: F401  (fixture re-used)


class ByBrightness:
    """Stand-in classifier: P(uniform) from the crop itself (bright crop = 'uniform'), so a result can only
    come from the pixels of THAT person's crop."""
    name = "fake"

    def __init__(self):
        self.calls = []

    def predict(self, crops):
        self.calls.append(len(crops))
        return [{"uniform": 0.95, "non_uniform": 0.05} if c.mean() > 128 else {"uniform": 0.08, "non_uniform": 0.92}
                for c in crops]


def scene():
    img = np.full((600, 900, 3), 60, np.uint8)
    img[100:500, 80:260] = 230                  # bright shirt person, left
    img[120:520, 600:780] = 20                  # dark shirt person, right
    noise = np.random.default_rng(0).integers(-15, 16, img.shape)       # texture, so the blur check passes
    return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)


def detector():
    d = UniformDetector(None, untrained_mode="baseline")
    d.backend = ByBrightness()
    return d


@pytest.mark.parametrize("order", [0, 1])
def test_each_person_gets_their_own_result_regardless_of_position_or_order(order):
    left, right, tiny = Person((80, 100, 260, 500), "", 0), Person((600, 120, 780, 520), "", 0), Person((420, 300, 440, 330), "", 0)
    people = [left, right, tiny] if order == 0 else [tiny, right, left]
    d = detector()
    d.classify(scene(), people, uniform_conf=0.70, non_uniform_conf=0.60, min_height=50)
    assert (left.label, round(left.conf, 2)) == ("uniform", 0.95)
    assert (right.label, round(right.conf, 2)) == ("non_uniform", 0.92)
    assert right.extra["uniform_prob"] == pytest.approx(0.08)           # the CSV score is P(uniform) of THIS person
    assert (tiny.label, tiny.reason) == ("uncertain", "too small / far")  # not forced into either class
    assert d.backend.calls == [2]                                        # one batch, the 2 judgeable crops only


def test_low_confidence_stays_uncertain():
    d = detector()
    d.backend.predict = lambda crops: [{"uniform": 0.55, "non_uniform": 0.45} for _ in crops]
    p = Person((80, 100, 260, 500), "", 0)
    d.classify(scene(), [p], uniform_conf=0.70, non_uniform_conf=0.60, min_height=50)
    assert (p.label, p.reason) == ("uncertain", "low confidence")


def test_boxes_are_green_red_amber_at_each_persons_own_location():
    img = np.zeros((600, 900, 3), np.uint8)
    u, n, q = Person((80, 100, 260, 500), "uniform", 0.93), Person((600, 120, 780, 520), "non_uniform", 0.88), \
        Person((380, 150, 520, 450), "uncertain", 0.55, "low confidence")
    out = annotate(img, [u, n, q], detect.UNIFORM_DETECTED, 0.70, "test model")
    sx1, sy1, sx2, sy2 = upper_body_box(u.box)                          # uniform: box around its shirt region
    assert tuple(out[(sy1 + sy2) // 2, sx1]) == GREEN and tuple(out[(sy1 + sy2) // 2, sx2]) == GREEN
    assert tuple(out[400, 600]) == RED and tuple(out[400, 780]) == RED   # non-uniform: its own person box
    assert tuple(out[350, 380]) == AMBER and tuple(out[350, 520]) == AMBER
    assert tuple(out[400, 600]) != GREEN and tuple(out[(sy1 + sy2) // 2, sx1]) != RED
    assert not img.any()                                                 # the original frame is untouched


# ------------------------------------------------------------------ counts and "not scored" reasons
def test_zero_and_several_uniform_wearers_are_counted_as_they_are():
    from backend.extraction.detect import NO_UNIFORM_DETECTED, UNIFORM_DETECTED, outcome_for
    none = [Person((0, 0, 50, 120), "non_uniform", 0.9), Person((60, 0, 110, 120), "non_uniform", 0.8)]
    assert outcome_for(none)[0] == NO_UNIFORM_DETECTED and sum(p.label == "uniform" for p in none) == 0
    two = [Person((0, 0, 50, 120), "uniform", 0.9), Person((60, 0, 110, 120), "non_uniform", 0.8),
           Person((120, 0, 170, 120), "uniform", 0.85)]
    assert outcome_for(two)[0] == UNIFORM_DETECTED and sum(p.label == "uniform" for p in two) == 2
    img = np.zeros((300, 400, 3), np.uint8)
    assert np.any(annotate(img, none, NO_UNIFORM_DETECTED, 0.7, "m") != annotate(img, two, UNIFORM_DETECTED, 0.7, "m"))


def test_not_scored_people_keep_their_reason_in_results_and_csv(client, real_model, monkeypatch):  # noqa: F811
    from tests.test_compliance_reports import csv_rows, run_job
    def frame(img, models, cfg, thr):
        tiny = Person((650, 60, 670, 90), "uncertain", 0.0, "too small / far")
        tiny.extra.pop("uniform_prob")              # like the real classifier: no crop -> never scored
        return [Person((40, 40, 140, 400), "uniform", 0.93), Person((300, 50, 400, 420), "uniform", 0.81),
                Person((500, 50, 600, 420), "non_uniform", 0.10), tiny]
    monkeypatch.setattr(detect, "classify_frame", frame)
    j, run = run_job(client, {"random_seed": 301, "camera_id": "CAM-NS"})
    assert run["state"] == "done" and run["detections"] == 8                   # 2 uniform wearers x 4 screenshots
    rows = csv_rows(client, run["results_csv_url"])
    tiny = [r for r in rows if r["detection_type"] == "uncertain"]
    assert len(tiny) == 4 and all(r["person_reason"] == "too small / far" and r["confidence"] == "" for r in tiny)
    assert all(r["person_reason"] == "" for r in rows if r["detection_type"] != "uncertain")
    assert {r["detection_count"] for r in rows} == {"2"}
    biz = client.get(f"/api/extractions/{j['id']}/uniform-compliance").json()["records"][0]["row"]
    assert biz["confidence score"] == "Person 1: 93.0%; Person 2: 81.0%; Person 3: 10.0%; Person 4: not scored"
    assert biz["remark (Example)"].endswith("Person 4 requires manual review (too small / far).")
