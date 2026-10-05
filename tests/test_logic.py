"""Unit tests for grid/ROI geometry, tracking votes, temporal presence and the status rule."""
import numpy as np

from backend.analysis import compliance
from backend.analysis.temporal import PresenceTimeline
from backend.detection.grid import AnalysisArea, detect_layout, grid_cells, validate_area
from backend.detection.person import Person
from backend.tracking.memory import TrackMemory
from backend.uniform.classifier import StripePatternBaseline


def P(tid, label, conf=0.9):
    p = Person(tid, (10, 10, 60, 200), 0.9)
    p.label, p.conf = label, conf
    return p


def test_grid_cells_row_major():
    cells = grid_cells(2, 2)
    assert cells[0] == [0, 0, 0.5, 0.5] and cells[1] == [0.5, 0, 1, 0.5] and cells[3] == [0.5, 0.5, 1, 1]


def test_area_crop_and_polygon_roi():
    a = AnalysisArea([0.5, 0, 1, 0.5], roi=[[0.5, 0], [1, 0], [1, 1], [0.5, 1]]).bind(1280, 960)
    assert a.rect == (640, 0, 1280, 480) and a.tile_size == (640, 480)
    assert a.contains((400, 100, 500, 400))          # feet x=450 > 320 -> inside
    assert not a.contains((50, 100, 150, 400))
    assert AnalysisArea([0, 0, 1, 1]).bind(100, 100).contains((0, 0, 10, 10))   # no ROI -> whole tile


def test_validate_area():
    validate_area([0, 0, 0.5, 0.5], None)
    for bad in ([0.5, 0, 0.4, 1], [0, 0, 0.01, 0.01], [0, 0, 1.2, 1]):
        try:
            validate_area(bad, None)
            raise AssertionError(bad)
        except ValueError:
            pass


def test_layout_detection():
    rng = np.random.default_rng(1)
    tiles = [np.full((240, 320, 3), v, np.uint8) + rng.integers(0, 20, (240, 320, 3), dtype=np.uint8) for v in (40, 120, 180, 90)]
    assert detect_layout([np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])]) == "2x2"
    assert detect_layout([np.hstack(tiles[:2])]) == "1x2"
    assert detect_layout([tiles[0]]) == "1x1"


def test_same_person_counted_once_and_confirmed():
    m = TrackMemory(5, 3)
    for t in range(100):
        m.update([P(1, "uniform")], t * 0.2)
    assert len(m.people()) == 1 and m.verdict(m.people()[0])[0] == "uniform"


def test_confirmation_needs_multiple_frames():
    m = TrackMemory(5, 3)
    p = P(1, "uniform")
    m.update([p], 0)
    assert p.smoothed_label != "uniform"          # one frame is never enough
    for t in (1, 2):
        p = P(1, "uniform")
        m.update([p], t)
    assert p.smoothed_label == "uniform"


def test_blurred_frame_does_not_flip_uniform():
    m = TrackMemory(5, 3)
    for t in range(4):
        m.update([P(1, "uniform")], t)
    p = P(1, "non_uniform")
    m.update([p], 5)
    assert p.smoothed_label == "uniform"


def test_occlusion_gap_bridged():
    tl = PresenceTimeline(gap_tolerance=3.0)
    for t in (1, 2):
        tl.update(t, True)
    tl.update(3, False)
    tl.update(4, False)          # not visible
    tl.update(5, True)           # back
    assert len(tl.segments) == 1 and tl.longest == 4.0
    assert tl.present_at(4.0)


def test_status_rule():
    assert compliance.decide(["non_uniform", "uniform"]) == compliance.COMPLIANT
    assert compliance.decide(["non_uniform", "uncertain"]) == compliance.NO_UNIFORM
    assert compliance.decide(["uncertain"]) == compliance.UNCERTAIN
    assert compliance.decide([]) == compliance.NO_PERSON
    assert "owner" not in compliance.message(compliance.NO_UNIFORM, 0, 3).lower()


def _stripes(vertical=True, period=6):
    img = np.full((140, 110, 3), 240, np.uint8)
    for k in range(0, 110 if vertical else 140, period):
        if vertical:
            img[:, k:k + period // 2] = 25
        else:
            img[k:k + period // 2, :] = 25
    return img


def test_baseline_rejects_black_white_lookalikes():
    b = StripePatternBaseline()
    p = lambda im: b.predict([im])[0]["uniform"]  # noqa: E731
    assert p(_stripes(True)) > 0.7
    assert p(_stripes(False)) < 0.5                                   # horizontal stripes
    assert p(np.full((140, 110, 3), 25, np.uint8)) < 0.5              # plain black
    assert p(np.full((140, 110, 3), 235, np.uint8)) < 0.5             # plain white
    blue = _stripes(True).copy(); blue[..., 1:] //= 3                 # blue/black stripes
    assert p(blue) < 0.5
