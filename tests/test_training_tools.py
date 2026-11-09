"""training/apply_labels.py (reviewed crops -> train / frozen test set) and the metrics of
training/compare_models.py. Temporary folders only: the project's data/ is never touched."""
import csv

import cv2
import numpy as np
import pytest

from training import apply_labels, compare_models


@pytest.fixture()
def roots(tmp_path, monkeypatch):
    monkeypatch.setattr(apply_labels, "TRAIN_ROOT", tmp_path / "data")
    monkeypatch.setattr(apply_labels, "TEST_ROOT", tmp_path / "data" / "validation")
    return tmp_path


def queue(tmp_path, rows, name="q"):
    d = tmp_path / name
    (d / "crops").mkdir(parents=True)
    for i, r in enumerate(rows):
        cv2.imwrite(str(d / "crops" / f"shot{i}__p1.jpg"), np.full((40, 30, 3), 10 * i, np.uint8))
        r.setdefault("crop", f"crops/shot{i}__p1.jpg")
    with open(d / "cctv_label_queue.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["crop", "source_video", "reviewer_label", "reviewer_notes"])
        w.writeheader()
        w.writerows(rows)
    return d / "cctv_label_queue.csv"


def test_copies_only_reviewed_two_class_labels_grouped_by_video(roots):
    q = queue(roots, [{"source_video": "cam 1.mp4", "reviewer_label": "uniform"},
                      {"source_video": "cam 1.mp4", "reviewer_label": "non_uniform"},
                      {"source_video": "cam 1.mp4", "reviewer_label": "uncertain"},
                      {"source_video": "cam 1.mp4", "reviewer_label": ""}])
    dry = apply_labels.apply(q, "train", None, dry_run=True)
    assert dry["would copy -> data/uniform"] == 1 and dry["would copy -> data/non_uniform"] == 1
    assert not (roots / "data").exists()                                   # dry run writes nothing
    st = apply_labels.apply(q, "train", None)
    files = sorted(p.relative_to(roots / "data").as_posix() for p in (roots / "data").rglob("*.jpg"))
    assert files == ["non_uniform/cam-1__all__shot1__p1.jpg", "uniform/cam-1__all__shot0__p1.jpg"]
    assert st["skipped (uncertain)"] == 1 and st["skipped (not labelled)"] == 1
    assert (roots / "q" / "crops" / "shot0__p1.jpg").exists()             # copied, never moved
    again = apply_labels.apply(q, "train", None)                           # repeat: nothing duplicated / overwritten
    assert again["already there (same file)"] == 2


def test_never_overwrites_a_different_file(roots):
    q = queue(roots, [{"source_video": "v.mp4", "reviewer_label": "uniform"}])
    dest = roots / "data" / "uniform" / "v__all__shot0__p1.jpg"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"someone else's file")
    assert apply_labels.apply(q, "train", None)["NAME CLASH - not overwritten"] == 1
    assert dest.read_bytes() == b"someone else's file"


def test_test_set_and_training_never_share_a_video(roots):
    q_train = queue(roots, [{"source_video": "dayA.mp4", "reviewer_label": "uniform"}], "q1")
    apply_labels.apply(q_train, "train", None)
    q_test = queue(roots, [{"source_video": "dayA.mp4", "reviewer_label": "non_uniform"}], "q2")
    with pytest.raises(SystemExit, match="already in the TRAINING data"):
        apply_labels.apply(q_test, "test", "store50")
    q_new = queue(roots, [{"source_video": "dayB.mp4", "reviewer_label": "non_uniform"}], "q3")
    apply_labels.apply(q_new, "test", "store50")
    assert (roots / "data" / "validation" / "store50" / "non_uniform" / "dayB__all__shot0__p1.jpg").exists()
    q_leak = queue(roots, [{"source_video": "dayB.mp4", "reviewer_label": "uniform"}], "q4")
    with pytest.raises(SystemExit, match="already in the TEST set"):
        apply_labels.apply(q_leak, "train", None)
    with pytest.raises(SystemExit, match="--subset"):
        apply_labels.apply(q_new, "test", None)


def test_metrics_treat_review_as_not_approved():
    rows = [{"actual": "uniform", "m": "uniform"}, {"actual": "uniform", "m": "uncertain"},
            {"actual": "uniform", "m": "non_uniform"}, {"actual": "non_uniform", "m": "uniform"},
            {"actual": "non_uniform", "m": "non_uniform"}, {"actual": "non_uniform", "m": "non_uniform"}]
    m = compare_models.binary(rows, "m")
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 2, 2)
    assert m["precision"] == 0.5 and m["recall"] == round(1 / 3, 4) and m["false_positive_rate"] == round(1 / 3, 4)
    assert m["violations_wrongly_reported"] == 1 and m["review_rate"] == round(1 / 6, 4)
