"""Per-person memory on top of ByteTrack IDs.

ByteTrack (inside PersonDetector.track) keeps the same ID for the same person,
so 100 frames of one person = one person. This module stores each ID's
classification history, smooths it (short window with hysteresis) and builds
a whole-video verdict per person.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class TrackRecord:
    track_id: int
    first_seen: float
    last_seen: float
    observations: int = 0
    counts: dict = field(default_factory=lambda: {"uniform": 0, "non_uniform": 0, "uncertain": 0})
    conf_sum: dict = field(default_factory=lambda: {"uniform": 0.0, "non_uniform": 0.0, "uncertain": 0.0})
    window: deque = field(default_factory=deque)
    confirmed_uniform: bool = False
    ever_confirmed_uniform: bool = False
    best_det_conf: float = 0.0

    def mean_conf(self, label: str) -> float:
        n = self.counts[label]
        return self.conf_sum[label] / n if n else 0.0


class TrackMemory:
    def __init__(self, window: int = 5, min_uniform_votes: int = 3):
        self.window = max(1, window)
        self.min_uniform_votes = max(1, min(min_uniform_votes, self.window))
        self.tracks: dict[int, TrackRecord] = {}

    def update(self, people: list, t: float) -> None:
        for p in people:
            r = self.tracks.get(p.track_id)
            if r is None:
                r = self.tracks[p.track_id] = TrackRecord(p.track_id, t, t, window=deque(maxlen=self.window))
            r.last_seen = t
            r.observations += 1
            r.counts[p.label] += 1
            r.conf_sum[p.label] += p.conf
            r.best_det_conf = max(r.best_det_conf, p.det_conf)
            r.window.append((p.label, p.conf))
            self._smooth(r, p)

    def _smooth(self, r: TrackRecord, p) -> None:
        labels = [l for l, _ in r.window]
        n_u, n_n = labels.count("uniform"), labels.count("non_uniform")
        # confirm after N uniform votes; once confirmed only a non-uniform
        # MAJORITY clears it (turning around / one blurred frame doesn't)
        if n_u >= self.min_uniform_votes:
            r.confirmed_uniform = r.ever_confirmed_uniform = True
        elif r.confirmed_uniform and n_n > len(labels) / 2:
            r.confirmed_uniform = False
        if r.confirmed_uniform:
            confs = [c for l, c in r.window if l == "uniform"]
            p.smoothed_label, p.smoothed_conf = "uniform", (sum(confs) / len(confs) if confs else p.conf)
        elif n_n >= self.min_uniform_votes and n_n > n_u:
            # NON-UNIFORM needs the same number of votes as UNIFORM: never from one frame
            confs = [c for l, c in r.window if l == "non_uniform"]
            p.smoothed_label, p.smoothed_conf = "non_uniform", sum(confs) / len(confs)
        else:
            p.smoothed_label, p.smoothed_conf = "uncertain", p.conf
            if p.label in ("uniform", "non_uniform"):
                # not yet enough votes either way -> shown as "checking", not as a verdict
                p.extra["pending"] = (p.label, n_u if p.label == "uniform" else n_n, self.min_uniform_votes)

    def verdict(self, r: TrackRecord) -> tuple[str, float]:
        """Whole-video label for one person."""
        u, n = r.counts["uniform"], r.counts["non_uniform"]
        if r.ever_confirmed_uniform and u >= max(n * 0.5, 1):
            return "uniform", r.mean_conf("uniform")
        if n >= self.min_uniform_votes and n > u:
            return "non_uniform", r.mean_conf("non_uniform")
        return "uncertain", max(r.mean_conf("uniform"), r.mean_conf("non_uniform"), r.mean_conf("uncertain"))

    def people(self, min_observations: int = 1) -> list[TrackRecord]:
        return [r for r in sorted(self.tracks.values(), key=lambda r: r.first_seen)
                if r.observations >= min_observations]
