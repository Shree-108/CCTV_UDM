"""Store-level uniform presence over time.

A confirmed uniform that disappears for less than `gap_tolerance` seconds
(walking behind a shelf, occlusion, missed detections) is treated as
continuous presence - no false "no uniform" gap is created.
"""
from __future__ import annotations


class PresenceTimeline:
    def __init__(self, gap_tolerance: float = 3.0):
        self.gap = gap_tolerance
        self.segments: list[list[float]] = []   # [start, end] of confirmed presence

    def update(self, t: float, uniform_present: bool) -> None:
        if not uniform_present:
            return
        if self.segments and t - self.segments[-1][1] <= self.gap:
            self.segments[-1][1] = t
        else:
            self.segments.append([t, t])

    @property
    def longest(self) -> float:
        return max((e - s for s, e in self.segments), default=0.0)

    @property
    def total(self) -> float:
        return sum(e - s for s, e in self.segments)

    def present_at(self, t: float) -> bool:
        return bool(self.segments) and t - self.segments[-1][1] <= self.gap
