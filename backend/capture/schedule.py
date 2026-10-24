"""Capture time slots (pure functions, local time of this PC).

Slots are aligned to the window start: with start 09:00, end 21:00 and a 15-minute interval
the slots are 09:00, 09:15, ... 20:45 (the end is exclusive). A window whose end is earlier
than its start runs overnight (e.g. 22:00-06:00).

The scheduler captures the CURRENT slot (the latest slot that has started) once per camera.
After a restart it therefore catches up on the slot in progress but never back-fills older
slots, and a slot already captured is never captured again.
"""
from __future__ import annotations

from datetime import date, datetime, time, timedelta


def parse_hhmm(s: str) -> time:
    h, m = s.split(":")
    return time(int(h), int(m))


def _bounds(day: date, start: time, end: time) -> tuple[datetime, datetime]:
    s = datetime.combine(day, start)
    e = datetime.combine(day, end)
    if e <= s:          # overnight window
        e += timedelta(days=1)
    return s, e


def current_slot(now: datetime, interval_min: int, start: time, end: time) -> datetime | None:
    """The latest slot that has started at `now` (naive local time), or None outside the window."""
    step = timedelta(minutes=interval_min)
    for day in (now.date(), now.date() - timedelta(days=1)):   # an overnight window may have begun yesterday
        s, e = _bounds(day, start, end)
        if s <= now < e:
            return s + step * int((now - s) / step)
    return None


def next_slot(now: datetime, interval_min: int, start: time, end: time) -> datetime:
    """The first slot strictly after `now` (naive local time)."""
    step = timedelta(minutes=interval_min)
    cur = current_slot(now, interval_min, start, end)
    if cur is not None:
        cand = cur + step
        for day in (now.date(), now.date() - timedelta(days=1)):
            s, e = _bounds(day, start, end)
            if s <= cur < e and cand < e:
                return cand
    for k in range(0, 3):                                        # next window start
        s, _ = _bounds(now.date() + timedelta(days=k), start, end)
        if s > now:
            return s
    raise ValueError("no next slot")                             # unreachable for valid windows


def slots_per_day(interval_min: int, start: time, end: time) -> int:
    s, e = _bounds(date(2000, 1, 1), start, end)
    return -(-int((e - s).total_seconds()) // (interval_min * 60))   # ceil
