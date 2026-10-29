"""Which frames to extract (pure functions, no I/O).

Scheduled samples are taken at interval, 2*interval, ... while that time is at or before the
LAST frame of the video. The optional initial frame (t = 0) is recorded separately as sample
type "initial" - it is not counted as a scheduled sample.

Example: a 15:00 video (900 s, last frame at 899.96 s at 25 fps) sampled every 15 minutes has
NO scheduled sample (the first would be at 15:00, after the last frame). With the initial frame
it yields exactly one image. It never yields "four images".

Random samples (sample type "random") are `count` distinct frames drawn uniformly from frame 1 to
the last frame with random.Random(seed): the same video, count and seed always give the same
frames. Frame 0 is never drawn - it belongs to the optional initial frame.

Random across video (strategy "random_segments", the default for uploads): the usable frames are
split into `count` equal consecutive segments and ONE frame is drawn at random from each segment
with random.Random(seed). The segments do not overlap, so the frames are always distinct and
spread over the whole video. Usable frames are 0 .. last frame, or 1 .. last frame when the
initial frame is also saved (so it is never drawn twice). Fewer usable frames than `count` is an
error - fewer images are never returned silently.

Neither strategy guarantees that every person who appears in the video is captured: only the
sampled frames are ever looked at.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta

EPS = 1e-6


@dataclass(frozen=True)
class Sample:
    index: int               # 0 = initial frame, 1.. = scheduled / random samples in time order
    sample_type: str         # initial | scheduled | random
    target_seconds: float
    target_frame: int


def hhmmss(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def parse_recording_start(value: str | None) -> datetime | None:
    """ISO 8601 clock time of the video's first frame, as supplied by the user / CCTV adapter."""
    if value is None or not str(value).strip():
        return None
    try:
        return datetime.fromisoformat(str(value).strip())
    except ValueError:
        raise ValueError("The recording start time must be an ISO 8601 date-time, e.g. 2026-10-25T14:00:00")


def clock_time(recording_start: str | None, seconds: float) -> str:
    """Real clock time of a frame - only when the recording start time is known, otherwise ''.
    A video-relative timestamp is never presented as a clock time."""
    start = parse_recording_start(recording_start)
    return (start + timedelta(seconds=seconds)).isoformat(timespec="milliseconds") if start else ""


def last_frame_index(fps: float, frames: int | None, duration: float) -> int:
    return (frames - 1) if frames else max(0, int(duration * fps) - 1)


def plan(duration: float, fps: float, frames: int | None, interval_seconds: float,
         include_initial: bool) -> list[Sample]:
    if fps <= 0 or duration <= 0:
        raise ValueError("The video has no usable duration or frame rate")
    if interval_seconds <= 0:
        raise ValueError("The interval must be greater than 0")
    last = last_frame_index(fps, frames, duration)
    last_t = last / fps
    out: list[Sample] = []
    if include_initial:
        out.append(Sample(0, "initial", 0.0, 0))
    k = 1
    while k * interval_seconds <= last_t + EPS:
        t = k * interval_seconds
        out.append(Sample(k, "scheduled", round(t, 6), min(last, int(round(t * fps)))))
        k += 1
    return out


def plan_random(duration: float, fps: float, frames: int | None, count: int, seed: int,
                include_initial: bool) -> list[Sample]:
    if fps <= 0 or duration <= 0:
        raise ValueError("The video has no usable duration or frame rate")
    if count < 1:
        raise ValueError("The number of random samples must be at least 1")
    last = last_frame_index(fps, frames, duration)
    if count > last:
        raise ValueError(f"The video has only {last} frames after the first one; ask for fewer random samples")
    picks = sorted(random.Random(seed).sample(range(1, last + 1), count))
    out = [Sample(0, "initial", 0.0, 0)] if include_initial else []
    out += [Sample(i, "random", round(f / fps, 6), f) for i, f in enumerate(picks, 1)]
    return out


def segments(frames_first: int, frames_last: int, count: int) -> list[tuple[int, int]]:
    """`count` equal, consecutive, non-empty frame ranges covering frames_first..frames_last."""
    n = frames_last - frames_first + 1
    return [(frames_first + i * n // count, frames_first + (i + 1) * n // count - 1) for i in range(count)]


def plan_segments(duration: float, fps: float, frames: int | None, count: int, seed: int,
                  include_initial: bool) -> list[Sample]:
    if fps <= 0 or duration <= 0:
        raise ValueError("The video has no usable duration or frame rate")
    if count < 1:
        raise ValueError("The number of screenshots must be at least 1")
    last = last_frame_index(fps, frames, duration)
    first = 1 if include_initial else 0
    usable = last - first + 1
    if usable < count:
        raise ValueError(f"The video is too short for {count} distinct screenshots: it has only {usable} usable "
                         f"frame{'s' if usable != 1 else ''} ({hhmmss(duration)} long)")
    rng = random.Random(seed)
    out = [Sample(0, "initial", 0.0, 0)] if include_initial else []
    for i, (lo, hi) in enumerate(segments(first, last, count), 1):
        f = rng.randint(lo, hi)
        out.append(Sample(i, "random", round(f / fps, 6), f))
    return out


RANDOM_STRATEGIES = ("random", "random_segments")

COVERAGE_NOTE = ("Only the sampled frames are analysed, so a person who appears only between samples is not "
                 "captured.")


def describe(duration: float, fps: float, frames: int | None, interval_seconds: float | None,
             include_initial: bool, samples: list[Sample], strategy: str = "interval",
             seed: int | None = None, count: int | None = None) -> dict:
    """Counts and a plain-language explanation of the plan (shown before extraction starts)."""
    scheduled = sum(s.sample_type != "initial" for s in samples)
    initial = len(samples) - scheduled
    last_t = last_frame_index(fps, frames, duration) / fps
    common = {"total": len(samples), "scheduled": scheduled, "initial": initial, "strategy": strategy,
              "interval_seconds": interval_seconds if strategy == "interval" else None,
              "sample_count": count if strategy in RANDOM_STRATEGIES else None, "random_seed": seed,
              "duration_seconds": duration, "last_frame_seconds": round(last_t, 3), "coverage_note": COVERAGE_NOTE}
    if strategy == "random_segments":
        segs = segments(1 if initial else 0, last_frame_index(fps, frames, duration), count)
        seg_txt = [f"{hhmmss(lo / fps)[:8]}–{hhmmss(hi / fps)[:8]}" for lo, hi in segs]
        parts = [f"Video length {hhmmss(duration)} (last frame at {hhmmss(last_t)}).",
                 f"Split into {count} equal parts ({', '.join(seg_txt)}); one random screenshot is taken from each, "
                 + (f"seed {seed} (the same seed and settings give the same timestamps)." if seed is not None
                    else "with a seed that is generated and recorded when the job starts.")]
        if seed is not None:
            parts.append("Timestamps: " + ", ".join(hhmmss(s.target_seconds) for s in samples if s.sample_type != "initial") + ".")
        if initial:
            parts.append("Plus the initial frame at 00:00:00.000, recorded separately (not one of the "
                         f"{count}).")
        parts.append(f"Total: {len(samples)} image{'s' if len(samples) != 1 else ''}.")
        return common | {"explanation": " ".join(parts), "segments": [[round(lo / fps, 3), round(hi / fps, 3)] for lo, hi in segs],
                         "first_timestamps": [s.target_seconds for s in samples[:10]] if seed is not None else []}
    if strategy == "random":
        parts = [f"Video length {hhmmss(duration)} (last frame at {hhmmss(last_t)}).",
                 f"{scheduled} random frame{'s' if scheduled != 1 else ''} drawn between frame 1 and the last frame, "
                 + (f"seed {seed} (the same seed and settings give the same frames)." if seed is not None
                    else "with a seed that is generated and recorded when the job starts.")]
        if initial:
            parts.append("Plus the initial frame at 00:00:00.000, recorded separately.")
        parts.append(f"Total: {len(samples)} image{'s' if len(samples) != 1 else ''}.")
        return common | {"explanation": " ".join(parts),
                         "first_timestamps": [s.target_seconds for s in samples[:10]] if seed is not None else []}
    first_sched = interval_seconds
    parts = [f"Video length {hhmmss(duration)} (last frame at {hhmmss(last_t)}), interval {_human(interval_seconds)}."]
    if scheduled == 0:
        parts.append(f"No scheduled sample: the first one would be at {hhmmss(first_sched)}, after the last frame.")
    elif scheduled == 1:
        parts.append(f"1 scheduled sample, at {hhmmss(samples[initial].target_seconds)}.")
    else:
        parts.append(f"{scheduled} scheduled samples, from {hhmmss(samples[initial].target_seconds)} "
                     f"to {hhmmss(samples[-1].target_seconds)}.")
    if initial:
        parts.append("Plus the initial frame at 00:00:00.000, recorded separately.")
    parts.append(f"Total: {len(samples)} image{'s' if len(samples) != 1 else ''}.")
    return common | {"explanation": " ".join(parts), "first_timestamps": [s.target_seconds for s in samples[:10]]}


def _human(sec: float) -> str:
    if sec >= 60 and abs(sec / 60 - round(sec / 60)) < EPS:
        m = int(round(sec / 60))
        return f"{m} minute{'s' if m != 1 else ''}"
    return f"{sec:g} second{'s' if sec != 1 else ''}"


def as_dicts(samples: list[Sample]) -> list[dict]:
    return [asdict(s) for s in samples]
