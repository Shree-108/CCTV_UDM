"""Interface for a FUTURE authorised CCTV / NVR integration - nothing here contacts a camera.

No recorder API is configured: the vendor, its API documentation and credentials have not been
provided, so no endpoint is assumed and no camera retrieval is simulated. What exists:

  * `CctvAdapter`  the contract a real integration must implement: given a store, camera and
                   time range, return a local video file (`VideoClip`) plus the facts that only
                   the recorder knows (camera id, store id, real recording start time);
  * `ingest_clip`  hands such a clip to the SAME pipeline as a manual upload (media record ->
                   extraction job with store / camera / recording start time -> optional
                   detection). It is exercised by the tests with a local video file.

The uploaded-video workflow does not depend on this module.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol

from backend.analysis.media import kind_of, new_upload_path, probe

# What is needed before a real adapter can be written (shown by GET /api/frame-results -> "cctv").
REQUIREMENTS = [
    "Recorder / VMS vendor and model, firmware version, and written authorisation to access it",
    "The vendor's official API documentation for exporting recorded footage (or RTSP playback URLs) "
    "for a camera and time range",
    "A service account with read-only playback permission (stored outside the code, e.g. environment / secret store)",
    "Network route from this server to the recorder (VPN / firewall rules), and the expected bandwidth",
    "Mapping of recorder channels to store_id / camera_id, and the recorder's time zone / clock source",
    "A test window of footage to verify an export end to end before it is called working",
]


@dataclass(frozen=True)
class VideoClip:
    path: Path                          # local video file supplied by the integration
    store_id: str | None = None
    camera_id: str | None = None
    recording_start_time: str | None = None   # ISO 8601 clock time of the first frame, from the recorder
    source_ref: str = ""                # recorder / export reference, for traceability


class CctvAdapter(Protocol):
    name: str

    def fetch_clip(self, store_id: str, camera_id: str, start: datetime, end: datetime) -> VideoClip:
        """Export the recorded footage of one camera for [start, end) to a local file."""
        ...


class CctvNotConfigured(RuntimeError):
    pass


def status() -> dict:
    return {"state": "not_configured", "adapter": None,
            "note": "No authorised CCTV / recorder API is configured. Videos are processed from manual uploads; "
                    "live camera retrieval is not implemented and has not been tested.",
            "requirements": REQUIREMENTS}


def get_adapter() -> CctvAdapter:
    raise CctvNotConfigured("No CCTV adapter is configured: " + "; ".join(REQUIREMENTS))


def ingest_clip(db, runner, clip: VideoClip, *, interval_seconds: float | None = None, include_initial: bool = True,
                strategy: str = "interval", sample_count: int | None = None, seed: int | None = None,
                run_detection: bool = False, threshold: float | None = None) -> tuple[dict, bool]:
    """Register a supplied clip like an upload and start an extraction job for it. The clip is
    copied, so the pipeline never depends on the integration keeping the file."""
    src = Path(clip.path)
    if not src.is_file() or kind_of(src.name) != "video":
        raise ValueError(f"Not a supported video file: {src.name}")
    dst = new_upload_path(src.name)
    shutil.copy2(src, dst)
    try:
        meta = probe(dst, src.name, "video", dst.stat().st_size)
    except ValueError:
        dst.unlink(missing_ok=True)
        raise
    db.add_media(meta)
    return runner.create(db.media(meta["id"]), interval_seconds, include_initial, run_detection=run_detection,
                         threshold=threshold, strategy=strategy, sample_count=sample_count, seed=seed,
                         store_id=clip.store_id, camera_id=clip.camera_id,
                         recording_start_time=clip.recording_start_time, source="cctv_adapter")
