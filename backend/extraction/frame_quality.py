"""Is a decoded video frame a real picture? Checked before a screenshot is saved and before inference.

Two kinds of frame carry no evidence and must never be reported as "no person detected" (or as a
uniform result of any kind):
  decoder grey fill  where the video data a frame depends on is missing (a gap in an NVR recording, or a
                     decoder started in the middle of a group of pictures), the HEVC / H.264 decoder paints
                     the missing picture flat neutral mid-grey and draws only the changes on top: a grey
                     frame with ghost outlines, sometimes with part of a person.
  blank picture      (nearly) one flat colour over the whole frame: covered lens, "no video" screen, ...

Both are recognised from 16x16 blocks of a 640-px-wide copy. A block is flat when its brightness varies
by less than FLAT_STD levels; it is grey fill when it is also neutral (no colour) and close to mid-grey.
Real CCTV pictures - colour or grayscale (IR night) - have sensor noise and texture. Calibration on
14,000+ sequentially decoded frames (cart 46 / 50 / 78 NVR exports, a 1080p MP4 CCTV clip, H.264 and
MPEG-4 clips): every readable frame had 0 % grey-fill blocks, every broken one 50 % or more.
"""
from __future__ import annotations

import cv2
import numpy as np

WIDTH, BLOCK = 640, 16
FLAT_STD = 3.0             # brightness std inside a block (0-255 levels)
GREY_MID, GREY_TOL, NEUTRAL = 128, 20, 6
GREY_FILL_LIMIT = 0.25     # share of the picture painted with decoder grey fill
BLANK_LIMIT = 0.98         # share of the picture that is flat (any colour)


def problem(frame) -> str | None:
    """Why `frame` (BGR image) is not a usable picture, or None when it is readable."""
    if frame is None or getattr(frame, "size", 0) == 0 or frame.ndim != 3 or min(frame.shape[:2]) < BLOCK:
        return "the decoder returned no usable image data"
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (WIDTH, max(BLOCK, round(h * WIDTH / w))), interpolation=cv2.INTER_AREA)
    ycc = cv2.cvtColor(small, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    bh, bw = ycc.shape[0] // BLOCK, WIDTH // BLOCK
    blocks = ycc[:bh * BLOCK, :bw * BLOCK].reshape(bh, BLOCK, bw, BLOCK, 3).swapaxes(1, 2).reshape(bh, bw, -1, 3)
    y, chroma = blocks[..., 0], np.abs(blocks[..., 1:] - 128).max(axis=-1)
    flat = y.std(axis=-1) < FLAT_STD
    grey = flat & (np.abs(y.mean(axis=-1) - GREY_MID) < GREY_TOL) & (chroma.mean(axis=-1) < NEUTRAL)
    if grey.mean() >= GREY_FILL_LIMIT:
        return (f"corrupted frame: {grey.mean():.0%} of the picture is decoder grey fill (the video data this "
                "frame depends on is missing, e.g. a gap in the recording)")
    if flat.mean() >= BLANK_LIMIT:
        return "blank frame: the picture is one flat colour (no video signal, covered lens or complete darkness)"
    return None
