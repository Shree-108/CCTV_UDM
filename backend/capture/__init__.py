"""Scheduled capture: grab a frame from a camera source on a fixed schedule and run it
through the existing image-analysis pipeline (backend/analysis/runner.py).

Phase 2 supports LOCAL sources only (a folder of images or a video file) so the whole flow
can be tested with recorded footage. RTSP/NVR sources are added once CCTV access is authorised.
"""
