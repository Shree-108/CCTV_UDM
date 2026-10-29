"""Frame extraction from uploaded CCTV videos.

Upload (existing /api/media) -> plan the sample times -> extract the real frames with OpenCV,
one at a time -> save them unchanged under storage/extractions/<job>/ -> image_manifest.csv
(successful frames) + failures.csv (failed samples).

Separate from the uniform analysis pipeline: extracting frames never runs the model.
"""
