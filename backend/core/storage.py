"""Where evidence files are kept, behind one small interface.

The pipeline writes every file to local disk first and then calls `storage.put(path, kind)`;
the returned URI is what goes into the CSVs (`original_image_path`, `annotated_image_path`).

  LocalStorage   the only backend that exists: the file stays where it was written and the URI is
                 its project-relative path (storage/extractions/...).

Evidence storage on S3 is NOT implemented. S3 holds only the CSV / xlsx reports (every analysed video's CSV,
backend/extraction/s3_output.py; the cart batches' xlsx) and the admin uploads' input videos. A future S3 backend would implement the same two methods: upload the file
to a private bucket (credentials from the standard AWS provider chain - never from code or
config files) and return the real object URI (s3://bucket/key). SU_S3_BUCKET does not change where
evidence files are stored.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Protocol

from backend.core.config import ROOT, STORAGE

FileKind = Literal["original", "annotated", "export"]


def rel(p: Path) -> str:
    """Project-relative path (forward slashes); the full path when the file is outside the project."""
    try:
        return p.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(p)


class StorageBackend(Protocol):
    name: str

    def put(self, local_path: Path, kind: FileKind) -> str:
        """Store a file that was written locally; return the URI recorded in the CSVs."""
        ...

    def status(self) -> dict:
        ...


class LocalStorage:
    name = "local"

    def put(self, local_path: Path, kind: FileKind) -> str:
        return rel(local_path)

    def status(self) -> dict:
        bucket = os.environ.get("SU_S3_BUCKET", "").strip()
        return {
            "mode": self.name, "path": rel(STORAGE), "frames_path": rel(STORAGE / "extractions"),
            "s3": {"configured": bool(bucket), "bucket": bucket or None, "uploads_enabled": False,
                   "note": ("Screenshots and annotated images are kept on this PC's local disk. Every analysed video's "
                            "CSV rows are appended to today's MM-YYYY/DD-MM-YYYY/card_<N>/output_data/card<N>_YYYY_MM_DD.csv in "
                            "S3 with each row's detection image next to it (card<N>_YYYY_MM_DD_pic<K>.jpg), verified "
                            "there; the admin S3 Upload page also stores the input video in input_data.")},
        }


_storage: StorageBackend = LocalStorage()


def get_storage() -> StorageBackend:
    return _storage
