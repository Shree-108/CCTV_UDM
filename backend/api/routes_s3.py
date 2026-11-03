"""S3 cart batches: analyse the videos in a cart's input_data folder and write the results to its
output_data folder (backend/extraction/s3_batch.py)."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from backend.core import s3

router = APIRouter(prefix="/api/s3")


def st(request: Request):
    return request.app.state.app_state


class ProcessIn(BaseModel):
    date: str = Field(description="Day folder, DD-MM-YYYY, e.g. 30-10-2026 (the month folder 10-2026 is derived)")
    carts: list[str] = Field(min_length=1, max_length=100, description="e.g. [\"cart_78\", \"cart_50\"]; a bare number means cart_<n>")
    bucket: Optional[str] = Field(default=None, max_length=63, description="Default: SU_S3_BUCKET or yk-cctv-prediction")
    threshold: Optional[float] = Field(default=None, ge=0.5, le=0.99, description="Detection threshold (default: Settings)")
    output_folder: Optional[str] = Field(default=None, max_length=40,
                                         description="Test runs only: upload to <cart>/<output_folder>/ instead of "
                                                     "<cart>/output_data/, e.g. output_data_test (default: output_data)")


@router.get("/status")
def s3_status(request: Request, bucket: Optional[str] = None):
    """Whether AWS credentials are present and the bucket is reachable (the credentials are never returned)."""
    return st(request).s3_batches.check(bucket or s3.bucket_name())


@router.post("/process", status_code=202)
def process(request: Request, body: ProcessIn):
    """Queue a batch: for each cart, every video in <MM-YYYY>/<DD-MM-YYYY>/<cart>/input_data/ is analysed
    and the results are uploaded to <MM-YYYY>/<DD-MM-YYYY>/<cart>/output_data/. Poll GET /api/s3/process/{id}."""
    runner = st(request).s3_batches
    try:
        day = s3.parse_day(body.date)
        carts = list(dict.fromkeys(s3.cart_name(c) for c in body.carts))
        folder = s3.other_output_folder(body.output_folder) if body.output_folder is not None else s3.OUTPUT_FOLDER
    except ValueError as e:
        raise HTTPException(422, str(e))
    bucket = (body.bucket or "").strip() or s3.bucket_name()
    check = runner.check(bucket)
    if not check["ok"]:
        raise HTTPException(503, check["message"])
    return runner.create(day, carts, bucket, body.threshold, folder)


@router.get("/process")
def list_batches(request: Request):
    return {"batches": st(request).s3_batches.list()}


@router.get("/process/{bid}")
def get_batch(request: Request, bid: str):
    b = st(request).s3_batches.get(bid)
    if not b:
        raise HTTPException(404, "Batch not found")
    return b
