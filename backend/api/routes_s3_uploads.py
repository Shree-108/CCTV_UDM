"""Admin video uploads: upload one video for a Card ID, have it stored in S3, analysed, and its CSV report
uploaded next to it (backend/extraction/s3_upload.py). Every route needs the admin login."""
from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from backend.analysis.media import kind_of, probe
from backend.api.routes import receive
from backend.core import s3
from backend.core.auth import require_admin
from backend.extraction import cart_report
from backend.extraction.s3_upload import INCOMING, UploadRejected, card_cart

router = APIRouter(prefix="/api/s3", dependencies=[Depends(require_admin)])


def runner(request: Request):
    return request.app.state.app_state.s3_uploads


@router.get("/cards")
def cards():
    """Card IDs that can be selected (config/carts.yaml)."""
    return {"cards": [{"card_id": cart_report.cart_id(c), "cart": c, "store_name": name}
                      for c, name in cart_report.known_carts().items()]}


@router.post("/uploads", status_code=202)
async def upload(request: Request, card_id: int = Form(ge=1, le=99999), file: UploadFile = File(...),
                 date: Optional[str] = Form(default=None, description="S3 day folder DD-MM-YYYY; default: the "
                                            "recording date in the CCTV file name"),
                 user: str = Depends(require_admin)):
    """Receive a video for a card. 202 with the job (poll GET /api/s3/uploads/{id}); 200 with the existing
    job when this video was already uploaded for this card ("how": "duplicate" / "retried")."""
    name = file.filename or ""
    if kind_of(name) != "video":
        raise HTTPException(415, "Choose a video file (MP4, AVI, MOV, MKV, M4V, WEBM, MPG, TS).")
    try:
        card_cart(card_id)
        if date and date.strip():
            day, day_source = s3.parse_day(date), "selected"
        else:
            start = cart_report.recording_start(Path(name).name)
            if not start:
                raise UploadRejected("The recording date is not in the file name: choose the date folder (DD-MM-YYYY).")
            day, day_source = datetime.fromisoformat(start).date(), "file_name"
    except (UploadRejected, ValueError) as e:
        raise HTTPException(422, str(e))
    run = runner(request)
    bucket = s3.bucket_name()
    check = run.batches.check(bucket)
    if not check["ok"]:
        raise HTTPException(503, check["message"])

    INCOMING.mkdir(parents=True, exist_ok=True)
    path = INCOMING / f"{uuid.uuid4().hex}{Path(name).suffix.lower()}"
    size, sha = await receive(file, path, "video")
    try:
        probe(path, name, "video", size)                  # decodable, real length (NVR headers can be wrong)
    except ValueError as e:
        path.unlink(missing_ok=True)
        raise HTTPException(422, str(e))
    try:
        job, how = run.create(card_id=card_id, path=path, filename=name, size=size, sha256=sha, day=day,
                              day_source=day_source, bucket=bucket, user=user)
    except UploadRejected as e:
        path.unlink(missing_ok=True)
        raise HTTPException(422, str(e))
    return JSONResponse(job | {"how": how}, status_code=202 if how != "duplicate" else 200)


@router.get("/uploads")
def list_uploads(request: Request, limit: int = 50):
    return {"uploads": runner(request).list(max(1, min(limit, 200)))}


@router.get("/uploads/{uid}")
def get_upload(request: Request, uid: str):
    run = runner(request)
    job = run.get(uid)
    if not job:
        raise HTTPException(404, "Upload not found")
    return job | {"report_rows": run.report_rows(uid)}


@router.post("/uploads/{uid}/retry", status_code=202)
def retry_upload(request: Request, uid: str):
    try:
        return runner(request).retry(uid)
    except KeyError:
        raise HTTPException(404, "Upload not found")
    except UploadRejected as e:
        raise HTTPException(409, str(e))


@router.get("/uploads/{uid}/report.csv")
def report_csv(request: Request, uid: str):
    path = runner(request).report_path(uid)
    if not path:
        raise HTTPException(404, "No report: the job has not completed")
    return FileResponse(path, media_type="text/csv", filename=path.name)
