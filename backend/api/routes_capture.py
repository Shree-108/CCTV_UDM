"""Cameras and scheduled captures (Phase 2: local test sources only)."""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.api.routes import AreaIn
from backend.capture.service import CaptureBusy
from backend.capture.sources import validate_source
from backend.detection.grid import validate_area

router = APIRouter(prefix="/api")
ID_PATTERN = r"^[A-Za-z0-9_-]{1,32}$"


def st(request: Request):
    return request.app.state.app_state


class CameraIn(BaseModel):
    id: str = Field(pattern=ID_PATTERN, description="Camera id, e.g. CAM-01")
    store_id: Optional[str] = Field(default=None, pattern=ID_PATTERN, description="Store/card id; default from Settings")
    name: Optional[str] = Field(default=None, max_length=80)
    source_type: Literal["folder", "video"]
    source: str = Field(min_length=1, max_length=500, description="Local folder of images or video file")
    area: Optional[AreaIn] = Field(default=None, description="Camera tile + store ROI; default: full frame")
    enabled: bool = True


class CameraPatch(BaseModel):
    store_id: Optional[str] = Field(default=None, pattern=ID_PATTERN)
    name: Optional[str] = Field(default=None, max_length=80)
    source_type: Optional[Literal["folder", "video"]] = None
    source: Optional[str] = Field(default=None, min_length=1, max_length=500)
    area: Optional[AreaIn] = None
    enabled: Optional[bool] = None


def _check_area(area: AreaIn | None) -> None:
    if area is not None:
        try:
            validate_area(area.rect, area.roi)
        except ValueError as e:
            raise HTTPException(422, str(e))


def _camera_out(s, cam: dict) -> dict:
    last, n = s.db.captures(limit=1, camera_id=cam["id"])
    return cam | {"last_capture": _capture_out(last[0]) if last else None, "captures_total": n}


def _capture_out(c: dict) -> dict:
    return c | {"image_url": f"/api/captures/{c['id']}/image" if c.get("media_id") else None,
                "preview_url": f"/api/media/{c['media_id']}/preview.jpg" if c.get("media_id") else None}


@router.get("/capture/status")
def capture_status(request: Request):
    return st(request).capture.status()


@router.get("/cameras")
def list_cameras(request: Request):
    s = st(request)
    return {"cameras": [_camera_out(s, c) for c in s.db.cameras()]}


@router.get("/cameras/{cam_id}")
def get_camera(request: Request, cam_id: str):
    s = st(request)
    cam = s.db.camera(cam_id) or _404("Camera")
    return _camera_out(s, cam)


@router.post("/cameras", status_code=201)
def create_camera(request: Request, body: CameraIn):
    s = st(request)
    if s.db.camera(body.id):
        raise HTTPException(409, f"Camera {body.id} already exists")
    _check_area(body.area)
    try:
        source = validate_source(body.source_type, body.source)
    except ValueError as e:
        raise HTTPException(422, str(e))
    s.db.upsert_camera({"id": body.id, "store_id": body.store_id or s.settings.get()["default_store_id"],
                        "name": body.name or body.id, "source_type": body.source_type, "source": source,
                        "area": body.area.model_dump() if body.area else None, "enabled": body.enabled, "cursor": 0})
    return _camera_out(s, s.db.camera(body.id))


@router.put("/cameras/{cam_id}")
def update_camera(request: Request, cam_id: str, body: CameraPatch):
    s = st(request)
    cam = s.db.camera(cam_id) or _404("Camera")
    changes = body.model_dump(exclude_unset=True)
    _check_area(body.area)
    if "source" in changes or "source_type" in changes:
        try:
            changes["source"] = validate_source(changes.get("source_type", cam["source_type"]),
                                                changes.get("source", cam["source"]))
        except ValueError as e:
            raise HTTPException(422, str(e))
        changes["cursor"] = 0                      # new source: start from its beginning
    s.db.upsert_camera({"id": cam_id, **changes})
    return _camera_out(s, s.db.camera(cam_id))


@router.delete("/cameras/{cam_id}")
def delete_camera(request: Request, cam_id: str):
    """Removes the camera; its captures and analyses are kept as history."""
    s = st(request)
    s.db.camera(cam_id) or _404("Camera")
    s.db.delete_camera(cam_id)
    return {"ok": True}


@router.post("/cameras/{cam_id}/capture")
def capture_now(request: Request, cam_id: str):
    s = st(request)
    s.db.camera(cam_id) or _404("Camera")
    try:
        return _capture_out(s.capture.capture_now(cam_id, "manual"))
    except CaptureBusy as e:
        raise HTTPException(409, str(e))


@router.get("/captures")
def list_captures(request: Request, camera_id: Optional[str] = None, store_id: Optional[str] = None,
                  status: Optional[str] = None, start: Optional[str] = None, end: Optional[str] = None,
                  limit: int = 100, offset: int = 0):
    rows, total = st(request).db.captures(min(limit, 500), offset, camera_id, store_id, status, start, end)
    return {"captures": [_capture_out(c) for c in rows], "total": total}


@router.get("/captures/{cap_id}/image")
def capture_image(request: Request, cap_id: str):
    """The ORIGINAL captured frame (before analysis)."""
    s = st(request)
    c = s.db.capture(cap_id) or _404("Capture")
    m = s.db.media(c["media_id"]) if c.get("media_id") else None
    if not m or not Path(m["path"]).exists():
        _404("Captured image")
    return FileResponse(m["path"], media_type="image/jpeg")


def _404(what: str):
    raise HTTPException(404, f"{what} not found")
