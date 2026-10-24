"""Sightline · Uniform Compliance - FastAPI backend.

Run from the project root:
    uvicorn backend.main:app --host 127.0.0.1 --port 8010

Inputs: uploaded images/videos, and scheduled captures from LOCAL camera sources
(backend/capture/: a folder of images or a video file). Live CCTV (RTSP/ONVIF/NVR) is
not enabled until authorised access exists; it would plug in as another capture source.

SU_CAPTURE_SCHEDULER=off disables the background capture scheduler (manual captures still work).
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.analysis.jobs import JobRunner
from backend.analysis.pipeline import Models
from backend.api.routes import router
from backend.api.routes_capture import router as capture_router
from backend.capture.service import CaptureService
from backend.core.config import CORS_ORIGINS, DB_PATH
from backend.core.settings import Settings
from backend.database.db import DB

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")


class AppState:
    def __init__(self, db_path=DB_PATH):
        self.db = DB(db_path)
        self.db.interrupt_running()
        self.settings = Settings()
        self.models = Models()
        self.jobs = JobRunner(self.db, self.settings, self.models)
        self.db.interrupt_captures()
        self.capture = CaptureService(self.db, self.settings, self.models, self.jobs)

    def shutdown(self) -> None:
        self.capture.stop()
        self.jobs.shutdown()


def create_app(state: AppState | None = None, preload: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        s = state or AppState()
        app.state.app_state = s
        if preload:
            try:
                cfg = s.settings.get()
                s.models.get(cfg).warmup(cfg)
            except Exception as e:  # noqa: BLE001 - reported by /api/health
                logging.error("model load failed: %s", e)
        if os.environ.get("SU_CAPTURE_SCHEDULER", "on").lower() not in ("off", "0", "false"):
            s.capture.start()
        yield
        s.shutdown()

    app = FastAPI(title="Sightline Uniform Compliance API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"],
                       allow_headers=["Content-Type", "Authorization"])
    app.include_router(router)
    app.include_router(capture_router)
    return app


app = create_app()
