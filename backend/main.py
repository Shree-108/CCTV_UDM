"""Sightline · Uniform Compliance - FastAPI backend.

Run from the project root:
    uvicorn backend.main:app --host 127.0.0.1 --port 8010

Inputs: uploaded images/videos, and scheduled captures from LOCAL camera sources
(backend/capture/: a folder of images or a video file). Live CCTV (RTSP/ONVIF/NVR) is
not enabled until authorised access exists; it would plug in as another capture source.
S3: POST /api/s3/process analyses a cart's input_data videos and uploads the results to output_data;
POST /api/s3/uploads (admin login) stores one uploaded video in S3, analyses it and uploads its CSV.

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
from backend.api.routes_extraction import router as extraction_router
from backend.api.routes_reports import router as reports_router
from backend.api.routes_results import router as results_router
from backend.api.routes_auth import router as auth_router
from backend.api.routes_s3 import router as s3_router
from backend.api.routes_s3_uploads import router as s3_uploads_router
from backend.capture.service import CaptureService
from backend.core.config import CORS_ORIGINS, DB_PATH
from backend.core.settings import Settings
from backend.database.db import DB
from backend.extraction.runner import ExtractionRunner
from backend.extraction.s3_batch import S3BatchRunner
from backend.extraction.s3_upload import S3UploadRunner

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
        self.db.interrupt_extractions()
        self.db.interrupt_detection_runs()
        self.extractions = ExtractionRunner(self.db, self.settings, self.models)
        self.s3_batches = S3BatchRunner(self.db, self.extractions)
        self.db.interrupt_s3_uploads()
        self.s3_uploads = S3UploadRunner(self.db, self.s3_batches)

    def shutdown(self) -> None:
        self.capture.stop()
        self.s3_uploads.shutdown()
        self.s3_batches.shutdown()
        self.extractions.shutdown()
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
        from backend.extraction import master_csv
        logging.info("master CSV sync: %s", master_csv.sync(s.db))    # backfill / recover finished jobs
        if os.environ.get("SU_CAPTURE_SCHEDULER", "on").lower() not in ("off", "0", "false"):
            s.capture.start()
        yield
        s.shutdown()

    app = FastAPI(title="Sightline Uniform Compliance API", version="1.0.0", lifespan=lifespan)
    # credentials: the admin session cookie (dashboard on :3010, API on :8010 = same site, other origin)
    app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_credentials=True,
                       allow_headers=["Content-Type", "Authorization"])
    app.include_router(router)
    app.include_router(capture_router)
    app.include_router(extraction_router)
    app.include_router(results_router)
    app.include_router(reports_router)
    app.include_router(s3_router)
    app.include_router(auth_router)
    app.include_router(s3_uploads_router)
    return app


app = create_app()
