"""Business reports for the Results Dashboard: overview + latest analysis, months, all historical rows.

Everything is derived from the database and the stored result files (backend.extraction.reports_service);
nothing here writes data. The six-column CSVs use exactly the business schema."""
from __future__ import annotations

import csv
import io
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Path as P, Request
from fastapi.responses import Response

from backend.extraction import reports_service as RS
from backend.extraction.business_csv import BUSINESS_FIELDS

router = APIRouter(prefix="/api/reports")
By = Literal["completion", "recording"]
MONTH = P(pattern=r"^\d{4}-(0[1-9]|1[0-2])$", description="YYYY-MM")


def db(request: Request):
    return request.app.state.app_state.db


@router.get("/overview")
def overview(request: Request):
    """Summary cards, the latest completed analysis (with its evidence and CSV), the newest failed /
    running job when it is not that one, and the months that contain records."""
    return RS.overview(db(request))


@router.get("/months/{month}.csv")
def month_csv(request: Request, month: str = MONTH, by: By = "completion", download: bool = False):
    r = RS.month_report(db(request), month, by)
    if not r["analyses"] and not r["videos_received"]:
        raise HTTPException(404, f"No records for {month} ({by} date)")
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=BUSINESS_FIELDS)
    w.writeheader()
    w.writerows(r["_rows"])
    name = f"uniform_compliance_{month}{'_by_recording_date' if by == 'recording' else ''}.csv"
    headers = {"Content-Disposition": f'attachment; filename="{name}"'} if download else {}
    return Response(buf.getvalue(), media_type="text/csv", headers=headers)


@router.get("/months/{month}")
def month(request: Request, month: str = MONTH, by: By = "completion"):
    r = RS.month_report(db(request), month, by)
    if not r["analyses"] and not r["videos_received"]:
        raise HTTPException(404, f"No records for {month} ({by} date)")
    return {k: v for k, v in r.items() if not k.startswith("_")}


@router.get("/records")
def records(request: Request, month: Optional[str] = None, by: By = "completion", store: Optional[str] = None,
            video: Optional[str] = None, result: Optional[str] = None, status: Optional[str] = None,
            q: Optional[str] = None, limit: int = 500, offset: int = 0):
    """All business rows of completed analyses (the master CSV's content), filterable."""
    rows = RS.history_records(db(request), month, by, store, video, result, status, q)
    return {"total": len(rows), "fields": BUSINESS_FIELDS, "records": rows[offset: offset + min(limit, 5000)]}
