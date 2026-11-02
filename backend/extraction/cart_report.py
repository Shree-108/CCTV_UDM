"""The cart report uploaded to S3 by the cart batches: <cart>/output_data/<cart>.xlsx, in the layout of
the business sample (cart_50.xlsx): one row per sampled screenshot, in recording-time order.

  cart_id           the cart number (cart_50 -> 50)
  Store Name        from config/carts.yaml; blank when the cart is not listed
  Date / Time Stamp recording clock time (from the video file name); blank Date and the position inside
                    the video ("video 00:01:23") when the recording time is unknown
  img name          pic1, pic2, ... (which screenshot each one is: pictures.csv in the local batch folder)
  confidence score  mean of the uniform classifier's P(uniform) over the scored people, 0-100;
                    0 when nobody was detected; blank when the screenshot could not be analysed
  remark (Example)  "1. person1 wearing the dress", "2. person2 is not wearing the dress, others wearing", ...
                    A person the model could not judge "needs manual review" - never "not wearing".
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import yaml

from backend.core.config import ROOT

CARTS_CONFIG = ROOT / "config" / "carts.yaml"
HEADERS = ["cart_id", "Store Name ", "Date ", "Time Stamp ", "img name ", "confidence score ", "remark (Example)"]
WIDTHS = [8.73, 20.0, 13.0, 12.0, 11.0, 17.0, 70.0]
FILENAME_TIME = re.compile(r"^\d{8}_(\d{14})_\d{14}_")       # 20261029_20261029183009_20261029183120_183009.mp4


def store_name(cart: str) -> str:
    try:
        names = yaml.safe_load(CARTS_CONFIG.read_text(encoding="utf-8")) or {}
    except FileNotFoundError:
        names = {}
    return str(names.get(cart) or "")


def cart_id(cart: str) -> int | str:
    digits = re.sub(r"\D", "", cart)
    return int(digits) if digits else cart


def recording_start(filename: str) -> str | None:
    """Recording start time encoded in the CCTV file name (ISO), or None when the name has none."""
    m = FILENAME_TIME.match(filename)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y%m%d%H%M%S").isoformat()
    except ValueError:
        return None


def _names(nums: list[int]) -> str:
    names = [f"person{n}" for n in nums]
    return names[0] if len(names) == 1 else f"{', '.join(names[:-1])} and {names[-1]}"


def remark(rec: dict) -> str:
    """Remark of one screenshots.screenshot_results record, worded like the business sample."""
    if rec["processing_status"] not in ("ok", "annotation_failed"):
        return "uniform status could not be determined (processing failed)"
    types = [p["type"] for p in rec["people"]]
    if not types:
        return "no person detected in frame"
    if len(types) == 1:
        return {"uniform_shirt": "person1 wearing the dress",
                "not_uniform": "person1 not wearing the dress"}.get(types[0], "person1 needs manual review")
    group = "both persons" if len(types) == 2 else f"all {len(types)} persons"
    if all(t == "uniform_shirt" for t in types):
        return f"{group} are wearing the dress"
    if all(t == "not_uniform" for t in types):
        return f"{group} are not wearing the dress"
    nots = [i for i, t in enumerate(types, 1) if t == "not_uniform"]
    unsure = [i for i, t in enumerate(types, 1) if t not in ("uniform_shirt", "not_uniform")]
    parts = []
    if nots:
        parts.append(f"{_names(nots)} {'is' if len(nots) == 1 else 'are'} not wearing the dress")
    if unsure:
        parts.append(f"{_names(unsure)} {'needs' if len(unsure) == 1 else 'need'} manual review")
    if "uniform_shirt" in types:
        parts.append("others wearing")
    return ", ".join(parts)


def score(rec: dict) -> int | None:
    if rec["processing_status"] not in ("ok", "annotation_failed"):
        return None
    scores = [p["confidence"] for p in rec["people"] if p["confidence"] is not None]
    if not rec["people"]:
        return 0
    return round(100 * sum(scores) / len(scores)) if scores else None


def rows(cart: str, records: list[dict]) -> list[dict]:
    """Report rows for screenshot records of one cart (any order); also what each picN is."""
    def key(r):
        return (r["recording_timestamp"] or "~", r["source_video"], r["video_timestamp_seconds"])
    out, store, cid = [], store_name(cart), cart_id(cart)
    for n, r in enumerate(sorted(records, key=key), 1):
        when = datetime.fromisoformat(r["recording_timestamp"]) if r["recording_timestamp"] else None
        out.append({"cart_id": cid, "store": store, "date": when.replace(hour=0, minute=0, second=0, microsecond=0) if when else None,
                    "time": when.time().replace(microsecond=0) if when else f"video {r['video_timestamp_hhmmss'][:8]}",
                    "img": f"pic{n}", "score": score(r), "remark": f"{n}. {remark(r)}",
                    "source_video": r["source_video"], "image_filename": r["image_filename"],
                    "original_image_path": r["original_image_path"]})
    return out


def write_xlsx(path: Path, report_rows: list[dict]) -> Path:
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side

    line = Side(style="thin", color="FFBFBFBF")
    box = Border(left=line, right=line, top=line, bottom=line)
    body, center = Font(name="Arial", size=10, color="FF000000"), Alignment(horizontal="center")
    wb = Workbook()
    ws = wb.active
    ws.title = "Sheet1"
    ws.append(HEADERS)
    for c in ws[1]:
        c.alignment = center
        if c.column > 1:                                 # the sample leaves cart_id unstyled
            c.font = Font(name="Arial", size=10, bold=True, color="FFFFFFFF")
            c.fill = PatternFill("solid", fgColor="FF305496")
            c.border = box
    for r in report_rows:
        ws.append([r["cart_id"], r["store"], r["date"], r["time"], r["img"], r["score"], r["remark"]])
        row = ws[ws.max_row]
        row[0].alignment = center
        row[1].font = Font(name="Arial", size=10)
        for c in row[2:]:
            c.font, c.border = body, box
            if c.column < 7:
                c.alignment = center
        row[2].number_format = "dd\\-mm\\-yyyy"
        if not isinstance(r["time"], str):
            row[3].number_format = "h:mm:ss"
    for i, w in enumerate(WIDTHS):
        ws.column_dimensions[chr(65 + i)].width = w
    ws.freeze_panes = "A2"
    path.parent.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path
