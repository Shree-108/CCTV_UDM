"""Analysis reports: JSON (complete), CSV (one row per person) and a plain-text summary."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from backend.analysis import compliance


def _clock(sec) -> str:
    if sec is None:
        return "—"
    sec = int(round(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def write_reports(db, aid: str, out_dir: Path, annotated: Path | None, keyframe: Path | None) -> dict:
    a = db.analysis(aid)
    persons = db.persons(aid)
    alerts, _ = db.alerts(limit=500, analysis_id=aid)
    files = {"annotated": str(annotated) if annotated else None, "keyframe": str(keyframe) if keyframe else None}

    data = {"analysis": {k: v for k, v in a.items() if k not in ("live", "files")},
            "status_title": compliance.TITLE.get(a["status"], a["status"]),
            "segments_sec": (a.get("live") or {}).get("segments"),
            "persons": persons, "alerts": alerts,
            "note": "Clothing only. The system does not identify who a person is."}
    jp = out_dir / "report.json"
    jp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    cp = out_dir / "people.csv"
    cols = ["track_id", "label", "confidence", "det_confidence", "first_seen_sec", "last_seen_sec", "frames",
            "uniform_votes", "non_uniform_votes", "uncertain_votes", "note"]
    with open(cp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(persons)

    area = a.get("area") or {}
    avg = "—" if a["avg_uniform_conf"] is None else f"{a['avg_uniform_conf'] * 100:.1f}%"
    lines = [
        "SIGHTLINE · UNIFORM COMPLIANCE REPORT", "=" * 46,
        f"Analysis ID        {aid}",
        f"File               {a['filename']} ({a['kind']})",
        f"Analysed at        {a['created_at']}",
        f"Camera / area      {area.get('label', '—')}" + ("  + store ROI" if area.get("roi") else ""),
        "",
        f"FINAL STATUS       {compliance.TITLE.get(a['status'], a['status'])}",
        f"                   {compliance.message(a['status'], a['uniformed'] or 0, a['people'] or 0)}",
        "",
        f"People detected    {a['people']}",
        f"Uniformed          {a['uniformed']}",
        f"Non-uniform        {a['non_uniform']}",
        f"Uncertain          {a['uncertain']}",
        f"Avg uniform conf.  {avg}",
        f"Alerts             {a['alerts']}",
    ]
    if a["kind"] == "video":
        lines += [
            f"Video duration     {_clock(a['duration_sec'])}",
            f"Processing time    {a['processing_sec']} s  ({a['processing_fps']} analysed frames/s on {str(a['device']).upper()})",
            f"Frames processed   {a['frames_processed']} of {a['frames_total']}",
            f"Longest confirmed uniform presence  {_clock(a['longest_presence_sec'])}",
        ]
    lines += ["", f"Uniform model      {a.get('model_version') or a['classifier']}", "", "PEOPLE", "-" * 46]
    for p in persons:
        seen = f"  seen {_clock(p['first_seen_sec'])}–{_clock(p['last_seen_sec'])}" if p["first_seen_sec"] is not None else ""
        lines.append(f"Person #{p['track_id']:02d}  {p['label'].replace('_', '-').upper():12s} {p['confidence'] * 100:5.1f}%{seen}")
    if alerts:
        lines += ["", "ALERTS", "-" * 46]
        for al in alerts:
            at = f" @ {_clock(al['media_time_sec'])}" if al["media_time_sec"] is not None else ""
            lines.append(f"[{al['severity'].upper()}] {al['title']}{at} - {al['message']}")
    lines += ["", "Clothing only. The system does not identify who a person is."]
    tp = out_dir / "report.txt"
    tp.write_text("\n".join(lines), encoding="utf-8")
    files.update(report_json=str(jp), report_csv=str(cp), report_txt=str(tp))
    return files
