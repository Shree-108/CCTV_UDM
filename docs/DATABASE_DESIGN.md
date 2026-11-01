# Database and storage design

## 1. Who stores what

| Layer | Owner | Holds | Does not hold |
|---|---|---|---|
| **AWS S3** `yk-cctv-prediction` | shared | input CCTV videos; per-analysis original + annotated screenshots, CSV reports, manifest JSON | databases, secrets |
| **AI service SQLite** `storage/sightline_uniform.db` (on the EC2 instance) | AI service | analysis IDs, status, attempts, video ↔ analysis ↔ screenshot ↔ prediction links, errors | videos (only temporary copies while processing) |
| **MongoDB** | Admin backend | per store × hour summaries for the Admin dashboard, report S3 keys | screenshots, full per-person records, videos |

## 2. Existing SQLite schema (unchanged)

| Table | Purpose | Key columns |
|---|---|---|
| `media` | one row per received file | `id`, `filename`, `path`, `size_bytes`, `fps`, `frames`, `duration_sec`, `sha256` |
| `extractions` | one quick-analysis job (one video) | `id` (`ext-…`), `media_id`, `state`, `planned`, `extracted`, `sampling_strategy`, `sample_count`, `random_seed`, `store_id`, `camera_id`, `recording_start_time`, `plan` (JSON) |
| `frame_detection_runs` | one model run over a job's screenshots | `id` (`det-…`), `extraction_id`, `state`, `threshold`, `model_version`, `model_sha256`, counts |
| `analyses`, `persons`, `alerts` | full-video pipeline | — |
| `cameras`, `captures` | scheduled capture | — |

Per-screenshot and per-person details live in files next to the job (`image_manifest.csv`,
`detection_results.csv`), keyed by `image_id` = `<extraction_id>-R<nnnn>`.

## 3. New tables (additive migration, automatic backup first)

```sql
CREATE TABLE service_analyses (
  id TEXT PRIMARY KEY,                 -- an-YYYYMMDD-HHMMSS-xxxxxx
  idempotency_key TEXT NOT NULL,       -- SHA-256 of the request identity (API_DESIGN.md)
  request_id TEXT,
  store_id TEXT NOT NULL, store_name TEXT,
  window_start TEXT NOT NULL, window_end TEXT NOT NULL,   -- ISO 8601 with offset
  input_prefix TEXT NOT NULL, output_prefix TEXT NOT NULL,
  sample_count INTEGER NOT NULL, random_seed INTEGER NOT NULL,
  status TEXT NOT NULL,                -- queued | running | completed | completed_with_errors | no_footage | failed | timed_out
  stage TEXT, error TEXT, attempts INTEGER NOT NULL DEFAULT 1,
  video_count INTEGER, screenshots_planned INTEGER, screenshots_ok INTEGER, screenshots_failed INTEGER,
  person_count INTEGER, uniform_count INTEGER, non_uniform_count INTEGER, uncertain_count INTEGER,
  model_version TEXT, model_sha256 TEXT,
  csv_s3_key TEXT, person_csv_s3_key TEXT, manifest_s3_key TEXT,
  created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT
);
CREATE UNIQUE INDEX ux_service_idem ON service_analyses(idempotency_key) WHERE status NOT IN ('failed','timed_out');
CREATE INDEX ix_service_store_window ON service_analyses(store_id, window_start);

CREATE TABLE service_analysis_videos (
  analysis_id TEXT NOT NULL, attempt INTEGER NOT NULL,
  s3_key TEXT NOT NULL, etag TEXT, size_bytes INTEGER,
  recording_start TEXT, duration_sec REAL, fps REAL, frames INTEGER,
  media_id TEXT, extraction_id TEXT, detection_run_id TEXT,   -- links into the existing tables
  status TEXT NOT NULL, error TEXT,
  PRIMARY KEY (analysis_id, attempt, s3_key)
);
```

The partial unique index stops two live analyses for the same request; a failed one can be retried or
re-requested.

## 4. Identifier chain (CSV row → evidence)

```
analysis_id  an-20261030-140002-7c1e2a
 └─ video     s3_key …/input_data/cam01_20261030_1200.mp4  (+ ETag)      → service_analysis_videos
     └─ job    extraction_id ext-…  (existing)                            → extractions
         └─ frame_id  an-…-v01-f0004681   (= screenshot file stem)        → image_manifest.csv (image_id, frame_number, timestamp)
             ├─ original_s3_key   …/screenshots/original/an-…-v01-f0004681.png
             ├─ annotated_s3_key  …/screenshots/annotated/an-…-v01-f0004681.jpg
             ├─ business CSV row  (img name = an-…-v01-f0004681.png)
             └─ person_id  an-…-v01-f0004681-p1 … pN                       → detection_results.csv / person_results.csv
```

The business CSV keeps its six columns. The stable IDs are in `person_results.csv` and
`analysis_metadata.json`, so the Admin import never has to rely on file names alone.

## 5. S3 layout

Input (written by the recorder side, read-only for the AI service):

```
{MM-YYYY}/{DD-MM-YYYY}/{store_id}/input_data/<video files>      e.g. 10-2026/30-10-2026/cart_50/input_data/
```

Output (written by the AI service only; nothing is overwritten):

```
{output_prefix}{analysis_id}/
   screenshots/original/{frame_id}.png
   screenshots/annotated/{frame_id}.jpg
   reports/uniform_detection_report.csv      six business columns, one row per screenshot
   reports/person_results.csv                one row per person, with all stable IDs
   manifests/analysis_metadata.json          request, videos, plan, seed, model, counts, object keys + SHA-256
{output_prefix}{analysis_id}/attempt-2/...    only for retries
```

Dates in prefixes come from the request (the caller decides the folder); the service never hardcodes them.

## 6. MongoDB (Admin side) — proposed summary document

Collection `uniform_hourly_summaries` (one document per analysis; verify against the Admin app):

```json
{
  "_id": "an-20261030-140002-7c1e2a",
  "store_id": "cart_50",
  "store_name": "Store Name",
  "analysis_id": "an-20261030-140002-7c1e2a",
  "date": "2026-10-30",
  "hour": 12,
  "window_start": "2026-10-30T12:00:00+05:30",
  "window_end": "2026-10-30T13:00:00+05:30",
  "analysis_type": "sampled_frames",
  "screenshots": 6,
  "total_people_detected": 11,
  "compliant_count": 4,
  "non_compliant_count": 5,
  "uncertain_count": 2,
  "compliance_percentage": 44.4,
  "report_s3_key": "…/reports/uniform_detection_report.csv",
  "manifest_s3_key": "…/manifests/analysis_metadata.json",
  "model_version": "0.2.0",
  "analysis_status": "completed",
  "created_at": "2026-10-30T14:01:10+05:30",
  "updated_at": "2026-10-30T14:01:10+05:30"
}
```

- `compliance_percentage = compliant / (compliant + non_compliant) × 100`; **uncertain people are excluded**
  from the denominator and shown separately; `null` when both are 0. (Needs the business's confirmation.)
- Counts are people **in the sampled screenshots**, not unique people in the hour.
- Indexes: unique on `analysis_id`; `{store_id: 1, date: 1, hour: 1}`; `{date: 1}`.
- Import is an **upsert by `analysis_id`**, so running cron job 2 twice changes nothing. A separate
  `uniform_import_log` collection records `analysis_id`, `csv_etag`, `imported_at`, `status`, `error`.
