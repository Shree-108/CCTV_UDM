# API design — AI microservice (`/api/v1`)

Status: **proposed contract.** Field names follow the brief where they fit the existing code; anything the
Admin team has not confirmed is marked *TBC*.

## Conventions

- Base URL: `https://<ai-service-host>` (internal). JSON, UTF-8. Times are ISO 8601 **with a time zone**
  (`2026-10-30T13:00:00+05:30`); a time without zone is rejected.
- Authentication: header `X-API-Key: <key>` on every `/api/v1` call (401 if missing / wrong). `/health` is
  open but returns no sensitive detail.
- Every response carries `X-Request-ID` (echoed from the request, or generated).
- Errors use one shape:

```json
{ "error": { "code": "invalid_time_range", "message": "end_time must be after start_time", "request_id": "b1f…" } }
```

| HTTP | `code` examples |
|---|---|
| 400 / 422 | `validation_error`, `invalid_time_range`, `prefix_not_allowed`, `sample_count_out_of_range` |
| 401 | `unauthorized` |
| 404 | `analysis_not_found` |
| 409 | `analysis_running` (retry while running) |
| 429 | `queue_full` |
| 503 | `model_not_ready`, `storage_unavailable` |

## `GET /health`

```json
{
  "status": "ok",
  "service": "yatrikart-uniform-ai",
  "version": "1.0.0",
  "mode": "service",
  "model": { "loaded": true, "name": "yatrikart-uniform-cls", "version": "0.2.0", "status": "experimental", "sha256_verified": true },
  "person_detector": { "loaded": true, "name": "yolo11n" },
  "s3": { "configured": true, "reachable": true, "bucket": "yk-cctv-prediction", "region": "ap-south-1" },
  "queue": { "queued": 0, "running": 1, "workers": 1 },
  "timestamp": "2026-10-30T14:00:02+05:30"
}
```

`status` is `ok`, `degraded` (e.g. S3 unreachable) or `error` (model not loaded). HTTP 200 for ok/degraded,
503 for error.

## `POST /api/v1/analyze`

Request:

```json
{
  "store_id": "cart_50",
  "store_name": "Store Name",
  "start_time": "2026-10-30T12:00:00+05:30",
  "end_time": "2026-10-30T13:00:00+05:30",
  "input_s3_prefix": "10-2026/30-10-2026/cart_50/input_data/",
  "output_s3_prefix": "10-2026/30-10-2026/cart_50/output_data/",
  "sample_count": 6,
  "random_seed": null,
  "force_new": false
}
```

Validation:

| Field | Rule |
|---|---|
| `store_id` | 1–64 chars, `[A-Za-z0-9._ -]` (same rule as today) |
| `store_name` | optional, ≤ 128 chars; blank → `Store Name` column blank |
| `start_time`, `end_time` | zone-aware, `end > start`, window ≤ 24 h (configurable), not in the future |
| `input_s3_prefix`, `output_s3_prefix` | relative keys in the configured bucket, must match `S3_ALLOWED_PREFIX_PATTERN`, no `..`, must end with `/`; the bucket is never taken from the request |
| `sample_count` | integer 1–60; default from Settings (currently **4**) |
| `random_seed` | optional 0–2³¹-1; generated and stored when omitted |
| `force_new` | `true` creates a new analysis even if an identical one exists |

Duplicate protection: the analysis key is `SHA-256(store_id, start_time, end_time, input_prefix,
output_prefix, sample_count, model_sha256)`. Optional header `Idempotency-Key` overrides it. An identical
request returns the existing analysis with **200** and `"duplicate": true`; a new one returns **202**.

Response (202):

```json
{
  "analysis_id": "an-20261030-140002-7c1e2a",
  "status": "queued",
  "duplicate": false,
  "store_id": "cart_50",
  "requested_window": { "start_time": "2026-10-30T12:00:00+05:30", "end_time": "2026-10-30T13:00:00+05:30" },
  "sample_count": 6,
  "random_seed": 1928374,
  "status_url": "/api/v1/analysis/an-20261030-140002-7c1e2a",
  "request_id": "b1f0…"
}
```

## `GET /api/v1/analysis/{analysis_id}`

```json
{
  "analysis_id": "an-20261030-140002-7c1e2a",
  "store_id": "cart_50",
  "store_name": "Store Name",
  "status": "completed",
  "stage": "done",
  "requested_window": { "start_time": "…", "end_time": "…" },
  "video_count": 1,
  "screenshots_planned": 6,
  "screenshot_count": 6,
  "screenshots_failed": 0,
  "person_count": 11,
  "compliant_count": 4,
  "non_compliant_count": 5,
  "uncertain_count": 2,
  "images_with_uniform": 4,
  "model_version": "yatrikart-uniform-cls v0.2.0 (experimental) · sha256 5d785e8bd947",
  "analysis_type": "sampled_frames",
  "csv_s3_key": "10-2026/30-10-2026/cart_50/output_data/an-20261030-140002-7c1e2a/reports/uniform_detection_report.csv",
  "manifest_s3_key": "…/manifests/analysis_metadata.json",
  "attempts": 1,
  "created_at": "…", "started_at": "…", "finished_at": "…",
  "error": null
}
```

`status`: `queued` → `running` → `completed` | `completed_with_errors` (some screenshots failed) |
`no_footage` (no video overlaps the window) | `failed` | `timed_out`.
`stage` while running: `listing_videos`, `reading_videos`, `analysing`, `writing_reports`, `uploading`.
Counts are **people in the sampled screenshots**, not unique people in the hour (the same person in two
screenshots counts twice).

## `GET /api/v1/analysis/{analysis_id}/results`

```json
{
  "analysis_id": "an-…",
  "csv_fields": ["Store Name", "Date", "Time Stamp", "img name", "confidence score", "remark (Example)"],
  "screenshots": [
    {
      "frame_id": "an-…-v01-f0004681",
      "video_s3_key": "…/input_data/cam01_20261030_1200.mp4",
      "frame_index": 4681,
      "video_offset_seconds": 187.24,
      "frame_time": "2026-10-30T12:03:07+05:30",
      "original_s3_key": "…/screenshots/original/an-…-v01-f0004681.png",
      "annotated_s3_key": "…/screenshots/annotated/an-…-v01-f0004681.jpg",
      "outcome": "UNIFORM_DETECTED",
      "processing_status": "ok",
      "csv_row": { "Store Name": "Store Name", "Date": "2026-10-30", "Time Stamp": "12:03:07", "img name": "an-…-v01-f0004681.png", "confidence score": "Person 1: 97.4%; Person 2: 12.0%", "remark (Example)": "Person 1 detected wearing the designated uniform shirt; Person 2 detected without the designated uniform shirt." },
      "people": [
        { "person_id": "an-…-v01-f0004681-p1", "prediction": "UNIFORM", "p_uniform": 0.974, "person_confidence": 0.84, "bbox": [846, 224, 1210, 609], "reason": "" },
        { "person_id": "an-…-v01-f0004681-p2", "prediction": "NON_UNIFORM", "p_uniform": 0.12, "person_confidence": 0.77, "bbox": [954, 200, 1364, 540], "reason": "" }
      ]
    }
  ],
  "missing_samples": [ { "planned_time": "2026-10-30T12:47:10+05:30", "reason": "no footage covers this time" } ]
}
```

`?presign=true` adds short-lived (default 15 min) presigned GET URLs for the screenshot keys; URLs are never
stored.

## `GET /api/v1/analyses`

Filters: `store_id`, `date` (YYYY-MM-DD, by window start in the service time zone), `month` (YYYY-MM),
`year` (YYYY), `status`, `limit` (≤ 200), `offset`. Returns `{ "total": n, "analyses": [ <status objects> ] }`.

## `POST /api/v1/analysis/{analysis_id}/retry`

Only for `failed`, `timed_out` or `completed_with_errors`. Creates a **new attempt** under the same
`analysis_id` (`attempts` + 1), writes to a new `attempt-N/` sub-folder, and keeps the earlier outputs.
409 while the analysis is queued or running; 200 with the status object otherwise.

## Limits and timeouts

| Setting | Default |
|---|---|
| Max request body | 16 KB |
| Queue size | 200 analyses (429 when full) |
| Workers | 1 (`SERVICE_WORKERS`) |
| Per-analysis timeout | 15 min (`SERVICE_ANALYSIS_TIMEOUT_S`) |
| Presigned URL lifetime | 900 s |

OpenAPI / Swagger: `/docs` and `/openapi.json`, disabled in service mode unless `SERVICE_EXPOSE_DOCS=true`.
