"""S3 access for the cart batch endpoint (POST /api/s3/process).

Credentials, bucket and region come ONLY from the project's .env (loaded into the environment by
backend.core.config): AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY (+ AWS_SESSION_TOKEN if used), SU_S3_BUCKET,
AWS_REGION / AWS_DEFAULT_REGION. They are passed to boto3 explicitly (no ~/.aws files, no instance role) and
are never logged, returned or written anywhere.

Bucket layout:
    s3://<bucket>/<MM-YYYY>/<DD-MM-YYYY>/<cart>/input_data/   videos to analyse (the recording's day)
    s3://<bucket>/<MM-YYYY>/<DD-MM-YYYY>/card_<N>/output_data/  results, filed under the PROCESSING day
                                                              (today, Asia/Kolkata - never the video's date)
        card<N>_YYYY_MM_DD.csv          the day's CSV report; the day's first run creates it, every later run
                                        appends its rows (append_csv) - earlier rows are never changed
        card<N>_YYYY_MM_DD_pic<K>.jpg   the detection image of CSV row pic<K> (put_new: never replaced)
    S3 has no real folders: the month / day / card_<N> / output_data "folders" are created by the first object
    written under them and reused after. Nothing is ever deleted.
"""
from __future__ import annotations

import base64
import os
import re
from datetime import date, datetime, timedelta, timezone

import backend.core.config  # noqa: F401 - loads the project's .env (AWS_* variables)

DEFAULT_BUCKET = "yk-cctv-prediction"
CART_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class S3Unavailable(RuntimeError):
    """S3 cannot be used (no credentials, no access, boto3 missing); the message is shown as is."""


def bucket_name() -> str:
    return os.environ.get("SU_S3_BUCKET", "").strip() or DEFAULT_BUCKET


def client():
    """An S3 client built from the .env values only. Raises S3Unavailable when a key is missing."""
    try:
        import boto3
    except ImportError:
        raise S3Unavailable("boto3 is not installed: run pip install -r requirements.txt")
    key_id = os.environ.get("AWS_ACCESS_KEY_ID", "").strip()
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip()
    if not key_id or not secret:
        raise S3Unavailable("No AWS credentials found: set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY (and AWS_REGION) "
                            "in the project's .env, then retry.")
    return boto3.client("s3", aws_access_key_id=key_id, aws_secret_access_key=secret,
                        aws_session_token=os.environ.get("AWS_SESSION_TOKEN", "").strip() or None,
                        region_name=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or None)


IST = timezone(timedelta(hours=5, minutes=30), "Asia/Kolkata")     # India has no daylight saving time


def today() -> date:
    """The processing day the output is filed under: today's date in Asia/Kolkata, whatever the video's date."""
    return datetime.now(IST).date()


def parse_day(value: str) -> date:
    """The day folder as written in the bucket, DD-MM-YYYY (e.g. 30-10-2026)."""
    try:
        return datetime.strptime(value.strip(), "%d-%m-%Y").date()
    except ValueError:
        raise ValueError(f"date must be DD-MM-YYYY, e.g. 30-10-2026 (got {value!r})")


def cart_name(value: str) -> str:
    """'cart_78' as is; a bare number '78' becomes 'cart_78'."""
    v = value.strip()
    if v.isdigit():
        v = f"cart_{v}"
    if not CART_PATTERN.match(v):
        raise ValueError(f"cart may contain letters, digits, '_' and '-' only (got {value!r})")
    return v


def day_prefix(day: date) -> str:
    return f"{day:%m-%Y}/{day:%d-%m-%Y}/"


def input_prefix(day: date, cart: str) -> str:
    return f"{day_prefix(day)}{cart}/input_data/"


OUTPUT_FOLDER = "output_data"


def card_number(cart: str) -> str:
    """cart_50 -> "50" (the cart's number; a name without digits is kept as is)."""
    digits = re.sub(r"\D", "", cart)
    return digits or cart


def output_prefix(day: date, cart: str) -> str:
    """The processing day's output folder: MM-YYYY/DD-MM-YYYY/card_<N>/output_data/ (e.g.
    11-2026/09-11-2026/card_50/output_data/) - the same month / day folders as the input videos (day_prefix),
    so an existing month folder is reused, never duplicated. The earlier YYYY_MM/DD/ layout is no longer
    written to (its files are left as they are)."""
    return f"{day_prefix(day)}card_{card_number(cart)}/{OUTPUT_FOLDER}/"


def check(s3c, bucket: str) -> dict:
    """Whether credentials exist and the bucket can be reached. Never returns the credentials themselves."""
    from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError, PartialCredentialsError
    try:
        s3c.head_bucket(Bucket=bucket)
    except (NoCredentialsError, PartialCredentialsError):
        return {"ok": False, "bucket": bucket, "credentials": False,
                "message": "No AWS credentials found. Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY (and AWS_REGION) "
                           "in the environment or .env, or configure ~/.aws/credentials."}
    except ClientError as e:
        code = str(e.response.get("Error", {}).get("Code", ""))
        msg = {"403": f"The AWS credentials have no access to bucket {bucket}.",
               "404": f"Bucket {bucket} does not exist (or is in another account)."}.get(
            code, f"S3 refused the request for bucket {bucket}: {code or e}")
        return {"ok": False, "bucket": bucket, "credentials": True, "message": msg}
    except BotoCoreError as e:                       # network, endpoint, region
        return {"ok": False, "bucket": bucket, "credentials": None, "message": f"S3 could not be reached: {e}"}
    return {"ok": True, "bucket": bucket, "credentials": True, "message": f"Credentials found; bucket {bucket} is reachable."}


SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str) -> str:
    """The base name of an uploaded file reduced to letters, digits, '.', '_' and '-' (no folders, no
    leading dots), at most 120 characters, extension kept. NVR names (20261104_..._192126.mp4) are unchanged."""
    base = name.replace("\\", "/").rsplit("/", 1)[-1]
    stem, dot, ext = base.rpartition(".")
    if not dot:
        stem, ext = base, ""
    stem = SAFE_NAME.sub("_", stem).strip("._-")[:100] or "video"
    ext = SAFE_NAME.sub("", ext)[:10].lower()
    return f"{stem}.{ext}" if ext else stem


class ObjectConflict(RuntimeError):
    """A DIFFERENT object already exists under the key; it is never overwritten."""


def _b64_sha256(sha256_hex: str) -> str:
    return base64.b64encode(bytes.fromhex(sha256_hex)).decode()


def object_sha256(s3c, bucket: str, key: str) -> tuple[int, str | None] | None:
    """(size, SHA-256 hex computed by S3) of an object, None when it does not exist. The checksum is only
    known for objects written with ChecksumSHA256 in one PUT (as put_new does)."""
    from botocore.exceptions import ClientError
    try:
        h = s3c.head_object(Bucket=bucket, Key=key, ChecksumMode="ENABLED")
    except ClientError as e:
        if str(e.response.get("Error", {}).get("Code", "")) in ("404", "NoSuchKey", "NotFound"):
            return None
        raise
    cs = h.get("ChecksumSHA256") or ""
    sha = base64.b64decode(cs).hex() if cs and "-" not in cs else None
    return int(h.get("ContentLength", -1)), sha


def put_new(s3c, bucket: str, key: str, path, sha256_hex: str, content_type: str) -> bool:
    """Write a local file to S3 ONLY if the key is free (conditional PUT, If-None-Match: *). S3 checks the
    body against the SHA-256 sent with it. Returns True when written, False when the identical file (same
    SHA-256) is already there; raises ObjectConflict when a different object has the key."""
    from botocore.exceptions import ClientError
    try:
        with open(path, "rb") as f:
            s3c.put_object(Bucket=bucket, Key=key, Body=f, ContentType=content_type, IfNoneMatch="*",
                           ChecksumSHA256=_b64_sha256(sha256_hex), Metadata={"sha256": sha256_hex})
        return True
    except ClientError as e:
        if str(e.response.get("Error", {}).get("Code", "")) not in (
                "PreconditionFailed", "412", "ConditionalRequestConflict", "409"):
            raise
    existing = object_sha256(s3c, bucket, key)
    if existing and existing[1] == sha256_hex:
        return False
    raise ObjectConflict(f"s3://{bucket}/{key} already exists with different content; it was not overwritten. "
                         "Rename the file if it is a different recording.")


_CONFLICT_CODES = ("PreconditionFailed", "412", "ConditionalRequestConflict", "409")


def daily_csv_name(day: date, cart: str) -> str:
    """The day's CSV report: card<N>_YYYY_MM_DD.csv (e.g. card50_2026_11_08.csv)."""
    return f"card{card_number(cart)}_{day:%Y_%m_%d}.csv"


def daily_csv_key(day: date, cart: str) -> str:
    return output_prefix(day, cart) + daily_csv_name(day, cart)


def _get(s3c, bucket: str, key: str) -> tuple[bytes, str] | None:
    """(body, ETag) of an object, None when it does not exist."""
    from botocore.exceptions import ClientError
    try:
        r = s3c.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if str(e.response.get("Error", {}).get("Code", "")) in ("404", "NoSuchKey", "NotFound"):
            return None
        raise
    return r["Body"].read(), r["ETag"]


PIC = re.compile(r"(?:^|_)pic(\d+)(?:_[^/]*)?(?:\.\w+)?$")


def highest_pic(values) -> int:
    """Highest K of pic<K> among daily-CSV rows (lists: the "img name" column, index 4) or file names / keys
    (strings); 0 when there is none."""
    best = 0
    for v in values:
        for cell in (v[4:5] if isinstance(v, list) else [v.rsplit("/", 1)[-1]]):
            m = PIC.search(str(cell).strip())
            if m:
                best = max(best, int(m.group(1)))
    return best


def append_csv(s3c, bucket: str, key: str, header: list[str], make_rows, local, *,
               on_prepared=None, already: str | None = None, attempts: int = 5, floor: int = 0) -> tuple[int, str]:
    """Append rows to the CSV at `key` (S3 has no append: read, add, conditional write back).

    Numbering: `make_rows(first)` gets first = 1 + the highest of: the number of rows already in the file, the
    highest pic<K> in its "img name" column, and `floor` (e.g. the highest pic<K> image already in the folder),
    so a new row never reuses an existing picture number.

    The existing object's bytes are kept exactly; its header must equal `header` (otherwise nothing is
    written). `make_rows(first)` returns the new rows as lists, numbered from `first` (the number of rows
    already in the file + 1). The write is conditional on the object being unchanged since it was read
    (If-Match on its ETag, If-None-Match when it did not exist), so a concurrent writer's rows are never
    lost: on a conflict the file is read again. The merged file is written to `local` first and
    `on_prepared(size, sha256)` is called before the PUT, so a caller can remember what it is about to write;
    `already` is such a SHA-256 from an earlier attempt: when the object has exactly that content - or starts
    with it (`local` still holds it: that write landed and other runs appended after it) - this append already
    happened and is not repeated (no duplicate rows on retries). Returns (size, SHA-256) of the object."""
    import csv
    import hashlib
    import io
    from pathlib import Path

    from botocore.exceptions import ClientError
    local = Path(local)
    previous = None                                       # what an earlier attempt wrote, when local still has it
    if already and local.is_file():
        b = local.read_bytes()
        previous = b if hashlib.sha256(b).hexdigest() == already else None
    for _ in range(attempts):
        got = _get(s3c, bucket, key)
        body, etag = got if got else (b"", None)
        if got and already and hashlib.sha256(body).hexdigest() == already:
            return len(body), already
        if got and previous and body.startswith(previous):  # landed earlier; other runs appended after it
            return len(body), hashlib.sha256(body).hexdigest()
        existing = list(csv.reader(io.StringIO(body.decode("utf-8-sig"))))
        if existing and existing[0] != header:
            raise ObjectConflict(f"s3://{bucket}/{key} exists with other columns ({existing[0]}); it was not changed")
        buf = io.StringIO()
        w = csv.writer(buf)
        if not existing:
            w.writerow(header)
        elif body and not body.endswith(b"\n"):
            buf.write("\r\n")
        w.writerows(make_rows(max(len(existing) - 1, highest_pic(existing[1:]), floor, 0) + 1))
        data = body + buf.getvalue().encode("utf-8")
        sha = hashlib.sha256(data).hexdigest()
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(data)
        if on_prepared:
            on_prepared(len(data), sha)
        cond = {"IfMatch": etag} if etag else {"IfNoneMatch": "*"}
        try:
            s3c.put_object(Bucket=bucket, Key=key, Body=data, ContentType="text/csv", **cond,
                           ChecksumSHA256=_b64_sha256(sha), Metadata={"sha256": sha})
            return len(data), sha
        except Exception as e:
            if isinstance(e, ClientError) and str(e.response.get("Error", {}).get("Code", "")) in _CONFLICT_CODES:
                continue                                  # changed since it was read: merge again
            try:                                          # the write may have landed (e.g. the response was lost)
                landed = object_sha256(s3c, bucket, key) == (len(data), sha)
            except Exception:  # noqa: BLE001 - the original error is the one to report
                landed = False
            if landed:
                return len(data), sha
            raise
    raise RuntimeError(f"s3://{bucket}/{key} kept changing while appending; retry")


def verify(s3c, bucket: str, key: str, size: int, sha256_hex: str) -> None:
    """Raises RuntimeError unless the object exists with exactly this size and SHA-256."""
    got = object_sha256(s3c, bucket, key)
    if got is None:
        raise RuntimeError(f"s3://{bucket}/{key} is not in S3 after the upload")
    if got != (size, sha256_hex):
        raise RuntimeError(f"s3://{bucket}/{key} does not match the local file "
                           f"(size {got[0]} vs {size}, sha256 {got[1] or 'unknown'} vs {sha256_hex})")


def list_keys(s3c, bucket: str, prefix: str) -> list[str]:
    """Object keys under a prefix (all pages), without folder placeholder objects."""
    keys = []
    for page in s3c.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys += [o["Key"] for o in page.get("Contents", []) if not o["Key"].endswith("/")]
    return sorted(keys)
