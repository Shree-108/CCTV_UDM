"""S3 access for the cart batch endpoint (POST /api/s3/process).

Credentials come ONLY from the standard AWS provider chain: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
(environment or the project's .env), ~/.aws/credentials, or an instance role - never from code or
config files. Region: AWS_REGION / AWS_DEFAULT_REGION (optional).

Bucket layout (one folder per day, one folder per cart):
    s3://<bucket>/<MM-YYYY>/<DD-MM-YYYY>/<cart>/input_data/     videos to analyse
    s3://<bucket>/<MM-YYYY>/<DD-MM-YYYY>/<cart>/output_data/    results written by this app
"""
from __future__ import annotations

import base64
import os
import re
from datetime import date, datetime

import backend.core.config  # noqa: F401 - loads the project's .env (AWS_* variables)

DEFAULT_BUCKET = "yk-cctv-prediction"
CART_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


class S3Unavailable(RuntimeError):
    """S3 cannot be used (no credentials, no access, boto3 missing); the message is shown as is."""


def bucket_name() -> str:
    return os.environ.get("SU_S3_BUCKET", "").strip() or DEFAULT_BUCKET


def client():
    try:
        import boto3
    except ImportError:
        raise S3Unavailable("boto3 is not installed: run pip install -r requirements.txt")
    return boto3.client("s3", region_name=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or None)


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
FOLDER_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


def output_prefix(day: date, cart: str, folder: str = OUTPUT_FOLDER) -> str:
    return f"{day_prefix(day)}{cart}/{folder}/"


def other_output_folder(value: str) -> str:
    """An opt-in folder next to output_data for test runs (e.g. "output_data_test"), never the production
    output_data folder or the input_data folder."""
    v = value.strip()
    if not FOLDER_PATTERN.match(v) or v.lower() in (OUTPUT_FOLDER, "input_data"):
        raise ValueError("output_folder must be 1-40 letters, digits, _ or - and not output_data / input_data "
                         f"(got {value!r})")
    return v


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
