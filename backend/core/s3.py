"""S3 access for the cart batch endpoint (POST /api/s3/process).

Credentials come ONLY from the standard AWS provider chain: AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
(environment or the project's .env), ~/.aws/credentials, or an instance role - never from code or
config files. Region: AWS_REGION / AWS_DEFAULT_REGION (optional).

Bucket layout (one folder per day, one folder per cart):
    s3://<bucket>/<MM-YYYY>/<DD-MM-YYYY>/<cart>/input_data/     videos to analyse
    s3://<bucket>/<MM-YYYY>/<DD-MM-YYYY>/<cart>/output_data/    results written by this app
"""
from __future__ import annotations

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


def output_prefix(day: date, cart: str) -> str:
    return f"{day_prefix(day)}{cart}/output_data/"


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


def list_keys(s3c, bucket: str, prefix: str) -> list[str]:
    """Object keys under a prefix (all pages), without folder placeholder objects."""
    keys = []
    for page in s3c.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        keys += [o["Key"] for o in page.get("Contents", []) if not o["Key"].endswith("/")]
    return sorted(keys)
