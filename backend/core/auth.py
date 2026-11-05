"""Admin login for the S3 upload workflow (backend/api/routes_auth.py, backend/api/routes_s3_uploads.py).

One admin account, configured on the server only (environment or the git-ignored .env):
    SU_ADMIN_USERNAME        default "admin"
    SU_ADMIN_PASSWORD_HASH   pbkdf2_sha256$<iterations>$<salt hex>$<hash hex>; create it with
                             python -m backend.core.auth            (asks for the password, prints the line)
    SU_SESSION_SECRET        random string that signs the session cookie (python -c "import secrets;
                             print(secrets.token_urlsafe(48))"); without it sessions end when the server restarts
    SU_SESSION_HOURS         session length, default 12
    SU_COOKIE_SECURE         1 = cookie only over HTTPS (set it behind HTTPS); default 0 for local http

Without SU_ADMIN_PASSWORD_HASH the protected routes refuse every request (503) - they never fall open.
The session is an HMAC-signed, HttpOnly, SameSite=Strict cookie; the password is never stored or returned.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time

from fastapi import HTTPException, Request

import backend.core.config  # noqa: F401 - loads the project's .env

COOKIE = "yk_admin_session"
ITERATIONS = 600_000
MAX_FAILURES, LOCKOUT_SECONDS = 5, 60
_PROCESS_SECRET = secrets.token_bytes(32)          # used only when SU_SESSION_SECRET is not set
_failures: dict[str, list[float]] = {}
_failures_lock = threading.Lock()


def hash_password(password: str, iterations: int = ITERATIONS) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iterations, salt, digest = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        got = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iterations))
        return hmac.compare_digest(got.hex(), digest)
    except (ValueError, TypeError):
        return False


def username() -> str:
    return os.environ.get("SU_ADMIN_USERNAME", "").strip() or "admin"


def configured() -> bool:
    return bool(os.environ.get("SU_ADMIN_PASSWORD_HASH", "").strip())


def session_hours() -> float:
    try:
        return max(0.1, float(os.environ.get("SU_SESSION_HOURS", "12")))
    except ValueError:
        return 12.0


def cookie_secure() -> bool:
    return os.environ.get("SU_COOKIE_SECURE", "0").strip().lower() in ("1", "true", "yes")


def _secret() -> bytes:
    s = os.environ.get("SU_SESSION_SECRET", "").strip()
    return s.encode() if s else _PROCESS_SECRET


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def make_token(user: str, now: float | None = None) -> str:
    body = _b64(json.dumps({"u": user, "exp": int((now or time.time()) + session_hours() * 3600)}).encode())
    return f"{body}.{_b64(hmac.new(_secret(), body.encode(), hashlib.sha256).digest())}"


def read_token(token: str | None, now: float | None = None) -> str | None:
    """The user name in a valid, unexpired token signed for the CURRENT admin user, else None."""
    if not token or token.count(".") != 1:
        return None
    body, sig = token.split(".")
    try:
        if not hmac.compare_digest(_unb64(sig), hmac.new(_secret(), body.encode(), hashlib.sha256).digest()):
            return None
        data = json.loads(_unb64(body))
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or data.get("exp", 0) < (now or time.time()) or data.get("u") != username():
        return None
    return data["u"]


def check_login(client: str, user: str, password: str) -> bool:
    """Constant-time check; after MAX_FAILURES wrong attempts from one client it is locked for a while (429)."""
    now = time.time()
    with _failures_lock:
        recent = [t for t in _failures.get(client, []) if now - t < LOCKOUT_SECONDS]
        _failures[client] = recent
        if len(recent) >= MAX_FAILURES:
            raise HTTPException(429, f"Too many failed logins. Try again in {LOCKOUT_SECONDS} seconds.")
    ok_user = hmac.compare_digest(user.strip().encode(), username().encode())
    ok_pass = verify_password(password, os.environ.get("SU_ADMIN_PASSWORD_HASH", "").strip())
    if ok_user and ok_pass:
        with _failures_lock:
            _failures.pop(client, None)
        return True
    with _failures_lock:
        _failures.setdefault(client, []).append(now)
    return False


def require_admin(request: Request) -> str:
    """FastAPI dependency: the logged-in admin's user name, or 401 / 503."""
    if not configured():
        raise HTTPException(503, "Admin login is not configured on the server (SU_ADMIN_PASSWORD_HASH).")
    user = read_token(request.cookies.get(COOKIE))
    if not user:
        raise HTTPException(401, "Please log in as administrator.")
    return user


if __name__ == "__main__":                         # python -m backend.core.auth
    import getpass
    pw = getpass.getpass("New admin password: ")
    if len(pw) < 10 or pw != getpass.getpass("Repeat: "):
        raise SystemExit("Passwords differ or are shorter than 10 characters.")
    print(f"SU_ADMIN_PASSWORD_HASH={hash_password(pw)}")
