"""Admin login / logout / session check (backend/core/auth.py)."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel, Field

from backend.core import auth

router = APIRouter(prefix="/api/auth")


class LoginIn(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


@router.post("/login")
def login(request: Request, response: Response, body: LoginIn):
    if not auth.configured():
        raise HTTPException(503, "Admin login is not configured on the server (SU_ADMIN_PASSWORD_HASH).")
    client = request.client.host if request.client else "unknown"
    if not auth.check_login(client, body.username, body.password):
        raise HTTPException(401, "Wrong user name or password.")
    response.set_cookie(auth.COOKIE, auth.make_token(auth.username()), max_age=int(auth.session_hours() * 3600),
                        httponly=True, secure=auth.cookie_secure(), samesite="strict", path="/")
    return {"username": auth.username()}


@router.post("/logout")
def logout(response: Response):
    response.delete_cookie(auth.COOKIE, path="/", httponly=True, secure=auth.cookie_secure(), samesite="strict")
    return {"ok": True}


@router.get("/me")
def me(request: Request):
    """200 {username} when logged in; 401 otherwise (503 when no admin account is configured)."""
    return {"username": auth.require_admin(request)}
