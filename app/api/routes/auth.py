"""
auth.py

Public registration/login for people using the API/docs — separate
from the single admin account in admin.py. Gates access to /docs so
anyone visiting must sign in first; the admin panel can then see who's
registered and what they've done.

Also issues JWT bearer tokens via /token for programmatic clients that
can't hold a browser session cookie.

RBAC: get_current_user() returns just the username (unchanged, so
every existing route keeps working). require_role(min_role) builds on
top of it and additionally checks the caller's role — either from the
session (looked up fresh via user_store) or from the "role" claim
embedded in the JWT access token at issue time. Use require_role() as
a route dependency to gate an endpoint to investigator/admin, etc.:

    @router.post("/search-face/")
    def search_face(..., user: str = Depends(require_role("analyst"))):
        ...
"""

from typing import Optional

from fastapi import APIRouter, Request, Form, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.services.user_store import (
    create_user, verify_user, user_exists, is_locked_out,
    get_user_role, has_role_at_least, DEFAULT_ROLE, ROLE_RANK,
)
from app.services.jwt_utils import (
    create_access_token,
    decode_access_token,
    decode_access_token_payload,
    create_refresh_token,
    decode_refresh_token,
    revoke_refresh_token,
)
from pydantic import BaseModel

router = APIRouter(include_in_schema=False)
templates = Jinja2Templates(directory="app/templates")

USER_SESSION_KEY = "user_authenticated"
USERNAME_SESSION_KEY = "username"

token_router = APIRouter(tags=["auth"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token", auto_error=False)

# Same in-memory, IP-keyed rate limiting strategy as admin.py / main.py.
limiter = Limiter(key_func=get_remote_address)


def _lockout_message(username: str) -> Optional[str]:
    seconds = is_locked_out(username)
    if seconds is None:
        return None
    minutes = max(1, seconds // 60)
    return f"Too many failed attempts. Try again in about {minutes} minute(s)."


@token_router.post("/token")
@limiter.limit("5/minute")
def issue_token(request: Request, form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Exchanges username/password for a JWT bearer token, for
    programmatic clients that can't hold a browser session cookie.
    Use the returned access_token as: Authorization: Bearer <token>
    The issued access token embeds the user's current role.
    """
    lockout_msg = _lockout_message(form_data.username)
    if lockout_msg:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=lockout_msg)

    if not verify_user(form_data.username, form_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
        )
    access_token = create_access_token(form_data.username)
    refresh_token = create_refresh_token(form_data.username)
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
    }


class RefreshRequest(BaseModel):
    refresh_token: str


@token_router.post("/token/refresh")
@limiter.limit("5/minute")
def refresh_token_endpoint(request: Request, body: RefreshRequest):
    """
    Exchanges a still-valid refresh token for a new access token,
    without requiring the client to re-send username/password.
    Use like: POST /token/refresh {"refresh_token": "<refresh token>"}
    The new access token picks up the user's CURRENT role at refresh
    time — this is how a role change eventually propagates to a client
    holding an old access token, once it expires and is refreshed.
    """
    username = decode_refresh_token(body.refresh_token)
    if not username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token. Sign in again via /token.",
        )
    new_access_token = create_access_token(username)
    return {"access_token": new_access_token, "token_type": "bearer"}


@token_router.post("/token/revoke")
@limiter.limit("5/minute")
def revoke_token_endpoint(request: Request, body: RefreshRequest):
    """
    Revokes a refresh token so it can no longer be exchanged for new
    access tokens — the JWT equivalent of "logging out" a programmatic
    client. The access token(s) already issued from it stay valid until
    their own (short) expiry, but no new ones can be minted from it.
    Use like: POST /token/revoke {"refresh_token": "<refresh token>"}
    """
    revoked = revoke_refresh_token(body.refresh_token)
    if not revoked:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Malformed or invalid refresh token — nothing to revoke.",
        )
    return {"status": "revoked"}


async def get_current_user(request: Request, token: str = Depends(oauth2_scheme)) -> str:
    """
    Combined auth dependency: accepts EITHER a valid session cookie
    (browser/Swagger UI login) OR a valid JWT bearer token
    (programmatic clients). Use this on endpoints that should work
    both ways instead of only checking the session. Returns just the
    username — use require_role() instead if the endpoint needs a
    minimum role.
    """
    if request.session.get(USER_SESSION_KEY):
        return request.session.get(USERNAME_SESSION_KEY, "unknown")

    if token:
        username = decode_access_token(token)
        if username:
            return username

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated. Sign in via /login or obtain a token via /token.",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def _get_current_user_and_role(request: Request, token: str = Depends(oauth2_scheme)) -> tuple[str, str]:
    """
    Like get_current_user, but also returns the caller's role.
    - Session auth: role is looked up fresh from user_store on every
      request, so a role change takes effect on the caller's very next
      request (no stale-role window for browser/session users).
    - Bearer token auth: role comes from the "role" claim embedded in
      the access token at issue time — see the staleness note in
      jwt_utils.py's docstring.
    """
    if request.session.get(USER_SESSION_KEY):
        username = request.session.get(USERNAME_SESSION_KEY, "unknown")
        role = get_user_role(username) or DEFAULT_ROLE
        return username, role

    if token:
        payload = decode_access_token_payload(token)
        if payload:
            username = payload.get("sub", "unknown")
            role = payload.get("role", DEFAULT_ROLE)
            return username, role

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated. Sign in via /login or obtain a token via /token.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def require_role(min_role: str):
    """
    Route dependency factory: require_role("investigator") returns a
    dependency that authenticates the caller (session or bearer token,
    same as get_current_user) AND requires their role to be at least
    `min_role` in the ROLES rank order (viewer < analyst < investigator
    < admin). Raises 403 (not 401) if authenticated but underprivileged
    — the caller is a known, valid identity; they're just not allowed
    to do this specific thing.

    Usage:
        @router.post("/search-face/")
        def search_face(..., user: str = Depends(require_role("analyst"))):
            ...
    """
    if min_role not in ROLE_RANK:
        raise ValueError(f"Unknown role '{min_role}' — must be one of {list(ROLE_RANK)}")

    async def _dependency(request: Request, token: str = Depends(oauth2_scheme)) -> str:
        username, role = await _get_current_user_and_role(request, token)
        if ROLE_RANK.get(role, -1) < ROLE_RANK[min_role]:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This action requires the '{min_role}' role or higher (you have '{role}').",
            )
        return username

    return _dependency


def is_user_logged_in(request: Request) -> bool:
    return request.session.get(USER_SESSION_KEY) is True


def current_username(request: Request) -> str:
    return request.session.get(USERNAME_SESSION_KEY, "unknown")


@router.get("/login", response_class=HTMLResponse)
def user_login_page(request: Request):
    if is_user_logged_in(request):
        return RedirectResponse(url="/docs", status_code=303)
    return templates.TemplateResponse(request, "user_login.html", {"error": None})


@router.post("/login")
@limiter.limit("5/minute")
def user_login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    lockout_msg = _lockout_message(username)
    if lockout_msg:
        return templates.TemplateResponse(
            request, "user_login.html", {"error": lockout_msg}, status_code=429,
        )

    if verify_user(username, password):
        request.session[USER_SESSION_KEY] = True
        request.session[USERNAME_SESSION_KEY] = username
        return RedirectResponse(url="/docs", status_code=303)
    return templates.TemplateResponse(
        request, "user_login.html", {"error": "Invalid username or password."}, status_code=401,
    )


@router.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    if is_user_logged_in(request):
        return RedirectResponse(url="/docs", status_code=303)
    return templates.TemplateResponse(request, "register.html", {"error": None})


@router.post("/register")
@limiter.limit("5/minute")
def register_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    confirm_password: str = Form(...),
):
    if len(username) < 3:
        return templates.TemplateResponse(
            request, "register.html", {"error": "Username must be at least 3 characters."}, status_code=400,
        )
    if len(password) < 6:
        return templates.TemplateResponse(
            request, "register.html", {"error": "Password must be at least 6 characters."}, status_code=400,
        )
    if password != confirm_password:
        return templates.TemplateResponse(
            request, "register.html", {"error": "Passwords do not match."}, status_code=400,
        )
    if user_exists(username):
        return templates.TemplateResponse(
            request, "register.html", {"error": "That username is already taken."}, status_code=409,
        )

    # Self-registration always creates the lowest-privilege role
    # (DEFAULT_ROLE, "viewer"). Elevating to analyst/investigator/admin
    # is an admin-only action — see admin.py.
    create_user(username, password, role=DEFAULT_ROLE)
    request.session[USER_SESSION_KEY] = True
    request.session[USERNAME_SESSION_KEY] = username
    return RedirectResponse(url="/docs", status_code=303)


@router.get("/logout")
def user_logout(request: Request):
    request.session.pop(USER_SESSION_KEY, None)
    request.session.pop(USERNAME_SESSION_KEY, None)
    return RedirectResponse(url="/login", status_code=303)
