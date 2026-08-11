"""
admin.py

Admin portal: login (with MFA), dashboard (with pagination, search,
thumbnails, bulk delete, inline metadata editing), export, activity
log view, and user role management (RBAC).

Admin login is now two steps when admin_mfa_enabled=True:
1. POST /admin/login — verifies username/password. On success, does
   NOT set admin_authenticated yet. Instead sets a short-lived
   "admin_mfa_pending" session flag and redirects to either:
   - /admin/mfa-setup (first time ever — no confirmed secret exists), or
   - /admin/mfa-verify (secret already confirmed from a past login)
2. POST /admin/mfa-setup or /admin/mfa-verify — checks the 6-digit
   TOTP code. Only on success does admin_authenticated get set.

is_logged_in() is unchanged in meaning (True = fully authenticated,
past MFA) so every existing route that gates on it keeps working
without modification.

CONSENT: deleting a face now also deletes its consent record (see
consent_store.delete_consent()), so a removed face never leaves an
orphaned consent entry behind.
"""

import csv
import json
import io
import math
import secrets
from typing import List, Optional

from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from slowapi import Limiter
from slowapi.util import get_remote_address

import pyotp
import qrcode
import qrcode.image.svg
import io as _io
import base64

from app.core.config import settings
from app.services.vector_store import (
    get_all_faces, delete_face, delete_many, count_faces, update_metadata,
)
from app.services.activity_log import get_recent_activity, log_activity
from app.services.user_store import get_all_users, count_users, set_user_role, ROLES
from app.services import admin_mfa_store
from app.services import consent_store
from app.services import calibration

router = APIRouter(include_in_schema=False)
templates = Jinja2Templates(directory="app/templates")

# Uses the same in-memory, IP-keyed limiting strategy as app.state.limiter
# in main.py. A second Limiter instance here is fine — slowapi routes the
# decorator through request.app.state.limiter under the hood, so this just
# needs to exist for the @limiter.limit(...) decorator syntax to work.
limiter = Limiter(key_func=get_remote_address)

SESSION_KEY = "admin_authenticated"
MFA_PENDING_KEY = "admin_mfa_pending"
PAGE_SIZE = 10


def is_logged_in(request: Request) -> bool:
    return request.session.get(SESSION_KEY) is True


def _mfa_pending(request: Request) -> bool:
    return request.session.get(MFA_PENDING_KEY) is True


@router.get("/admin/login", response_class=HTMLResponse)
def login_page(request: Request):
    if is_logged_in(request):
        return RedirectResponse(url="/admin/", status_code=303)
    return templates.TemplateResponse(request, "login.html", {"error": None})


@router.post("/admin/login")
@limiter.limit("5/minute")
def login_submit(request: Request, username: str = Form(...), password: str = Form(...)):
    # secrets.compare_digest avoids leaking timing information about how
    # many leading characters of username/password matched, unlike `==`.
    valid_username = secrets.compare_digest(username, settings.admin_username)
    valid_password = secrets.compare_digest(password, settings.admin_password)

    if not (valid_username and valid_password):
        return templates.TemplateResponse(
            request, "login.html", {"error": "Invalid username or password."}, status_code=401,
        )

    if not settings.admin_mfa_enabled:
        # MFA disabled entirely via config — fall back to single-step login.
        request.session[SESSION_KEY] = True
        return RedirectResponse(url="/admin/", status_code=303)

    # Password correct — do NOT fully authenticate yet. Require a second
    # factor before setting SESSION_KEY.
    request.session[MFA_PENDING_KEY] = True

    if admin_mfa_store.is_confirmed():
        return RedirectResponse(url="/admin/mfa-verify", status_code=303)
    else:
        return RedirectResponse(url="/admin/mfa-setup", status_code=303)


@router.get("/admin/mfa-setup", response_class=HTMLResponse)
def mfa_setup_page(request: Request):
    if not _mfa_pending(request):
        return RedirectResponse(url="/admin/login", status_code=303)
    if admin_mfa_store.is_confirmed():
        # Already enrolled from a previous session — nothing to set up.
        return RedirectResponse(url="/admin/mfa-verify", status_code=303)

    secret = admin_mfa_store.generate_and_store_pending_secret()
    provisioning_uri = pyotp.TOTP(secret).provisioning_uri(
        name=settings.admin_username,
        issuer_name=settings.admin_mfa_issuer_name,
    )

    # Generate QR code as inline SVG (no image library / PNG round-trip
    # needed beyond what `qrcode` already provides).
    qr_img = qrcode.make(provisioning_uri, image_factory=qrcode.image.svg.SvgImage)
    buf = _io.BytesIO()
    qr_img.save(buf)
    qr_svg_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return templates.TemplateResponse(
        request,
        "admin_mfa_setup.html",
        {
            "secret": secret,
            "qr_svg_b64": qr_svg_b64,
            "error": None,
        },
    )


@router.post("/admin/mfa-setup")
@limiter.limit("5/minute")
def mfa_setup_submit(request: Request, code: str = Form(...)):
    if not _mfa_pending(request):
        return RedirectResponse(url="/admin/login", status_code=303)

    if admin_mfa_store.confirm_setup(code):
        request.session.pop(MFA_PENDING_KEY, None)
        request.session[SESSION_KEY] = True
        log_activity("admin_mfa_enrolled", "admin", "MFA enrollment completed")
        return RedirectResponse(url="/admin/", status_code=303)

    secret = admin_mfa_store.get_secret()
    provisioning_uri = pyotp.TOTP(secret).provisioning_uri(
        name=settings.admin_username,
        issuer_name=settings.admin_mfa_issuer_name,
    )
    qr_img = qrcode.make(provisioning_uri, image_factory=qrcode.image.svg.SvgImage)
    buf = _io.BytesIO()
    qr_img.save(buf)
    qr_svg_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return templates.TemplateResponse(
        request,
        "admin_mfa_setup.html",
        {
            "secret": secret,
            "qr_svg_b64": qr_svg_b64,
            "error": "Incorrect code. Make sure your device's clock is accurate and try the newest code shown.",
        },
        status_code=401,
    )


@router.get("/admin/mfa-verify", response_class=HTMLResponse)
def mfa_verify_page(request: Request):
    if not _mfa_pending(request):
        return RedirectResponse(url="/admin/login", status_code=303)
    if not admin_mfa_store.is_confirmed():
        return RedirectResponse(url="/admin/mfa-setup", status_code=303)
    return templates.TemplateResponse(request, "admin_mfa_verify.html", {"error": None})


@router.post("/admin/mfa-verify")
@limiter.limit("5/minute")
def mfa_verify_submit(request: Request, code: str = Form(...)):
    if not _mfa_pending(request):
        return RedirectResponse(url="/admin/login", status_code=303)

    if admin_mfa_store.verify_code(code):
        request.session.pop(MFA_PENDING_KEY, None)
        request.session[SESSION_KEY] = True
        return RedirectResponse(url="/admin/", status_code=303)

    return templates.TemplateResponse(
        request, "admin_mfa_verify.html", {"error": "Incorrect code. Try again."}, status_code=401,
    )


@router.get("/admin/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/admin/login", status_code=303)


@router.get("/admin/", response_class=HTMLResponse)
def dashboard(request: Request, page: int = 1, q: Optional[str] = None):
    if not is_logged_in(request):
        return RedirectResponse(url="/admin/login", status_code=303)

    all_faces = get_all_faces()

    if q:
        q_lower = q.lower()
        all_faces = [
            f for f in all_faces
            if q_lower in f["id"].lower()
            or q_lower in (f.get("person_id") or "").lower()
            or q_lower in (f.get("source_url") or "").lower()
        ]

    total = len(all_faces)
    total_pages = max(1, math.ceil(total / PAGE_SIZE))
    page = max(1, min(page, total_pages))
    start = (page - 1) * PAGE_SIZE
    page_faces = all_faces[start:start + PAGE_SIZE]

    message = request.query_params.get("message")

    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "faces": page_faces,
            "total_count": count_faces(),
            "filtered_count": total,
            "model_name": settings.model_name,
            "match_threshold": settings.match_threshold,
            "message": message,
            "query": q or "",
            "page": page,
            "total_pages": total_pages,
        },
    )


@router.post("/admin/faces/{face_id}/delete")
def admin_delete_face(request: Request, face_id: str):
    if not is_logged_in(request):
        return RedirectResponse(url="/admin/login", status_code=303)
    deleted = delete_face(face_id)
    consent_store.delete_consent(face_id)
    log_activity("delete", face_id, "via admin panel")
    msg = f"Deleted '{face_id}'" if deleted else f"'{face_id}' not found"
    return RedirectResponse(url=f"/admin/?message={msg}", status_code=303)


@router.post("/admin/faces/bulk-delete")
def admin_bulk_delete(request: Request, face_ids: List[str] = Form(...)):
    if not is_logged_in(request):
        return RedirectResponse(url="/admin/login", status_code=303)
    deleted_count = delete_many(face_ids)
    for fid in face_ids:
        consent_store.delete_consent(fid)
        log_activity("bulk_delete", fid, "via admin panel bulk action")
    return RedirectResponse(url=f"/admin/?message=Deleted {deleted_count} face(s)", status_code=303)


@router.post("/admin/faces/{face_id}/edit")
def admin_edit_face(request: Request, face_id: str, source_url: str = Form(""), person_id: str = Form("")):
    if not is_logged_in(request):
        return RedirectResponse(url="/admin/login", status_code=303)
    updated = update_metadata(face_id, source_url=source_url or None, person_id=person_id or None)
    log_activity("edit_metadata", face_id, f"source_url={source_url}, person_id={person_id}")
    msg = f"Updated '{face_id}'" if updated else f"'{face_id}' not found"
    return RedirectResponse(url=f"/admin/?message={msg}", status_code=303)


@router.get("/admin/export")
def admin_export(request: Request, format: str = "json"):
    if not is_logged_in(request):
        return RedirectResponse(url="/admin/login", status_code=303)

    faces = get_all_faces()
    slim = [{k: v for k, v in f.items() if k != "thumbnail"} for f in faces]

    if format == "csv":
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=["id", "person_id", "source_url", "created_at"])
        writer.writeheader()
        writer.writerows(slim)
        buffer.seek(0)
        return StreamingResponse(
            iter([buffer.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=faces_export.csv"},
        )

    json_bytes = json.dumps(slim, indent=2).encode("utf-8")
    return StreamingResponse(
        iter([json_bytes]),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=faces_export.json"},
    )


@router.get("/admin/activity", response_class=HTMLResponse)
def admin_activity(request: Request):
    if not is_logged_in(request):
        return RedirectResponse(url="/admin/login", status_code=303)
    entries = get_recent_activity(limit=100)
    return templates.TemplateResponse(request, "activity_log.html", {"entries": entries})


@router.get("/admin/users", response_class=HTMLResponse)
def admin_users(request: Request):
    if not is_logged_in(request):
        return RedirectResponse(url="/admin/login", status_code=303)

    users = get_all_users()
    all_activity = get_recent_activity(limit=10_000)

    activity_by_user = {}
    for entry in all_activity:
        uname = entry.get("username", "unknown")
        activity_by_user.setdefault(uname, []).append(entry)

    message = request.query_params.get("message")

    return templates.TemplateResponse(
        request,
        "users_page.html",
        {
            "users": users,
            "total_users": count_users(),
            "activity_by_user": activity_by_user,
            "available_roles": ROLES,
            "message": message,
        },
    )


@router.post("/admin/users/{username}/role")
def admin_set_user_role(request: Request, username: str, role: str = Form(...)):
    """
    RBAC role change — admin-only, since this whole router is gated by
    is_logged_in() (the single admin-panel session, now behind MFA),
    not a regular user's role.
    """
    if not is_logged_in(request):
        return RedirectResponse(url="/admin/login", status_code=303)

    updated = set_user_role(username, role)
    if updated:
        log_activity("role_change", username, f"role set to '{role}' via admin panel")
        msg = f"Set '{username}' to role '{role}'"
    else:
        msg = f"Could not update role for '{username}' (invalid role or user not found)"

    return RedirectResponse(url=f"/admin/users?message={msg}", status_code=303)

@router.get("/admin/calibration", response_class=HTMLResponse)
def admin_calibration(request: Request):
    """
    Runs the FAR/FRR calibration harness over the labeled test photo
    set at settings.calibration_data_path and shows the results —
    genuine vs. impostor distance distributions, an FAR/FRR curve
    across candidate thresholds, and where the CURRENT match_threshold
    (from config.py) actually lands on that curve.

    This recomputes embeddings and re-runs the full analysis on every
    page load (no caching) — acceptable for a small admin-only test
    set, but if your calibration set grows large, this will get slow
    and should be moved to a background job with cached results.
    """
    if not is_logged_in(request):
        return RedirectResponse(url="/admin/login", status_code=303)

    result = calibration.run_calibration()

    return templates.TemplateResponse(
        request,
        "calibration.html",
        {"result": result, "calibration_data_path": settings.calibration_data_path},
    )
