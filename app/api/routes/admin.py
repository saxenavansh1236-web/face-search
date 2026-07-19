"""
admin.py

Admin portal: login, dashboard (with pagination, search, thumbnails,
bulk delete, inline metadata editing), export, and an activity log view.
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

from app.core.config import settings
from app.services.vector_store import (
    get_all_faces, delete_face, delete_many, count_faces, update_metadata,
)
from app.services.activity_log import get_recent_activity, log_activity
from app.services.user_store import get_all_users, count_users

router = APIRouter(include_in_schema=False)
templates = Jinja2Templates(directory="app/templates")

# Uses the same in-memory, IP-keyed limiting strategy as app.state.limiter
# in main.py. A second Limiter instance here is fine — slowapi routes the
# decorator through request.app.state.limiter under the hood, so this just
# needs to exist for the @limiter.limit(...) decorator syntax to work.
limiter = Limiter(key_func=get_remote_address)

SESSION_KEY = "admin_authenticated"
PAGE_SIZE = 10


def is_logged_in(request: Request) -> bool:
    return request.session.get(SESSION_KEY) is True


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

    if valid_username and valid_password:
        request.session[SESSION_KEY] = True
        return RedirectResponse(url="/admin/", status_code=303)

    return templates.TemplateResponse(
        request, "login.html", {"error": "Invalid username or password."}, status_code=401,
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
    log_activity("delete", face_id, "via admin panel")
    msg = f"Deleted '{face_id}'" if deleted else f"'{face_id}' not found"
    return RedirectResponse(url=f"/admin/?message={msg}", status_code=303)


@router.post("/admin/faces/bulk-delete")
def admin_bulk_delete(request: Request, face_ids: List[str] = Form(...)):
    if not is_logged_in(request):
        return RedirectResponse(url="/admin/login", status_code=303)
    deleted_count = delete_many(face_ids)
    for fid in face_ids:
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

    return templates.TemplateResponse(
        request,
        "users_page.html",
        {
            "users": users,
            "total_users": count_users(),
            "activity_by_user": activity_by_user,
        },
    )