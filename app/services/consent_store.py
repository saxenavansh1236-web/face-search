"""
consent_store.py

SQLite-backed consent records for every indexed face. This project is
scoped to closed, consenting datasets (see README's "Scope note") —
every face indexed via /index-face/ or /bulk-index/ must have a
recorded consent before it's allowed into the vector store.

One consent record per face_id (1:1 with the embedding stored in
vector_store.py). If a face is deleted, its consent record should be
deleted too (see delete_consent(), called from admin.py's delete
routes — wire that in if not already done).
"""

import sqlite3
import os
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from app.core.config import settings

_DB_PATH = os.path.normpath(os.path.join(settings.db_path, "..", "consent.db"))

# Recognized ways consent was obtained. Free-text is intentionally NOT
# allowed here — a closed dropdown keeps the audit trail consistent
# and machine-checkable, rather than relying on someone typing
# "yeah they said it's fine" into a text box.
VALID_CONSENT_METHODS = {
    "written_form",      # signed physical or digital consent form
    "verbal_recorded",   # verbal consent, recorded/witnessed
    "self_registered",   # the person themself is the one uploading/indexing their own photo
    "institutional",     # covered by an org-level agreement (e.g. employee handbook, class syllabus)
}


def _get_conn():
    os.makedirs(os.path.dirname(_DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS consent_records (
                face_id TEXT PRIMARY KEY,
                consent_given_by TEXT NOT NULL,
                consent_method TEXT NOT NULL,
                purpose TEXT NOT NULL,
                recorded_by_username TEXT NOT NULL,
                recorded_at TEXT NOT NULL
            )
        """)


def record_consent(
    face_id: str,
    consent_given_by: str,
    consent_method: str,
    purpose: str,
    recorded_by_username: str,
) -> None:
    """
    Stores a consent record for a face_id. Overwrites any existing
    record for the same face_id (relevant on PUT /index-face/{id}
    re-index, where consent should be re-confirmed for the new photo).
    """
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO consent_records
                (face_id, consent_given_by, consent_method, purpose, recorded_by_username, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(face_id) DO UPDATE SET
                consent_given_by = excluded.consent_given_by,
                consent_method = excluded.consent_method,
                purpose = excluded.purpose,
                recorded_by_username = excluded.recorded_by_username,
                recorded_at = excluded.recorded_at
            """,
            (
                face_id,
                consent_given_by,
                consent_method,
                purpose,
                recorded_by_username,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def get_consent(face_id: str) -> Optional[Dict[str, Any]]:
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM consent_records WHERE face_id = ?", (face_id,)
        ).fetchone()
    return dict(row) if row else None


def has_consent(face_id: str) -> bool:
    return get_consent(face_id) is not None


def delete_consent(face_id: str) -> bool:
    """Call this alongside delete_face()/delete_many() in admin.py so a
    deleted face doesn't leave an orphaned consent record behind."""
    with _get_conn() as conn:
        cur = conn.execute("DELETE FROM consent_records WHERE face_id = ?", (face_id,))
    return cur.rowcount > 0


def get_all_consents() -> List[Dict[str, Any]]:
    with _get_conn() as conn:
        rows = conn.execute("SELECT * FROM consent_records ORDER BY recorded_at DESC").fetchall()
    return [dict(r) for r in rows]


def is_valid_method(method: str) -> bool:
    return method in VALID_CONSENT_METHODS


init_db()
