"""
token_store.py

Tracks revoked refresh tokens by their unique "jti" (JWT ID) claim, so
a refresh token can be invalidated before its natural expiry — e.g. on
logout, or if a client reports one as compromised.

Stateless JWTs otherwise can't be revoked early (there's no way to
"delete" a signed token that's already out in the world) — this table
is the one piece of server-side state that makes early revocation
possible: decode_refresh_token() in jwt_utils.py checks here before
trusting an otherwise-valid, unexpired token.

Uses the same SQLite file as user_store.py (users.db), just a
different table, to avoid managing a second database file.
"""

import sqlite3
import os
from datetime import datetime, timezone

from app.core.config import settings

_DB_PATH = os.path.normpath(os.path.join(settings.db_path, "..", "users.db"))


def _get_conn():
    os.makedirs(os.path.dirname(_DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS revoked_tokens (
                jti TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                revoked_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
        """)


def revoke_jti(jti: str, username: str, expires_at: datetime) -> None:
    """Marks a token's jti as revoked. expires_at should match the
    token's own exp claim, so cleanup_expired() can later purge it
    once it would have expired naturally anyway."""
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO revoked_tokens (jti, username, revoked_at, expires_at) VALUES (?, ?, ?, ?)",
            (jti, username, datetime.now(timezone.utc).isoformat(), expires_at.isoformat()),
        )


def is_revoked(jti: str) -> bool:
    with _get_conn() as conn:
        row = conn.execute("SELECT 1 FROM revoked_tokens WHERE jti = ?", (jti,)).fetchone()
    return row is not None


def cleanup_expired() -> int:
    """Deletes revoked-token records whose underlying token would have
    expired naturally anyway (no need to remember them forever). Not
    called automatically — wire up to a periodic task/cron if desired."""
    now = datetime.now(timezone.utc).isoformat()
    with _get_conn() as conn:
        cursor = conn.execute("DELETE FROM revoked_tokens WHERE expires_at < ?", (now,))
        return cursor.rowcount


init_db()