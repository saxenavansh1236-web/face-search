"""
user_store.py

A minimal SQLite-backed user table for API/docs login and registration.
Separate from the admin panel's single admin_username/admin_password —
these are regular users who register themselves to use the API/docs,
and the admin can see who they are and what they've done.

Passwords are hashed with PBKDF2-SHA256 (stdlib hashlib, no extra deps).

Also tracks failed login attempts per-username and locks an account
temporarily after too many failures in a row — this protects a single
targeted account even if an attacker spreads requests across many IPs
to dodge the IP-based rate limiting in auth.py.
"""

import sqlite3
import hashlib
import os
import secrets
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any

from app.core.config import settings

_DB_PATH = os.path.normpath(os.path.join(settings.db_path, "..", "users.db"))

# Lockout policy: after this many consecutive failed attempts, the
# account is locked for LOCKOUT_MINUTES. A successful login, or the
# lockout window expiring, resets the failure count.
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_MINUTES = 15


def _get_conn():
    os.makedirs(os.path.dirname(_DB_PATH) or ".", exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL,
                failed_attempts INTEGER NOT NULL DEFAULT 0,
                locked_until TEXT
            )
        """)
        # Add columns for pre-existing databases created before lockout
        # tracking was introduced (ALTER TABLE fails silently if the
        # column already exists).
        for column, coltype in [("failed_attempts", "INTEGER NOT NULL DEFAULT 0"), ("locked_until", "TEXT")]:
            try:
                conn.execute(f"ALTER TABLE users ADD COLUMN {column} {coltype}")
            except sqlite3.OperationalError:
                pass  # column already exists


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()


def create_user(username: str, password: str) -> bool:
    """Returns False if the username is already taken."""
    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)
    try:
        with _get_conn() as conn:
            conn.execute(
                "INSERT INTO users (username, password_hash, salt, created_at) VALUES (?, ?, ?, ?)",
                (username, password_hash, salt, datetime.now(timezone.utc).isoformat()),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def is_locked_out(username: str) -> Optional[int]:
    """Returns remaining lockout seconds if locked, else None."""
    with _get_conn() as conn:
        row = conn.execute("SELECT locked_until FROM users WHERE username = ?", (username,)).fetchone()
    if not row or not row["locked_until"]:
        return None
    locked_until = datetime.fromisoformat(row["locked_until"])
    now = datetime.now(timezone.utc)
    if now < locked_until:
        return int((locked_until - now).total_seconds())
    return None


def _record_failed_attempt(username: str) -> None:
    with _get_conn() as conn:
        row = conn.execute("SELECT failed_attempts FROM users WHERE username = ?", (username,)).fetchone()
        if not row:
            return  # unknown username: nothing to lock, verify_user already returns False
        new_count = row["failed_attempts"] + 1
        if new_count >= MAX_FAILED_ATTEMPTS:
            locked_until = (datetime.now(timezone.utc) + timedelta(minutes=LOCKOUT_MINUTES)).isoformat()
            conn.execute(
                "UPDATE users SET failed_attempts = ?, locked_until = ? WHERE username = ?",
                (new_count, locked_until, username),
            )
        else:
            conn.execute(
                "UPDATE users SET failed_attempts = ? WHERE username = ?",
                (new_count, username),
            )


def _reset_failed_attempts(username: str) -> None:
    with _get_conn() as conn:
        conn.execute(
            "UPDATE users SET failed_attempts = 0, locked_until = NULL WHERE username = ?",
            (username,),
        )


def verify_user(username: str, password: str) -> bool:
    lockout_seconds = is_locked_out(username)
    if lockout_seconds is not None:
        return False

    with _get_conn() as conn:
        row = conn.execute("SELECT password_hash, salt FROM users WHERE username = ?", (username,)).fetchone()
    if not row:
        return False

    computed_hash = _hash_password(password, row["salt"])
    # secrets.compare_digest avoids leaking timing information about how
    # many leading characters of the computed hash matched the stored one.
    is_valid = secrets.compare_digest(computed_hash, row["password_hash"])

    if is_valid:
        _reset_failed_attempts(username)
    else:
        _record_failed_attempt(username)

    return is_valid


def user_exists(username: str) -> bool:
    with _get_conn() as conn:
        row = conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
    return row is not None


def get_all_users() -> List[Dict[str, Any]]:
    with _get_conn() as conn:
        rows = conn.execute("SELECT username, created_at FROM users ORDER BY created_at DESC").fetchall()
    return [dict(r) for r in rows]


def count_users() -> int:
    with _get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) as c FROM users").fetchone()
    return row["c"]


init_db()