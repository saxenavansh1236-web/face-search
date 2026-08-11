"""
admin_mfa_store.py

Stores the admin account's TOTP (Time-based One-Time Password) secret
for MFA. There's only ever one admin account (see admin.py), so this
is a single JSON file rather than a full table — the secret is
generated once on first MFA setup and persists across restarts.

The secret itself is sensitive (equivalent to a password) — this file
should be treated like users.db: not committed to version control, and
protected by filesystem permissions in any real deployment.
"""

import json
import os
from typing import Optional

import pyotp

from app.core.config import settings

_MFA_PATH = os.path.normpath(os.path.join(settings.db_path, "..", "admin_mfa.json"))


def _read() -> dict:
    if not os.path.exists(_MFA_PATH):
        return {}
    with open(_MFA_PATH, "r") as f:
        return json.load(f)


def _write(data: dict) -> None:
    os.makedirs(os.path.dirname(_MFA_PATH) or ".", exist_ok=True)
    with open(_MFA_PATH, "w") as f:
        json.dump(data, f)


def has_mfa_configured() -> bool:
    """True once the admin has completed the one-time QR enrollment."""
    return bool(_read().get("secret"))


def get_secret() -> Optional[str]:
    return _read().get("secret")


def generate_and_store_pending_secret() -> str:
    """
    Generates a new TOTP secret and stores it, but callers should treat
    it as PENDING until confirm_setup() succeeds — the admin hasn't
    proven they can generate valid codes yet. We store it eagerly
    (rather than only in the session) so the QR code stays scannable
    if the admin refreshes the enrollment page.
    """
    secret = pyotp.random_base32()
    _write({"secret": secret, "confirmed": False})
    return secret


def confirm_setup(code: str) -> bool:
    """Verifies the enrollment code against the pending secret and, if
    valid, marks MFA as fully confirmed/active. Returns False if no
    setup is in progress or the code is wrong."""
    data = _read()
    secret = data.get("secret")
    if not secret:
        return False
    if pyotp.TOTP(secret).verify(code, valid_window=1):
        data["confirmed"] = True
        _write(data)
        return True
    return False


def is_confirmed() -> bool:
    return _read().get("confirmed", False)


def verify_code(code: str) -> bool:
    """Verifies a 6-digit code against the stored, confirmed secret at
    normal login time. valid_window=1 tolerates ~30s of clock drift
    between server and phone, a standard TOTP allowance."""
    data = _read()
    secret = data.get("secret")
    if not secret or not data.get("confirmed"):
        return False
    return pyotp.TOTP(secret).verify(code, valid_window=1)


def reset_mfa() -> None:
    """Wipes MFA enrollment entirely, forcing re-setup on next login.
    Not exposed via any route yet — intended for manual recovery if
    the admin loses their authenticator device (run via a one-off
    python -c or Python shell, same pattern as set_user_role)."""
    if os.path.exists(_MFA_PATH):
        os.remove(_MFA_PATH)
