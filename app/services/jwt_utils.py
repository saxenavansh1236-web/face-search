"""
jwt_utils.py

Stateless JWT access tokens, as an alternative to the session-cookie
login for programmatic API clients (scripts, other services) that
can't hold a browser session. Issued via POST /token after verifying
username/password against the same user_store used for /login.

Also issues longer-lived refresh tokens (POST /token/refresh) so a
client can obtain a new access token without re-sending a
username/password every time the access token expires.

Both token types are plain JWTs signed with the same secret/algorithm,
distinguished by a "type" claim ("access" vs "refresh") so one can
never be mistaken for or used in place of the other — e.g. a leaked
refresh token can't be used directly to call /index-face/, since
get_current_user() only accepts "access" tokens.

Refresh tokens also carry a unique "jti" (JWT ID) claim. Revoking a
refresh token (e.g. on logout, via POST /token/revoke) records its jti
in token_store.py's revoked_tokens table; decode_refresh_token() checks
that table before trusting an otherwise-valid, unexpired token. This is
the one piece of server-side state needed to support early revocation
of an otherwise-stateless JWT. Access tokens are NOT individually
revocable — they're short-lived by design, so the exposure window from
a leaked one is small; only refresh tokens (which are long-lived) get
this protection.

RBAC: access tokens embed the user's `role` at issue-time as a "role"
claim, so role-gated endpoints (see auth.py: require_role()) don't
need a user_store lookup per request. This means a role change (e.g.
admin promotes a user to investigator) does NOT take effect for that
user's already-issued access tokens until they expire and are
refreshed — a deliberate tradeoff for statelessness. Keep
jwt_expiry_minutes reasonably short if you need role changes to
propagate quickly; revoking the user's refresh token forces a full
re-login sooner if immediate effect is required.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional
import secrets
import jwt as pyjwt

from app.core.config import settings
from app.services import token_store
from app.services.user_store import get_user_role, DEFAULT_ROLE


def create_access_token(username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=settings.jwt_expiry_minutes)
    role = get_user_role(username) or DEFAULT_ROLE
    payload = {"sub": username, "exp": expire, "type": "access", "role": role}
    return pyjwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> Optional[str]:
    """Returns the username if the token is a valid, unexpired ACCESS
    token, else None. Rejects refresh tokens even if otherwise valid."""
    payload = decode_access_token_payload(token)
    return payload.get("sub") if payload else None


def decode_access_token_payload(token: str) -> Optional[dict]:
    """Like decode_access_token, but returns the full payload (so
    callers can read the "role" claim) instead of just the username."""
    try:
        payload = pyjwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except pyjwt.PyJWTError:
        return None
    if payload.get("type") != "access":
        return None
    return payload


def create_refresh_token(username: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(days=settings.jwt_refresh_expiry_days)
    payload = {
        "sub": username,
        "exp": expire,
        "type": "refresh",
        "jti": secrets.token_hex(16),
    }
    return pyjwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def _decode_refresh_payload(token: str) -> Optional[dict]:
    """Decodes and validates signature/expiry/type only — does NOT
    check revocation. Used internally by both decode_refresh_token
    (which adds the revocation check) and revoke_refresh_token (which
    needs the jti of a token regardless of whether it's already
    revoked, so revoking twice is harmless)."""
    try:
        payload = pyjwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except pyjwt.PyJWTError:
        return None
    if payload.get("type") != "refresh":
        return None
    return payload


def decode_refresh_token(token: str) -> Optional[str]:
    """Returns the username if the token is a valid, unexpired,
    NOT-REVOKED refresh token, else None."""
    payload = _decode_refresh_payload(token)
    if not payload:
        return None
    jti = payload.get("jti")
    if jti and token_store.is_revoked(jti):
        return None
    return payload.get("sub")


def revoke_refresh_token(token: str) -> bool:
    """Marks a refresh token as revoked so it can no longer be used,
    even though it hasn't naturally expired yet. Returns True if the
    token was well-formed and got revoked, False if it was invalid/
    malformed (nothing to revoke)."""
    payload = _decode_refresh_payload(token)
    if not payload:
        return False
    jti = payload.get("jti")
    if not jti:
        return False  # older-format refresh token without a jti: can't be individually revoked
    expires_at = datetime.fromtimestamp(payload["exp"], tz=timezone.utc)
    token_store.revoke_jti(jti, payload.get("sub", "unknown"), expires_at)
    return True
