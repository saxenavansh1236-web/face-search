import hashlib
import warnings

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Central configuration for the face search tool.
    Values can be overridden via environment variables or a .env file.
    """

    db_path: str = "./face_db"
    collection_name: str = "faces"
    model_name: str = "Facenet512"
    distance_metric: str = "cosine"
    default_top_k: int = 3

    high_confidence_threshold: float = 0.30
    possible_match_threshold: float = 0.45
    match_threshold: float = 0.4

    duplicate_warning_threshold: float = 0.25

    max_upload_size_mb: float = 15.0
    thumbnail_size: int = 96

    admin_username: str = "saxenavansh1236@gmail.com"
    admin_password: str = "vansh@123"
    session_secret_key: str = "dev-only-change-this-secret-key"

    activity_log_path: str = "./face_db/activity_log.jsonl"
    calibration_data_path: str = "./calibration_data"

    # --- Advanced matching pipeline ---

    # Face detector backend used by DeepFace before embedding.
    # Options include: "opencv" (default, fast, lower accuracy),
    # "retinaface" (slower, much better on small/angled/occluded faces),
    # "mtcnn", "ssd", "yolov8". RetinaFace requires the `retina-face`
    # package (installed automatically as a DeepFace dependency).
    detector_backend: str = "retinaface"

    # Whether DeepFace should align faces (rotate/crop to a canonical
    # pose) before embedding. Improves matching consistency across
    # different head angles. Adds a small amount of processing time.
    align_faces: bool = True

    # Face quality gates applied before indexing. Rejects images where
    # the detected face is too small, too blurry, or too dark/bright to
    # produce a reliable embedding — catches bad inputs before they
    # pollute the database with a low-quality reference photo.
    min_face_size_px: int = 60
    min_sharpness_score: float = 40.0   # Laplacian variance; lower = blurrier
    min_brightness: float = 25.0         # 0-255 grayscale mean
    max_brightness: float = 230.0

    # Re-ranking: after the initial ANN search, re-verify the top
    # candidate with DeepFace.verify() (a stricter, purpose-built
    # comparison) before finalizing "high" confidence. Adds latency
    # but reduces false positives on borderline matches.
    enable_reranking: bool = True
    rerank_top_n: int = 3
    rerank_skip_below_distance: float = 0.15  # already near-certain match
    rerank_skip_above_distance: float = 0.45  # already confidently no match

    # JWT settings for stateless API access tokens (in addition to the
    # existing session-cookie login) — lets programmatic clients (not
    # just a logged-in browser) call the API with a bearer token.
    jwt_secret_key: str = "dev-only-change-this-jwt-secret"
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60 * 24  # 24 hours

    # Refresh tokens: longer-lived than access tokens, used only to
    # obtain a new access token via POST /token/refresh without asking
    # the client to re-send a username/password. Kept separate from
    # jwt_expiry_minutes so access tokens can be made short-lived
    # (reducing exposure if one leaks) without forcing frequent re-logins.
    jwt_refresh_expiry_days: int = 30

    # --- Admin MFA (TOTP) ---
    # Requires the admin account to enter a 6-digit code from an
    # authenticator app (Google Authenticator, Authy, etc.) in addition
    # to the username/password. The TOTP secret is generated on first
    # login after this feature is enabled and stored server-side (see
    # app/services/admin_mfa_store.py) — never in this config file.
    admin_mfa_enabled: bool = True
    admin_mfa_issuer_name: str = "Face Search Tool"

    class Config:
        env_file = ".env"


settings = Settings()


# --- Startup security check ---
# Warns loudly (visible in the uvicorn startup logs) if any secret or
# credential is still on its insecure development default. This does
# NOT block startup — local dev should keep working out of the box —
# it just makes it impossible to miss before a real deployment.

_INSECURE_DEFAULTS = {
    "admin_password": "change-me",
    "session_secret_key": "dev-only-change-this-secret-key",
    "jwt_secret_key": "dev-only-change-this-jwt-secret",
}


def _warn_insecure_defaults(s: Settings) -> None:
    for field, default_value in _INSECURE_DEFAULTS.items():
        if getattr(s, field) == default_value:
            warnings.warn(
                f"SECURITY WARNING: '{field}' is still set to its insecure default value. "
                f"Set it via an environment variable or .env file before deploying this "
                f"beyond your own machine.",
                stacklevel=2,
            )


_warn_insecure_defaults(settings)


def hash_admin_password(password: str) -> str:
    """Used only to compare against the configured admin password without
    ever branching on raw string equality (see admin.py login check)."""
    return hashlib.sha256(password.encode()).hexdigest()
