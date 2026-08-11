"""
POST /index-face/
PUT  /index-face/{face_id}   (re-index: replace an existing face's photo)

Ingestion endpoint. Accepts a face image plus a source reference and an
ID, generates its embedding, and stores it in the vector database.

Also supports:
- person_id: group multiple photos under the same person for best-of-N
  matching in /search-face/
- duplicate detection: before adding, checks if a very similar face is
  already indexed and returns a warning (still indexes unless force=false)

RBAC: requires the 'analyst' role or higher (analyst, investigator,
admin) — self-registered 'viewer' accounts can search but cannot add
or modify indexed faces. Adjust the require_role(...) argument below
if you want indexing gated to a different minimum role.

CONSENT: this project is scoped to closed, consenting datasets (see
README's "Scope note"). Every face indexed here requires proof of
consent — consent_given_by, consent_method, and purpose are all
required form fields. If consent_given=false is passed, or the
consent_method isn't one of consent_store.VALID_CONSENT_METHODS, the
request is rejected with a 422 BEFORE the image is ever processed or
embedded — no unconsented face ever reaches the vector store.
"""

from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, Depends

from app.services.embedding_service import get_embedding, EmbeddingError
from app.services.vector_store import add_face, update_face, search_faces
from app.services.upload_utils import save_upload_to_tempfile
from app.services.thumbnail_utils import make_thumbnail_b64
from app.services.activity_log import log_activity
from app.services.ai_detection import detect_ai_generated
from app.services.face_quality import assess_quality
from app.services import consent_store
from app.api.routes.auth import require_role
from app.core.config import settings
from app.schemas import IndexFaceResponse

router = APIRouter()


def _validate_consent(consent_given: bool, consent_given_by: str, consent_method: str, purpose: str) -> None:
    """Raises HTTPException(422) if consent is missing or malformed.
    Called before any image processing happens, so a rejected request
    never touches DeepFace or the vector store."""
    if not consent_given:
        raise HTTPException(
            status_code=422,
            detail=(
                "Consent is required to index a face. Set consent_given=true "
                "and provide consent_given_by, consent_method, and purpose. "
                "This tool is scoped to closed, consenting datasets — see the "
                "project README."
            ),
        )
    if not consent_given_by or not consent_given_by.strip():
        raise HTTPException(status_code=422, detail="consent_given_by is required (who gave consent).")
    if not consent_store.is_valid_method(consent_method):
        raise HTTPException(
            status_code=422,
            detail=(
                f"consent_method must be one of: {', '.join(sorted(consent_store.VALID_CONSENT_METHODS))}. "
                f"Got '{consent_method}'."
            ),
        )
    if not purpose or not purpose.strip():
        raise HTTPException(status_code=422, detail="purpose is required (why this face is being indexed).")


@router.post("/index-face/", response_model=IndexFaceResponse)
async def index_face(
    username: str = Depends(require_role("analyst")),
    face_id: str = Form(..., description="Unique ID for this face, e.g. 'student_042'"),
    source_url: str = Form(..., description="Where this image came from (URL, file path, etc.)"),
    person_id: Optional[str] = Form(None, description="Groups multiple photos of the same person. Defaults to face_id."),
    image: UploadFile = File(...),
    force: bool = Query(True, description="If false, indexing is skipped when a near-duplicate is found."),
    consent_given: bool = Form(..., description="Must be true — indexing is refused otherwise."),
    consent_given_by: str = Form(..., description="Name/ID of the person who gave consent (usually the subject themself)."),
    consent_method: str = Form(..., description="One of: written_form, verbal_recorded, self_registered, institutional."),
    purpose: str = Form(..., description="Why this face is being indexed, e.g. 'class roster face search project'."),
):
    _validate_consent(consent_given, consent_given_by, consent_method, purpose)

    tmp_path = await save_upload_to_tempfile(image)

    quality = assess_quality(tmp_path)
    if not quality["passed"]:
        Path(tmp_path).unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail="Image failed quality checks: " + " ".join(quality["reasons"]),
        )

    try:
        embedding = get_embedding(tmp_path)
        thumbnail = make_thumbnail_b64(tmp_path)
    except EmbeddingError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    # AI-generated image detection (metadata + filename-pattern heuristic)
    ai_check = detect_ai_generated(tmp_path, original_filename=image.filename or "")

    # Duplicate detection: search before adding
    duplicate_warning = None
    existing_count_check = search_faces(embedding, top_k=1)
    distances = existing_count_check.get("distances", [[]])[0]
    ids = existing_count_check.get("ids", [[]])[0]
    if distances and distances[0] < settings.duplicate_warning_threshold:
        # Clamp tiny floating-point negatives (e.g. -0.0000006) to a clean 0.000
        # so the message never shows a confusing "-0.000".
        clean_distance = max(distances[0], 0.0)
        duplicate_warning = (
            f"This face looks very similar to already-indexed '{ids[0]}' "
            f"(distance {clean_distance:.3f}). Indexed anyway."
        )
        if not force:
            log_activity("index_skipped_duplicate", face_id, duplicate_warning, username=username)
            return IndexFaceResponse(
                status="skipped_duplicate",
                id=face_id,
                person_id=person_id or face_id,
                duplicate_warning=duplicate_warning,
                ai_generated_warning=ai_check["reason"] if ai_check["is_likely_ai"] else None,
                ai_confidence=ai_check["confidence"],
            )

    add_face(
        face_id=face_id,
        embedding=embedding,
        source_url=source_url,
        person_id=person_id or face_id,
        thumbnail=thumbnail,
    )

    # Consent is recorded AFTER a successful add_face() — if indexing
    # somehow failed above, we don't want an orphaned consent record
    # for a face_id that was never actually stored.
    consent_store.record_consent(
        face_id=face_id,
        consent_given_by=consent_given_by,
        consent_method=consent_method,
        purpose=purpose,
        recorded_by_username=username,
    )

    ai_note = f", ai_flag={ai_check['confidence']}" if ai_check["is_likely_ai"] else ""
    log_activity(
        "index",
        face_id,
        f"person_id={person_id or face_id}, source={source_url}, consent_method={consent_method}{ai_note}",
        username=username,
    )

    return IndexFaceResponse(
        status="indexed",
        id=face_id,
        person_id=person_id or face_id,
        duplicate_warning=duplicate_warning,
        ai_generated_warning=ai_check["reason"] if ai_check["is_likely_ai"] else None,
        ai_confidence=ai_check["confidence"],
    )


@router.put("/index-face/{face_id}", response_model=IndexFaceResponse)
async def reindex_face(
    face_id: str,
    username: str = Depends(require_role("analyst")),
    source_url: Optional[str] = Form(None),
    person_id: Optional[str] = Form(None),
    image: UploadFile = File(...),
    consent_given: bool = Form(..., description="Must be true — re-indexing is refused otherwise."),
    consent_given_by: str = Form(..., description="Name/ID of the person who gave consent for this new photo."),
    consent_method: str = Form(..., description="One of: written_form, verbal_recorded, self_registered, institutional."),
    purpose: str = Form(..., description="Why this face is being re-indexed."),
):
    """
    Replaces an existing face's photo/embedding without changing its ID.
    Consent must be re-confirmed for the NEW photo being uploaded — an
    old consent record doesn't automatically cover a different image.
    """
    _validate_consent(consent_given, consent_given_by, consent_method, purpose)

    tmp_path = await save_upload_to_tempfile(image)

    try:
        embedding = get_embedding(tmp_path)
        thumbnail = make_thumbnail_b64(tmp_path)
    except EmbeddingError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    updated = update_face(
        face_id=face_id,
        embedding=embedding,
        source_url=source_url,
        person_id=person_id,
        thumbnail=thumbnail,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Face '{face_id}' not found. Use POST /index-face/ to create it first.")

    consent_store.record_consent(
        face_id=face_id,
        consent_given_by=consent_given_by,
        consent_method=consent_method,
        purpose=purpose,
        recorded_by_username=username,
    )

    log_activity("reindex", face_id, f"photo replaced, consent_method={consent_method}", username=username)

    return IndexFaceResponse(status="reindexed", id=face_id, person_id=person_id or face_id)
