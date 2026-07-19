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
from app.api.routes.auth import get_current_user
from app.core.config import settings
from app.schemas import IndexFaceResponse

router = APIRouter()


@router.post("/index-face/", response_model=IndexFaceResponse)
async def index_face(
    username: str = Depends(get_current_user),
    face_id: str = Form(..., description="Unique ID for this face, e.g. 'student_042'"),
    source_url: str = Form(..., description="Where this image came from (URL, file path, etc.)"),
    person_id: Optional[str] = Form(None, description="Groups multiple photos of the same person. Defaults to face_id."),
    image: UploadFile = File(...),
    force: bool = Query(True, description="If false, indexing is skipped when a near-duplicate is found."),
):
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
    ai_note = f", ai_flag={ai_check['confidence']}" if ai_check["is_likely_ai"] else ""
    log_activity("index", face_id, f"person_id={person_id or face_id}, source={source_url}{ai_note}", username=username)

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
    username: str = Depends(get_current_user),
    source_url: Optional[str] = Form(None),
    person_id: Optional[str] = Form(None),
    image: UploadFile = File(...),
):
    """Replaces an existing face's photo/embedding without changing its ID."""
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

    log_activity("reindex", face_id, "photo replaced", username=username)

    return IndexFaceResponse(status="reindexed", id=face_id, person_id=person_id or face_id)