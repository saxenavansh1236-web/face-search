"""
POST /bulk-index/

Indexes multiple face images in one request, instead of calling
/index-face/ once per photo through Swagger UI. Each file gets an
auto-generated face_id (person_id + running number), all sharing the
same person_id, so this doubles as the easiest way to add several
reference photos of one person for best-of-N matching.

RBAC: requires the 'analyst' role or higher, same as /index-face/,
since this also adds indexed biometric data.

CONSENT: one consent record is required per BATCH (i.e. per person_id),
not per individual photo — all photos in a batch are treated as
different pictures of the same already-consented person. If you need
different photos in the same batch to represent different consent
events, index them individually via POST /index-face/ instead.
"""

from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends

from app.services.embedding_service import get_embedding, EmbeddingError
from app.services.vector_store import add_face
from app.services.upload_utils import save_upload_to_tempfile
from app.services.thumbnail_utils import make_thumbnail_b64
from app.services.activity_log import log_activity
from app.services import consent_store
from app.api.routes.auth import require_role
from app.schemas import BulkIndexResponse, BulkIndexResult

router = APIRouter()


@router.post("/bulk-index/", response_model=BulkIndexResponse)
async def bulk_index(
    username: str = Depends(require_role("analyst")),
    person_id: str = Form(..., description="All uploaded photos are grouped under this person."),
    source_url: str = Form("bulk_upload", description="Shared source label for this batch."),
    images: List[UploadFile] = File(..., description="Multiple photos of the same person."),
    consent_given: bool = Form(..., description="Must be true — indexing is refused otherwise."),
    consent_given_by: str = Form(..., description="Name/ID of the person who gave consent (usually the subject themself)."),
    consent_method: str = Form(..., description="One of: written_form, verbal_recorded, self_registered, institutional."),
    purpose: str = Form(..., description="Why these faces are being indexed."),
):
    if not consent_given:
        raise HTTPException(
            status_code=422,
            detail=(
                "Consent is required to index faces. Set consent_given=true "
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
        raise HTTPException(status_code=422, detail="purpose is required (why these faces are being indexed).")

    results = []
    indexed_count = 0
    failed_count = 0

    for i, image in enumerate(images, start=1):
        face_id = f"{person_id}_{i:02d}"
        tmp_path = await save_upload_to_tempfile(image)
        try:
            embedding = get_embedding(tmp_path)
            thumbnail = make_thumbnail_b64(tmp_path)
            add_face(
                face_id=face_id,
                embedding=embedding,
                source_url=source_url,
                person_id=person_id,
                thumbnail=thumbnail,
            )
            # One consent record per resulting face_id, all sharing the
            # same consent_given_by/method/purpose supplied for this batch.
            consent_store.record_consent(
                face_id=face_id,
                consent_given_by=consent_given_by,
                consent_method=consent_method,
                purpose=purpose,
                recorded_by_username=username,
            )
            log_activity("bulk_index", face_id, f"person_id={person_id}, consent_method={consent_method}", username=username)
            results.append(BulkIndexResult(id=face_id, status="indexed"))
            indexed_count += 1
        except EmbeddingError as exc:
            results.append(BulkIndexResult(id=face_id, status="failed", detail=str(exc)))
            failed_count += 1
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    return BulkIndexResponse(
        results=results,
        indexed_count=indexed_count,
        failed_count=failed_count,
    )
