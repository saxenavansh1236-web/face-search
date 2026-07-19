"""
POST /bulk-index/

Indexes multiple face images in one request, instead of calling
/index-face/ once per photo through Swagger UI. Each file gets an
auto-generated face_id (person_id + running number), all sharing the
same person_id, so this doubles as the easiest way to add several
reference photos of one person for best-of-N matching.
"""

from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, Form, Depends

from app.services.embedding_service import get_embedding, EmbeddingError
from app.services.vector_store import add_face
from app.services.upload_utils import save_upload_to_tempfile
from app.services.thumbnail_utils import make_thumbnail_b64
from app.services.activity_log import log_activity
from app.api.routes.auth import get_current_user
from app.schemas import BulkIndexResponse, BulkIndexResult

router = APIRouter()


@router.post("/bulk-index/", response_model=BulkIndexResponse)
async def bulk_index(
    username: str = Depends(get_current_user),
    person_id: str = Form(..., description="All uploaded photos are grouped under this person."),
    source_url: str = Form("bulk_upload", description="Shared source label for this batch."),
    images: List[UploadFile] = File(..., description="Multiple photos of the same person."),
):
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
            log_activity("bulk_index", face_id, f"person_id={person_id}", username=username)
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