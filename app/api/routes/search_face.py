"""
POST /search-face/

Query endpoint. Accepts a new face image, generates its embedding, and
finds the closest stored match(es) by cosine distance.

Results are grouped by person_id, keeping each person's single best
(lowest-distance) match — this is the "best-of-N" behavior: if someone
has 3 reference photos indexed, the closest of the 3 represents them.

Each result gets a confidence label using two thresholds:
  - distance < high_confidence_threshold  -> "high"
  - distance < possible_match_threshold   -> "possible"
  - otherwise                              -> "none"

RBAC: requires the 'viewer' role or higher — i.e. any authenticated,
registered user can search, since search alone doesn't add or modify
indexed data. Raise this to require_role("investigator") if you want
searching itself restricted to a higher-privilege role (e.g. if this
tool is used for case-based investigations rather than a general
closed-dataset lookup).
"""

from pathlib import Path
import base64
import tempfile

from fastapi import APIRouter, UploadFile, File, Query, HTTPException, Depends

from app.services.embedding_service import get_embedding, EmbeddingError, verify_faces
from app.services.vector_store import search_faces, get_all_faces
from app.services.upload_utils import save_upload_to_tempfile
from app.core.config import settings
from app.services.activity_log import log_activity
from app.api.routes.auth import require_role
from app.services.ai_detection import detect_ai_generated
from app.schemas import SearchFaceResponse, MatchResult

router = APIRouter()


def _confidence_label(distance: float) -> str:
    if distance < settings.high_confidence_threshold:
        return "high"
    if distance < settings.possible_match_threshold:
        return "possible"
    return "none"


@router.post("/search-face/", response_model=SearchFaceResponse)
async def search_face(
    username: str = Depends(require_role("viewer")),
    image: UploadFile = File(...),
    top_k: int = Query(20, ge=1, le=50, description="How many raw candidates to fetch before grouping by person."),
    threshold: float = Query(
        settings.match_threshold,
        ge=0.0,
        description="Cosine distance below which a result counts as a confident match.",
    ),
):
    tmp_path = await save_upload_to_tempfile(image)

    ai_check = detect_ai_generated(tmp_path, original_filename=image.filename or "")

    tmp_path_for_rerank = tmp_path  # keep path for re-ranking below, deleted at the end
    try:
        embedding = get_embedding(tmp_path)
    except EmbeddingError as exc:
        Path(tmp_path).unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc))

    raw_results = search_faces(embedding, top_k=top_k)

    ids = raw_results.get("ids", [[]])[0]
    distances = raw_results.get("distances", [[]])[0]
    metadatas = raw_results.get("metadatas", [[]])[0]

    # Group by person_id, keep only the best (lowest-distance) hit per person
    best_per_person = {}
    for face_id, distance, meta in zip(ids, distances, metadatas):
        meta = meta or {}
        person_id = meta.get("person_id", face_id)
        if person_id not in best_per_person or distance < best_per_person[person_id]["distance"]:
            best_per_person[person_id] = {
                "id": face_id,
                "person_id": person_id,
                "source_url": meta.get("source_url"),
                "distance": distance,
            }

    ranked = sorted(best_per_person.values(), key=lambda m: m["distance"])

    matches = [
        MatchResult(
            id=m["id"],
            person_id=m["person_id"],
            source_url=m["source_url"],
            distance=m["distance"],
            confidence=_confidence_label(m["distance"]),
            match_found=m["distance"] < threshold,
        )
        for m in ranked
    ]

    # Re-ranking: only worth running when the primary distance is
    # genuinely ambiguous. A very low distance is already a confident
    # match (re-checking against a small, lossy thumbnail only adds
    # noise, as observed in testing — the thumbnail's low resolution
    # can make DeepFace.verify() misjudge an already-clear match). A
    # distance already well above possible_match_threshold is already
    # confidently "not a match." Only the in-between "possible" zone
    # benefits from a second, independent opinion.
    rerank_note = None
    if (
        settings.enable_reranking
        and matches
        and settings.rerank_skip_below_distance <= matches[0].distance <= settings.rerank_skip_above_distance
    ):
        top_thumbnail_b64 = None
        top_faces = get_all_faces()
        for f in top_faces:
            if f["id"] == matches[0].id:
                top_thumbnail_b64 = f.get("thumbnail")
                break

        if top_thumbnail_b64 and top_thumbnail_b64.startswith("data:image"):
            try:
                header, encoded = top_thumbnail_b64.split(",", 1)
                candidate_bytes = base64.b64decode(encoded)
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as cand_tmp:
                    cand_tmp.write(candidate_bytes)
                    cand_tmp_path = cand_tmp.name

                verify_result = verify_faces(tmp_path_for_rerank, cand_tmp_path)
                rerank_note = (
                    f"Re-ranking check against top candidate '{matches[0].id}': "
                    f"verified={verify_result.get('verified')}, "
                    f"distance={verify_result.get('distance'):.4f}"
                )
                Path(cand_tmp_path).unlink(missing_ok=True)
            except Exception as exc:
                rerank_note = f"Re-ranking check could not run: {exc}"
    elif matches:
        if matches[0].distance < settings.rerank_skip_below_distance:
            rerank_note = "Re-ranking skipped: primary distance already confidently indicates a match."
        elif matches[0].distance > settings.rerank_skip_above_distance:
            rerank_note = "Re-ranking skipped: primary distance already confidently indicates no match."

    best_match_found = bool(matches) and matches[0].distance < threshold

    top_result = matches[0].id if matches else "no_match"
    log_activity("search", top_result, f"best_match_found={best_match_found}", username=username)

    Path(tmp_path_for_rerank).unlink(missing_ok=True)

    return SearchFaceResponse(
        query_ok=True,
        threshold_used=threshold,
        best_match_found=best_match_found,
        matches=matches,
        query_image_ai_warning=ai_check["reason"] if ai_check["is_likely_ai"] else None,
        query_image_ai_confidence=ai_check["confidence"],
        rerank_note=rerank_note,
    )
