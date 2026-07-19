"""
embedding_service.py

Wraps DeepFace so the rest of the app never has to think about model
details. Given any face image path, returns a 512-dimensional embedding.

Configurable via app/core/config.py:
- model_name: "Facenet512" (default) or "ArcFace" (InsightFace's model,
  bundled through DeepFace rather than a separate insightface install —
  avoids a second heavy ML dependency stack while still giving access
  to ArcFace's embeddings, which are widely regarded as state-of-art
  for face verification).
- detector_backend: "opencv" (fast, default fallback) or "retinaface"
  (recommended — much better on small, angled, or partially occluded
  faces; this is what "RetinaFace for higher-quality detection" means
  in practice without hand-rolling a detector from scratch).
- align_faces: whether to rotate/crop faces to a canonical pose before
  embedding (face alignment / pose normalization).

This same function is used by BOTH the ingestion endpoint and the
search endpoint, which is what guarantees a query face and a stored
face end up in the same vector space and can be meaningfully compared.
"""

from typing import List
from deepface import DeepFace

from app.core.config import settings


class EmbeddingError(Exception):
    """Raised when a face embedding could not be generated (e.g. no face found)."""
    pass


def get_embedding(image_path: str) -> List[float]:
    """
    Detects the face in `image_path` (using settings.detector_backend),
    aligns it if settings.align_faces is True, and returns its
    embedding as a plain list of floats.
    """
    try:
        result = DeepFace.represent(
            img_path=image_path,
            model_name=settings.model_name,
            detector_backend=settings.detector_backend,
            align=settings.align_faces,
            enforce_detection=True,
        )
    except Exception as exc:
        raise EmbeddingError(f"Could not generate embedding for '{image_path}': {exc}") from exc

    if not result:
        raise EmbeddingError(f"No face detected in '{image_path}'")

    return result[0]["embedding"]


def verify_faces(image_path_1: str, image_path_2: str) -> dict:
    """
    Dedicated pairwise verification (used for re-ranking search
    results). DeepFace.verify() runs its own purpose-built comparison
    logic rather than a raw cosine distance on cached embeddings,
    which can catch borderline cases the initial ANN search missed.
    Returns DeepFace's native result dict, including 'verified' (bool)
    and 'distance'.
    """
    return DeepFace.verify(
        img1_path=image_path_1,
        img2_path=image_path_2,
        model_name=settings.model_name,
        detector_backend=settings.detector_backend,
        align=settings.align_faces,
    )