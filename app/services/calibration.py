"""
calibration.py

FAR/FRR (False Accept Rate / False Reject Rate) testing harness, plus
threshold calibration support.

Reads a labeled test set from settings.calibration_data_path, laid
out as:

    calibration_data/
      person_001/
        photo1.jpg
        photo2.jpg
      person_002/
        photo1.jpg

Every pair of photos within the SAME person folder is a "genuine pair"
(same person, different photo) — used to measure the False Reject
Rate (how often the system WRONGLY says "not a match" for two photos
of the same actual person).

Every pair of photos across DIFFERENT person folders is an "impostor
pair" (different people) — used to measure the False Accept Rate (how
often the system WRONGLY says "match" for two different people).

This does NOT touch the live vector_store — it's a standalone,
read-only analysis over embeddings computed just for this test run,
so running it never pollutes or depends on your indexed production
data.
"""

import itertools
import os
from typing import List, Dict, Any, Tuple

import numpy as np

from app.core.config import settings
from app.services.embedding_service import get_embedding, EmbeddingError


def _cosine_distance(a: List[float], b: List[float]) -> float:
    a, b = np.array(a), np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 1.0
    return float(1 - np.dot(a, b) / denom)


def _load_test_photos() -> Dict[str, List[str]]:
    """Returns {person_id: [photo_path, ...]} for every subfolder
    under calibration_data_path that contains at least one image."""
    base = settings.calibration_data_path
    if not os.path.isdir(base):
        return {}

    photos_by_person: Dict[str, List[str]] = {}
    for person_id in sorted(os.listdir(base)):
        person_dir = os.path.join(base, person_id)
        if not os.path.isdir(person_dir):
            continue
        files = [
            os.path.join(person_dir, f)
            for f in sorted(os.listdir(person_dir))
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".heic", ".heif"))
        ]
        if files:
            photos_by_person[person_id] = files
    return photos_by_person


def run_calibration() -> Dict[str, Any]:
    """
    Computes embeddings for every test photo, builds genuine and
    impostor distance lists, and returns a summary including per-
    threshold FAR/FRR so the admin dashboard can render a curve and
    pick an informed operating threshold.
    """
    photos_by_person = _load_test_photos()

    if len(photos_by_person) < 2:
        return {
            "ok": False,
            "error": (
                f"Need at least 2 person folders with photos under "
                f"'{settings.calibration_data_path}' to run calibration "
                f"(found {len(photos_by_person)}). See calibration.py's "
                f"docstring for the expected folder layout."
            ),
        }

    # Compute embeddings once per photo, skipping any that fail
    # detection (recorded as skipped, not silently dropped).
    embeddings: Dict[str, List[float]] = {}
    skipped: List[Dict[str, str]] = []
    for person_id, paths in photos_by_person.items():
        for path in paths:
            try:
                embeddings[path] = get_embedding(path)
            except EmbeddingError as exc:
                skipped.append({"path": path, "reason": str(exc)})

    genuine_distances: List[float] = []
    impostor_distances: List[float] = []

    # Genuine pairs: all photo pairs WITHIN the same person folder.
    for person_id, paths in photos_by_person.items():
        usable = [p for p in paths if p in embeddings]
        for p1, p2 in itertools.combinations(usable, 2):
            genuine_distances.append(_cosine_distance(embeddings[p1], embeddings[p2]))

    # Impostor pairs: one photo from each of two DIFFERENT person folders.
    person_ids = list(photos_by_person.keys())
    for i, j in itertools.combinations(range(len(person_ids)), 2):
        usable_i = [p for p in photos_by_person[person_ids[i]] if p in embeddings]
        usable_j = [p for p in photos_by_person[person_ids[j]] if p in embeddings]
        for p1 in usable_i:
            for p2 in usable_j:
                impostor_distances.append(_cosine_distance(embeddings[p1], embeddings[p2]))

    if not genuine_distances or not impostor_distances:
        return {
            "ok": False,
            "error": (
                "Not enough usable photos after embedding — need at least one "
                "person folder with 2+ photos (for genuine pairs) AND at least "
                "2 person folders total (for impostor pairs)."
            ),
            "skipped": skipped,
        }

    # Sweep candidate thresholds and compute FAR/FRR at each.
    candidate_thresholds = [round(t, 2) for t in np.arange(0.10, 0.71, 0.02)]
    curve = []
    for t in candidate_thresholds:
        frr = sum(1 for d in genuine_distances if d >= t) / len(genuine_distances)
        far = sum(1 for d in impostor_distances if d < t) / len(impostor_distances)
        curve.append({"threshold": t, "far": round(far, 4), "frr": round(frr, 4)})

    # Equal Error Rate point: threshold where FAR and FRR are closest —
    # a common single-number summary of overall separability, and a
    # reasonable starting point for match_threshold if you have no
    # other preference (e.g. prioritizing fewer false accepts).
    eer_point = min(curve, key=lambda c: abs(c["far"] - c["frr"]))

    current_threshold = settings.match_threshold
    current_frr = sum(1 for d in genuine_distances if d >= current_threshold) / len(genuine_distances)
    current_far = sum(1 for d in impostor_distances if d < current_threshold) / len(impostor_distances)

    return {
        "ok": True,
        "num_people": len(photos_by_person),
        "num_photos_used": len(embeddings),
        "num_genuine_pairs": len(genuine_distances),
        "num_impostor_pairs": len(impostor_distances),
        "genuine_distance_min": round(min(genuine_distances), 4),
        "genuine_distance_max": round(max(genuine_distances), 4),
        "genuine_distance_mean": round(float(np.mean(genuine_distances)), 4),
        "impostor_distance_min": round(min(impostor_distances), 4),
        "impostor_distance_max": round(max(impostor_distances), 4),
        "impostor_distance_mean": round(float(np.mean(impostor_distances)), 4),
        "curve": curve,
        "eer_point": eer_point,
        "current_threshold": current_threshold,
        "current_threshold_far": round(current_far, 4),
        "current_threshold_frr": round(current_frr, 4),
        "skipped": skipped,
    }
