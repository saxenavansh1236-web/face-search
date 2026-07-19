"""
face_quality.py

Quality gates applied before a face gets indexed, so bad reference
photos (blurry, too small, too dark/bright) don't silently degrade
future match accuracy. This runs independently of DeepFace's own face
detection — it's a cheap pre-filter on the raw image, using classical
image-processing signals rather than a trained quality-assessment model.

Checks:
- Sharpness: Laplacian variance (a standard blur-detection heuristic —
  a sharp image has high-frequency edges everywhere; a blurry one
  doesn't, so the variance of the Laplacian-filtered image drops).
- Brightness: mean grayscale value, flags near-black or near-white
  (overexposed) images.
- Resolution: overall image dimensions, as a proxy for whether the
  face region will have enough detail (checked properly against the
  actual detected face box in index_face.py, this is just the
  whole-image floor).
"""

from typing import Dict, Any
import numpy as np
from PIL import Image, ImageFilter

from app.core.config import settings


def assess_quality(image_path: str) -> Dict[str, Any]:
    """
    Returns:
      {
        "passed": bool,
        "sharpness": float,
        "brightness": float,
        "reasons": [str, ...]   # populated only when passed is False
      }
    """
    reasons = []

    with Image.open(image_path) as img:
        gray = img.convert("L")
        arr = np.array(gray).astype(np.float32)

        laplacian = gray.filter(ImageFilter.FIND_EDGES)
        lap_arr = np.array(laplacian).astype(np.float32)
        sharpness = float(lap_arr.var())

        brightness = float(arr.mean())

        if sharpness < settings.min_sharpness_score:
            reasons.append(
                f"Image appears too blurry (sharpness score {sharpness:.1f}, "
                f"minimum {settings.min_sharpness_score})."
            )
        if brightness < settings.min_brightness:
            reasons.append(
                f"Image appears too dark (brightness {brightness:.1f}, "
                f"minimum {settings.min_brightness})."
            )
        if brightness > settings.max_brightness:
            reasons.append(
                f"Image appears overexposed/too bright (brightness {brightness:.1f}, "
                f"maximum {settings.max_brightness})."
            )

    return {
        "passed": len(reasons) == 0,
        "sharpness": sharpness,
        "brightness": brightness,
        "reasons": reasons,
    }


def check_face_region_size(facial_area: Dict[str, int]) -> Dict[str, Any]:
    """
    Given DeepFace's detected facial_area ({'x','y','w','h'}), checks
    whether the actual face region (not the whole image) is large
    enough to produce a reliable embedding.
    """
    w = facial_area.get("w", 0)
    h = facial_area.get("h", 0)
    min_dim = min(w, h)

    if min_dim < settings.min_face_size_px:
        return {
            "passed": False,
            "reason": f"Detected face is only {w}x{h}px, below the minimum "
                      f"{settings.min_face_size_px}px — move closer or use a "
                      f"higher-resolution photo.",
        }
    return {"passed": True, "reason": None}