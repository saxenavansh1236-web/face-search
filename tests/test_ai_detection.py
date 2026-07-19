"""
test_ai_detection.py

Unit tests for the AI/edit-detection heuristics.
"""

from PIL import Image, PngImagePlugin
from app.services.ai_detection import detect_ai_generated


def test_plain_image_only_weak_signal_at_most(tmp_image_path):
    """
    A synthetic test image naturally has no camera EXIF (same as many
    real photos that were edited/exported), so the low-confidence
    "missing EXIF" signal is expected to fire — that's working as
    designed, not a false positive. What matters is it should NEVER
    reach "high" or "medium" confidence without an actual AI/edit
    signal being present.
    """
    result = detect_ai_generated(tmp_image_path)
    assert result["confidence"] in ("none", "low")


def test_ai_filename_pattern_flagged(tmp_image_path):
    result = detect_ai_generated(tmp_image_path, original_filename="ChatGPT Image Jul 1, 2026.png")
    assert result["is_likely_ai"] is True
    assert result["confidence"] == "high"


def test_stable_diffusion_metadata_flagged(tmp_path):
    img_path = tmp_path / "sd_output.png"
    img = Image.new("RGB", (64, 64), color=(100, 110, 120))
    meta = PngImagePlugin.PngInfo()
    meta.add_text("parameters", "a photo of a person, masterpiece, Steps: 20")
    img.save(img_path, pnginfo=meta)

    result = detect_ai_generated(str(img_path))
    assert result["is_likely_ai"] is True
    assert result["confidence"] == "high"
    assert "stable diffusion" in result["reason"].lower()