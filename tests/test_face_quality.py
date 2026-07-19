"""
test_face_quality.py

Unit tests for the quality-gating heuristics (no DeepFace/TensorFlow
needed, so these run fast in CI).
"""

from app.services.face_quality import assess_quality, check_face_region_size


def test_normal_image_passes(tmp_image_path):
    result = assess_quality(tmp_image_path)
    assert result["passed"] is True
    assert result["reasons"] == []


def test_dark_image_fails(dark_image_path):
    result = assess_quality(dark_image_path)
    assert result["passed"] is False
    assert any("dark" in r.lower() for r in result["reasons"])


def test_face_region_size_pass():
    result = check_face_region_size({"x": 0, "y": 0, "w": 200, "h": 200})
    assert result["passed"] is True


def test_face_region_size_fail_when_too_small():
    result = check_face_region_size({"x": 0, "y": 0, "w": 20, "h": 20})
    assert result["passed"] is False
    assert "minimum" in result["reason"]