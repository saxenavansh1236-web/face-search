"""
conftest.py

Shared pytest fixtures. Sets required environment variables before
any app module is imported (config.py reads them at import time).
"""

import os
import sys
import tempfile

_test_dir = tempfile.mkdtemp(prefix="face_search_test_")
os.environ.setdefault("DB_PATH", os.path.join(_test_dir, "face_db"))
os.environ.setdefault("ACTIVITY_LOG_PATH", os.path.join(_test_dir, "face_db", "activity_log.jsonl"))
os.environ.setdefault("SESSION_SECRET_KEY", "test-secret-key")
os.environ.setdefault("JWT_SECRET_KEY", "test-jwt-secret")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from PIL import Image


@pytest.fixture
def tmp_image_path(tmp_path):
    """Creates a simple synthetic RGB test image and returns its path."""
    img_path = tmp_path / "test_image.jpg"
    arr = (np.random.rand(200, 200, 3) * 255).astype(np.uint8)
    Image.fromarray(arr).save(img_path, quality=90)
    return str(img_path)


@pytest.fixture
def dark_image_path(tmp_path):
    """A near-black image, expected to fail quality checks."""
    img_path = tmp_path / "dark_image.jpg"
    arr = np.full((200, 200, 3), 5, dtype=np.uint8)
    Image.fromarray(arr).save(img_path, quality=90)
    return str(img_path)