"""
thumbnail_utils.py

Generates a small base64-encoded JPEG thumbnail from an uploaded image,
so the admin dashboard can show a visual preview per face instead of
just an ID string. Stored directly in ChromaDB metadata (as a string),
so no separate file storage is needed.
"""

import base64
from io import BytesIO
from PIL import Image

from app.core.config import settings


def make_thumbnail_b64(image_path: str) -> str:
    """
    Opens the image at `image_path`, resizes it to a square thumbnail
    (settings.thumbnail_size), and returns a base64 JPEG data URI ready
    to drop straight into an <img src="..."> tag.
    """
    size = settings.thumbnail_size
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        w, h = img.size
        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        img = img.resize((size, size))

        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=80)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return f"data:image/jpeg;base64,{encoded}"