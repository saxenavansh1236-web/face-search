"""
upload_utils.py

Shared helper for saving an uploaded image to a temp file, enforcing a
max file-size limit. Image dimensions are never restricted here.

HEIC/HEIF uploads (e.g. straight from an iPhone) are transparently
converted to JPEG on save. This isn't just for the quality-check step —
DeepFace loads images via OpenCV (cv2.imread), which has no HEIC
support at all, regardless of any Pillow-side HEIF plugin being
registered. Converting once here means every downstream consumer
(face_quality, embedding_service, thumbnail_utils) just sees a normal
JPEG and never has to know HEIC was involved.
"""

import tempfile
from pathlib import Path

from fastapi import UploadFile, HTTPException
from PIL import Image
import pillow_heif

from app.core.config import settings

pillow_heif.register_heif_opener()

HEIC_SUFFIXES = {".heic", ".heif"}


async def save_upload_to_tempfile(image: UploadFile) -> str:
    max_bytes = int(settings.max_upload_size_mb * 1024 * 1024)
    suffix = Path(image.filename).suffix or ".jpg"

    total = 0
    chunk_size = 1024 * 1024

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        while True:
            chunk = await image.read(chunk_size)
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                tmp.close()
                Path(tmp.name).unlink(missing_ok=True)
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"Image exceeds max upload size of "
                        f"{settings.max_upload_size_mb} MB."
                    ),
                )
            tmp.write(chunk)
        tmp_path = tmp.name

    if suffix.lower() in HEIC_SUFFIXES:
        tmp_path = _convert_heic_to_jpeg(tmp_path)

    return tmp_path


def _convert_heic_to_jpeg(heic_path: str) -> str:
    """
    Converts a HEIC/HEIF file at heic_path to a new JPEG temp file and
    deletes the original. Returns the new file's path. Raises
    HTTPException(422) if the file can't be decoded (e.g. corrupt or
    not actually HEIC despite its extension).
    """
    jpeg_fd_path = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg").name

    try:
        with Image.open(heic_path) as img:
            img.convert("RGB").save(jpeg_fd_path, format="JPEG", quality=95)
    except Exception as exc:
        Path(jpeg_fd_path).unlink(missing_ok=True)
        Path(heic_path).unlink(missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail=f"Could not decode HEIC/HEIF image: {exc}",
        )

    Path(heic_path).unlink(missing_ok=True)
    return jpeg_fd_path
