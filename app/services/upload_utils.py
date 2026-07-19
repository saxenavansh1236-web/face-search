"""
upload_utils.py

Shared helper for saving an uploaded image to a temp file, enforcing a
max file-size limit. Image dimensions are never restricted here.
"""

import tempfile
from pathlib import Path

from fastapi import UploadFile, HTTPException

from app.core.config import settings


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

    return tmp_path