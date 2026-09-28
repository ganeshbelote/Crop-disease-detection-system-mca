"""
Upload validation utilities: file type, size and basic content checks.

Kept deliberately simple and dependency-light (no arbitrary filesystem
access, no trusting client-supplied content-type alone) since this is the
main untrusted-input boundary of the API.
"""

import io
import os
from dataclasses import dataclass

from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from app.config import settings


class UploadValidationError(ValueError):
    """Raised when an uploaded file fails validation. Message is safe to
    show directly to the client."""


@dataclass
class ValidatedImage:
    content: bytes
    safe_filename: str


def _safe_filename(original_filename: str) -> str:
    """Strip directory components and any characters that are not
    alphanumeric, dash, underscore or dot, to prevent path traversal and
    filesystem-unsafe names. Never trust a client-supplied filename as-is.
    """
    base = os.path.basename(original_filename or "upload")
    cleaned = "".join(c for c in base if c.isalnum() or c in ("-", "_", "."))
    return cleaned[-200:] or "upload.jpg"


async def validate_and_read_upload(file: UploadFile) -> ValidatedImage:
    """Validate content-type, extension, size and that the bytes actually
    decode as an image. Returns the raw bytes and a sanitized filename.

    Raises UploadValidationError with a client-safe message on any failure.
    """
    filename = file.filename or "upload"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in settings.ALLOWED_EXTENSIONS:
        raise UploadValidationError(
            f"Unsupported file extension '{ext}'. Allowed: {', '.join(sorted(settings.ALLOWED_EXTENSIONS))}"
        )

    if file.content_type and file.content_type not in settings.ALLOWED_CONTENT_TYPES:
        raise UploadValidationError(
            f"Unsupported content type '{file.content_type}'. Please upload a JPEG or PNG image."
        )

    content = await file.read()

    if len(content) == 0:
        raise UploadValidationError("Uploaded file is empty.")

    if len(content) > settings.MAX_UPLOAD_BYTES:
        raise UploadValidationError(
            f"File too large ({len(content) / (1024 * 1024):.1f} MB). "
            f"Maximum allowed size is {settings.MAX_UPLOAD_MB} MB."
        )

    # Verify the bytes actually decode as a real image (defends against a
    # renamed non-image file passing the extension/content-type checks).
    try:
        with Image.open(io.BytesIO(content)) as img:
            img.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise UploadValidationError(f"The uploaded file is not a valid image: {exc}") from exc

    return ValidatedImage(content=content, safe_filename=_safe_filename(filename))
