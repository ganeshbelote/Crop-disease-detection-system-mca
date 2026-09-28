"""
Small filesystem helpers: unique-but-readable filenames and guaranteed
temporary file cleanup.
"""

import os
import tempfile
import uuid
from contextlib import contextmanager


def unique_filename(safe_filename: str) -> str:
    """Prefix a sanitized filename with a short random id so concurrent
    uploads never collide, while keeping the original name for readability
    in the database/history view."""
    return f"{uuid.uuid4().hex[:8]}_{safe_filename}"


@contextmanager
def temporary_file(content: bytes, suffix: str = ".jpg"):
    """Write `content` to a temp file for the duration of the `with` block
    and guarantee it is deleted afterwards, even if an exception occurs.
    Used for any processing step that needs a real file path rather than
    in-memory bytes.
    """
    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(content)
        yield path
    finally:
        if os.path.exists(path):
            os.remove(path)
