"""Image validation, resizing, and preparation for multimodal LLM input."""

from __future__ import annotations

import io
import logging

from PIL import Image

log = logging.getLogger(__name__)

# Pillow format → MIME type
_MIME_MAP = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "GIF": "image/gif",
}


def validate_and_detect_mime(data: bytes) -> str | None:
    """Open image with Pillow and return MIME type, or None if invalid."""
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        return _MIME_MAP.get(img.format, f"image/{(img.format or 'unknown').lower()}")
    except Exception:
        return None


def resize_if_needed(data: bytes, max_dim: int = 1024) -> tuple[bytes, str]:
    """Resize image proportionally if larger than max_dim. Returns (bytes, mime)."""
    img = Image.open(io.BytesIO(data))
    mime = _MIME_MAP.get(img.format, "image/jpeg")

    # Convert palette/CMYK to RGB for consistent encoding
    if img.mode in ("P", "CMYK"):
        img = img.convert("RGBA" if "transparency" in img.info else "RGB")

    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.LANCZOS)

    buf = io.BytesIO()
    if img.mode == "RGBA":
        img.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    else:
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(buf, format="JPEG", quality=85)
        mime = "image/jpeg"

    return buf.getvalue(), mime


def prepare_images_for_llm(
    raw_images: list[bytes],
    max_dim: int = 1024,
    max_count: int = 4,
) -> list[tuple[bytes, str]]:
    """Validate, resize, and limit images for LLM input.

    Returns list of (bytes, mime_type) tuples ready for base64 encoding.
    """
    max_bytes = 20 * 1024 * 1024  # 20MB raw limit
    result = []
    for data in raw_images[:max_count]:
        if len(data) > max_bytes:
            log.warning("Skipping oversized image (%d bytes)", len(data))
            continue
        if not validate_and_detect_mime(data):
            log.warning("Skipping invalid image (%d bytes)", len(data))
            continue
        try:
            img_bytes, mime = resize_if_needed(data, max_dim)
            result.append((img_bytes, mime))
        except Exception:
            log.warning("Failed to process image (%d bytes)", len(data), exc_info=True)
    return result
