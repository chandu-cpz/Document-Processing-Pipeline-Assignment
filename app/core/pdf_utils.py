"""
app/core/pdf_utils.py
---------------------
Async PDF → base64-encoded JPEG page converter.

Uses PyMuPDF (fitz) at the configured DPI (default 150).
Outputs JPEG at 85% quality — same resolution as PNG but ~6-8x smaller
payload, keeping 18-page batches well within OpenRouter's request limits.

CPU-bound render is offloaded to a thread pool via asyncio.to_thread
so it never blocks the async event loop.
"""
import asyncio
import base64
import logging

import fitz  # PyMuPDF

logger = logging.getLogger("uvicorn.error")

from app.graph.state import PageData  # canonical definition

# PageData is imported from app.graph.state (canonical definition).
# It has: page_idx (int), b64_image (str), mime_type (str).


def _render_pdf_sync(pdf_bytes: bytes, dpi: int) -> list[PageData]:
    """
    CPU-bound: renders every PDF page to JPEG and base64-encodes it.

    Args:
        pdf_bytes: Raw PDF file bytes.
        dpi:       Render resolution. 150 DPI gives clear, readable images
                   while JPEG compression keeps each page ~100-300 KB.

    Returns:
        List of PageData dicts, one per page, in order.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise ValueError(f"Cannot open PDF: {exc}") from exc

    pages: list[PageData] = []
    scale = dpi / 72.0
    matrix = fitz.Matrix(scale, scale)

    for idx in range(len(doc)):
        try:
            page = doc.load_page(idx)
            pixmap = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False)
            # JPEG at q=85: sharp enough for OCR, ~6-8x smaller than lossless PNG
            jpeg_bytes = pixmap.tobytes("jpeg", jpg_quality=85)
            b64 = base64.b64encode(jpeg_bytes).decode("ascii")
            pages.append({
                "page_idx": idx,
                "b64_image": b64,
                "mime_type": "image/jpeg",
            })
            logger.debug(
                f"Page {idx}: {pixmap.width}x{pixmap.height}px "
                f"→ JPEG {len(jpeg_bytes):,} bytes"
            )
            pixmap = None  # free memory immediately
        except Exception as exc:
            raise RuntimeError(f"Failed to render page {idx}: {exc}") from exc

    doc.close()
    total_b64_kb = sum(len(p["b64_image"]) * 3 // 4 for p in pages) // 1024
    logger.info(
        f"PDF rendered: {len(pages)} pages at {dpi} DPI "
        f"(total payload ≈ {total_b64_kb:,} KB)"
    )
    return pages


async def pdf_to_pages(pdf_bytes: bytes, dpi: int = 150) -> list[PageData]:
    """
    Async entry-point: offloads the CPU-bound PDF render to a thread pool.
    """
    return await asyncio.to_thread(_render_pdf_sync, pdf_bytes, dpi)
