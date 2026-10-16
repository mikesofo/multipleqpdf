"""Shared pytest fixtures for PDF generation."""

from __future__ import annotations

from collections.abc import Callable

import fitz
import pytest


def make_sample_pdf(pages: int = 2, encrypt: bool = False) -> bytes:
    """Create a small in-memory PDF with text and vector shapes for testing."""
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=200, height=200)
        page.insert_text((20, 40), f"Page {i + 1} content")
        page.draw_rect(fitz.Rect(20, 60, 120, 140), color=(1, 0, 0), fill=(0, 0, 1))
        page.draw_circle((100, 100), 40, color=(0, 0.5, 0))
    if encrypt:
        return doc.tobytes(
            encryption=fitz.PDF_ENCRYPT_AES_256,
            owner_pw="owner",
            user_pw="secret",
        )
    return doc.tobytes()


def _png_bytes(width: int, height: int, color: tuple[int, int, int]) -> bytes:
    """Build a solid-color RGB PNG of the given size using PyMuPDF."""
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, width, height))
    pix.set_rect(pix.irect, color)
    return pix.tobytes("png")


def make_pdf_with_image(
    image_page_1based: int = 1,
    pages: int = 1,
    image_width: int = 120,
    image_height: int = 120,
) -> bytes:
    """Create a PDF that embeds a real raster image on one page, text-only elsewhere."""
    doc = fitz.open()
    png = _png_bytes(image_width, image_height, (10, 120, 60))
    for i in range(pages):
        page = doc.new_page(width=200, height=200)
        page.insert_text((20, 40), f"Page {i + 1} content")
        if i + 1 == image_page_1based:
            page.insert_image(fitz.Rect(20, 60, 180, 180), stream=png)
    return doc.tobytes()


@pytest.fixture
def pdf_factory() -> Callable[..., bytes]:
    """Return a factory function that builds test PDFs."""
    return make_sample_pdf
