"""Shared pytest fixtures for PDF generation."""

from __future__ import annotations

from collections.abc import Callable

import pymupdf as fitz
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
            encryption=fitz.PDF_ENCRYPT_AES_256,  # type: ignore[attr-defined]
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


def make_pdf_with_two_images(
    image_width: int = 100,
    image_height: int = 100,
) -> bytes:
    """Create a 1-page PDF (400x800) with top and bottom text + image pairs.

    Top region:
      - Text "Top anchor text about glomerulus" at (20, 30)
      - Smaller image (80x80 box) at Rect(20, 50, 100, 130)
    Bottom region (far away):
      - Text "Bottom anchor text about myocardium" at (20, 620)
      - Larger image (160x140 box) at Rect(200, 640, 360, 780)
    """
    doc = fitz.open()
    page = doc.new_page(width=400, height=800)
    top_png = _png_bytes(image_width, image_height, (10, 120, 60))
    bot_png = _png_bytes(image_width, image_height, (120, 10, 60))

    page.insert_text((20, 30), "Top anchor text about glomerulus")
    page.insert_image(fitz.Rect(20, 50, 100, 130), stream=top_png)

    page.insert_text((20, 620), "Bottom anchor text about myocardium")
    page.insert_image(fitz.Rect(200, 640, 360, 780), stream=bot_png)
    return doc.tobytes()


def make_pdf_with_captioned_image(
    image_width: int = 160,
    image_height: int = 160,
) -> bytes:
    """A 1-page PDF (300x400) with a substantial image and a caption below it."""
    doc = fitz.open()
    page = doc.new_page(width=300, height=400)
    png = _png_bytes(image_width, image_height, (10, 120, 60))
    page.insert_text((20, 30), "Relevant lead-in text about the specimen")
    page.insert_image(fitz.Rect(40, 60, 260, 280), stream=png)
    page.insert_text((40, 300), "Figure 1. Photomicrograph of the specimen.")
    return doc.tobytes()


def make_pdf_with_nearby_and_far_images(
    small_decorative: bool = False,
) -> bytes:
    """A tall 1-page PDF with a captioned figure near the top anchor text and a
    far unrelated image at the bottom.

    When ``small_decorative`` is True, the top image is a tiny decorative
    thumbnail that should be ignored by the area floor.
    """
    doc = fitz.open()
    page = doc.new_page(width=400, height=1000)
    page.insert_text((20, 30), "Top anchor text about glomerulus")
    if small_decorative:
        thumb = _png_bytes(70, 70, (10, 120, 60))
        page.insert_image(fitz.Rect(20, 50, 48, 78), stream=thumb)
    else:
        fig = _png_bytes(140, 140, (10, 120, 60))
        page.insert_image(fitz.Rect(20, 50, 160, 190), stream=fig)
        page.insert_text((20, 210), "Figure 2. Glomerulus at high power.")
    far = _png_bytes(140, 140, (120, 10, 60))
    page.insert_image(fitz.Rect(200, 840, 360, 980), stream=far)
    return doc.tobytes()


def make_pdf_with_labeled_figures() -> bytes:
    """A 1-page PDF (400x900) with two captioned, figure-numbered images.

    Top figure:   image at Rect(20,50,180,190), caption "Figure 1. Glomerulus."
    Bottom figure: image at Rect(20,520,180,660), caption "Figure 2. Myocardium."
    """
    doc = fitz.open()
    page = doc.new_page(width=400, height=900)
    top = _png_bytes(120, 120, (10, 120, 60))
    bot = _png_bytes(120, 120, (120, 10, 60))
    page.insert_image(fitz.Rect(20, 50, 180, 190), stream=top)
    page.insert_text((20, 210), "Figure 1. Glomerulus histology overview.")
    page.insert_image(fitz.Rect(20, 520, 180, 660), stream=bot)
    page.insert_text((20, 680), "Figure 2. Myocardium cross section detail.")
    return doc.tobytes()


def make_pdf_with_toc(pages: int = 6) -> bytes:
    """Create a PDF with a 3-chapter top-level outline (chapters at 1, 3, 5)."""
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=200, height=200)
        page.insert_text((20, 40), f"Page {i + 1} content")
    toc = [
        [1, "Introduction", 1],
        [2, "Intro subsection", 2],
        [1, "Methods", 3],
        [1, "Results", 5],
    ]
    doc.set_toc(toc)
    return doc.tobytes()


def make_large_pdf_with_toc(pages: int = 40) -> bytes:
    """A >25-page PDF with level-1 chapters at pages 1, 15, and 30."""
    doc = fitz.open()
    for i in range(pages):
        page = doc.new_page(width=200, height=200)
        page.insert_text((20, 40), f"Page {i + 1} content")
    toc = [
        [1, "Alpha", 1],
        [2, "Alpha sub", 5],
        [1, "Beta", 15],
        [1, "Gamma", 30],
    ]
    doc.set_toc(toc)
    return doc.tobytes()


@pytest.fixture
def pdf_factory() -> Callable[..., bytes]:
    """Return a factory function that builds test PDFs."""
    return make_sample_pdf
