"""Tests for PDF validation, hashing, and PyMuPDF page/crop rendering."""

from __future__ import annotations

import pytest

from conftest import make_sample_pdf as _make_pdf
from pdf2anki.pdf_media import (
    PDFValidationError,
    media_filename,
    render_crop,
    render_page,
    validate_pdf,
)
from pdf2anki.schemas import ImageCrop


def test_validate_pdf_accepts_valid_pdf() -> None:
    data = _make_pdf(pages=2)
    info = validate_pdf(data)
    assert info.page_count == 2
    assert len(info.pdf_hash) == 16


def test_validate_pdf_rejects_non_pdf_mime() -> None:
    data = _make_pdf(pages=1)
    with pytest.raises(PDFValidationError, match="not a PDF"):
        validate_pdf(data, mime_type="text/plain")


def test_validate_pdf_rejects_oversized() -> None:
    data = _make_pdf(pages=1)
    with pytest.raises(PDFValidationError, match="limit"):
        validate_pdf(data, max_bytes=10)


def test_validate_pdf_rejects_encrypted() -> None:
    data = _make_pdf(pages=1, encrypt=True)
    with pytest.raises(PDFValidationError, match="Encrypted"):
        validate_pdf(data)


def test_render_page_and_crop_produce_png() -> None:
    data = _make_pdf(pages=1)
    full_png = render_page(data, 1, dpi=72)
    assert full_png.startswith(b"\x89PNG")

    crop = ImageCrop(x0=0.1, y0=0.2, x1=0.7, y1=0.8)
    crop_png = render_crop(data, 1, crop, dpi=72)
    assert crop_png.startswith(b"\x89PNG")
    assert len(crop_png) < len(full_png)


def test_render_crop_falls_back_on_invalid_crop() -> None:
    data = _make_pdf(pages=1)
    bad_crop = ImageCrop.model_construct(x0=0.9, y0=0.9, x1=0.1, y1=0.1)
    fallback_png = render_crop(data, 1, bad_crop, dpi=72)
    full_png = render_page(data, 1, dpi=72)
    assert fallback_png == full_png


def test_media_filename_is_collision_proof() -> None:
    crop_a = ImageCrop(x0=0.0, y0=0.0, x1=0.5, y1=0.5)
    crop_b = ImageCrop(x0=0.1, y0=0.1, x1=0.5, y1=0.5)
    name_a = media_filename("deadbeef12345678", 1, "q1", crop_a)
    name_b = media_filename("deadbeef12345678", 1, "q1", crop_b)
    assert name_a != name_b
    assert name_a.endswith(".png")
