"""Tests for UI-side default image auto-selection logic."""

from __future__ import annotations

from app import _autoselect_images
from conftest import make_pdf_with_image, make_sample_pdf
from pdf2anki.schemas import MCQ


def _mcq(**overrides: object) -> MCQ:
    base = dict(
        id="q1",
        learning_objective="obj",
        topic="Topic",
        stem="Stem?",
        options=["A", "B", "C", "D"],
        correct_index=0,
        explanation="Because A.",
        source_pages=[1],
        image_required=False,
        image_page=None,
        image_crop=None,
    )
    base.update(overrides)
    return MCQ(**base)  # type: ignore[arg-type]


def test_autoselect_enables_image_when_source_page_has_figure() -> None:
    pdf_bytes = make_pdf_with_image(image_page_1based=1, pages=1)
    mcqs = [_mcq(source_pages=[1])]
    _autoselect_images(mcqs, pdf_bytes, page_count=1)
    assert mcqs[0].image_required is True
    assert mcqs[0].image_page == 1


def test_autoselect_skips_text_only_page() -> None:
    pdf_bytes = make_sample_pdf(pages=1)
    mcqs = [_mcq(source_pages=[1])]
    _autoselect_images(mcqs, pdf_bytes, page_count=1)
    assert mcqs[0].image_required is False
    assert mcqs[0].image_page is None


def test_autoselect_preserves_existing_model_choice() -> None:
    pdf_bytes = make_pdf_with_image(image_page_1based=1, pages=1)
    mcqs = [_mcq(source_pages=[1], image_required=True, image_page=1)]
    _autoselect_images(mcqs, pdf_bytes, page_count=1)
    assert mcqs[0].image_required is True
    assert mcqs[0].image_page == 1


def test_autoselect_ignores_out_of_range_page() -> None:
    pdf_bytes = make_pdf_with_image(image_page_1based=1, pages=1)
    mcqs = [_mcq(source_pages=[5])]
    _autoselect_images(mcqs, pdf_bytes, page_count=1)
    assert mcqs[0].image_required is False
