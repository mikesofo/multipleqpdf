"""Tests for UI-side default image auto-selection logic."""

from __future__ import annotations

import app as app_module
import pytest
from app import _autoselect_images

from conftest import make_pdf_with_image, make_pdf_with_two_images, make_sample_pdf
from pdf2anki.schemas import MCQ


@pytest.fixture(autouse=True)
def fake_session_state(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    state: dict[str, object] = {}
    monkeypatch.setattr(app_module.st, "session_state", state)
    return state


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
        source_support="Page 1 content",
        image_required=False,
        image_page=None,
        image_crop=None,
    )
    base.update(overrides)
    return MCQ(**base)  # type: ignore[arg-type]


def test_autoselect_enables_image_when_source_page_has_figure(
    fake_session_state: dict[str, object],
) -> None:
    pdf_bytes = make_pdf_with_image(image_page_1based=1, pages=1)
    mcqs = [_mcq(source_pages=[1])]
    _autoselect_images(mcqs, pdf_bytes, page_count=1)
    assert mcqs[0].image_required is True
    assert mcqs[0].image_page == 1
    assert fake_session_state["imgreq_q1_0"] is True
    assert fake_session_state["imgpage_q1_0"] == 1


def test_autoselect_skips_text_only_page(
    fake_session_state: dict[str, object],
) -> None:
    pdf_bytes = make_sample_pdf(pages=1)
    mcqs = [_mcq(source_pages=[1])]
    _autoselect_images(mcqs, pdf_bytes, page_count=1)
    assert mcqs[0].image_required is False
    assert mcqs[0].image_page is None
    assert fake_session_state["imgreq_q1_0"] is False


def test_autoselect_preserves_existing_model_choice(
    fake_session_state: dict[str, object],
) -> None:
    pdf_bytes = make_pdf_with_image(image_page_1based=1, pages=1)
    mcqs = [_mcq(source_pages=[1], image_required=True, image_page=1)]
    _autoselect_images(mcqs, pdf_bytes, page_count=1)
    assert mcqs[0].image_required is True
    assert mcqs[0].image_page == 1
    assert fake_session_state["imgreq_q1_0"] is True
    assert fake_session_state["imgpage_q1_0"] == 1


def test_autoselect_ignores_out_of_range_page(
    fake_session_state: dict[str, object],
) -> None:
    pdf_bytes = make_pdf_with_image(image_page_1based=1, pages=1)
    mcqs = [_mcq(source_pages=[5])]
    _autoselect_images(mcqs, pdf_bytes, page_count=1)
    assert mcqs[0].image_required is False
    assert fake_session_state["imgreq_q1_0"] is False


def test_autoselect_overwrites_stale_false_widget_state(
    fake_session_state: dict[str, object],
) -> None:
    fake_session_state["imgreq_q1_0"] = False
    fake_session_state["imgpage_q1_0"] = 1
    pdf_bytes = make_pdf_with_image(image_page_1based=1, pages=1)
    mcqs = [_mcq(source_pages=[1])]
    _autoselect_images(mcqs, pdf_bytes, page_count=1)
    assert mcqs[0].image_required is True
    assert fake_session_state["imgreq_q1_0"] is True
    assert fake_session_state["imgpage_q1_0"] == 1


def test_autoselect_does_not_enable_distant_unrelated_image(
    fake_session_state: dict[str, object],
) -> None:
    import pymupdf as fitz

    from conftest import _png_bytes

    doc = fitz.open()
    page = doc.new_page(width=400, height=800)
    page.insert_text((20, 30), "Top text about unrelated topic")
    png = _png_bytes(100, 100, (20, 40, 160))
    page.insert_image(fitz.Rect(200, 640, 360, 780), stream=png)
    pdf_bytes = doc.tobytes()

    mcqs = [_mcq(source_pages=[1], source_support="Top text about unrelated topic")]
    _autoselect_images(mcqs, pdf_bytes, page_count=1)
    assert mcqs[0].image_required is False
    assert fake_session_state["imgreq_q1_0"] is False


def test_autoselect_picks_nearest_image_among_multiple(
    fake_session_state: dict[str, object],
) -> None:
    pdf_bytes = make_pdf_with_two_images()
    mcqs = [
        _mcq(
            source_pages=[1],
            source_support="Top anchor text about glomerulus",
        )
    ]
    _autoselect_images(mcqs, pdf_bytes, page_count=1)
    assert mcqs[0].image_required is True
    assert mcqs[0].image_crop is not None
    assert mcqs[0].image_crop.y1 < 0.35
