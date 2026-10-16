"""Tests for UI-side default image auto-selection logic."""

from __future__ import annotations

import app as app_module
import pytest
from app import _autoselect_images

from conftest import (
    make_pdf_with_image,
    make_sample_pdf,
)
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


def _fig_mcq(**overrides: object) -> MCQ:
    overrides.setdefault("question_type", "figure_grounded")
    overrides.setdefault("image_required", True)
    overrides.setdefault("image_page", 1)
    return _mcq(**overrides)


def test_autoselect_text_based_never_enables_image(
    fake_session_state: dict[str, object],
) -> None:
    # Text-based question on a page WITH a figure must stay image-free.
    pdf_bytes = make_pdf_with_image(image_page_1based=1, pages=1)
    mcqs = [_mcq(source_pages=[1])]  # defaults to text_based
    _autoselect_images(mcqs, pdf_bytes, page_count=1, promote=False)
    assert mcqs[0].image_required is False
    assert mcqs[0].image_page is None
    assert fake_session_state["imgreq_q1_0"] is False


def test_autoselect_text_based_clears_stale_true_state(
    fake_session_state: dict[str, object],
) -> None:
    fake_session_state["imgreq_q1_0"] = True
    fake_session_state["imgpage_q1_0"] = 1
    pdf_bytes = make_pdf_with_image(image_page_1based=1, pages=1)
    mcqs = [_mcq(source_pages=[1])]
    _autoselect_images(mcqs, pdf_bytes, page_count=1, promote=False)
    assert fake_session_state["imgreq_q1_0"] is False
    assert "imgpage_q1_0" not in fake_session_state


def test_autoselect_figure_grounded_preserves_image(
    fake_session_state: dict[str, object],
) -> None:
    pdf_bytes = make_pdf_with_image(image_page_1based=1, pages=1)
    mcqs = [_fig_mcq(source_pages=[1])]
    _autoselect_images(mcqs, pdf_bytes, page_count=1, promote=True)
    assert mcqs[0].image_required is True
    assert mcqs[0].image_page == 1
    assert fake_session_state["imgreq_q1_0"] is True
    assert fake_session_state["imgpage_q1_0"] == 1


def test_autoselect_skips_text_only_page(
    fake_session_state: dict[str, object],
) -> None:
    pdf_bytes = make_sample_pdf(pages=1)
    mcqs = [_mcq(source_pages=[1])]
    _autoselect_images(mcqs, pdf_bytes, page_count=1, promote=False)
    assert mcqs[0].image_required is False
    assert mcqs[0].image_page is None
    assert fake_session_state["imgreq_q1_0"] is False
