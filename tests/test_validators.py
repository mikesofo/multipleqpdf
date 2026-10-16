"""Tests for MCQ validation rules."""

from __future__ import annotations

from pdf2anki.schemas import MCQ, ImageCrop
from pdf2anki.validators import filter_valid_mcqs, validate_mcq


def _mcq(**overrides: object) -> MCQ:
    base = dict(
        id="q1",
        learning_objective="obj",
        topic="topic",
        stem="What is correct?",
        options=["A", "B", "C", "D"],
        correct_index=0,
        explanation="Because A.",
        source_pages=[1],
        image_required=False,
    )
    base.update(overrides)
    return MCQ(**base)  # type: ignore[arg-type]


def test_valid_mcq_passes() -> None:
    result = validate_mcq(_mcq(), page_count=10)
    assert result.is_valid
    assert result.errors == []


def test_rejects_duplicate_options() -> None:
    result = validate_mcq(_mcq(options=["A", "A", "C", "D"]), page_count=10)
    assert not result.is_valid
    assert any("distinct" in e.lower() for e in result.errors)


def test_rejects_empty_option() -> None:
    result = validate_mcq(_mcq(options=["A", "", "C", "D"]), page_count=10)
    assert not result.is_valid
    assert any("non-empty" in e.lower() for e in result.errors)


def test_rejects_empty_explanation() -> None:
    result = validate_mcq(_mcq(explanation="   "), page_count=10)
    assert not result.is_valid
    assert any("explanation" in e.lower() for e in result.errors)


def test_rejects_out_of_range_source_page() -> None:
    result = validate_mcq(_mcq(source_pages=[99]), page_count=10)
    assert not result.is_valid
    assert any("out of range" in e.lower() for e in result.errors)


def test_rejects_missing_source_pages() -> None:
    result = validate_mcq(_mcq(source_pages=[]), page_count=10)
    assert not result.is_valid


def test_image_required_without_page_fails() -> None:
    result = validate_mcq(
        _mcq(
            question_type="figure_grounded",
            image_required=True,
            image_page=None,
        ),
        page_count=10,
    )
    assert not result.is_valid
    assert any("image_page" in e.lower() for e in result.errors)


def test_image_required_out_of_range_page_fails() -> None:
    result = validate_mcq(
        _mcq(
            question_type="figure_grounded",
            image_required=True,
            image_page=50,
        ),
        page_count=10,
    )
    assert not result.is_valid


def test_image_required_valid() -> None:
    result = validate_mcq(
        _mcq(
            question_type="figure_grounded",
            image_required=True,
            image_page=2,
            image_crop=ImageCrop(x0=0.0, y0=0.0, x1=0.5, y1=0.5),
        ),
        page_count=10,
    )
    assert result.is_valid


def test_filter_valid_mcqs_splits() -> None:
    good = _mcq(id="good")
    bad = _mcq(id="bad", source_pages=[999])
    valid, rejected = filter_valid_mcqs([good, bad], page_count=10)
    assert [m.id for m in valid] == ["good"]
    assert [m.id for m, _ in rejected] == ["bad"]


def test_source_pages_deduplicated_and_sorted() -> None:
    mcq = _mcq(source_pages=[3, 1, 3, 2])
    result = validate_mcq(mcq, page_count=10)
    assert result.is_valid
    assert mcq.source_pages == [1, 2, 3]


def test_image_crop_ignored_when_image_not_required() -> None:
    mcq = _mcq(
        image_required=False,
        image_crop=ImageCrop(x0=0.1, y0=0.1, x1=0.5, y1=0.5),
    )
    result = validate_mcq(mcq, page_count=10)
    assert result.is_valid
