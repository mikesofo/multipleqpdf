"""Tests for Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pdf2anki.schemas import MCQ, CoveragePlan, ImageCrop, MCQBatch, TopicAllocation


def test_coverage_plan_minimal() -> None:
    plan = CoveragePlan(
        title="Histology",
        subject="Pathology",
        learner_level="medical student",
        question_count=10,
    )
    assert plan.question_count == 10
    assert plan.topics == []


def test_coverage_plan_rejects_negative_count() -> None:
    with pytest.raises(ValidationError):
        CoveragePlan(
            title="X",
            subject="Y",
            learner_level="Z",
            question_count=-1,
        )


def test_topic_allocation() -> None:
    alloc = TopicAllocation(topic="Cells", question_count=3)
    assert alloc.question_count == 3


def test_mcq_valid() -> None:
    mcq = MCQ(
        id="q1",
        learning_objective="Identify cells",
        topic="Cells",
        stem="Which is a cell?",
        options=["A", "B", "C", "D"],
        correct_index=0,
        explanation="A is a cell.",
        source_pages=[1],
    )
    assert mcq.correct_index == 0


def test_mcq_rejects_bad_index() -> None:
    with pytest.raises(ValidationError):
        MCQ(
            id="q1",
            learning_objective="x",
            topic="t",
            stem="s",
            options=["A", "B", "C", "D"],
            correct_index=4,
            explanation="e",
            source_pages=[1],
        )


def test_image_crop_bounds() -> None:
    crop = ImageCrop(x0=0.0, y0=0.0, x1=1.0, y1=1.0)
    assert crop.x1 == 1.0
    with pytest.raises(ValidationError):
        ImageCrop(x0=0.0, y0=0.0, x1=1.5, y1=1.0)


def test_image_crop_rejects_inverted_or_degenerate_rectangle() -> None:
    with pytest.raises(ValidationError):
        ImageCrop(x0=0.8, y0=0.8, x1=0.1, y1=0.1)
    with pytest.raises(ValidationError):
        ImageCrop(x0=0.5, y0=0.2, x1=0.5, y1=0.8)


def test_mcq_batch_default_empty() -> None:
    batch = MCQBatch()
    assert batch.questions == []


def test_text_based_mcq_clears_image_fields() -> None:
    mcq = MCQ(
        id="q1",
        learning_objective="x",
        topic="t",
        stem="s",
        options=["A", "B", "C", "D"],
        correct_index=0,
        explanation="e",
        source_pages=[1],
        question_type="text_based",
        image_required=True,
        image_page=3,
        image_crop=ImageCrop(x0=0.0, y0=0.0, x1=0.5, y1=0.5),
        image_alt_text="x",
    )
    assert mcq.image_required is False
    assert mcq.image_page is None
    assert mcq.image_crop is None
    assert mcq.image_alt_text is None


def test_figure_grounded_mcq_keeps_image_fields() -> None:
    crop = ImageCrop(x0=0.0, y0=0.0, x1=0.5, y1=0.5)
    mcq = MCQ(
        id="q1",
        learning_objective="x",
        topic="t",
        stem="s",
        options=["A", "B", "C", "D"],
        correct_index=0,
        explanation="e",
        source_pages=[1],
        question_type="figure_grounded",
        image_required=True,
        image_page=3,
        image_crop=crop,
    )
    assert mcq.image_required is True
    assert mcq.image_page == 3
    assert mcq.image_crop == crop


def test_mcq_defaults_to_text_based() -> None:
    mcq = MCQ(
        id="q1",
        learning_objective="x",
        topic="t",
        stem="s",
        options=["A", "B", "C", "D"],
        correct_index=0,
        explanation="e",
        source_pages=[1],
    )
    assert mcq.question_type == "text_based"
