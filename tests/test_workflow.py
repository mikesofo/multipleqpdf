"""Tests for the generation workflow using a mocked Anthropic client."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest
from conftest import make_sample_pdf

from pdf2anki.anthropic_client import AnthropicClient
from pdf2anki.config import SUPPORTED_MODELS
from pdf2anki.schemas import MCQ, CoveragePlan, MCQBatch, TopicAllocation
from pdf2anki.workflow import _batches, generate_deck


def _mcq(qid: str, topic: str, bad: bool = False) -> MCQ:
    return MCQ(
        id=qid,
        learning_objective="obj",
        topic=topic,
        stem="Stem?",
        options=["A", "B", "C", "D"],
        correct_index=0,
        explanation="Because A.",
        source_pages=[999] if bad else [1],
    )


def test_batches_splits_evenly() -> None:
    assert _batches(25, 10) == [10, 10, 5]
    assert _batches(0, 10) == []
    assert _batches(10, 10) == [10]
    assert sum(_batches(500, 10)) == 100


def test_generate_deck_runs_two_passes() -> None:
    plan = CoveragePlan(
        title="T",
        subject="S",
        learner_level="student",
        question_count=4,
        allocation=[
            TopicAllocation(topic="Alpha", question_count=2),
            TopicAllocation(topic="Beta", question_count=2),
        ],
    )

    client = MagicMock()
    client.plan_coverage.return_value = plan
    client.generate_mcqs.side_effect = [
        MCQBatch(questions=[_mcq("a1", "Alpha"), _mcq("a2", "Alpha")]),
        MCQBatch(questions=[_mcq("b1", "Beta"), _mcq("b2", "Beta")]),
    ]

    result = generate_deck(
        client=client,
        file_id="file_123",
        page_count=10,
        max_questions=50,
        batch_size=10,
    )

    client.plan_coverage.assert_called_once_with("file_123", 50)
    assert client.generate_mcqs.call_count == 2
    assert [m.id for m in result.mcqs] == ["a1", "a2", "b1", "b2"]
    assert result.rejected == []
    assert result.plan is plan


def test_generate_deck_filters_invalid() -> None:
    plan = CoveragePlan(
        title="T",
        subject="S",
        learner_level="student",
        question_count=2,
        allocation=[TopicAllocation(topic="Alpha", question_count=2)],
    )
    client = MagicMock()
    client.plan_coverage.return_value = plan
    client.generate_mcqs.return_value = MCQBatch(
        questions=[_mcq("good", "Alpha"), _mcq("bad", "Alpha", bad=True)]
    )

    result = generate_deck(
        client=client,
        file_id="f",
        page_count=10,
        max_questions=10,
        batch_size=10,
    )

    assert [m.id for m in result.mcqs] == ["good"]
    assert [m.id for m, _ in result.rejected] == ["bad"]


def test_generate_deck_without_allocation_uses_subject() -> None:
    plan = CoveragePlan(
        title="T",
        subject="Cardiology",
        learner_level="resident",
        question_count=3,
        allocation=[],
    )
    client = MagicMock()
    client.plan_coverage.return_value = plan
    client.generate_mcqs.return_value = MCQBatch(
        questions=[_mcq("q1", "Cardiology")]
    )

    result = generate_deck(
        client=client,
        file_id="f",
        page_count=10,
        max_questions=10,
        batch_size=10,
    )

    assert client.generate_mcqs.call_count == 1
    _, kwargs = client.generate_mcqs.call_args
    assert kwargs["topic"] == "Cardiology"
    assert [m.id for m in result.mcqs] == ["q1"]


def test_generate_deck_recovers_from_partial_batch_failure_and_dedupes_ids() -> None:
    plan = CoveragePlan(
        title="T",
        subject="S",
        learner_level="student",
        question_count=4,
        allocation=[
            TopicAllocation(topic="Alpha", question_count=2),
            TopicAllocation(topic="Beta", question_count=2),
        ],
    )
    client = MagicMock()
    client.plan_coverage.return_value = plan
    client.generate_mcqs.side_effect = [
        MCQBatch(questions=[_mcq("dup", "Alpha"), _mcq("dup", "Alpha")]),
        RuntimeError("transient failure"),
    ]

    result = generate_deck(
        client=client,
        file_id="f",
        page_count=10,
        max_questions=10,
        batch_size=10,
    )

    assert len(result.mcqs) == 2
    assert len({m.id for m in result.mcqs}) == 2
    assert len(result.batch_errors) == 1


@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY")
    or os.environ.get("RUN_ANTHROPIC_INTEGRATION") != "1",
    reason="Set ANTHROPIC_API_KEY and RUN_ANTHROPIC_INTEGRATION=1 to run live API test",
)
def test_live_integration_generate_deck() -> None:
    api_key = os.environ["ANTHROPIC_API_KEY"]
    client = AnthropicClient.from_api_key(api_key, SUPPORTED_MODELS[-1])
    pdf_bytes = make_sample_pdf(pages=1)
    file_id: str | None = None
    try:
        file_id = client.upload_pdf(pdf_bytes, "sample.pdf")
        result = generate_deck(
            client=client,
            file_id=file_id,
            page_count=1,
            max_questions=1,
            batch_size=1,
        )
        assert result.plan is not None
        assert len(result.mcqs) >= 1
    finally:
        if file_id:
            assert client.delete_file(file_id)
