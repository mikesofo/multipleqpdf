"""Tests for Claude model-family selection and propagation."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from pdf2anki.anthropic_client import AnthropicClient, AnthropicError
from pdf2anki.config import (
    DEFAULT_MODEL_CHOICE,
    MODEL_CHOICES,
    SUPPORTED_MODELS,
    model_id_for_choice,
)
from pdf2anki.schemas import CoveragePlan, MCQBatch


def test_default_choice_is_sonnet() -> None:
    assert DEFAULT_MODEL_CHOICE == "Sonnet"
    assert model_id_for_choice("Sonnet") == "claude-sonnet-5-5"


@pytest.mark.parametrize(
    "choice,model_id",
    [
        ("Haiku", "claude-haiku-5-5"),
        ("Sonnet", "claude-sonnet-5-5"),
        ("Opus", "claude-opus-5-5"),
        ("Fable", "claude-fable-5-1"),
    ],
)
def test_model_choice_mapping(choice: str, model_id: str) -> None:
    assert model_id_for_choice(choice) == model_id
    assert model_id in SUPPORTED_MODELS


def test_unknown_choice_falls_back_to_default() -> None:
    assert model_id_for_choice("Nonexistent") == "claude-sonnet-5-5"


def test_all_choices_have_descriptions() -> None:
    for _name, (model_id, desc) in MODEL_CHOICES.items():
        assert model_id.startswith("claude-")
        assert desc  # non-empty friendly description


def test_client_rejects_unsupported_model() -> None:
    with pytest.raises(AnthropicError):
        AnthropicClient(MagicMock(), "gpt-4o")


def test_selected_model_reaches_generate_call() -> None:
    captured: dict[str, Any] = {}
    sdk = MagicMock()

    class _Usage:
        cache_creation_input_tokens = 0
        cache_read_input_tokens = 0
        input_tokens = 10

    class _Msg:
        parsed_output = MCQBatch(questions=[])
        usage = _Usage()

    def _parse(**kwargs: Any) -> _Msg:
        captured.update(kwargs)
        return _Msg()

    sdk.beta.messages.parse.side_effect = _parse

    model_id = model_id_for_choice("Opus")
    client = AnthropicClient(sdk, model_id)
    client.generate_mcqs(
        file_id="f",
        topic="Alpha",
        count=1,
        learner_level="student",
        id_prefix="t0-",
    )
    assert captured["model"] == "claude-opus-5-5"


def test_selected_model_reaches_plan_call() -> None:
    captured: dict[str, Any] = {}
    sdk = MagicMock()

    class _Msg:
        parsed_output = CoveragePlan(
            title="T",
            subject="S",
            learner_level="student",
            question_count=1,
        )
        usage = None

    def _parse(**kwargs: Any) -> _Msg:
        captured.update(kwargs)
        return _Msg()

    sdk.beta.messages.parse.side_effect = _parse

    client = AnthropicClient(sdk, model_id_for_choice("Haiku"))
    client.plan_coverage("f", 1)
    assert captured["model"] == "claude-haiku-5-5"
