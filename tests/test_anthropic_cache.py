"""Tests for prompt-cache plumbing in AnthropicClient._parse."""

from __future__ import annotations

import logging
from typing import Any
from unittest.mock import MagicMock

from pdf2anki.anthropic_client import AnthropicClient
from pdf2anki.config import SUPPORTED_MODELS
from pdf2anki.schemas import CoveragePlan, MCQBatch


class _Usage:
    def __init__(self) -> None:
        self.cache_creation_input_tokens = 1200
        self.cache_read_input_tokens = 0
        self.input_tokens = 1500


class _Msg:
    def __init__(self, parsed: Any) -> None:
        self.parsed_output = parsed
        self.usage = _Usage()


def _client_with_capture() -> tuple[AnthropicClient, dict[str, Any]]:
    captured: dict[str, Any] = {}
    sdk = MagicMock()

    def _parse(**kwargs: Any) -> _Msg:
        captured.update(kwargs)
        return _Msg(MCQBatch(questions=[]))

    sdk.beta.messages.parse.side_effect = _parse
    client = AnthropicClient(sdk, SUPPORTED_MODELS[-1])
    return client, captured


def test_generate_mcqs_attaches_cache_controls() -> None:
    client, captured = _client_with_capture()
    client.generate_mcqs(
        file_id="file_1",
        topic="Alpha",
        count=2,
        learner_level="student",
        id_prefix="t0-",
    )
    # System prompt is sent as a cacheable block list.
    system = captured["system"]
    assert isinstance(system, list)
    assert system[0]["cache_control"] == {"type": "ephemeral"}
    # Document block carries a cache breakpoint; the user text does not.
    content = captured["messages"][0]["content"]
    assert content[0]["type"] == "document"
    assert content[0]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in content[1]


def test_plan_coverage_does_not_cache() -> None:
    client, captured = _client_with_capture()
    sdk_msg = _Msg(
        CoveragePlan(
            title="T",
            subject="S",
            learner_level="student",
            question_count=1,
        )
    )
    client._client.beta.messages.parse.side_effect = lambda **kw: (  # type: ignore[attr-defined]
        captured.update(kw) or sdk_msg
    )
    client.plan_coverage("file_1", 1)
    assert isinstance(captured["system"], str)
    content = captured["messages"][0]["content"]
    assert "cache_control" not in content[0]


def test_parse_falls_back_when_cache_unsupported() -> None:
    captured: dict[str, Any] = {}
    sdk = MagicMock()
    calls = {"n": 0}

    def _parse(**kwargs: Any) -> _Msg:
        calls["n"] += 1
        if calls["n"] == 1:
            raise TypeError("unexpected keyword 'cache_control'")
        captured.update(kwargs)
        return _Msg(MCQBatch(questions=[]))

    sdk.beta.messages.parse.side_effect = _parse
    client = AnthropicClient(sdk, SUPPORTED_MODELS[-1])
    client.generate_mcqs(
        file_id="file_1",
        topic="Alpha",
        count=1,
        learner_level="student",
        id_prefix="t0-",
    )
    # Fell back to a plain (uncached) call shape.
    assert calls["n"] == 2
    assert isinstance(captured["system"], str)
    content = captured["messages"][0]["content"]
    assert "cache_control" not in content[0]


def test_log_cache_usage_emits_telemetry(caplog: Any) -> None:
    client, _ = _client_with_capture()
    with caplog.at_level(logging.INFO, logger="pdf2anki.anthropic"):
        client.generate_mcqs(
            file_id="file_1",
            topic="Alpha",
            count=1,
            learner_level="student",
            id_prefix="t0-",
        )
    assert any("prompt_cache usage" in rec.message for rec in caplog.records)
