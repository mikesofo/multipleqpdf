"""Thin abstraction over the Anthropic SDK for PDF upload and structured calls."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

logger = logging.getLogger("pdf2anki.anthropic")

try:
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback when tenacity absent

    def retry(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[no-redef]
        """No-op decorator used when tenacity is not installed."""

        def _decorator(func: Any) -> Any:
            return func

        return _decorator

    def retry_if_exception_type(  # type: ignore[no-redef]
        *_args: Any, **_kwargs: Any
    ) -> Any:
        return None

    def stop_after_attempt(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[no-redef]
        return None

    def wait_exponential(*_args: Any, **_kwargs: Any) -> Any:  # type: ignore[no-redef]
        return None

from .config import SUPPORTED_MODELS
from .pdf_media import validate_pdf
from .prompts import (
    COVERAGE_SYSTEM_PROMPT,
    MCQ_SYSTEM_PROMPT,
    coverage_user_prompt,
    mcq_user_prompt,
)
from .schemas import CoveragePlan, MCQBatch

if TYPE_CHECKING:  # pragma: no cover - typing only
    from anthropic import Anthropic


class AnthropicError(RuntimeError):
    """Raised when an Anthropic operation fails in a non-retryable way."""


def _transient_exceptions() -> tuple[type[Exception], ...]:
    """Return the Anthropic exception types that are safe to retry."""
    try:
        from anthropic import APIStatusError, APITimeoutError, RateLimitError
    except Exception:  # pragma: no cover - SDK not installed
        return ()
    return (RateLimitError, APITimeoutError, APIStatusError)


_TRANSIENT = _transient_exceptions()
_RETRY_PREDICATE = retry_if_exception_type(_TRANSIENT) if _TRANSIENT else None


def _retry_decorator(func: Any) -> Any:
    if _RETRY_PREDICATE is None:  # pragma: no cover
        return func
    return retry(
        retry=_RETRY_PREDICATE,
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        reraise=True,
    )(func)


class AnthropicClient:
    """Wraps PDF upload and the two-pass MCQ pipeline.

    The client is injected so tests can pass a mock without a real API key.
    """

    def __init__(self, client: Anthropic, model: str) -> None:
        if not model or not (
            model in SUPPORTED_MODELS or model.startswith("claude-")
        ):
            raise AnthropicError(
                f"Unsupported model '{model}'. Expected a Claude model such as "
                f"{SUPPORTED_MODELS[0]}."
            )
        self._client = client
        self._model = model

    @classmethod
    def from_api_key(cls, api_key: str, model: str) -> AnthropicClient:
        """Build a client from an API key using the real Anthropic SDK."""
        from anthropic import Anthropic

        return cls(Anthropic(api_key=api_key), model)

    def upload_pdf(self, data: bytes, filename: str) -> str:
        """Validate and upload PDF bytes to the Files API and return the file_id."""
        validate_pdf(data)
        return self._upload_validated(data, filename)

    @_retry_decorator
    def _upload_validated(self, data: bytes, filename: str) -> str:
        result = self._client.beta.files.upload(
            file=(filename, data, "application/pdf"),
        )
        return result.id

    def delete_file(self, file_id: str) -> bool:
        """Delete a remote file; return True on success, False on failure."""
        try:
            self._client.beta.files.delete(file_id)
            return True
        except Exception:
            return False

    def _file_content_block(
        self, file_id: str, cache: bool = False
    ) -> dict[str, Any]:
        """Build a document content block that references an uploaded file.

        When ``cache`` is True, a prompt-cache breakpoint is attached so the
        (large) document prefix is written once and read on later calls that
        reuse the same prefix.
        """
        block: dict[str, Any] = {
            "type": "document",
            "source": {"type": "file", "file_id": file_id},
        }
        if cache:
            block["cache_control"] = {"type": "ephemeral"}
        return block

    def _system_blocks(self, system: str, cache: bool) -> Any:
        """Return the system prompt, as cacheable blocks when requested."""
        if not cache:
            return system
        return [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]

    @staticmethod
    def _log_cache_usage(message: Any) -> None:
        """Emit lightweight telemetry about prompt-cache hits, if available."""
        usage = getattr(message, "usage", None)
        if usage is None:
            return
        created = getattr(usage, "cache_creation_input_tokens", None)
        read = getattr(usage, "cache_read_input_tokens", None)
        if created is None and read is None:
            return
        logger.info(
            "prompt_cache usage: created=%s read=%s input=%s",
            created,
            read,
            getattr(usage, "input_tokens", None),
        )

    @_retry_decorator
    def _parse(
        self,
        *,
        system: str,
        user_text: str,
        file_id: str,
        schema: type[Any],
        max_tokens: int = 8192,
        cache: bool = False,
    ) -> Any:
        """Call messages.parse with a document block and a response schema.

        Stable content (document block, system prompt) is placed first and, when
        ``cache`` is True, marked with a prompt-cache breakpoint. The variable
        per-request text is appended after the cached prefix. If the installed
        SDK does not support cache controls, the call is retried without them so
        behavior is unchanged when caching is unavailable.
        """
        content: list[dict[str, Any]] = [
            self._file_content_block(file_id, cache=cache),
            {"type": "text", "text": user_text},
        ]
        messages: Any = [{"role": "user", "content": content}]
        parse = self._client.beta.messages.parse

        def _call(system_arg: Any, messages_arg: Any) -> Any:
            return parse(
                model=self._model,
                max_tokens=max_tokens,
                system=system_arg,
                messages=messages_arg,
                output_format=schema,
            )

        try:
            message = _call(self._system_blocks(system, cache), messages)
        except TypeError:
            # Older SDK that does not understand cache_control shapes.
            if cache:
                plain_content: list[dict[str, Any]] = [
                    self._file_content_block(file_id, cache=False),
                    {"type": "text", "text": user_text},
                ]
                message = _call(system, [{"role": "user", "content": plain_content}])
            else:
                raise

        self._log_cache_usage(message)
        parsed = getattr(message, "parsed_output", None)
        if parsed is None:
            parsed = getattr(message, "parsed", None)
        return parsed

    def plan_coverage(self, file_id: str, max_questions: int) -> CoveragePlan:
        """Run Pass 1 and return a validated CoveragePlan."""
        return self._parse(
            system=COVERAGE_SYSTEM_PROMPT,
            user_text=coverage_user_prompt(max_questions),
            file_id=file_id,
            schema=CoveragePlan,
        )

    def generate_mcqs(
        self,
        file_id: str,
        topic: str,
        count: int,
        learner_level: str,
        id_prefix: str,
        focus_pages: list[int] | None = None,
    ) -> MCQBatch:
        """Run Pass 2 for a single topic batch and return validated MCQs.

        The document block and system prompt form a stable prefix reused across
        every topic batch, so prompt caching is enabled to avoid re-sending the
        full PDF on each batch.
        """
        scaled_tokens = min(16384, max(2048, 1200 + 800 * max(1, count)))
        return self._parse(
            system=MCQ_SYSTEM_PROMPT,
            user_text=mcq_user_prompt(
                topic, count, learner_level, id_prefix, focus_pages
            ),
            file_id=file_id,
            schema=MCQBatch,
            max_tokens=scaled_tokens,
            cache=True,
        )
