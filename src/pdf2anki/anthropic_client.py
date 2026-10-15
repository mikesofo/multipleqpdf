"""Thin abstraction over the Anthropic SDK for PDF upload and structured calls."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    from tenacity import (
        retry,
        retry_if_exception_type,
        stop_after_attempt,
        wait_exponential,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback when tenacity absent

    def retry(*_args: Any, **_kwargs: Any):  # type: ignore[no-redef]
        """No-op decorator used when tenacity is not installed."""

        def _decorator(func):
            return func

        return _decorator

    def retry_if_exception_type(
        *_args: Any, **_kwargs: Any
    ):  # type: ignore[no-redef]
        return None

    def stop_after_attempt(*_args: Any, **_kwargs: Any):  # type: ignore[no-redef]
        return None

    def wait_exponential(*_args: Any, **_kwargs: Any):  # type: ignore[no-redef]
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

    @retry(
        retry=retry_if_exception_type(_transient_exceptions() or Exception),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        reraise=True,
    )
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

    def _file_content_block(self, file_id: str) -> dict[str, Any]:
        """Build a document content block that references an uploaded file."""
        return {
            "type": "document",
            "source": {"type": "file", "file_id": file_id},
        }

    @retry(
        retry=retry_if_exception_type(_transient_exceptions() or Exception),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=1, max=20),
        reraise=True,
    )
    def _parse(
        self,
        *,
        system: str,
        user_text: str,
        file_id: str,
        schema: type[Any],
        max_tokens: int = 8192,
    ) -> Any:
        """Call messages.parse with a document block and a response schema."""
        message = self._client.beta.messages.parse(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=[
                {
                    "role": "user",
                    "content": [
                        self._file_content_block(file_id),
                        {"type": "text", "text": user_text},
                    ],
                }
            ],
            response_format=schema,
        )
        return message.parsed

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
    ) -> MCQBatch:
        """Run Pass 2 for a single topic batch and return validated MCQs."""
        scaled_tokens = min(16384, max(2048, 1200 + 800 * max(1, count)))
        return self._parse(
            system=MCQ_SYSTEM_PROMPT,
            user_text=mcq_user_prompt(topic, count, learner_level, id_prefix),
            file_id=file_id,
            schema=MCQBatch,
            max_tokens=scaled_tokens,
        )
