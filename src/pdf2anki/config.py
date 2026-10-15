"""Application configuration and runtime limits."""

from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from dotenv import load_dotenv

    load_dotenv()
except ModuleNotFoundError:  # pragma: no cover - dotenv optional at runtime
    pass

# Anthropic PDF limits (see spec.md).
MAX_PDF_BYTES: int = 32 * 1024 * 1024  # 32 MB per request
MAX_PDF_PAGES: int = 600

# Safety limit on generated questions to avoid runaway cost/output.
MAX_QUESTIONS: int = 100
DEFAULT_MAX_QUESTIONS: int = 50

# Questions generated per MCQ batch when the plan requires many questions.
QUESTIONS_PER_BATCH: int = 10

ACCEPTED_MIME_TYPE: str = "application/pdf"

SUPPORTED_MODELS: tuple[str, ...] = (
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
)


@dataclass(frozen=True)
class Settings:
    """Runtime settings resolved from the environment."""

    anthropic_api_key: str | None = None
    model: str = "claude-opus-4-20250514"
    max_pdf_bytes: int = MAX_PDF_BYTES
    max_pdf_pages: int = MAX_PDF_PAGES
    max_questions: int = MAX_QUESTIONS


def get_api_key(override: str | None = None) -> str | None:
    """Return the Anthropic API key from an override or the environment."""
    if override:
        return override
    return os.environ.get("ANTHROPIC_API_KEY")


def load_settings(api_key_override: str | None = None) -> Settings:
    """Build a Settings object, preferring an explicit override for the key."""
    return Settings(anthropic_api_key=get_api_key(api_key_override))
