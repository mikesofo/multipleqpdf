"""Application configuration and runtime limits."""

from __future__ import annotations

import math
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

# Absolute hard ceiling on generated questions to avoid runaway cost/output.
# The effective per-upload cap is computed dynamically from selected pages.
MAX_QUESTIONS: int = 300
DEFAULT_MAX_QUESTIONS: int = 20

# (max_selected_pages_inclusive, max_questions) tiers; last tier is the ceiling.
_MAX_CAP_TABLE: tuple[tuple[int, int], ...] = (
    (25, 20),
    (50, 35),
    (100, 60),
    (200, 100),
    (350, 150),
    (500, 220),
)


def max_questions_for_pages(selected_pages: int) -> int:
    """Return the dynamic maximum question count for a selected page count."""
    pages = max(0, selected_pages)
    for threshold, cap in _MAX_CAP_TABLE:
        if pages <= threshold:
            return cap
    return 300


def recommended_range_for_pages(selected_pages: int) -> tuple[int, int]:
    """Return a (low, high) recommended question range for a page count.

    Low is ~1 question per 3 pages, high ~1 per 2 pages. The high value is the
    raw recommendation and is not clamped to the table cap (the cap is applied
    separately to the allowed maximum).
    """
    pages = max(0, selected_pages)
    if pages == 0:
        return (0, 0)
    low = math.ceil(pages / 3)
    high = math.ceil(pages / 2)
    return (low, high)


# Questions generated per MCQ batch when the plan requires many questions.
QUESTIONS_PER_BATCH: int = 10

ACCEPTED_MIME_TYPE: str = "application/pdf"

SUPPORTED_MODELS: tuple[str, ...] = (
    "claude-opus-4-8",
    "claude-sonnet-4-20250514",
    "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
)


@dataclass(frozen=True)
class Settings:
    """Runtime settings resolved from the environment."""

    anthropic_api_key: str | None = None
    model: str = "claude-opus-4-8"
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
