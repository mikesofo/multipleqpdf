"""Tests for dynamic question-cap and recommended-range helpers."""

from __future__ import annotations

import pytest

from pdf2anki.config import (
    max_questions_for_pages,
    recommended_range_for_pages,
)


@pytest.mark.parametrize(
    "pages,expected",
    [
        (1, 20),
        (25, 20),
        (26, 35),
        (50, 35),
        (51, 60),
        (100, 60),
        (101, 100),
        (200, 100),
        (201, 150),
        (350, 150),
        (351, 220),
        (500, 220),
        (501, 300),
        (2000, 300),
    ],
)
def test_max_questions_for_pages(pages: int, expected: int) -> None:
    assert max_questions_for_pages(pages) == expected


def test_max_questions_allows_far_more_than_50_for_large_selection() -> None:
    assert max_questions_for_pages(500) == 220
    assert max_questions_for_pages(1000) == 300
    assert max_questions_for_pages(500) > 50


@pytest.mark.parametrize(
    "pages,low,high",
    [
        (50, 17, 25),
        (200, 67, 100),
        (500, 167, 250),
    ],
)
def test_recommended_range_matches_examples(
    pages: int, low: int, high: int
) -> None:
    assert recommended_range_for_pages(pages) == (low, high)


def test_recommended_range_zero_pages() -> None:
    assert recommended_range_for_pages(0) == (0, 0)
