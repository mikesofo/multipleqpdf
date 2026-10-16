"""Tests for deterministic figure bundle extraction and figure-grounded generation."""

from __future__ import annotations

from unittest.mock import MagicMock

from conftest import (
    make_pdf_with_captioned_image,
    make_pdf_with_image,
    make_pdf_with_labeled_figures,
    make_sample_pdf,
)
from pdf2anki.pdf_media import extract_figure_bundles
from pdf2anki.schemas import MCQ, MCQBatch
from pdf2anki.workflow import generate_figure_grounded_deck


def test_extract_figure_bundles_finds_captioned_figures() -> None:
    pdf_bytes = make_pdf_with_labeled_figures()
    bundles = extract_figure_bundles(pdf_bytes)
    assert len(bundles) == 2
    assert [b.figure_label for b in bundles] == ["figure 1", "figure 2"]
    assert "Glomerulus" in bundles[0].caption_text
    assert "Myocardium" in bundles[1].caption_text
    assert bundles[0].image_crop is not None
    assert bundles[1].image_crop is not None
    assert bundles[0].image_crop.y1 < 0.5
    assert bundles[1].image_crop.y0 > 0.5


def test_extract_figure_bundles_collects_nearby_text_and_references() -> None:
    pdf_bytes = make_pdf_with_captioned_image()
    bundles = extract_figure_bundles(pdf_bytes)
    assert len(bundles) == 1
    b = bundles[0]
    assert b.figure_label == "figure 1"
    assert "Photomicrograph" in b.caption_text
    assert "Relevant lead-in text" in b.nearby_text
    assert any("Figure 1" in r for r in b.references)


def test_extract_figure_bundles_empty_when_uncaptioned_or_text_only() -> None:
    assert extract_figure_bundles(make_sample_pdf(pages=1)) == []
    assert extract_figure_bundles(make_pdf_with_image(image_page_1based=1, pages=1)) == []


def test_generate_figure_grounded_deck_stamps_bundle_crop_and_page() -> None:
    pdf_bytes = make_pdf_with_labeled_figures()
    client = MagicMock()
    client.generate_figure_mcqs.side_effect = [
        MCQBatch(
            questions=[
                MCQ(
                    id="q1",
                    learning_objective="Identify glomerulus",
                    topic="Renal",
                    stem="Which structure is shown in Figure 1?",
                    options=["Glomerulus", "Tubule", "Loop", "Duct"],
                    correct_index=0,
                    explanation="Figure 1 shows the glomerulus.",
                    source_pages=[1],
                )
            ]
        ),
        MCQBatch(
            questions=[
                MCQ(
                    id="q2",
                    learning_objective="Identify myocardium",
                    topic="Cardiac",
                    stem="Which tissue is shown in Figure 2?",
                    options=["Myocardium", "Epicardium", "Endocardium", "Valve"],
                    correct_index=0,
                    explanation="Figure 2 shows myocardium.",
                    source_pages=[1],
                )
            ]
        ),
    ]

    result = generate_figure_grounded_deck(
        client=client,
        file_id="file_abc",
        pdf_bytes=pdf_bytes,
        page_count=1,
        max_questions=2,
    )

    assert not result.no_captioned_figures
    assert len(result.mcqs) == 2
    assert result.mcqs[0].image_required is True
    assert result.mcqs[0].image_page == 1
    assert result.mcqs[0].image_crop is not None
    assert result.mcqs[0].image_crop.y1 < 0.5
    assert result.mcqs[1].image_crop is not None
    assert result.mcqs[1].image_crop.y0 > 0.5


def test_generate_figure_grounded_deck_no_captioned_figures_flag() -> None:
    pdf_bytes = make_sample_pdf(pages=1)
    client = MagicMock()

    result = generate_figure_grounded_deck(
        client=client,
        file_id="file_abc",
        pdf_bytes=pdf_bytes,
        page_count=1,
        max_questions=5,
    )

    assert result.no_captioned_figures is True
    assert result.mcqs == []
    client.generate_figure_mcqs.assert_not_called()
