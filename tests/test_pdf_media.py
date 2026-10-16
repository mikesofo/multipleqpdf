"""Tests for PDF validation, hashing, and PyMuPDF page/crop rendering."""

from __future__ import annotations

import pytest

from conftest import (
    make_large_pdf_with_toc,
    make_pdf_with_captioned_image,
    make_pdf_with_conflicting_caption,
    make_pdf_with_image,
    make_pdf_with_labeled_figures,
    make_pdf_with_matching_caption,
    make_pdf_with_nearby_and_far_images,
    make_pdf_with_toc,
    make_pdf_with_two_images,
)
from conftest import make_sample_pdf as _make_pdf
from pdf2anki.pdf_media import (
    PDFValidationError,
    best_image_crop_for_text,
    chunk_chapters,
    count_page_images,
    extract_chapters,
    extract_figure_metadata,
    find_image_crop_by_caption,
    find_image_crop_by_reference,
    largest_image_crop,
    media_filename,
    plan_chapters,
    render_crop,
    render_page,
    select_image_crop,
    validate_pdf,
)
from pdf2anki.schemas import ImageCrop


def test_validate_pdf_accepts_valid_pdf() -> None:
    data = _make_pdf(pages=2)
    info = validate_pdf(data)
    assert info.page_count == 2
    assert len(info.pdf_hash) == 16


def test_validate_pdf_rejects_non_pdf_mime() -> None:
    data = _make_pdf(pages=1)
    with pytest.raises(PDFValidationError, match="not a PDF"):
        validate_pdf(data, mime_type="text/plain")


def test_validate_pdf_rejects_oversized() -> None:
    data = _make_pdf(pages=1)
    with pytest.raises(PDFValidationError, match="limit"):
        validate_pdf(data, max_bytes=10)


def test_validate_pdf_rejects_encrypted() -> None:
    data = _make_pdf(pages=1, encrypt=True)
    with pytest.raises(PDFValidationError, match="Encrypted"):
        validate_pdf(data)


def test_render_page_and_crop_produce_png() -> None:
    data = _make_pdf(pages=1)
    full_png = render_page(data, 1, dpi=72)
    assert full_png.startswith(b"\x89PNG")

    crop = ImageCrop(x0=0.1, y0=0.2, x1=0.7, y1=0.8)
    crop_png = render_crop(data, 1, crop, dpi=72)
    assert crop_png.startswith(b"\x89PNG")
    assert len(crop_png) < len(full_png)


def test_render_crop_falls_back_on_invalid_crop() -> None:
    data = _make_pdf(pages=1)
    bad_crop = ImageCrop.model_construct(x0=0.9, y0=0.9, x1=0.1, y1=0.1)
    fallback_png = render_crop(data, 1, bad_crop, dpi=72)
    full_png = render_page(data, 1, dpi=72)
    assert fallback_png == full_png


def test_media_filename_is_collision_proof() -> None:
    crop_a = ImageCrop(x0=0.0, y0=0.0, x1=0.5, y1=0.5)
    crop_b = ImageCrop(x0=0.1, y0=0.1, x1=0.5, y1=0.5)
    name_a = media_filename("deadbeef12345678", 1, "q1", crop_a)
    name_b = media_filename("deadbeef12345678", 1, "q1", crop_b)
    assert name_a != name_b
    assert name_a.endswith(".png")


def test_count_page_images_detects_embedded_image() -> None:
    data = make_pdf_with_image(image_page_1based=1, pages=1)
    assert count_page_images(data, 1) >= 1


def test_count_page_images_ignores_text_only_page() -> None:
    data = _make_pdf(pages=1)
    assert count_page_images(data, 1) == 0


def test_count_page_images_ignores_trivial_image() -> None:
    data = make_pdf_with_image(
        image_page_1based=1, pages=1, image_width=16, image_height=16
    )
    assert count_page_images(data, 1) == 0


def test_count_page_images_out_of_range_raises() -> None:
    data = _make_pdf(pages=1)
    with pytest.raises(ValueError, match="out of range"):
        count_page_images(data, 5)


def test_largest_image_crop_returns_subregion_of_page() -> None:
    # Image is inserted at Rect(20, 60, 180, 180) on a 200x200 page.
    data = make_pdf_with_image(image_page_1based=1, pages=1)
    crop = largest_image_crop(data, 1, margin=0.0)
    assert crop is not None
    # Must be a strict sub-region, not the whole page.
    assert 0.0 <= crop.x0 < crop.x1 <= 1.0
    assert 0.0 <= crop.y0 < crop.y1 <= 1.0
    assert crop.x0 > 0.0 or crop.y0 > 0.0 or crop.x1 < 1.0 or crop.y1 < 1.0
    assert crop.y0 == pytest.approx(0.3, abs=0.02)
    assert crop.y1 == pytest.approx(0.9, abs=0.02)


def test_largest_image_crop_applies_margin_and_clamps() -> None:
    data = make_pdf_with_image(image_page_1based=1, pages=1)
    crop = largest_image_crop(data, 1, margin=0.05)
    assert crop is not None
    assert 0.0 <= crop.x0 < crop.x1 <= 1.0
    assert 0.0 <= crop.y0 < crop.y1 <= 1.0
    # Margin pushes bounds outward versus the no-margin crop.
    base = largest_image_crop(data, 1, margin=0.0)
    assert base is not None
    assert crop.x0 <= base.x0
    assert crop.y1 >= base.y1


def test_largest_image_crop_none_on_text_only_page() -> None:
    data = _make_pdf(pages=1)
    assert largest_image_crop(data, 1) is None


def test_largest_image_crop_out_of_range_raises() -> None:
    data = _make_pdf(pages=1)
    with pytest.raises(ValueError, match="out of range"):
        largest_image_crop(data, 5)


def test_best_image_crop_for_text_picks_nearest_over_largest() -> None:
    data = make_pdf_with_two_images()
    # Top image is smaller (80x80) than bottom image (160x140), but is right next
    # to the top anchor text.
    crop = best_image_crop_for_text(
        data, 1, "Top anchor text about glomerulus", margin=0.0
    )
    assert crop is not None
    # Top image sits in y=[50, 130] on an 800pt tall page -> y in [0.0625, 0.1625].
    assert crop.y1 < 0.30

    # Bottom anchor text should select the bottom image instead.
    bot_crop = best_image_crop_for_text(
        data, 1, "Bottom anchor text about myocardium", margin=0.0
    )
    assert bot_crop is not None
    assert bot_crop.y0 > 0.70


def test_best_image_crop_for_text_ignores_distant_unrelated_image() -> None:
    # Single page (400x800): put text at the very top, image only at the bottom.
    import pymupdf as fitz

    from conftest import _png_bytes

    doc = fitz.open()
    page = doc.new_page(width=400, height=800)
    page.insert_text((20, 30), "Isolated top paragraph with no figure nearby")
    png = _png_bytes(100, 100, (20, 40, 160))
    page.insert_image(fitz.Rect(200, 640, 360, 780), stream=png)
    pdf_bytes = doc.tobytes()

    crop = best_image_crop_for_text(
        pdf_bytes, 1, "Isolated top paragraph with no figure nearby"
    )
    assert crop is None


def test_best_image_crop_for_text_returns_none_without_anchor() -> None:
    data = make_pdf_with_image(image_page_1based=1, pages=1)
    assert best_image_crop_for_text(data, 1, "") is None
    assert best_image_crop_for_text(data, 1, "text not on page") is None


def test_select_image_crop_tier1_anchors_on_source_text() -> None:
    data = make_pdf_with_two_images()
    crop = select_image_crop(data, 1, "Top anchor text about glomerulus")
    assert crop is not None
    # Nearest image to the top text is the top figure.
    assert crop.y1 < 0.35


def test_select_image_crop_tier2_anchors_on_fragment() -> None:
    # source_support is a paraphrase whose only literal overlap is a fragment
    # that does appear verbatim on the page.
    data = make_pdf_with_two_images()
    paraphrase = (
        "This question concerns the glomerulus; see Top anchor text about "
        "glomerulus, which labels the figure."
    )
    crop = select_image_crop(data, 1, paraphrase)
    assert crop is not None
    assert crop.y1 < 0.35


def test_select_image_crop_tier3_requires_caption_for_fallback() -> None:
    # One substantial but UNcaptioned image + non-matching text -> no fallback.
    data = make_pdf_with_image(image_page_1based=1, pages=1)
    assert select_image_crop(data, 1, "completely unrelated description") is None


def test_select_image_crop_tier3_captioned_single_image_fallback() -> None:
    # One substantial CAPTIONED image + non-matching text -> fallback selects it.
    data = make_pdf_with_captioned_image()
    crop = select_image_crop(data, 1, "text that does not appear on the page")
    assert crop is not None


def test_select_image_crop_prefers_nearby_captioned_over_far_image() -> None:
    data = make_pdf_with_nearby_and_far_images()
    crop = select_image_crop(data, 1, "Top anchor text about glomerulus")
    assert crop is not None
    # Nearby captioned figure is in the top region, not the far bottom one.
    assert crop.y0 < 0.35


def test_select_image_crop_ignores_small_decorative_image() -> None:
    # Top image is a tiny decorative thumbnail (area floor filters it); the only
    # other image is far away and unrelated -> attach nothing.
    data = make_pdf_with_nearby_and_far_images(small_decorative=True)
    assert select_image_crop(data, 1, "Top anchor text about glomerulus") is None


def test_captioned_image_selected_via_source_anchor() -> None:
    data = make_pdf_with_captioned_image()
    crop = best_image_crop_for_text(
        data, 1, "Relevant lead-in text about the specimen"
    )
    assert crop is not None


def test_select_image_crop_no_fallback_with_multiple_unanchored_images() -> None:
    # Two images, no anchor match -> prefer no image over an unrelated one.
    data = make_pdf_with_two_images()
    assert select_image_crop(data, 1, "no matching text whatsoever") is None


def test_select_image_crop_none_on_text_only_page() -> None:
    data = _make_pdf(pages=1)
    assert select_image_crop(data, 1, "Page 1 content") is None


def test_extract_figure_metadata_detects_labels_and_captions() -> None:
    data = make_pdf_with_labeled_figures()
    metas = extract_figure_metadata(data, 1)
    labels = sorted(m.figure_label for m in metas)
    assert labels == ["figure 1", "figure 2"]
    captions = " ".join(m.caption_text for m in metas)
    assert "Glomerulus" in captions
    assert "Myocardium" in captions


def test_tier1_explicit_figure_reference_match() -> None:
    data = make_pdf_with_labeled_figures()
    # Reference Figure 2 explicitly -> must select the bottom figure.
    crop = find_image_crop_by_reference(
        data, 1, "As shown in Figure 2, the tissue is striated.", margin=0.0
    )
    assert crop is not None
    assert crop.y0 > 0.5


def test_select_image_crop_prefers_referenced_figure() -> None:
    data = make_pdf_with_labeled_figures()
    # Support text references Figure 1 -> select the top figure, not the bottom.
    crop = select_image_crop(
        data, 1, "See Figure 1 for the structure described.", margin=0.0
    )
    assert crop is not None
    assert crop.y1 < 0.5


def test_select_image_crop_reference_in_extra_text() -> None:
    data = make_pdf_with_labeled_figures()
    # Figure ref only in the question stem (extra_text), not source_text.
    crop = select_image_crop(
        data,
        1,
        "unrelated supporting evidence",
        extra_text="Which structure is shown in Fig. 2?",
        margin=0.0,
    )
    assert crop is not None
    assert crop.y0 > 0.5


def test_tier2_caption_similarity_match_without_figure_number() -> None:
    data = make_pdf_with_labeled_figures()
    # No explicit figure number; caption-word overlap should pick Myocardium.
    crop = find_image_crop_by_caption(
        data,
        1,
        "This question is about the myocardium cross section detail.",
        margin=0.0,
    )
    assert crop is not None
    assert crop.y0 > 0.5


def test_multiple_figures_different_captions_resolve_distinctly() -> None:
    data = make_pdf_with_labeled_figures()
    top = select_image_crop(data, 1, "Figure 1 glomerulus", margin=0.0)
    bottom = select_image_crop(data, 1, "Figure 2 myocardium", margin=0.0)
    assert top is not None and bottom is not None
    assert top.y1 < 0.5
    assert bottom.y0 > 0.5


def test_low_confidence_returns_no_image() -> None:
    data = make_pdf_with_labeled_figures()
    # No figure reference, no caption overlap, no nearby anchor text match.
    crop = select_image_crop(
        data, 1, "entirely unrelated content with no overlap tokens"
    )
    assert crop is None


def test_caption_match_rejects_conflicting_medical_term() -> None:
    # Question is about lichen; the only caption mentions syphilis -> reject.
    data = make_pdf_with_conflicting_caption()
    query = "Which feature is characteristic of lichen planus in this patient?"
    assert find_image_crop_by_caption(data, 1, query) is None


def test_caption_match_succeeds_on_shared_salient_term() -> None:
    # Caption shares the salient term 'lichen' with the question -> match.
    data = make_pdf_with_matching_caption()
    query = "Which feature is characteristic of lichen planus in this patient?"
    crop = find_image_crop_by_caption(data, 1, query)
    assert crop is not None


def test_caption_match_rejects_low_lexical_overlap() -> None:
    # Only stopword/structural overlap with the caption -> no salient term -> reject.
    data = make_pdf_with_matching_caption()  # caption: "Lichen planus papules..."
    query = "This figure is shown at high power in the following section."
    assert find_image_crop_by_caption(data, 1, query) is None


def test_select_image_crop_conflicting_caption_returns_no_image() -> None:
    # End-to-end: lichen question, syphilis caption, no figure ref, no anchor
    # match -> lone-captioned Tier 4 still fires only via caption adjacency, but
    # the caption tier (Tier 2) must not select the conflicting figure.
    data = make_pdf_with_conflicting_caption()
    crop = find_image_crop_by_caption(
        data, 1, "Describe the lichen planus lesion morphology."
    )
    assert crop is None


def test_extract_chapters_from_toc() -> None:
    data = make_pdf_with_toc(pages=6)
    chapters = extract_chapters(data)
    titles = [c.title for c in chapters]
    assert titles == ["Introduction", "Methods", "Results"]


def test_chapter_ranges_computed_from_successive_entries() -> None:
    data = make_pdf_with_toc(pages=6)
    chapters = extract_chapters(data)
    ranges = [(c.start_page, c.end_page) for c in chapters]
    # Level-1 entries start at 1, 3, 5; level-2 entry is ignored.
    assert ranges == [(1, 2), (3, 4), (5, 6)]
    assert chapters[0].pages == {1, 2}
    assert chapters[2].pages == {5, 6}


def test_extract_chapters_empty_without_toc() -> None:
    data = _make_pdf(pages=3)
    assert extract_chapters(data) == []


def test_chunk_chapters_fixed_ranges() -> None:
    chapters = chunk_chapters(60, chunk_size=25)
    ranges = [(c.start_page, c.end_page) for c in chapters]
    assert ranges == [(1, 25), (26, 50), (51, 60)]
    titles = [c.title for c in chapters]
    assert titles == ["Pages 1-25", "Pages 26-50", "Pages 51-60"]


def test_plan_chapters_small_pdf_no_split() -> None:
    # 25 pages or fewer -> whole-document flow (no chapters/chunks).
    data = make_pdf_with_toc(pages=6)
    assert plan_chapters(data) == []
    data2 = _make_pdf(pages=10)
    assert plan_chapters(data2) == []


def test_plan_chapters_large_pdf_with_toc_uses_chapters() -> None:
    data = make_large_pdf_with_toc(pages=40)
    chapters = plan_chapters(data)
    titles = [c.title for c in chapters]
    ranges = [(c.start_page, c.end_page) for c in chapters]
    assert titles == ["Alpha", "Beta", "Gamma"]
    # Level-1 starts at 1, 15, 30 over 40 pages.
    assert ranges == [(1, 14), (15, 29), (30, 40)]


def test_plan_chapters_large_pdf_without_toc_uses_chunks() -> None:
    data = _make_pdf(pages=55)
    chapters = plan_chapters(data)
    ranges = [(c.start_page, c.end_page) for c in chapters]
    assert ranges == [(1, 25), (26, 50), (51, 55)]
