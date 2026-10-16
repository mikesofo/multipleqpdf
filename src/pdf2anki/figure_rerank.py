"""Claude-backed structured reranking over deterministic figure candidates."""

from __future__ import annotations

from typing import Protocol, cast

import pymupdf as fitz

from .pdf_media import (
    MIN_IMAGE_DIMENSION,
    _candidate_image_rects,
    _caption_lines,
    _nearest_caption_for_image,
    _normalize_figure_refs,
    _rect_to_crop,
)
from .schemas import FigureRerankResult, ImageCrop

DEFAULT_MIN_CONFIDENCE: int = 60
MAX_RERANK_CANDIDATES: int = 8


class _RerankClient(Protocol):
    """Minimal client protocol so tests can pass a lightweight stub."""

    def rerank_figures(
        self,
        question_text: str,
        support_text: str,
        candidates: list[dict[str, object]],
    ) -> FigureRerankResult: ...


def build_candidates(
    pdf_bytes: bytes,
    page_number_1based: int,
    min_dim: int = MIN_IMAGE_DIMENSION,
) -> list[dict[str, object]]:
    """Deterministically build figure candidates (with bbox) for one page."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if page_number_1based < 1 or page_number_1based > doc.page_count:
            return []
        page = doc[page_number_1based - 1]
        prect = page.rect
        cap_rects, cap_texts = _caption_lines(page)
        candidates: list[dict[str, object]] = []
        for i, rect in enumerate(
            _candidate_image_rects(doc, page, min_dim=min_dim)
        ):
            caption = _nearest_caption_for_image(rect, cap_rects, cap_texts, prect)
            labels = _normalize_figure_refs(caption)
            label = sorted(labels)[0] if labels else ""
            candidates.append(
                {
                    "candidate_id": f"c{i + 1}",
                    "figure_label": label,
                    "caption_text": caption,
                    "page_number": page_number_1based,
                    "nearby_reference_text": caption,
                    "local_text_snippet": "",
                    "_bbox": rect,
                }
            )
            if len(candidates) >= MAX_RERANK_CANDIDATES:
                break
        return candidates


def rerank_select_crop(
    client: _RerankClient,
    pdf_bytes: bytes,
    page_number_1based: int,
    question_text: str,
    support_text: str,
    min_confidence: int = DEFAULT_MIN_CONFIDENCE,
    margin: float = 0.03,
) -> ImageCrop | None:
    """Use Claude to rerank candidates and return the selected crop, or None.

    Figure-number matches outrank semantic picks. Returns None when the client
    fails, no candidate is kept, or confidence is below ``min_confidence``.
    """
    candidates = build_candidates(pdf_bytes, page_number_1based)
    if not candidates:
        return None

    payload = [{k: v for k, v in c.items() if k != "_bbox"} for c in candidates]
    try:
        result = client.rerank_figures(question_text, support_text, payload)
    except Exception:
        return None

    by_id = {str(c["candidate_id"]): c for c in candidates}
    kept = [
        item
        for item in result.items
        if item.keep
        and item.score_0_to_100 >= min_confidence
        and item.candidate_id in by_id
    ]
    if not kept:
        return None

    # Figure-number evidence outranks semantic score.
    kept.sort(key=lambda it: (it.figure_number_match, it.score_0_to_100), reverse=True)

    selected_id = result.final_selected_candidate_id
    chosen = None
    if selected_id and selected_id in {it.candidate_id for it in kept}:
        chosen = next(it for it in kept if it.candidate_id == selected_id)
    else:
        chosen = kept[0]

    rect = cast(fitz.Rect, by_id[chosen.candidate_id]["_bbox"])
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        prect = doc[page_number_1based - 1].rect
    return _rect_to_crop(rect, prect, margin)
