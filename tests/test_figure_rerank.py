"""Tests for the Claude-backed figure reranker orchestration."""

from __future__ import annotations

from conftest import make_pdf_with_labeled_figures
from pdf2anki.figure_rerank import build_candidates, rerank_select_crop
from pdf2anki.pdf_media import select_image_crop
from pdf2anki.schemas import FigureRerankItem, FigureRerankResult


class _StubClient:
    def __init__(self, result: FigureRerankResult) -> None:
        self._result = result
        self.last_candidates: list[dict[str, object]] | None = None

    def rerank_figures(
        self,
        question_text: str,
        support_text: str,
        candidates: list[dict[str, object]],
    ) -> FigureRerankResult:
        self.last_candidates = candidates
        return self._result


def test_build_candidates_includes_label_caption_and_bbox() -> None:
    data = make_pdf_with_labeled_figures()
    cands = build_candidates(data, 1)
    assert len(cands) == 2
    labels = sorted(str(c["figure_label"]) for c in cands)
    assert labels == ["figure 1", "figure 2"]
    assert all("_bbox" in c for c in cands)
    captions = " ".join(str(c["caption_text"]) for c in cands)
    assert "Glomerulus" in captions and "Myocardium" in captions


def _result_select(cid: str, **item_kw: object) -> FigureRerankResult:
    items = [
        FigureRerankItem(candidate_id="c1", score_0_to_100=10, keep=False),
        FigureRerankItem(candidate_id="c2", score_0_to_100=10, keep=False),
    ]
    for it in items:
        if it.candidate_id == cid:
            for k, v in item_kw.items():
                setattr(it, k, v)
    return FigureRerankResult(items=items, final_selected_candidate_id=cid)


def test_rerank_figure_number_driven_selection() -> None:
    data = make_pdf_with_labeled_figures()
    # c2 is the bottom (Figure 2) image.
    result = _result_select(
        "c2", figure_number_match=True, score_0_to_100=95, keep=True
    )
    client = _StubClient(result)
    crop = rerank_select_crop(client, data, 1, "See Figure 2.", "", margin=0.0)
    assert crop is not None
    assert crop.y0 > 0.5


def test_rerank_caption_driven_selection() -> None:
    data = make_pdf_with_labeled_figures()
    result = _result_select(
        "c1", score_0_to_100=88, keep=True, caption_overlap_terms=["glomerulus"]
    )
    client = _StubClient(result)
    crop = rerank_select_crop(
        client, data, 1, "Which structure filters blood?", "glomerulus", margin=0.0
    )
    assert crop is not None
    assert crop.y1 < 0.5


def test_rerank_mismatch_rejection_despite_semantic_closeness() -> None:
    data = make_pdf_with_labeled_figures()
    result = FigureRerankResult(
        items=[
            FigureRerankItem(
                candidate_id="c1",
                score_0_to_100=20,
                keep=False,
                contradiction_terms=["myocardium"],
            ),
            FigureRerankItem(candidate_id="c2", score_0_to_100=15, keep=False),
        ],
        final_selected_candidate_id=None,
    )
    client = _StubClient(result)
    crop = rerank_select_crop(
        client, data, 1, "Describe hepatocytes.", "liver lobule", margin=0.0
    )
    assert crop is None


def test_rerank_low_confidence_returns_none() -> None:
    data = make_pdf_with_labeled_figures()
    result = _result_select("c1", score_0_to_100=55, keep=True)  # below 60
    client = _StubClient(result)
    crop = rerank_select_crop(client, data, 1, "Anything", "", margin=0.0)
    assert crop is None


def test_rerank_client_failure_falls_back_to_none() -> None:
    class _Boom:
        def rerank_figures(self, *a: object, **k: object) -> FigureRerankResult:
            raise RuntimeError("api down")

    data = make_pdf_with_labeled_figures()
    assert rerank_select_crop(_Boom(), data, 1, "q", "s") is None


def test_select_image_crop_tier0_uses_client_when_provided() -> None:
    data = make_pdf_with_labeled_figures()
    result = _result_select(
        "c2", figure_number_match=True, score_0_to_100=99, keep=True
    )
    client = _StubClient(result)
    # No figure ref in text, but the client reranker selects c2 (bottom figure).
    crop = select_image_crop(
        data, 1, "ambiguous support", extra_text="pick best", margin=0.0, client=client
    )
    assert crop is not None
    assert crop.y0 > 0.5


def test_parsing_structured_rerank_output() -> None:
    # Ensure strict schema parsing of a full reranker payload.
    payload = {
        "items": [
            {
                "candidate_id": "c1",
                "figure_number_match": True,
                "caption_overlap_terms": ["glomerulus"],
                "contradiction_terms": [],
                "score_0_to_100": 92,
                "keep": True,
                "short_reason": "figure-number + caption",
            }
        ],
        "final_selected_candidate_id": "c1",
    }
    result = FigureRerankResult.model_validate(payload)
    assert result.items[0].figure_number_match is True
    assert result.items[0].score_0_to_100 == 92
    assert result.final_selected_candidate_id == "c1"
