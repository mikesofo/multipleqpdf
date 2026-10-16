"""Pydantic schemas for the coverage plan and MCQ output."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class TopicAllocation(BaseModel):
    """Number of questions allocated to a single topic."""

    topic: str
    question_count: int = Field(ge=0)


class CoveragePlan(BaseModel):
    """First-pass plan describing the document and question budget."""

    title: str
    subject: str
    learner_level: str
    topics: list[str] = Field(default_factory=list)
    learning_objectives: list[str] = Field(default_factory=list)
    complexity: str = ""
    visual_dependence: str = ""
    question_count: int = Field(ge=0)
    allocation: list[TopicAllocation] = Field(default_factory=list)
    visual_pages: list[int] = Field(default_factory=list)


class ImageCrop(BaseModel):
    """Normalized crop rectangle in 0..1 coordinates."""

    x0: float = Field(ge=0.0, le=1.0)
    y0: float = Field(ge=0.0, le=1.0)
    x1: float = Field(ge=0.0, le=1.0)
    y1: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_positive_area(self) -> ImageCrop:
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("ImageCrop requires x1 > x0 and y1 > y0.")
        return self


class MCQ(BaseModel):
    """A single multiple-choice question.

    ``question_type`` is a hard invariant: ``text_based`` questions may never
    carry any image data. The validator below clears all image fields for
    text-based questions regardless of what a model returned, so image leakage
    into text-only mode is structurally impossible.
    """

    id: str
    learning_objective: str
    topic: str
    stem: str
    options: list[str]
    correct_index: int = Field(ge=0, le=3)
    explanation: str
    source_pages: list[int] = Field(default_factory=list)
    source_support: str = ""
    question_type: Literal["text_based", "figure_grounded"] = "text_based"
    image_required: bool = False
    image_page: int | None = None
    image_crop: ImageCrop | None = None
    image_alt_text: str | None = None

    @model_validator(mode="after")
    def _enforce_text_based_has_no_image(self) -> MCQ:
        if self.question_type == "text_based":
            object.__setattr__(self, "image_required", False)
            object.__setattr__(self, "image_page", None)
            object.__setattr__(self, "image_crop", None)
            object.__setattr__(self, "image_alt_text", None)
        return self


class MCQBatch(BaseModel):
    """A batch of MCQs returned by a single generation call."""

    questions: list[MCQ] = Field(default_factory=list)


class FigureCandidate(BaseModel):
    """A deterministically extracted figure candidate shown to the reranker."""

    candidate_id: str
    figure_label: str = ""
    caption_text: str = ""
    page_number: int
    nearby_reference_text: str = ""
    local_text_snippet: str = ""


class FigureRerankItem(BaseModel):
    """Per-candidate structured judgment from Claude."""

    candidate_id: str
    figure_number_match: bool = False
    caption_overlap_terms: list[str] = Field(default_factory=list)
    contradiction_terms: list[str] = Field(default_factory=list)
    score_0_to_100: int = Field(ge=0, le=100)
    keep: bool = False
    short_reason: str = ""


class FigureRerankResult(BaseModel):
    """Full reranker output: per-candidate items plus a final decision."""

    items: list[FigureRerankItem] = Field(default_factory=list)
    final_selected_candidate_id: str | None = None


class FigureBundle(BaseModel):
    """Deterministically extracted figure bundle with caption, crop, and context."""

    figure_id: str
    figure_label: str = ""
    caption_text: str
    page_number: int = Field(ge=1)
    image_crop: ImageCrop | None = None
    nearby_text: str = ""
    references: list[str] = Field(default_factory=list)
