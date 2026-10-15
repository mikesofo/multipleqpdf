"""Pydantic schemas for the coverage plan and MCQ output."""

from __future__ import annotations

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
    """A single multiple-choice question."""

    id: str
    learning_objective: str
    topic: str
    stem: str
    options: list[str]
    correct_index: int = Field(ge=0, le=3)
    explanation: str
    source_pages: list[int] = Field(default_factory=list)
    source_support: str = ""
    image_required: bool = False
    image_page: int | None = None
    image_crop: ImageCrop | None = None
    image_alt_text: str | None = None


class MCQBatch(BaseModel):
    """A batch of MCQs returned by a single generation call."""

    questions: list[MCQ] = Field(default_factory=list)
