"""Validation rules for generated MCQs."""

from __future__ import annotations

from dataclasses import dataclass, field

from .schemas import MCQ


@dataclass
class ValidationResult:
    """Outcome of validating a single MCQ."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)


def validate_mcq(mcq: MCQ, page_count: int) -> ValidationResult:
    """Validate a single MCQ against structural and grounding rules."""
    errors: list[str] = []

    options = [o.strip() for o in mcq.options]
    if len(mcq.options) != 4:
        errors.append("Question must have exactly four options.")
    if any(not o for o in options):
        errors.append("All four options must be non-empty.")
    if len(set(options)) != len(options):
        errors.append("Options must be distinct.")

    if not 0 <= mcq.correct_index <= 3:
        errors.append("correct_index must be between 0 and 3.")

    if not mcq.stem.strip():
        errors.append("Stem must not be empty.")

    if not mcq.explanation.strip():
        errors.append("Explanation must not be empty.")

    if not mcq.source_pages:
        errors.append("At least one source page is required.")
    else:
        mcq.source_pages = sorted(set(mcq.source_pages))
    for page in mcq.source_pages:
        if page < 1 or page > page_count:
            errors.append(f"Source page {page} is out of range (1-{page_count}).")

    if mcq.image_required:
        if mcq.image_page is None:
            errors.append("Image is required but image_page is missing.")
        elif mcq.image_page < 1 or mcq.image_page > page_count:
            errors.append(
                f"image_page {mcq.image_page} is out of range (1-{page_count})."
            )
        if mcq.image_crop is not None and (
            mcq.image_crop.x1 <= mcq.image_crop.x0
            or mcq.image_crop.y1 <= mcq.image_crop.y0
        ):
            errors.append("image_crop must have positive width and height.")

    return ValidationResult(is_valid=not errors, errors=errors)


def filter_valid_mcqs(
    mcqs: list[MCQ], page_count: int
) -> tuple[list[MCQ], list[tuple[MCQ, ValidationResult]]]:
    """Split MCQs into valid ones and rejected ones with their results."""
    valid: list[MCQ] = []
    rejected: list[tuple[MCQ, ValidationResult]] = []
    for mcq in mcqs:
        result = validate_mcq(mcq, page_count)
        if result.is_valid:
            valid.append(mcq)
        else:
            rejected.append((mcq, result))
    return valid, rejected
