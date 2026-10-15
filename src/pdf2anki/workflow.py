"""Orchestration of the two-pass pipeline over an uploaded PDF."""

from __future__ import annotations

from dataclasses import dataclass, field

from .anthropic_client import AnthropicClient
from .config import MAX_QUESTIONS, QUESTIONS_PER_BATCH
from .schemas import MCQ, CoveragePlan
from .validators import ValidationResult, filter_valid_mcqs, validate_mcq


@dataclass
class GenerationResult:
    """Result of running the full pipeline."""

    plan: CoveragePlan
    mcqs: list[MCQ] = field(default_factory=list)
    rejected: list[tuple[MCQ, ValidationResult]] = field(default_factory=list)
    batch_errors: list[str] = field(default_factory=list)


def _batches(count: int, size: int) -> list[int]:
    """Split a total count (clamped to MAX_QUESTIONS) into batch sizes of at most `size`."""
    if count <= 0 or size <= 0:
        return []
    clamped = min(count, MAX_QUESTIONS)
    full, remainder = divmod(clamped, size)
    sizes = [size] * full
    if remainder:
        sizes.append(remainder)
    return sizes


def _topic_slug(topic: str, index: int) -> str:
    """Build a filesystem/id-safe short prefix for a topic batch."""
    slug = "".join(c.lower() if c.isalnum() else "-" for c in topic).strip("-")
    slug = slug or f"topic{index}"
    return slug[:24]


def generate_deck(
    client: AnthropicClient,
    file_id: str,
    page_count: int,
    max_questions: int,
    batch_size: int = QUESTIONS_PER_BATCH,
) -> GenerationResult:
    """Run Pass 1 and Pass 2 and return validated questions.

    The caller owns the uploaded file lifecycle and remote cleanup.
    """
    max_questions = min(max_questions, MAX_QUESTIONS)
    plan = client.plan_coverage(file_id, max_questions)

    all_mcqs: list[MCQ] = []
    batch_errors: list[str] = []

    # Build the effective allocation: use the plan's allocation if present,
    # otherwise put the whole budget under the subject as a single topic.
    effective: list[tuple[str, int]]
    if plan.allocation:
        effective = [(a.topic, a.question_count) for a in plan.allocation]
    else:
        effective = [(plan.subject or "General", min(plan.question_count, max_questions))]

    remaining = max_questions
    seen_ids: set[str] = set()
    counter = 0

    for t_index, (topic, topic_count) in enumerate(effective):
        if remaining <= 0:
            break
        topic_count = min(topic_count, remaining)
        for b_index, size in enumerate(_batches(topic_count, batch_size)):
            id_prefix = f"t{t_index}-{_topic_slug(topic, t_index)}-{b_index}-"
            try:
                batch = client.generate_mcqs(
                    file_id=file_id,
                    topic=topic,
                    count=size,
                    learner_level=plan.learner_level,
                    id_prefix=id_prefix,
                )
            except Exception as exc:
                batch_errors.append(f"Batch '{topic}' #{b_index + 1} failed: {exc}")
                continue

            for mcq in batch.questions:
                counter += 1
                candidate_id = mcq.id.strip() or f"{id_prefix}{counter}"
                if candidate_id in seen_ids:
                    candidate_id = f"{id_prefix}{counter}"
                seen_ids.add(candidate_id)
                mcq.id = candidate_id
                all_mcqs.append(mcq)
        remaining -= topic_count

    if not all_mcqs and batch_errors:
        raise RuntimeError("; ".join(batch_errors))

    valid, rejected = filter_valid_mcqs(all_mcqs, page_count)
    return GenerationResult(
        plan=plan,
        mcqs=valid,
        rejected=rejected,
        batch_errors=batch_errors,
    )


def regenerate_single_mcq(
    client: AnthropicClient,
    pdf_bytes: bytes,
    pdf_name: str,
    topic: str,
    learner_level: str,
    page_count: int,
    question_id: str,
) -> MCQ:
    """Upload the PDF, generate a single replacement MCQ, validate, and clean up."""
    file_id: str | None = None
    try:
        file_id = client.upload_pdf(pdf_bytes, pdf_name)
        batch = client.generate_mcqs(
            file_id=file_id,
            topic=topic,
            count=1,
            learner_level=learner_level,
            id_prefix=f"{question_id}-regen-",
        )
    finally:
        if file_id:
            client.delete_file(file_id)

    if not batch.questions:
        raise RuntimeError("Claude returned no replacement question.")

    mcq = batch.questions[0]
    mcq.id = question_id
    res = validate_mcq(mcq, page_count)
    if not res.is_valid:
        raise ValueError(f"Regenerated question failed validation: {', '.join(res.errors)}")
    return mcq
