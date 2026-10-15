"""Orchestration of the two-pass pipeline over an uploaded PDF."""

from __future__ import annotations

from dataclasses import dataclass, field

from .anthropic_client import AnthropicClient
from .config import QUESTIONS_PER_BATCH
from .schemas import CoveragePlan, MCQ
from .validators import ValidationResult, filter_valid_mcqs


@dataclass
class GenerationResult:
    """Result of running the full pipeline."""

    plan: CoveragePlan
    mcqs: list[MCQ] = field(default_factory=list)
    rejected: list[tuple[MCQ, ValidationResult]] = field(default_factory=list)


def _batches(count: int, size: int) -> list[int]:
    """Split a total count into batch sizes of at most `size`."""
    if count <= 0:
        return []
    full, remainder = divmod(count, size)
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
    plan = client.plan_coverage(file_id, max_questions)

    all_mcqs: list[MCQ] = []

    # Build the effective allocation: use the plan's allocation if present,
    # otherwise put the whole budget under the subject as a single topic.
    effective: list[tuple[str, int]]
    if plan.allocation:
        effective = [(a.topic, a.question_count) for a in plan.allocation]
    else:
        effective = [(plan.subject or "General", min(plan.question_count, max_questions))]

    remaining = max_questions
    for t_index, (topic, topic_count) in enumerate(effective):
        if remaining <= 0:
            break
        topic_count = min(topic_count, remaining)
        for b_index, size in enumerate(_batches(topic_count, batch_size)):
            id_prefix = f"{_topic_slug(topic, t_index)}-{b_index}-"
            batch = client.generate_mcqs(
                file_id=file_id,
                topic=topic,
                count=size,
                learner_level=plan.learner_level,
                id_prefix=id_prefix,
            )
            all_mcqs.extend(batch.questions)
        remaining -= topic_count

    valid, rejected = filter_valid_mcqs(all_mcqs, page_count)
    return GenerationResult(plan=plan, mcqs=valid, rejected=rejected)
