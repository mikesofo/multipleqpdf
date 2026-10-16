"""Orchestration of the two-pass pipeline over an uploaded PDF."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

from .anthropic_client import AnthropicClient
from .config import MAX_QUESTIONS, QUESTIONS_PER_BATCH
from .schemas import MCQ, CoveragePlan
from .validators import ValidationResult, filter_valid_mcqs, validate_mcq


def shuffle_mcq_options(mcq: MCQ, seed_key: str) -> MCQ:
    """Return a copy of the MCQ with options deterministically permuted.

    The correct answer is tracked by content, so ``correct_index`` is rewritten
    to the shuffled position. The permutation is derived deterministically from
    ``seed_key`` so it is stable across reruns. MCQs that are structurally
    invalid (not exactly four options, or an out-of-range index) are returned
    unchanged.
    """
    if len(mcq.options) != 4 or not (0 <= mcq.correct_index <= 3):
        return mcq
    correct_text = mcq.options[mcq.correct_index]
    digest = hashlib.sha256(seed_key.encode()).hexdigest()
    rng = random.Random(digest)
    order = list(range(len(mcq.options)))
    rng.shuffle(order)
    new_options = [mcq.options[i] for i in order]
    try:
        new_index = new_options.index(correct_text)
    except ValueError:  # pragma: no cover - defensive
        return mcq
    return mcq.model_copy(
        update={"options": new_options, "correct_index": new_index}
    )


def shuffle_all_mcqs(mcqs: list[MCQ], seed_prefix: str = "") -> list[MCQ]:
    """Deterministically shuffle options for every MCQ, seeded by its id."""
    return [
        shuffle_mcq_options(mcq, f"{seed_prefix}:{mcq.id}") for mcq in mcqs
    ]


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
    allowed_pages: set[int] | None = None,
) -> GenerationResult:
    """Run Pass 1 and Pass 2 and return validated questions.

    The caller owns the uploaded file lifecycle and remote cleanup. When
    ``allowed_pages`` is given, generated questions are steered to and validated
    against that page set (selected chapter ranges).
    """
    max_questions = min(max_questions, MAX_QUESTIONS)
    focus_pages = sorted(allowed_pages) if allowed_pages else None
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
                    focus_pages=focus_pages,
                )
            except Exception as exc:
                batch_errors.append(f"Batch '{topic}' #{b_index + 1} failed: {exc}")
                continue

            for mcq in batch.questions:
                counter += 1
                candidate_id = mcq.id.strip() or f"{id_prefix}{counter}"
                while candidate_id in seen_ids:
                    candidate_id = f"{id_prefix}{counter}"
                    counter += 1
                seen_ids.add(candidate_id)
                mcq.id = candidate_id
                all_mcqs.append(mcq)
        remaining -= topic_count

    if not all_mcqs and batch_errors:
        raise RuntimeError("; ".join(batch_errors))

    valid, rejected = filter_valid_mcqs(
        all_mcqs, page_count, allowed_pages=allowed_pages
    )
    valid = shuffle_all_mcqs(valid)
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
