"""Orchestration of the two-pass pipeline over an uploaded PDF."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

from .anthropic_client import AnthropicClient
from .config import MAX_QUESTIONS, QUESTIONS_PER_BATCH
from .pdf_media import extract_figure_bundles
from .schemas import MCQ, CoveragePlan, FigureBundle
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
    figure_bundles: list[FigureBundle] = field(default_factory=list)
    no_captioned_figures: bool = False


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


def distribute_total(total: int, weights: list[float]) -> list[int]:
    """Split ``total`` across buckets proportional to ``weights``.

    Uses the largest-remainder method so the returned integers sum exactly to
    ``total``. Non-positive or empty weights fall back to an even split.
    """
    n = len(weights)
    if total <= 0 or n == 0:
        return [0] * n
    clean = [w if w > 0 else 0.0 for w in weights]
    weight_sum = sum(clean)
    if weight_sum <= 0:
        clean = [1.0] * n
        weight_sum = float(n)

    exact = [total * w / weight_sum for w in clean]
    floors = [int(x) for x in exact]
    remainder = total - sum(floors)
    # Distribute the remaining units to the largest fractional parts.
    order = sorted(
        range(n), key=lambda i: (exact[i] - floors[i]), reverse=True
    )
    for i in range(remainder):
        floors[order[i % n]] += 1
    return floors


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

    # Build the effective allocation: use the plan's allocation weights if
    # present, otherwise a single subject bucket. Rescale so the per-topic
    # counts sum to exactly ``max_questions``.
    if plan.allocation:
        topics = [a.topic for a in plan.allocation]
        weights = [float(max(0, a.question_count)) for a in plan.allocation]
    else:
        topics = [plan.subject or "General"]
        weights = [1.0]

    counts = distribute_total(max_questions, weights)
    effective: list[tuple[str, int]] = list(zip(topics, counts, strict=True))

    seen_ids: set[str] = set()
    counter = 0

    for t_index, (topic, topic_count) in enumerate(effective):
        if topic_count <= 0:
            continue
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


def split_mixed_counts(
    total: int,
    figure_pct: int,
    available_figures: int,
) -> tuple[int, int]:
    """Split ``total`` into (figure_grounded_count, text_based_count).

    ``figure_pct`` is an integer 0-100. The figure count is the rounded portion
    of the total, capped by both ``total`` and ``available_figures`` (limited
    figures shift their remainder to text). The returned pair always sums to
    exactly ``total``.
    """
    if total <= 0:
        return (0, 0)
    pct = max(0, min(100, figure_pct))
    desired_fig = int(round(total * pct / 100.0))
    fig = min(desired_fig, total, max(0, available_figures))
    text = total - fig
    return (fig, text)


def generate_mixed_deck(
    client: AnthropicClient,
    file_id: str,
    pdf_bytes: bytes,
    page_count: int,
    total: int,
    figure_pct: int,
    allowed_pages: set[int] | None = None,
) -> GenerationResult:
    """Generate a deck mixing figure-grounded and text-based questions.

    The figure/text split is computed from ``figure_pct`` and capped by the
    number of available captioned figure bundles, shifting any shortfall to the
    text-based generator so the overall total is preserved exactly.
    """
    bundles = extract_figure_bundles(pdf_bytes, allowed_pages=allowed_pages)
    available = len(bundles)
    fig_n, text_n = split_mixed_counts(total, figure_pct, available)

    mcqs: list[MCQ] = []
    rejected: list[tuple[MCQ, ValidationResult]] = []
    batch_errors: list[str] = []
    figure_bundles: list[FigureBundle] = []
    fig_plan: CoveragePlan | None = None
    text_plan: CoveragePlan | None = None

    if fig_n > 0:
        fig_result = generate_figure_grounded_deck(
            client=client,
            file_id=file_id,
            pdf_bytes=pdf_bytes,
            page_count=page_count,
            max_questions=fig_n,
            allowed_pages=allowed_pages,
        )
        mcqs.extend(fig_result.mcqs)
        rejected.extend(fig_result.rejected)
        batch_errors.extend(fig_result.batch_errors)
        figure_bundles = fig_result.figure_bundles
        fig_plan = fig_result.plan

    if text_n > 0:
        text_result = generate_deck(
            client=client,
            file_id=file_id,
            page_count=page_count,
            max_questions=text_n,
            allowed_pages=allowed_pages,
        )
        mcqs.extend(text_result.mcqs)
        rejected.extend(text_result.rejected)
        batch_errors.extend(text_result.batch_errors)
        text_plan = text_result.plan

    # Defensive cross-mode id dedup (prefixes differ, so collisions are rare).
    seen: set[str] = set()
    deduped: list[MCQ] = []
    counter = 0
    for mcq in mcqs:
        candidate = mcq.id
        while candidate in seen:
            counter += 1
            candidate = f"{mcq.id}-{counter}"
        seen.add(candidate)
        if candidate != mcq.id:
            mcq.id = candidate
        deduped.append(mcq)

    plan = text_plan or fig_plan or CoveragePlan(
        title="Mixed Deck",
        subject="Mixed",
        learner_level="student",
        question_count=len(deduped),
    )
    return GenerationResult(
        plan=plan,
        mcqs=deduped,
        rejected=rejected,
        batch_errors=batch_errors,
        figure_bundles=figure_bundles,
        no_captioned_figures=(available == 0),
    )


def generate_figure_grounded_deck(
    client: AnthropicClient,
    file_id: str,
    pdf_bytes: bytes,
    page_count: int,
    max_questions: int = MAX_QUESTIONS,
    allowed_pages: set[int] | None = None,
) -> GenerationResult:
    """Extract captioned figure bundles and generate MCQs grounded per figure."""
    bundles = extract_figure_bundles(pdf_bytes, allowed_pages=allowed_pages)
    if not bundles:
        empty_plan = CoveragePlan(
            title="Figure-Grounded Deck",
            subject="Visuals",
            learner_level="student",
            question_count=0,
        )
        return GenerationResult(
            plan=empty_plan,
            mcqs=[],
            rejected=[],
            batch_errors=[],
            figure_bundles=[],
            no_captioned_figures=True,
        )

    capped_total = min(max(1, max_questions), MAX_QUESTIONS)
    selected_bundles = bundles[:capped_total]
    weights = [1.0] * len(selected_bundles)
    counts = distribute_total(capped_total, weights)

    all_mcqs: list[MCQ] = []
    batch_errors: list[str] = []
    seen_ids: set[str] = set()
    counter = 0

    for idx, (bundle, count) in enumerate(
        zip(selected_bundles, counts, strict=True)
    ):
        if count <= 0:
            continue
        id_prefix = f"{bundle.figure_id}-"
        try:
            batch = client.generate_figure_mcqs(
                file_id=file_id,
                bundle=bundle,
                count=count,
                id_prefix=id_prefix,
            )
        except Exception as exc:
            batch_errors.append(
                f"Figure bundle '{bundle.figure_id}' (#{idx + 1}) failed: {exc}"
            )
            continue

        for mcq in batch.questions:
            counter += 1
            candidate_id = mcq.id.strip() or f"{id_prefix}{counter}"
            while candidate_id in seen_ids:
                candidate_id = f"{id_prefix}{counter}"
                counter += 1
            seen_ids.add(candidate_id)
            stamped = mcq.model_copy(
                update={
                    "id": candidate_id,
                    "source_pages": [bundle.page_number],
                    "source_support": mcq.source_support or bundle.caption_text,
                    "image_required": True,
                    "image_page": bundle.page_number,
                    "image_crop": bundle.image_crop,
                    "image_alt_text": mcq.image_alt_text or bundle.caption_text,
                }
            )
            all_mcqs.append(stamped)

    if not all_mcqs and batch_errors:
        raise RuntimeError("; ".join(batch_errors))

    valid, rejected = filter_valid_mcqs(
        all_mcqs, page_count, allowed_pages=allowed_pages
    )
    valid = shuffle_all_mcqs(valid)
    plan = CoveragePlan(
        title="Figure-Grounded Deck",
        subject="Visual Figures",
        learner_level="student",
        topics=[b.figure_label or b.figure_id for b in selected_bundles],
        question_count=len(valid),
        visual_pages=sorted({b.page_number for b in selected_bundles}),
    )
    return GenerationResult(
        plan=plan,
        mcqs=valid,
        rejected=rejected,
        batch_errors=batch_errors,
        figure_bundles=selected_bundles,
        no_captioned_figures=False,
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
