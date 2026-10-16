"""Streamlit UI for PDF-to-Anki MCQ Generator."""

from __future__ import annotations

import streamlit as st

from pdf2anki.anki_export import OPTION_LABELS, export_apkg_bytes
from pdf2anki.anthropic_client import AnthropicClient
from pdf2anki.config import (
    ACCEPTED_MIME_TYPE,
    DEFAULT_MAX_QUESTIONS,
    MAX_PDF_BYTES,
    MAX_PDF_PAGES,
    MAX_QUESTIONS,
    load_settings,
)
from pdf2anki.pdf_media import (
    PDFValidationError,
    count_page_images,
    render_crop,
    validate_pdf,
)
from pdf2anki.schemas import MCQ, ImageCrop
from pdf2anki.validators import validate_mcq
from pdf2anki.workflow import generate_deck, regenerate_single_mcq


def _parse_pages(raw: str) -> tuple[list[int], str | None]:
    """Parse a comma-separated list of 1-based page numbers, keeping valid integers."""
    pages: list[int] = []
    bad_tokens: list[str] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            pages.append(int(token))
        except ValueError:
            bad_tokens.append(token)
    error = "Source pages must be integers." if bad_tokens else None
    return sorted(set(pages)), error


def _autoselect_images(mcqs: list[MCQ], pdf_bytes: bytes, page_count: int) -> None:
    """Preselect images for generated MCQs whose source page has a real figure.

    Mutates the list in place. Only fills in defaults when the model did not
    already request an image, so manual/model choices are preserved.
    """
    for i, mcq in enumerate(mcqs):
        if mcq.image_required:
            continue
        effective_page = (
            mcq.image_page
            if mcq.image_page is not None
            else (mcq.source_pages[0] if mcq.source_pages else None)
        )
        if effective_page is None or not (1 <= effective_page <= page_count):
            continue
        try:
            has_image = count_page_images(pdf_bytes, effective_page) >= 1
        except (ValueError, RuntimeError):
            has_image = False
        if has_image:
            mcqs[i] = mcq.model_copy(
                update={"image_required": True, "image_page": effective_page}
            )


def _render_editor(
    pdf_bytes: bytes,
    page_count: int,
    client: AnthropicClient | None,
    pdf_name: str,
) -> bool:
    """Render editable MCQ cards in session state and return True if all are valid."""
    mcqs: list[MCQ] = st.session_state.get("mcqs", [])
    plan = st.session_state.get("plan")
    learner_level = plan.learner_level if plan else "student"

    all_valid = True
    delete_index: int | None = None
    regen_index: int | None = None

    for idx, mcq in enumerate(mcqs):
        with st.expander(f"Q{idx + 1} ({mcq.id}): {mcq.stem[:80]}", expanded=False):
            topic = st.text_input(
                "Topic", value=mcq.topic, key=f"topic_{mcq.id}_{idx}"
            )
            stem = st.text_area(
                "Question stem", value=mcq.stem, key=f"stem_{mcq.id}_{idx}"
            )

            new_options: list[str] = []
            cols = st.columns(2)
            for opt_i, label in enumerate(OPTION_LABELS):
                current_val = mcq.options[opt_i] if opt_i < len(mcq.options) else ""
                with cols[opt_i % 2]:
                    val = st.text_input(
                        f"Option {label}",
                        value=current_val,
                        key=f"opt_{mcq.id}_{idx}_{opt_i}",
                    )
                    new_options.append(val)

            safe_index = mcq.correct_index if 0 <= mcq.correct_index <= 3 else 0
            selected_label = st.radio(
                "Correct option",
                options=list(OPTION_LABELS),
                index=safe_index,
                horizontal=True,
                key=f"correct_{mcq.id}_{idx}",
            )
            correct_index = OPTION_LABELS.index(selected_label)

            explanation = st.text_area(
                "Explanation",
                value=mcq.explanation,
                key=f"exp_{mcq.id}_{idx}",
            )

            pages_str = st.text_input(
                "Source pages (comma-separated, 1-based)",
                value=", ".join(str(p) for p in mcq.source_pages),
                key=f"pages_{mcq.id}_{idx}",
            )
            parsed_pages, pages_parse_error = _parse_pages(pages_str)

            image_required = st.checkbox(
                "Include image on card",
                value=mcq.image_required,
                key=f"imgreq_{mcq.id}_{idx}",
            )

            image_page = mcq.image_page
            image_crop = mcq.image_crop
            if image_required:
                default_page = (
                    mcq.image_page
                    if mcq.image_page and 1 <= mcq.image_page <= page_count
                    else (parsed_pages[0] if parsed_pages else 1)
                )
                image_page = int(
                    st.number_input(
                        "Image page (1-based)",
                        min_value=1,
                        max_value=max(1, page_count),
                        value=min(max(1, default_page), max(1, page_count)),
                        key=f"imgpage_{mcq.id}_{idx}",
                    )
                )
                use_crop = st.checkbox(
                    "Use crop rectangle (uncheck for full page)",
                    value=mcq.image_crop is not None,
                    key=f"usecrop_{mcq.id}_{idx}",
                )
                if use_crop:
                    c0, c1, c2, c3 = st.columns(4)
                    default_crop = mcq.image_crop or ImageCrop(
                        x0=0.0, y0=0.0, x1=1.0, y1=1.0
                    )
                    x0 = c0.slider(
                        "x0",
                        0.0,
                        1.0,
                        float(default_crop.x0),
                        0.01,
                        key=f"x0_{mcq.id}_{idx}",
                    )
                    y0 = c1.slider(
                        "y0",
                        0.0,
                        1.0,
                        float(default_crop.y0),
                        0.01,
                        key=f"y0_{mcq.id}_{idx}",
                    )
                    x1 = c2.slider(
                        "x1",
                        0.0,
                        1.0,
                        float(default_crop.x1),
                        0.01,
                        key=f"x1_{mcq.id}_{idx}",
                    )
                    y1 = c3.slider(
                        "y1",
                        0.0,
                        1.0,
                        float(default_crop.y1),
                        0.01,
                        key=f"y1_{mcq.id}_{idx}",
                    )
                    if x1 > x0 and y1 > y0:
                        image_crop = ImageCrop(x0=x0, y0=y0, x1=x1, y1=y1)
                    else:
                        st.warning(
                            "Crop requires x1 > x0 and y1 > y0; falling back to full page."
                        )
                        image_crop = None
                else:
                    image_crop = None

                try:
                    preview_png = render_crop(pdf_bytes, image_page, image_crop)
                    st.image(preview_png, caption=f"Page {image_page} preview")
                except Exception as exc:
                    st.error(f"Could not render page preview: {exc}")
            else:
                image_page = None
                image_crop = None

            updated = mcq.model_copy(
                update={
                    "topic": topic,
                    "stem": stem,
                    "options": new_options,
                    "correct_index": correct_index,
                    "explanation": explanation,
                    "source_pages": parsed_pages,
                    "image_required": image_required,
                    "image_page": image_page,
                    "image_crop": image_crop,
                }
            )
            mcqs[idx] = updated

            validation = validate_mcq(updated, page_count)
            if pages_parse_error:
                validation.is_valid = False
                validation.errors.append(pages_parse_error)

            if not validation.is_valid:
                all_valid = False
                st.error("Validation errors: " + "; ".join(validation.errors))

            bcol1, bcol2 = st.columns(2)
            if bcol1.button("Delete question", key=f"del_{mcq.id}_{idx}"):
                delete_index = idx
            if bcol2.button(
                "Regenerate question",
                key=f"regen_{mcq.id}_{idx}",
                disabled=client is None,
            ):
                regen_index = idx

    if delete_index is not None:
        mcqs.pop(delete_index)
        st.session_state["mcqs"] = mcqs
        st.session_state.pop("apkg_bytes", None)
        st.rerun()

    if regen_index is not None and client is not None:
        target = mcqs[regen_index]
        try:
            with st.spinner(f"Regenerating {target.id}..."):
                replacement = regenerate_single_mcq(
                    client=client,
                    pdf_bytes=pdf_bytes,
                    pdf_name=pdf_name,
                    topic=target.topic,
                    learner_level=learner_level,
                    page_count=page_count,
                    question_id=target.id,
                )
            mcqs[regen_index] = replacement
            st.session_state["mcqs"] = mcqs
            st.session_state.pop("apkg_bytes", None)
            st.rerun()
        except Exception as exc:
            st.error(f"Regeneration failed: {exc}")

    return all_valid


def main() -> None:
    st.set_page_config(page_title="PDF to Anki MCQ", page_icon="📚")
    st.title("PDF → Anki MCQ Generator")
    st.caption("Upload a PDF, review & edit MCQs, and export an Anki .apkg deck.")

    st.warning(
        "Remove patient identifiers from clinical PDFs before uploading. "
        "Uploaded files are deleted from Anthropic after generation."
    )

    api_key_input = st.text_input(
        "Anthropic API key (optional if set in environment)",
        type="password",
        help="Read from ANTHROPIC_API_KEY if left blank.",
    )
    settings = load_settings(api_key_input or None)

    max_questions = st.slider(
        "Maximum questions",
        min_value=1,
        max_value=MAX_QUESTIONS,
        value=DEFAULT_MAX_QUESTIONS,
    )

    retain_remote = st.checkbox(
        "Temporarily retain the uploaded file on Anthropic (not recommended)",
        value=False,
    )

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded is None:
        st.info("Upload an unencrypted PDF to begin.")
        return

    data = uploaded.getvalue()

    try:
        pdf_info = validate_pdf(
            data,
            mime_type=uploaded.type or ACCEPTED_MIME_TYPE,
            max_bytes=MAX_PDF_BYTES,
            max_pages=MAX_PDF_PAGES,
        )
    except PDFValidationError as exc:
        st.error(str(exc))
        return

    pages = pdf_info.page_count
    st.success(f"PDF accepted: {pages} page(s), {len(data) // 1024} KB.")

    client: AnthropicClient | None = None
    if settings.anthropic_api_key:
        try:
            client = AnthropicClient.from_api_key(
                settings.anthropic_api_key, settings.model
            )
        except Exception as exc:
            st.error(str(exc))
            return
    else:
        st.info(
            "No Anthropic API key available. You can still edit existing "
            "questions and export, but generation and regeneration are disabled."
        )

    if client is not None and st.button("Generate MCQs"):
        file_id: str | None = None
        try:
            with st.spinner("Uploading PDF to Claude..."):
                file_id = client.upload_pdf(data, uploaded.name)
            with st.spinner("Planning coverage and generating questions..."):
                result = generate_deck(
                    client=client,
                    file_id=file_id,
                    page_count=pages,
                    max_questions=max_questions,
                )
            generated = list(result.mcqs)
            _autoselect_images(generated, data, pages)
            st.session_state["plan"] = result.plan
            st.session_state["mcqs"] = generated
            st.session_state["rejected"] = list(result.rejected)
            st.session_state["batch_errors"] = list(result.batch_errors)
            st.session_state["pdf_hash"] = pdf_info.pdf_hash
            st.session_state.pop("apkg_bytes", None)
        except Exception as exc:  # pragma: no cover - UI path
            st.error(f"Generation failed: {exc}")
            return
        finally:
            if file_id and not retain_remote:
                deleted = client.delete_file(file_id)
                if not deleted:
                    st.warning(
                        f"Remote file may not have been deleted from Anthropic: {file_id}"
                    )
            elif file_id and retain_remote:
                st.info(f"Remote file retained on Anthropic: {file_id}")

    if st.session_state.get("pdf_hash") != pdf_info.pdf_hash:
        return

    plan = st.session_state.get("plan")
    if plan is not None:
        st.subheader("Coverage plan")
        st.json(plan.model_dump())

    for err in st.session_state.get("batch_errors", []):
        st.warning(err)

    mcqs: list[MCQ] = st.session_state.get("mcqs", [])
    st.subheader(f"Editable questions ({len(mcqs)})")
    all_valid = _render_editor(
        pdf_bytes=data,
        page_count=pages,
        client=client,
        pdf_name=uploaded.name,
    )

    rejected = st.session_state.get("rejected", [])
    if rejected:
        st.subheader(f"Rejected questions ({len(rejected)})")
        for mcq, validation in rejected:
            st.write(f"**{mcq.id}** — {', '.join(validation.errors)}")

    if mcqs and all_valid:
        if st.button("Build Anki deck"):
            try:
                apkg_bytes = export_apkg_bytes(
                    mcqs=mcqs,
                    pdf_bytes=data,
                    pdf_hash=pdf_info.pdf_hash,
                    pdf_title=plan.title if plan and plan.title else uploaded.name,
                )
                st.session_state["apkg_bytes"] = apkg_bytes
            except Exception as exc:
                st.error(f"Failed to build Anki package: {exc}")
        stored_apkg = st.session_state.get("apkg_bytes")
        if isinstance(stored_apkg, bytes):
            st.download_button(
                label="Download Anki Deck (.apkg)",
                data=stored_apkg,
                file_name=f"{uploaded.name.rsplit('.', 1)[0]}.apkg",
                mime="application/octet-stream",
            )
    elif mcqs and not all_valid:
        st.session_state.pop("apkg_bytes", None)
        st.warning("Fix all validation errors above before exporting the Anki deck.")


if __name__ == "__main__":
    main()
