"""Minimal Streamlit UI for Phase 1 of pdf2anki-mcq."""

from __future__ import annotations

import streamlit as st

from pdf2anki.anthropic_client import AnthropicClient
from pdf2anki.config import (
    ACCEPTED_MIME_TYPE,
    DEFAULT_MAX_QUESTIONS,
    MAX_PDF_BYTES,
    MAX_QUESTIONS,
    load_settings,
)
from pdf2anki.workflow import generate_deck


def _page_count(data: bytes) -> int:
    """Return the PDF page count using PyMuPDF."""
    import fitz

    with fitz.open(stream=data, filetype="pdf") as doc:
        return doc.page_count


def main() -> None:
    st.set_page_config(page_title="PDF to Anki MCQ", page_icon="📚")
    st.title("PDF → Anki MCQ Generator")
    st.caption("Phase 1: upload, plan, and generate validated MCQs.")

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

    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded is None:
        st.info("Upload an unencrypted PDF to begin.")
        return

    data = uploaded.getvalue()

    if uploaded.type != ACCEPTED_MIME_TYPE:
        st.error("The uploaded file is not a PDF.")
        return
    if len(data) > MAX_PDF_BYTES:
        st.error("The PDF exceeds the 32 MB limit for a single request.")
        return

    try:
        pages = _page_count(data)
    except Exception:
        st.error("The PDF could not be read. It may be encrypted or corrupt.")
        return

    st.success(f"PDF accepted: {pages} page(s), {len(data) // 1024} KB.")

    if not settings.anthropic_api_key:
        st.error("No Anthropic API key available. Set it above or in the environment.")
        return

    if not st.button("Generate MCQs"):
        return

    client = AnthropicClient.from_api_key(settings.anthropic_api_key, settings.model)

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
    except Exception as exc:  # pragma: no cover - UI path
        st.error(f"Generation failed: {exc}")
        return
    finally:
        if file_id:
            client.delete_file(file_id)

    st.subheader("Coverage plan")
    st.json(result.plan.model_dump())

    st.subheader(f"Generated questions ({len(result.mcqs)})")
    for mcq in result.mcqs:
        with st.expander(f"{mcq.id}: {mcq.stem}"):
            st.json(mcq.model_dump())

    if result.rejected:
        st.subheader(f"Rejected questions ({len(result.rejected)})")
        for mcq, validation in result.rejected:
            st.write(f"**{mcq.id}** — {', '.join(validation.errors)}")


if __name__ == "__main__":
    main()
