"""Deterministic Anki .apkg generation with embedded images using genanki."""

from __future__ import annotations

import hashlib
import html
import tempfile
from pathlib import Path

import genanki

from .pdf_media import DEFAULT_DPI, media_filename, render_crop
from .schemas import MCQ

OPTION_LABELS: tuple[str, str, str, str] = ("A", "B", "C", "D")

CARD_CSS = """
.card {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    font-size: 18px;
    line-height: 1.5;
    color: #1f2937;
    background-color: #ffffff;
    padding: 16px;
}
.topic {
    font-size: 13px;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
}
.stem {
    font-weight: 600;
    margin-bottom: 14px;
}
.image {
    margin: 12px 0;
    text-align: center;
}
.image img {
    max-width: 100%;
    height: auto;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
}
.options {
    list-style-type: none;
    padding-left: 0;
}
.options li {
    margin: 6px 0;
    padding: 8px 10px;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
}
.answer-box {
    margin-top: 14px;
    padding: 12px;
    background-color: #ecfdf5;
    border-left: 4px solid #10b981;
    border-radius: 4px;
}
.meta {
    margin-top: 10px;
    font-size: 12px;
    color: #6b7280;
}
"""

FRONT_TEMPLATE = """
<div class="topic">{{Topic}}</div>
<div class="stem">{{Question}}</div>
{{#Image}}<div class="image">{{Image}}</div>{{/Image}}
<ul class="options">
  <li><strong>A.</strong> {{OptionA}}</li>
  <li><strong>B.</strong> {{OptionB}}</li>
  <li><strong>C.</strong> {{OptionC}}</li>
  <li><strong>D.</strong> {{OptionD}}</li>
</ul>
"""

BACK_TEMPLATE = """
{{FrontSide}}
<hr id="answer">
<div class="answer-box">
  <div><strong>Correct Answer:</strong> {{CorrectAnswer}}</div>
  <div style="margin-top: 6px;">{{Explanation}}</div>
</div>
<div class="meta">Source: {{PDFTitle}} — Page(s) {{SourcePages}} ({{QuestionID}})</div>
"""


def _stable_id(namespace: str, key: str) -> int:
    """Return a deterministic 31-bit positive integer for genanki IDs."""
    digest = hashlib.sha256(f"{namespace}:{key}".encode()).digest()
    value = int.from_bytes(digest[:4], "big") & 0x7FFFFFFF
    return value or 1


def build_anki_model(pdf_hash: str) -> genanki.Model:
    """Build the deterministic MCQ note model."""
    model_id = _stable_id("pdf2anki-model-v1", pdf_hash)
    return genanki.Model(
        model_id,
        "PDF2Anki MCQ Model",
        fields=[
            {"name": "QuestionID"},
            {"name": "Topic"},
            {"name": "Question"},
            {"name": "OptionA"},
            {"name": "OptionB"},
            {"name": "OptionC"},
            {"name": "OptionD"},
            {"name": "Image"},
            {"name": "CorrectAnswer"},
            {"name": "Explanation"},
            {"name": "SourcePages"},
            {"name": "PDFTitle"},
        ],
        templates=[
            {
                "name": "MCQ Card",
                "qfmt": FRONT_TEMPLATE,
                "afmt": BACK_TEMPLATE,
            }
        ],
        css=CARD_CSS,
    )


def build_note_fields(
    mcq: MCQ,
    pdf_title: str,
    image_basename: str | None = None,
) -> list[str]:
    """Escape all untrusted content and return the 12 note fields in order."""
    if len(mcq.options) != 4 or not (0 <= mcq.correct_index <= 3):
        raise ValueError(f"Invalid MCQ options or correct_index for {mcq.id}")

    escaped_options = [html.escape(opt.strip()) for opt in mcq.options]
    correct_label = OPTION_LABELS[mcq.correct_index]
    correct_text = escaped_options[mcq.correct_index]
    correct_answer = f"{correct_label}. {correct_text}"

    if image_basename:
        alt = html.escape(mcq.image_alt_text or "Question figure", quote=True)
        safe_src = html.escape(image_basename, quote=True)
        image_field = f'<img src="{safe_src}" alt="{alt}">'
    else:
        image_field = ""

    pages_str = ", ".join(str(p) for p in sorted(set(mcq.source_pages)))

    return [
        html.escape(mcq.id),
        html.escape(mcq.topic),
        html.escape(mcq.stem),
        escaped_options[0],
        escaped_options[1],
        escaped_options[2],
        escaped_options[3],
        image_field,
        correct_answer,
        html.escape(mcq.explanation),
        html.escape(pages_str),
        html.escape(pdf_title),
    ]


def export_apkg_bytes(
    mcqs: list[MCQ],
    pdf_bytes: bytes,
    pdf_hash: str,
    pdf_title: str,
    dpi: int = DEFAULT_DPI,
) -> bytes:
    """Package validated MCQs and their rendered images into an .apkg byte buffer."""
    model = build_anki_model(pdf_hash)
    deck_id = _stable_id("pdf2anki-deck-v1", pdf_hash)
    deck_title = pdf_title.strip() or "PDF2Anki MCQ Deck"
    deck = genanki.Deck(deck_id, deck_title)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        media_paths: list[str] = []

        for mcq in mcqs:
            image_basename: str | None = None
            if mcq.image_required and mcq.image_page is not None:
                png_bytes = render_crop(
                    pdf_bytes,
                    mcq.image_page,
                    mcq.image_crop,
                    dpi=dpi,
                )
                image_basename = media_filename(
                    pdf_hash,
                    mcq.image_page,
                    mcq.id,
                    mcq.image_crop,
                )
                img_file = tmp_path / image_basename
                img_file.write_bytes(png_bytes)
                media_paths.append(str(img_file))

            fields = build_note_fields(mcq, deck_title, image_basename=image_basename)
            note_guid = genanki.guid_for(pdf_hash, mcq.id)
            note = genanki.Note(
                model=model,
                fields=fields,
                guid=note_guid,
            )
            deck.add_note(note)

        package = genanki.Package(deck)
        package.media_files = media_paths
        out_file = tmp_path / "deck.apkg"
        package.write_to_file(str(out_file))
        return out_file.read_bytes()
