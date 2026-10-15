"""Tests for HTML escaping, deterministic IDs, and .apkg media packaging."""

from __future__ import annotations

import io
import json
import zipfile

import fitz

from pdf2anki.anki_export import build_note_fields, export_apkg_bytes
from pdf2anki.pdf_media import compute_pdf_hash, media_filename
from pdf2anki.schemas import MCQ, ImageCrop


def _sample_pdf() -> bytes:
    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    page.insert_text((20, 40), "Sample figure")
    page.draw_circle((100, 100), 40, color=(0, 0.5, 0))
    return doc.tobytes()


def test_build_note_fields_escapes_malicious_html() -> None:
    mcq = MCQ(
        id="q<1>",
        learning_objective="obj",
        topic="Pathology & <script>",
        stem="What is <script>alert(1)</script>?",
        options=["<b>A</b>", "<img src=x onerror=1>", "C", "D"],
        correct_index=0,
        explanation="Because <script>bad()</script>.",
        source_pages=[1],
        image_required=True,
        image_page=1,
        image_alt_text='Alt "quote" <tag>',
    )
    fields = build_note_fields(
        mcq,
        pdf_title="My <Title>",
        image_basename="fig_1.png",
    )
    joined = "\n".join(fields)
    assert "<script>" not in joined
    assert "<img src=x onerror=1>" not in joined
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in fields[2]
    assert '<img src="fig_1.png" alt="Alt &quot;quote&quot; &lt;tag&gt;">' == fields[7]


def test_export_apkg_embeds_image_media() -> None:
    pdf_bytes = _sample_pdf()
    pdf_hash = compute_pdf_hash(pdf_bytes)
    crop = ImageCrop(x0=0.1, y0=0.1, x1=0.9, y1=0.9)
    mcq = MCQ(
        id="img-q1",
        learning_objective="Identify figure",
        topic="Histology",
        stem="Which structure is shown?",
        options=["Alpha", "Beta", "Gamma", "Delta"],
        correct_index=1,
        explanation="Beta is the green circle.",
        source_pages=[1],
        image_required=True,
        image_page=1,
        image_crop=crop,
        image_alt_text="Green circle",
    )

    apkg_bytes = export_apkg_bytes(
        mcqs=[mcq],
        pdf_bytes=pdf_bytes,
        pdf_hash=pdf_hash,
        pdf_title="Histology Atlas",
    )

    expected_basename = media_filename(pdf_hash, 1, mcq.id, crop)

    with zipfile.ZipFile(io.BytesIO(apkg_bytes), "r") as zf:
        names = set(zf.namelist())
        assert "media" in names
        media_map = json.loads(zf.read("media").decode("utf-8"))
        assert expected_basename in media_map.values()
        for archive_key, mapped_name in media_map.items():
            if mapped_name == expected_basename:
                assert archive_key in names
                assert zf.read(archive_key).startswith(b"\x89PNG")
