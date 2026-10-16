"""Tests for HTML escaping, deterministic IDs, and .apkg media packaging."""

from __future__ import annotations

import io
import json
import sqlite3
import tempfile
import zipfile
from pathlib import Path

import re

from conftest import make_sample_pdf
from pdf2anki.anki_export import build_anki_model, build_note_fields, export_apkg_bytes
from pdf2anki.pdf_media import compute_pdf_hash, media_filename
from pdf2anki.schemas import MCQ, ImageCrop


def _sample_pdf() -> bytes:
    return make_sample_pdf(pages=1)


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

        db_member = "collection.anki21" if "collection.anki21" in names else "collection.anki2"
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "collection.sqlite"
            db_path.write_bytes(zf.read(db_member))
            with sqlite3.connect(db_path) as conn:
                row = conn.execute("SELECT flds FROM notes LIMIT 1").fetchone()
                assert row is not None
                note_fields = row[0].split("\x1f")
                assert f'<img src="{expected_basename}"' in note_fields[7]
                assert note_fields[11] == "Histology Atlas"


def test_export_apkg_auto_attaches_from_source_pages_when_image_page_missing() -> None:
    pdf_bytes = _sample_pdf()
    pdf_hash = compute_pdf_hash(pdf_bytes)
    mcq = MCQ(
        id="img-fallback-q1",
        learning_objective="Identify figure",
        topic="Histology",
        stem="Which structure is shown?",
        options=["Alpha", "Beta", "Gamma", "Delta"],
        correct_index=1,
        explanation="Beta is the green circle.",
        source_pages=[1],
        image_required=True,
        image_page=None,
        image_crop=None,
    )

    apkg_bytes = export_apkg_bytes(
        mcqs=[mcq],
        pdf_bytes=pdf_bytes,
        pdf_hash=pdf_hash,
        pdf_title="Histology Atlas",
    )

    expected_basename = media_filename(pdf_hash, 1, mcq.id, None)

    with zipfile.ZipFile(io.BytesIO(apkg_bytes), "r") as zf:
        names = set(zf.namelist())
        media_map = json.loads(zf.read("media").decode("utf-8"))
        assert expected_basename in media_map.values()
        for archive_key, mapped_name in media_map.items():
            if mapped_name == expected_basename:
                assert zf.read(archive_key).startswith(b"\x89PNG")

        db_member = "collection.anki21" if "collection.anki21" in names else "collection.anki2"
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "collection.sqlite"
            db_path.write_bytes(zf.read(db_member))
            with sqlite3.connect(db_path) as conn:
                row = conn.execute("SELECT flds FROM notes LIMIT 1").fetchone()
                assert row is not None
                note_fields = row[0].split("\x1f")
                assert f'<img src="{expected_basename}"' in note_fields[7]


def test_export_apkg_omits_image_when_not_required() -> None:
    pdf_bytes = _sample_pdf()
    pdf_hash = compute_pdf_hash(pdf_bytes)
    mcq = MCQ(
        id="no-img-q1",
        learning_objective="Recall fact",
        topic="Histology",
        stem="Which statement is true?",
        options=["Alpha", "Beta", "Gamma", "Delta"],
        correct_index=0,
        explanation="Alpha is true.",
        source_pages=[1],
        image_required=False,
        image_page=None,
    )

    apkg_bytes = export_apkg_bytes(
        mcqs=[mcq],
        pdf_bytes=pdf_bytes,
        pdf_hash=pdf_hash,
        pdf_title="Histology Atlas",
    )

    with zipfile.ZipFile(io.BytesIO(apkg_bytes), "r") as zf:
        names = set(zf.namelist())
        media_map = json.loads(zf.read("media").decode("utf-8"))
        assert media_map == {}

        db_member = "collection.anki21" if "collection.anki21" in names else "collection.anki2"
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "collection.sqlite"
            db_path.write_bytes(zf.read(db_member))
            with sqlite3.connect(db_path) as conn:
                row = conn.execute("SELECT flds FROM notes LIMIT 1").fetchone()
                assert row is not None
                note_fields = row[0].split("\x1f")
                assert note_fields[7] == ""


def test_answer_box_css_defines_explicit_text_color() -> None:
    model = build_anki_model("deadbeef")
    match = re.search(r"\.answer-box\s*\{([^}]*)\}", model.css)
    assert match is not None
    block = match.group(1)
    assert "background-color" in block
    assert re.search(r"(?<![-\w])color\s*:", block) is not None
