"""Tests for HTML escaping, deterministic IDs, and .apkg media packaging."""

from __future__ import annotations

import io
import json
import re
import sqlite3
import tempfile
import zipfile
from pathlib import Path

from conftest import make_pdf_with_image, make_pdf_with_two_images, make_sample_pdf
from pdf2anki.anki_export import build_anki_model, build_note_fields, export_apkg_bytes
from pdf2anki.pdf_media import compute_pdf_hash, media_filename
from pdf2anki.schemas import MCQ, ImageCrop


def _sample_pdf() -> bytes:
    return make_sample_pdf(pages=1)


def test_export_uses_shuffled_order_consistently() -> None:
    from pdf2anki.workflow import shuffle_mcq_options

    original = MCQ(
        id="shuf-q1",
        learning_objective="obj",
        topic="Topic",
        stem="Which is correct?",
        options=["CorrectAns", "W1", "W2", "W3"],
        correct_index=0,
        explanation="CorrectAns is right.",
        source_pages=[1],
    )
    shuffled = shuffle_mcq_options(original, seed_key="seed:shuf-q1")
    fields = build_note_fields(shuffled, pdf_title="Doc")

    # Field layout: 3..6 are OptionA..OptionD, field 8 is CorrectAnswer.
    option_fields = fields[3:7]
    correct_label = fields[8].split(".")[0]
    label_to_index = {"A": 0, "B": 1, "C": 2, "D": 3}
    idx = label_to_index[correct_label]
    assert option_fields[idx] == "CorrectAns"
    assert fields[8] == f"{correct_label}. CorrectAns"


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


def test_export_text_based_question_never_attaches_image() -> None:
    # A text-based question (image_required=False) must stay text-only even when
    # its source page contains a detectable figure.
    pdf_bytes = make_pdf_with_image(image_page_1based=1, pages=1)
    pdf_hash = compute_pdf_hash(pdf_bytes)
    mcq = MCQ(
        id="text-q1",
        learning_objective="Recall figure",
        topic="Histology",
        stem="Which structure is described?",
        options=["Alpha", "Beta", "Gamma", "Delta"],
        correct_index=1,
        explanation="Beta is described in the text.",
        source_pages=[1],
        source_support="Page 1 content",
        image_required=False,
        image_page=None,
        image_crop=None,
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


def test_export_figure_grounded_question_renders_image() -> None:
    # A figure-grounded question (image_required=True with a crop) is rendered.
    pdf_bytes = make_pdf_with_image(image_page_1based=1, pages=1)
    pdf_hash = compute_pdf_hash(pdf_bytes)
    crop = ImageCrop(x0=0.1, y0=0.3, x1=0.9, y1=0.9)
    mcq = MCQ(
        id="fig-q1",
        learning_objective="Identify figure",
        topic="Histology",
        stem="Which structure is shown in the figure?",
        options=["Alpha", "Beta", "Gamma", "Delta"],
        correct_index=1,
        explanation="Beta is the green square.",
        source_pages=[1],
        image_required=True,
        image_page=1,
        image_crop=crop,
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


def test_export_does_not_auto_attach_distant_unrelated_image() -> None:
    import pymupdf as fitz

    from conftest import _png_bytes

    doc = fitz.open()
    page = doc.new_page(width=400, height=1200)
    page.insert_text((20, 30), "Top text with no associated image")
    png_a = _png_bytes(100, 100, (20, 40, 160))
    png_b = _png_bytes(100, 100, (160, 40, 20))
    page.insert_image(fitz.Rect(40, 560, 200, 700), stream=png_a)
    page.insert_image(fitz.Rect(200, 1000, 360, 1140), stream=png_b)
    pdf_bytes = doc.tobytes()
    pdf_hash = compute_pdf_hash(pdf_bytes)

    mcq = MCQ(
        id="unrelated-img-q1",
        learning_objective="Recall fact",
        topic="Histology",
        stem="What does the top text state?",
        options=["Alpha", "Beta", "Gamma", "Delta"],
        correct_index=0,
        explanation="Alpha is stated at the top.",
        source_pages=[1],
        source_support="Top text with no associated image",
        image_required=False,
        image_page=None,
        image_crop=None,
    )

    apkg_bytes = export_apkg_bytes(
        mcqs=[mcq],
        pdf_bytes=pdf_bytes,
        pdf_hash=pdf_hash,
        pdf_title="Histology Atlas",
    )

    with zipfile.ZipFile(io.BytesIO(apkg_bytes), "r") as zf:
        media_map = json.loads(zf.read("media").decode("utf-8"))
        assert media_map == {}


def test_export_text_based_multi_image_page_stays_text_only() -> None:
    # Even on a page with multiple images, a text-based question attaches none.
    pdf_bytes = make_pdf_with_two_images()
    pdf_hash = compute_pdf_hash(pdf_bytes)
    mcq = MCQ(
        id="multi-text-q1",
        learning_objective="Recall glomerulus",
        topic="Renal",
        stem="Which structure is described near the top text?",
        options=["Alpha", "Beta", "Gamma", "Delta"],
        correct_index=0,
        explanation="Alpha is described.",
        source_pages=[1],
        source_support="Top anchor text about glomerulus",
        image_required=False,
        image_page=None,
        image_crop=None,
    )

    apkg_bytes = export_apkg_bytes(
        mcqs=[mcq],
        pdf_bytes=pdf_bytes,
        pdf_hash=pdf_hash,
        pdf_title="Renal Atlas",
    )

    with zipfile.ZipFile(io.BytesIO(apkg_bytes), "r") as zf:
        media_map = json.loads(zf.read("media").decode("utf-8"))
        assert media_map == {}


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
