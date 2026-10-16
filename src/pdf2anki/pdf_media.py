"""Local PDF validation, hashing, and page/crop rendering using PyMuPDF."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pymupdf as fitz

from .config import ACCEPTED_MIME_TYPE, MAX_PDF_BYTES, MAX_PDF_PAGES
from .schemas import ImageCrop

DEFAULT_DPI: int = 180
MIN_IMAGE_DIMENSION: int = 64
DEFAULT_MAX_DISTANCE_FRAC: float = 0.35


class PDFValidationError(ValueError):
    """Raised when an uploaded PDF fails size, encryption, or page checks."""


@dataclass(frozen=True)
class PDFInfo:
    """Validated metadata about a local PDF."""

    pdf_hash: str
    page_count: int
    size_bytes: int


def compute_pdf_hash(data: bytes) -> str:
    """Return a short deterministic hex digest for the PDF bytes."""
    return hashlib.sha256(data).hexdigest()[:16]


def validate_pdf(
    data: bytes,
    mime_type: str = ACCEPTED_MIME_TYPE,
    max_bytes: int = MAX_PDF_BYTES,
    max_pages: int = MAX_PDF_PAGES,
) -> PDFInfo:
    """Validate MIME type, size, encryption status, and page count."""
    if mime_type and mime_type not in (
        ACCEPTED_MIME_TYPE,
        "application/x-pdf",
        "application/octet-stream",
    ):
        raise PDFValidationError("The uploaded file is not a PDF.")
    if not data:
        raise PDFValidationError("The uploaded PDF is empty.")
    if len(data) > max_bytes:
        raise PDFValidationError(
            f"The PDF exceeds the {max_bytes // (1024 * 1024)} MB limit."
        )

    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise PDFValidationError(
            "The PDF could not be read. It may be corrupt or invalid."
        ) from exc

    with doc:
        if doc.needs_pass or doc.is_encrypted:
            raise PDFValidationError(
                "Encrypted or password-protected PDFs are not supported."
            )
        page_count = doc.page_count

    if page_count < 1:
        raise PDFValidationError("The PDF contains no pages.")
    if page_count > max_pages:
        raise PDFValidationError(
            f"The PDF has {page_count} pages, exceeding the {max_pages}-page limit."
        )

    return PDFInfo(
        pdf_hash=compute_pdf_hash(data),
        page_count=page_count,
        size_bytes=len(data),
    )


def media_filename(
    pdf_hash: str,
    page_number_1based: int,
    question_id: str,
    crop: ImageCrop | None = None,
) -> str:
    """Build a collision-proof PNG basename for an Anki media attachment."""
    safe_qid = "".join(c if c.isalnum() else "_" for c in question_id)[:32] or "q"
    if crop is not None:
        crop_tag = (
            f"{int(round(crop.x0 * 1000))}_{int(round(crop.y0 * 1000))}_"
            f"{int(round(crop.x1 * 1000))}_{int(round(crop.y1 * 1000))}"
        )
    else:
        crop_tag = "full"
    digest = hashlib.sha1(
        f"{pdf_hash}:{page_number_1based}:{question_id}:{crop_tag}".encode()
    ).hexdigest()[:10]
    return f"pdf2anki_{pdf_hash[:8]}_p{page_number_1based}_{safe_qid}_{crop_tag}_{digest}.png"


def count_page_images(
    pdf_bytes: bytes,
    page_number_1based: int,
    min_dim: int = MIN_IMAGE_DIMENSION,
) -> int:
    """Count non-trivial embedded raster images on a 1-based page.

    Images smaller than ``min_dim`` in either dimension are treated as
    decorative and ignored. Per-image extraction errors are skipped.
    """
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if page_number_1based < 1 or page_number_1based > doc.page_count:
            raise ValueError(
                f"Page {page_number_1based} out of range (1-{doc.page_count})."
            )
        page = doc[page_number_1based - 1]
        count = 0
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                info = doc.extract_image(xref)
            except Exception:
                continue
            if info.get("width", 0) >= min_dim and info.get("height", 0) >= min_dim:
                count += 1
        return count


def _candidate_image_rects(
    doc: fitz.Document,
    page: fitz.Page,
    min_dim: int = MIN_IMAGE_DIMENSION,
) -> list[fitz.Rect]:
    """Return clipped page rectangles for all non-trivial images on `page`."""
    prect = page.rect
    candidates: list[fitz.Rect] = []
    for img in page.get_images(full=True):
        xref = img[0]
        try:
            info = doc.extract_image(xref)
        except Exception:
            continue
        if info.get("width", 0) < min_dim or info.get("height", 0) < min_dim:
            continue
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            rects = []
        for rect in rects:
            r = fitz.Rect(rect) & prect
            if not r.is_empty and r.width > 0 and r.height > 0:
                candidates.append(r)
    return candidates


def _rect_to_crop(
    rect: fitz.Rect, prect: fitz.Rect, margin: float
) -> ImageCrop | None:
    """Convert a page-space rect into a normalized, margin-expanded ImageCrop."""
    x0 = (rect.x0 - prect.x0) / prect.width
    y0 = (rect.y0 - prect.y0) / prect.height
    x1 = (rect.x1 - prect.x0) / prect.width
    y1 = (rect.y1 - prect.y0) / prect.height

    x0 = max(0.0, x0 - margin)
    y0 = max(0.0, y0 - margin)
    x1 = min(1.0, x1 + margin)
    y1 = min(1.0, y1 + margin)

    if x1 <= x0 or y1 <= y0:
        return None
    return ImageCrop(x0=x0, y0=y0, x1=x1, y1=y1)


def _rect_distance(a: fitz.Rect, b: fitz.Rect) -> float:
    """Return Euclidean gap between two axis-aligned rectangles (0 if overlapping)."""
    dx = max(0.0, max(a.x0 - b.x1, b.x0 - a.x1))
    dy = max(0.0, max(a.y0 - b.y1, b.y0 - a.y1))
    return float((dx * dx + dy * dy) ** 0.5)


def largest_image_crop(
    pdf_bytes: bytes,
    page_number_1based: int,
    min_dim: int = MIN_IMAGE_DIMENSION,
    margin: float = 0.03,
) -> ImageCrop | None:
    """Return a normalized crop for the largest non-trivial image on a page.

    Inspects embedded raster images and their placement rectangles, selects the
    one covering the largest page area (ignoring images smaller than ``min_dim``
    in either source dimension), expands it by ``margin`` on all sides, clamps
    to the unit square, and returns it as an :class:`ImageCrop`. Returns ``None``
    when no qualifying image region can be located.
    """
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if page_number_1based < 1 or page_number_1based > doc.page_count:
            raise ValueError(
                f"Page {page_number_1based} out of range (1-{doc.page_count})."
            )
        page = doc[page_number_1based - 1]
        prect = page.rect
        if prect.width <= 0 or prect.height <= 0:
            return None

        best_area = 0.0
        best_rect: fitz.Rect | None = None
        for r in _candidate_image_rects(doc, page, min_dim=min_dim):
            area = r.width * r.height
            if area > best_area:
                best_area = area
                best_rect = r

        if best_rect is None:
            return None

        return _rect_to_crop(best_rect, prect, margin)


def best_image_crop_for_text(
    pdf_bytes: bytes,
    page_number_1based: int,
    source_text: str,
    *,
    min_dim: int = MIN_IMAGE_DIMENSION,
    margin: float = 0.03,
    max_distance_frac: float = DEFAULT_MAX_DISTANCE_FRAC,
) -> ImageCrop | None:
    """Return a normalized crop for the image nearest to `source_text` on the page.

    Locates `source_text` on the 1-based page via PyMuPDF text search. If the
    text is empty or not found on the page, returns ``None`` (conservative: do
    not guess when unanchored). Otherwise, finds all non-trivial candidate
    images on the page, measures their rectangle-to-rectangle distance to the
    text's bounding box normalized by the page diagonal, and selects the nearest
    image within ``max_distance_frac`` (breaking ties by larger area).
    """
    query = (source_text or "").strip()
    if not query:
        return None

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if page_number_1based < 1 or page_number_1based > doc.page_count:
            raise ValueError(
                f"Page {page_number_1based} out of range (1-{doc.page_count})."
            )
        page = doc[page_number_1based - 1]
        prect = page.rect
        if prect.width <= 0 or prect.height <= 0:
            return None

        candidates = _candidate_image_rects(doc, page, min_dim=min_dim)
        if not candidates:
            return None

        hits = page.search_for(query)
        if not hits:
            return None

        text_rect = fitz.Rect(hits[0])
        for h in hits[1:]:
            text_rect |= fitz.Rect(h)
        text_rect &= prect
        if text_rect.is_empty:
            return None

        diag = float((prect.width ** 2 + prect.height ** 2) ** 0.5)
        if diag <= 0:
            return None

        best_rect: fitz.Rect | None = None
        best_key: tuple[float, float] | None = None
        for r in candidates:
            norm_dist = _rect_distance(text_rect, r) / diag
            if norm_dist > max_distance_frac:
                continue
            key = (norm_dist, -(r.width * r.height))
            if best_key is None or key < best_key:
                best_key = key
                best_rect = r

        if best_rect is None:
            return None

        return _rect_to_crop(best_rect, prect, margin)


def _anchor_queries(source_text: str) -> list[str]:
    """Derive progressively shorter literal anchors from source text.

    Returns the full string first, then sentence/phrase fragments ordered by
    descending length, so the most specific literal match is tried first.
    """
    text = " ".join((source_text or "").split())
    if not text:
        return []
    queries: list[str] = [text]
    seen: set[str] = {text}
    separators = [". ", "; ", ", ", ": "]
    fragments: list[str] = [text]
    for sep in separators:
        next_fragments: list[str] = []
        for frag in fragments:
            next_fragments.extend(part for part in frag.split(sep))
        fragments = next_fragments
    for frag in sorted(
        (f.strip() for f in fragments), key=len, reverse=True
    ):
        # Require a reasonably specific fragment to avoid trivial matches.
        if len(frag) >= 12 and frag not in seen:
            seen.add(frag)
            queries.append(frag)
    return queries


def select_image_crop(
    pdf_bytes: bytes,
    page_number_1based: int,
    source_text: str,
    *,
    min_dim: int = MIN_IMAGE_DIMENSION,
    margin: float = 0.03,
    max_distance_frac: float = DEFAULT_MAX_DISTANCE_FRAC,
) -> ImageCrop | None:
    """Pick an image crop for a page using a tiered fallback strategy.

    Tier 1/2: anchor on the source text (full string, then longest literal
    fragments) and choose the nearest qualifying image within distance.
    Tier 3: if no anchor matches, fall back to the single substantial image on
    the page when (and only when) exactly one exists. Returns ``None`` when the
    page has no usable image, or has several images and no reliable anchor
    (preferring no image over an unrelated one).
    """
    for query in _anchor_queries(source_text):
        crop = best_image_crop_for_text(
            pdf_bytes,
            page_number_1based,
            query,
            min_dim=min_dim,
            margin=margin,
            max_distance_frac=max_distance_frac,
        )
        if crop is not None:
            return crop

    # Tier 3: conservative page-level fallback for a lone substantial image.
    if count_page_images(pdf_bytes, page_number_1based, min_dim=min_dim) == 1:
        return largest_image_crop(
            pdf_bytes, page_number_1based, min_dim=min_dim, margin=margin
        )
    return None


def render_page(
    pdf_bytes: bytes,
    page_number_1based: int,
    dpi: int = DEFAULT_DPI,
) -> bytes:
    """Render a full 1-based PDF page to PNG bytes."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if page_number_1based < 1 or page_number_1based > doc.page_count:
            raise ValueError(
                f"Page {page_number_1based} out of range (1-{doc.page_count})."
            )
        page = doc[page_number_1based - 1]
        scale = dpi / 72.0
        pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
        return pix.tobytes("png")


def render_crop(
    pdf_bytes: bytes,
    page_number_1based: int,
    crop: ImageCrop | None = None,
    dpi: int = DEFAULT_DPI,
) -> bytes:
    """Render a normalized crop of a 1-based page, falling back to full page."""
    if crop is None:
        return render_page(pdf_bytes, page_number_1based, dpi=dpi)

    if (
        crop.x0 < 0.0
        or crop.y0 < 0.0
        or crop.x1 > 1.0
        or crop.y1 > 1.0
        or crop.x1 <= crop.x0
        or crop.y1 <= crop.y0
    ):
        return render_page(pdf_bytes, page_number_1based, dpi=dpi)

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if page_number_1based < 1 or page_number_1based > doc.page_count:
            raise ValueError(
                f"Page {page_number_1based} out of range (1-{doc.page_count})."
            )
        page = doc[page_number_1based - 1]
        rect = page.rect
        clip = fitz.Rect(
            rect.x0 + crop.x0 * rect.width,
            rect.y0 + crop.y0 * rect.height,
            rect.x0 + crop.x1 * rect.width,
            rect.y0 + crop.y1 * rect.height,
        )
        if clip.is_empty or clip.width < 2 or clip.height < 2:
            return render_page(pdf_bytes, page_number_1based, dpi=dpi)
        scale = dpi / 72.0
        try:
            pix = page.get_pixmap(
                matrix=fitz.Matrix(scale, scale), clip=clip, alpha=False
            )
            if pix.width < 1 or pix.height < 1:
                return render_page(pdf_bytes, page_number_1based, dpi=dpi)
            return pix.tobytes("png")
        except Exception:
            return render_page(pdf_bytes, page_number_1based, dpi=dpi)
