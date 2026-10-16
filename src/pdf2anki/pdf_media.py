"""Local PDF validation, hashing, and page/crop rendering using PyMuPDF."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

import pymupdf as fitz

from .config import ACCEPTED_MIME_TYPE, MAX_PDF_BYTES, MAX_PDF_PAGES
from .schemas import ImageCrop

DEFAULT_DPI: int = 180
MIN_IMAGE_DIMENSION: int = 64
# Documents at or below this page count use the whole-document flow (no splitting).
SPLIT_PAGE_THRESHOLD: int = 25
# Fixed chunk size (pages) used when a large PDF has no usable TOC/outline.
CHUNK_SIZE: int = 25
DEFAULT_MAX_DISTANCE_FRAC: float = 0.35
# Minimum fraction of page area an image must cover to be considered a figure
# rather than a decorative thumbnail/icon.
MIN_IMAGE_AREA_FRAC: float = 0.02
# Distance reduction (in normalized-diagonal units) applied to a candidate that
# has caption-like text adjacent to it. Acts as a relevance boost.
CAPTION_DISTANCE_BONUS: float = 0.15
# An uncaptioned candidate must be quite close to the anchor to be accepted.
MAX_UNCAPTIONED_DISTANCE_FRAC: float = 0.18
CAPTION_PATTERN = r"(?i)\b(fig(?:ure)?\.?|plate|table|scheme)\b"
# Minimum Jaccard token overlap for a caption-similarity (Tier 2) match.
CAPTION_SIM_THRESHOLD: float = 0.25
# When several captions compete, a candidate whose shared salient fraction is
# below this floor is treated as dominated by unrelated entities and rejected.
CAPTION_CONFLICT_FLOOR: float = 0.2
# Minimum salient-term length (alphabetic) for lexical plausibility gating.
MIN_SALIENT_LEN: int = 4
# Common/caption-structural words that carry no topical signal and must not
# count as a lexical match between a question and a caption.
STOPWORDS: frozenset[str] = frozenset(
    {
        "figure", "figures", "fig", "figs", "plate", "table", "scheme",
        "photomicrograph", "photomicrographs", "micrograph", "image", "images",
        "picture", "panel", "inset", "arrow", "arrows", "scale", "bar",
        "shown", "show", "shows", "showing", "seen", "view", "views",
        "overview", "detail", "details", "section", "sections", "cross",
        "high", "low", "power", "magnification", "field", "level", "region",
        "left", "right", "upper", "lower", "middle", "center", "centre",
        "this", "that", "these", "those", "which", "with", "without", "from",
        "about", "near", "into", "onto", "over", "under", "between", "within",
        "there", "here", "where", "when", "what", "whose", "their", "them",
        "they", "the", "and", "for", "are", "was", "were", "has", "have",
        "had", "its", "also", "such", "than", "then", "both", "each", "some",
        "question", "questions", "answer", "stem", "above", "below", "following",
    }
)


def _salient_terms(text: str) -> set[str]:
    """Return topical content terms: alphabetic, length >= MIN_SALIENT_LEN,
    lowercased, with stopwords removed."""
    terms: set[str] = set()
    for tok in re.split(r"[^a-zA-Z]+", text or ""):
        low = tok.lower()
        if len(low) >= MIN_SALIENT_LEN and low not in STOPWORDS:
            terms.add(low)
    return terms


def _passes_lexical_gate(query_text: str, caption_text: str) -> bool:
    """Hard precision gate before semantic matching.

    Requires at least one strong salient-term overlap between the question/
    support text and the caption. When both sides have salient terms but share
    none, the candidate is rejected (conflict), preventing unrelated medical
    entities (e.g. 'lichen' vs 'syphilis') from matching.
    """
    q_terms = _salient_terms(query_text)
    c_terms = _salient_terms(caption_text)
    if not q_terms or not c_terms:
        return False
    return bool(q_terms & c_terms)


def _salient_overlap_score(query_text: str, caption_text: str) -> tuple[int, float]:
    """Return (shared_salient_count, shared_fraction_of_caption).

    ``shared_fraction_of_caption`` is the proportion of the caption's salient
    terms that are shared with the query, used to detect captions dominated by
    unrelated salient entities (contradiction penalty).
    """
    q_terms = _salient_terms(query_text)
    c_terms = _salient_terms(caption_text)
    if not q_terms or not c_terms:
        return (0, 0.0)
    shared = q_terms & c_terms
    fraction = len(shared) / len(c_terms) if c_terms else 0.0
    return (len(shared), fraction)
# Matches figure labels like "Figure 3", "Fig. 2.1", "FIG 4A", "Figs. 1-2".
FIGURE_LABEL_PATTERN = re.compile(
    r"(?i)\bfig(?:ure|s|\.)?\.?\s*([0-9]+(?:\.[0-9]+)*[a-z]?)"
)


class PDFValidationError(ValueError):
    """Raised when an uploaded PDF fails size, encryption, or page checks."""


@dataclass(frozen=True)
class PDFInfo:
    """Validated metadata about a local PDF."""

    pdf_hash: str
    page_count: int
    size_bytes: int


@dataclass(frozen=True)
class FigureMeta:
    """Metadata describing a figure: its label, caption, page, and bbox."""

    figure_label: str  # normalized, e.g. "figure 3" ("" if unlabeled)
    caption_text: str
    page_number: int  # 1-based
    image_bbox: fitz.Rect


@dataclass(frozen=True)
class Chapter:
    """A top-level outline chapter with an inclusive 1-based page range."""

    title: str
    start_page: int
    end_page: int

    @property
    def pages(self) -> set[int]:
        """Return the inclusive set of 1-based page numbers in this chapter."""
        return set(range(self.start_page, self.end_page + 1))


def extract_chapters(pdf_bytes: bytes) -> list[Chapter]:
    """Derive top-level (level-1) chapters from the PDF outline/TOC.

    Returns an empty list when the document has no usable outline, so callers
    can fall back to whole-document behavior. Chapter page ranges are inclusive
    and 1-based; each chapter runs until the page before the next level-1 entry
    (or the final page for the last chapter).
    """
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page_count = doc.page_count
        try:
            toc = doc.get_toc(simple=True)
        except Exception:
            return []

    starts: list[tuple[str, int]] = []
    for entry in toc:
        if len(entry) < 3:
            continue
        level, title, page = entry[0], entry[1], entry[2]
        if level != 1:
            continue
        if not isinstance(page, int) or page < 1 or page > page_count:
            continue
        starts.append((str(title).strip() or f"Chapter {len(starts) + 1}", page))

    if not starts:
        return []

    # Ensure start pages are non-decreasing in document order.
    chapters: list[Chapter] = []
    for i, (title, start) in enumerate(starts):
        if i + 1 < len(starts):
            next_start = starts[i + 1][1]
            end = max(start, next_start - 1)
        else:
            end = page_count
        chapters.append(Chapter(title=title, start_page=start, end_page=end))
    return chapters


def chunk_chapters(page_count: int, chunk_size: int = CHUNK_SIZE) -> list[Chapter]:
    """Split a document into fixed-size page chunks as pseudo-chapters.

    Each chunk spans ``chunk_size`` pages (the last one may be shorter) and is
    labeled "Pages START-END" with inclusive 1-based page ranges.
    """
    if page_count < 1 or chunk_size < 1:
        return []
    chapters: list[Chapter] = []
    start = 1
    while start <= page_count:
        end = min(start + chunk_size - 1, page_count)
        chapters.append(
            Chapter(title=f"Pages {start}-{end}", start_page=start, end_page=end)
        )
        start = end + 1
    return chapters


def plan_chapters(
    pdf_bytes: bytes,
    split_threshold: int = SPLIT_PAGE_THRESHOLD,
    chunk_size: int = CHUNK_SIZE,
) -> list[Chapter]:
    """Decide how to split a PDF for chapter/chunk selection.

    - ``page_count`` at or below ``split_threshold``: return ``[]`` (whole-document flow).
    - Above the threshold with a usable TOC/outline: return TOC-derived chapters.
    - Above the threshold without a TOC: return fixed ``chunk_size``-page chunks.
    """
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        page_count = doc.page_count

    if page_count <= split_threshold:
        return []

    toc_chapters = extract_chapters(pdf_bytes)
    if toc_chapters:
        return toc_chapters
    return chunk_chapters(page_count, chunk_size=chunk_size)


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
    min_area_frac: float = MIN_IMAGE_AREA_FRAC,
) -> list[fitz.Rect]:
    """Return clipped page rectangles for non-trivial, non-decorative images.

    Images below ``min_dim`` in either source dimension or covering less than
    ``min_area_frac`` of the page area are treated as decorative and excluded.
    """
    prect = page.rect
    page_area = float(prect.width * prect.height)
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
            if r.is_empty or r.width <= 0 or r.height <= 0:
                continue
            if page_area > 0 and (r.width * r.height) / page_area < min_area_frac:
                continue
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


def _caption_rects(page: fitz.Page) -> list[fitz.Rect]:
    """Return bounding rects of lines whose text looks like a figure caption."""
    rects: list[fitz.Rect] = []
    try:
        data = page.get_text("dict")
    except Exception:
        return rects
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            text = "".join(
                span.get("text", "") for span in line.get("spans", [])
            )
            if re.search(CAPTION_PATTERN, text):
                bbox = line.get("bbox")
                if bbox:
                    rects.append(fitz.Rect(bbox))
    return rects


def _has_adjacent_caption(
    image_rect: fitz.Rect,
    caption_rects: list[fitz.Rect],
    page_rect: fitz.Rect,
) -> bool:
    """True if a caption line sits just above or below the image, horizontally overlapping."""
    if not caption_rects:
        return False
    gap = 0.06 * page_rect.height  # captions typically hug the figure
    for cap in caption_rects:
        horiz_overlap = min(image_rect.x1, cap.x1) - max(image_rect.x0, cap.x0)
        if horiz_overlap <= 0:
            continue
        below = 0 <= (cap.y0 - image_rect.y1) <= gap
        above = 0 <= (image_rect.y0 - cap.y1) <= gap
        if below or above:
            return True
    return False


def _normalize_figure_refs(text: str) -> set[str]:
    """Extract normalized figure labels from free text.

    Returns a set like {"figure 3", "figure 2.1"}. Handles Figure/Fig./FIG./Figs.
    and numbering such as 3, 3A, 2.1, 4.2.3 (case/punctuation-insensitive).
    """
    refs: set[str] = set()
    for match in FIGURE_LABEL_PATTERN.finditer(text or ""):
        number = match.group(1).lower().strip()
        if number:
            refs.add(f"figure {number}")
    return refs


def _tokenize(text: str) -> set[str]:
    """Lowercase alphanumeric token set for similarity comparison."""
    return {
        tok
        for tok in re.split(r"[^a-z0-9]+", (text or "").lower())
        if len(tok) >= 3
    }


def _caption_similarity(a: str, b: str) -> float:
    """Return Jaccard token-overlap similarity between two strings (0..1)."""
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def _nearest_caption_for_image(
    image_rect: fitz.Rect,
    caption_rects: list[fitz.Rect],
    caption_texts: list[str],
    page_rect: fitz.Rect,
) -> str:
    """Return the text of a caption line adjacent to the image, or ""."""
    gap = 0.06 * page_rect.height
    for cap_rect, cap_text in zip(caption_rects, caption_texts, strict=True):
        horiz = min(image_rect.x1, cap_rect.x1) - max(image_rect.x0, cap_rect.x0)
        if horiz <= 0:
            continue
        below = 0 <= (cap_rect.y0 - image_rect.y1) <= gap
        above = 0 <= (image_rect.y0 - cap_rect.y1) <= gap
        if below or above:
            return cap_text
    return ""


def _caption_lines(page: fitz.Page) -> tuple[list[fitz.Rect], list[str]]:
    """Return (rects, texts) for caption-like lines on the page."""
    rects: list[fitz.Rect] = []
    texts: list[str] = []
    try:
        data = page.get_text("dict")
    except Exception:
        return rects, texts
    for block in data.get("blocks", []):
        for line in block.get("lines", []):
            text = "".join(
                span.get("text", "") for span in line.get("spans", [])
            )
            if re.search(CAPTION_PATTERN, text):
                bbox = line.get("bbox")
                if bbox:
                    rects.append(fitz.Rect(bbox))
                    texts.append(text.strip())
    return rects, texts


def extract_figure_metadata(
    pdf_bytes: bytes,
    page_number_1based: int,
    min_dim: int = MIN_IMAGE_DIMENSION,
) -> list[FigureMeta]:
    """Extract figure metadata (label, caption, bbox) for images on a page."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if page_number_1based < 1 or page_number_1based > doc.page_count:
            raise ValueError(
                f"Page {page_number_1based} out of range (1-{doc.page_count})."
            )
        page = doc[page_number_1based - 1]
        prect = page.rect
        cap_rects, cap_texts = _caption_lines(page)
        metas: list[FigureMeta] = []
        for r in _candidate_image_rects(doc, page, min_dim=min_dim):
            caption = _nearest_caption_for_image(r, cap_rects, cap_texts, prect)
            labels = _normalize_figure_refs(caption)
            label = sorted(labels)[0] if labels else ""
            metas.append(
                FigureMeta(
                    figure_label=label,
                    caption_text=caption,
                    page_number=page_number_1based,
                    image_bbox=r,
                )
            )
        return metas


def find_image_crop_by_reference(
    pdf_bytes: bytes,
    page_number_1based: int,
    query_text: str,
    margin: float = 0.03,
    min_dim: int = MIN_IMAGE_DIMENSION,
) -> ImageCrop | None:
    """Return the crop for the figure whose label is referenced in query_text."""
    refs = _normalize_figure_refs(query_text)
    if not refs:
        return None
    metas = extract_figure_metadata(pdf_bytes, page_number_1based, min_dim=min_dim)
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        prect = doc[page_number_1based - 1].rect
    for meta in metas:
        if meta.figure_label and meta.figure_label in refs:
            return _rect_to_crop(meta.image_bbox, prect, margin)
    return None


def find_image_crop_by_caption(
    pdf_bytes: bytes,
    page_number_1based: int,
    query_text: str,
    margin: float = 0.03,
    min_dim: int = MIN_IMAGE_DIMENSION,
    threshold: float = CAPTION_SIM_THRESHOLD,
) -> ImageCrop | None:
    """Return the crop whose caption best matches query_text above threshold."""
    query = (query_text or "").strip()
    if not query:
        return None
    metas = extract_figure_metadata(pdf_bytes, page_number_1based, min_dim=min_dim)

    # First pass: keep only captions that pass the hard lexical plausibility
    # gate (share at least one strong salient concept/entity with the query).
    eligible: list[tuple[FigureMeta, int, float, float]] = []
    for meta in metas:
        if not meta.caption_text:
            continue
        if not _passes_lexical_gate(query, meta.caption_text):
            continue
        shared_count, shared_fraction = _salient_overlap_score(
            query, meta.caption_text
        )
        sim = _caption_similarity(query, meta.caption_text)
        eligible.append((meta, shared_count, shared_fraction, sim))

    if not eligible:
        return None

    # Contradiction penalty: when multiple captions compete, drop candidates
    # dominated by unrelated salient entities (low shared fraction). A lone
    # plausible caption is kept regardless so short captions still match.
    if len(eligible) > 1:
        filtered = [
            e for e in eligible if e[2] >= CAPTION_CONFLICT_FLOOR
        ]
        if filtered:
            eligible = filtered

    # Semantic reranking only among lexically plausible survivors: prefer more
    # shared salient terms, then higher shared fraction, then Jaccard similarity.
    best_meta = max(
        eligible, key=lambda e: (e[1], e[2], e[3])
    )[0]
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        prect = doc[page_number_1based - 1].rect
    return _rect_to_crop(best_meta.image_bbox, prect, margin)


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

        caption_rects = _caption_rects(page)

        best_rect: fitz.Rect | None = None
        best_key: tuple[float, float] | None = None
        for r in candidates:
            norm_dist = _rect_distance(text_rect, r) / diag
            captioned = _has_adjacent_caption(r, caption_rects, prect)
            # Uncaptioned candidates must be clearly close to the anchor.
            dist_limit = (
                max_distance_frac
                if captioned
                else min(max_distance_frac, MAX_UNCAPTIONED_DISTANCE_FRAC)
            )
            if norm_dist > dist_limit:
                continue
            score = norm_dist - (CAPTION_DISTANCE_BONUS if captioned else 0.0)
            key = (score, -(r.width * r.height))
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
        # Also try progressively left-trimmed variants so leading filler words
        # (e.g. "see ...") do not prevent a literal page phrase from matching.
        words = frag.split(" ")
        for start in range(1, len(words)):
            suffix = " ".join(words[start:]).strip()
            if len(suffix) >= 12 and suffix not in seen:
                seen.add(suffix)
                queries.append(suffix)
    return queries


def select_image_crop(
    pdf_bytes: bytes,
    page_number_1based: int,
    source_text: str,
    *,
    extra_text: str = "",
    min_dim: int = MIN_IMAGE_DIMENSION,
    margin: float = 0.03,
    max_distance_frac: float = DEFAULT_MAX_DISTANCE_FRAC,
) -> ImageCrop | None:
    """Pick an image crop for a page using a ranked, grounded strategy.

    Ranking order (first match wins):
      1. Explicit figure-number reference (in ``source_text``/``extra_text``).
      2. Figure-caption similarity to the question/support text.
      3. Nearby paragraph/anchor-text proximity.
      4. Lone substantial captioned image on the page.

    Returns ``None`` when confidence is too low (preferring no image over a
    wrong image).
    """
    combined = f"{extra_text} {source_text}".strip()

    # Tier 1: explicit figure-number reference match (highest confidence).
    crop = find_image_crop_by_reference(
        pdf_bytes, page_number_1based, combined, margin=margin, min_dim=min_dim
    )
    if crop is not None:
        return crop

    # Tier 2: figure-caption similarity match.
    crop = find_image_crop_by_caption(
        pdf_bytes, page_number_1based, combined, margin=margin, min_dim=min_dim
    )
    if crop is not None:
        return crop

    # Tier 3: nearby anchor/proximity text match.
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

    # Tier 4: conservative page-level fallback for a lone captioned image.
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if page_number_1based < 1 or page_number_1based > doc.page_count:
            raise ValueError(
                f"Page {page_number_1based} out of range (1-{doc.page_count})."
            )
        page = doc[page_number_1based - 1]
        candidates = _candidate_image_rects(doc, page, min_dim=min_dim)
        if len(candidates) == 1:
            caption_rects = _caption_rects(page)
            if _has_adjacent_caption(candidates[0], caption_rects, page.rect):
                return _rect_to_crop(candidates[0], page.rect, margin)
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
