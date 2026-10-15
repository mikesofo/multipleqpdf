# Aider Build Plan: PDF-to-Anki MCQ Generator

## Direct recommendation

Build a Streamlit application that uploads the complete PDF to Anthropic, asks Claude first for a coverage plan and then for schema-constrained MCQs, renders any required PDF pages locally with PyMuPDF, and packages the cards and media into an `.apkg` file with `genanki`. The four Aider roles are development-time roles: Anthropic plans the application, Gemini implements it, DeepSeek handles mechanical work, and OpenAI reviews and repairs it; only Claude is called by the finished application at runtime.

This design meets the requested behavior without attempting to make Aider itself part of the deployed application. Claude's Files API supports uploading a PDF once and referencing it by `file_id`; Claude can process the PDF's text, images, charts, tables, and visual layout.[^1][^2]

## Functional specification

The application should provide the following workflow:

1. The user uploads one unencrypted PDF through Streamlit.
2. The app validates the MIME type, size, page count, and readability.
3. The app uploads the complete PDF to Claude through the Anthropic Files API.
4. Claude creates a coverage plan, identifies the major learning objectives, and chooses the number of questions required to cover them adequately.
5. Claude produces MCQs with exactly four options, exactly one correct option, a brief rationale, source-page metadata, and optional image instructions.
6. The app validates every generated question before presenting an editable preview.
7. For image-based questions, the app renders the relevant source page or crop locally from the original PDF.
8. The user can edit, delete, or regenerate individual questions.
9. The app generates a ready-to-import `.apkg` deck with all referenced images embedded.
10. The app deletes the remote Anthropic file in a `finally` block after generation unless the user explicitly opts to retain it temporarily.

Streamlit supplies native upload and binary-download widgets. Its uploader defaults to 200 MB per file, but the application should impose Anthropic's lower effective PDF limit before making an API call. Claude currently permits standard, unencrypted PDFs up to 32 MB per request and up to 600 pages on a 1-million-token context model, although dense PDFs may exhaust context earlier.[^2][^3][^4]

## Question-generation pipeline

A two-pass Claude pipeline is preferable to a single prompt. Prompt chaining gives each subtask more focused attention and improves consistency for complex workflows.[^5]

### Pass 1: coverage planning

Claude should return:

- Document title and subject
- Intended learner level inferred from the PDF
- Major topics and learning objectives
- Estimated complexity and visual dependence
- A justified `question_count`
- Allocation of questions by topic
- Pages containing useful diagrams, tables, photomicrographs, or other examinable visuals

The AI may choose the final count, but the application needs safety limits such as a configurable maximum to prevent accidental extreme cost or output size. If the chosen count exceeds a single reliable response, the application should generate questions in topic batches while reusing the same uploaded `file_id`.

### Pass 2: MCQ generation

Use Anthropic structured outputs with a Pydantic model. The Claude API can constrain its answer to a JSON schema, and the Python SDK's `messages.parse()` helper can return validated Pydantic output.[^6][^7]

Each question should contain:

```text
id
learning_objective
topic
stem
options[^4]
correct_index           # integer 0–3
explanation             # short explanation of why the answer is correct
source_pages[]          # 1-based page numbers
source_support          # short supporting passage or visual description
image_required          # boolean
image_page              # nullable 1-based page
image_crop               # nullable normalized x0,y0,x1,y1
image_alt_text           # nullable
```

Validation must reject or regenerate a question when:

- There are not exactly four non-empty, distinct choices.
- `correct_index` is outside 0–3.
- More than one choice is defensibly correct.
- The answer depends on information absent from the PDF.
- The stem contains answer cues or negative wording without clear emphasis.
- Distractors differ in grammatical form or conceptual category.
- The rationale merely repeats the answer rather than explaining it.
- The indicated source page is invalid.
- An image is essential but no usable image page is supplied.

Claude citations and structured outputs cannot be enabled in the same request, because citation blocks cannot be interleaved with strict JSON output. Therefore the implementation should use structured output and require Claude to return source-page fields inside the schema; a separate optional audit call can verify grounding if stronger source checking is required.[^8]

## Image handling

The complete PDF should be sent to Claude as requested, but the local uploaded bytes must also remain available so the application can create actual Anki media. Claude can identify pages containing useful visuals, but it does not return the original embedded image files.

Use PyMuPDF for local media creation:

- Default reliable behavior: render the full relevant page at 150–200 DPI.
- If Claude returns a plausible normalized crop rectangle, convert it to PDF coordinates and render that clipped region.
- Show every proposed crop in the Streamlit preview so the user can adjust or replace it.
- If the crop is missing or invalid, fall back to the full page rather than silently omitting essential visual context.
- Use collision-proof media names derived from the PDF hash, page number, question ID, and crop coordinates.

PyMuPDF can render complete pages or clipped rectangles through `Page.get_pixmap()`, and it can also extract embedded raster images where appropriate. Rendering the page or crop is generally safer than extracting a raw image because medical figures often combine labels, vector annotations, legends, and multiple panels.[^9][^10]

## Anki deck design

Use `genanki` to create an Anki deck package. An `.apkg` represents a packaged deck and can include images and other media.[^11][^12]

Recommended note fields:

```text
QuestionID
Topic
Question
OptionA
OptionB
OptionC
OptionD
Image
CorrectAnswer
Explanation
SourcePages
PDFTitle
```

The front template should display the stem, optional image, and four labeled choices. The back template should repeat the front, highlight the correct choice, and show the concise explanation plus source-page reference. The app should escape untrusted HTML before inserting generated content into card fields.

With `genanki`, media paths are attached to `Package.media_files`, while the card references each image by basename using an HTML `<img>` tag; media filenames must be unique. The package should use deterministic deck/model IDs and stable note GUIDs based on the PDF hash and question ID so regenerating the same deck does not create unnecessary duplicates.[^13][^14]

## Project structure

```text
pdf2anki-mcq/
├── app.py
├── pyproject.toml
├── .env.example
├── .gitignore
├── README.md
├── src/pdf2anki/
│   ├── __init__.py
│   ├── config.py
│   ├── schemas.py
│   ├── prompts.py
│   ├── anthropic_client.py
│   ├── pdf_media.py
│   ├── validators.py
│   ├── anki_export.py
│   └── workflow.py
└── tests/
    ├── fixtures/
    ├── test_schemas.py
    ├── test_validators.py
    ├── test_pdf_media.py
    ├── test_anki_export.py
    └── test_workflow.py
```

Suggested dependencies are `streamlit`, `anthropic`, `pydantic`, `genanki`, `pymupdf`, `pillow`, `python-dotenv`, and `tenacity`, with `pytest`, `ruff`, and `mypy` as development dependencies.

## Privacy and reliability

Do not store the API key in source code, session state logs, generated decks, or Git. Read `ANTHROPIC_API_KEY` from the environment, with an optional masked Streamlit input for local use.

Files uploaded through Anthropic's Files API persist until deleted or until an assigned expiry is reached, and the API provides a file-deletion endpoint. For clinical or patient-related PDFs, the application should display a warning to remove identifiers; Anthropic states that PDF HIPAA eligibility applies to PDFs sent inline through Messages, not through the Files API, which is particularly relevant for pathology material containing PHI.[^15][^16][^17]

The app should implement retries only for transient rate-limit and server errors, never blindly retry schema or validation failures. It should provide actionable messages for encrypted PDFs, over-limit PDFs, unsupported models, rate limits, malformed model output, and failed Anki packaging.

## Aider role workflow

The detailed downloadable specification should be added to the project as `SPEC.md`. Start the repository and ask the Anthropic architect plus Gemini programmer to implement it in phases rather than attempting the full application in one edit.

### Architecture and core implementation

```text
/model architect
/editor-model programmer
/architect Read SPEC.md completely. Design and implement Phase 1 of this application:
project scaffolding, Pydantic schemas, Anthropic PDF upload/client abstraction,
two-pass coverage and MCQ pipeline, validation, and mocked unit tests. Keep the
Streamlit UI minimal in this phase. Do not invent SDK methods: verify usage
against the installed anthropic package. Run the tests and lint checks.
```

Then implement the media and Anki layer:

```text
/model architect
/editor-model programmer
/architect Implement Phase 2 from SPEC.md: PyMuPDF page/crop rendering,
editable Streamlit preview, deterministic genanki .apkg generation with embedded
images, download flow, cleanup, error handling, and tests. Preserve separation
between UI, API, PDF, validation, and export layers. Run the entire test suite.
```

### Repetitive work

DeepSeek should handle low-risk, repetitive tasks rather than architectural or security-critical decisions:

```text
/model grunt
/code Add docstrings and type annotations to the public functions, complete the
.env.example and .gitignore, normalize imports, and add straightforward test
cases for validation boundaries. Do not change public behavior, prompts,
Anthropic calls, Anki templates, or dependency versions. Run ruff and pytest.
```

### Independent review

OpenAI should first review without editing:

```text
/model reviewer
/ask Audit the implementation against every requirement in SPEC.md. Focus on
Anthropic SDK correctness, PDF limits and cleanup, schema validation, MCQ quality
rules, prompt injection from PDF content, HTML escaping, image crop/page
correctness, APKG media integrity, stable Anki identifiers, Streamlit state,
error recovery, privacy, and test coverage. Do not edit. Return prioritized
findings with file names, line references, reproduction steps, and concrete fixes.
```

After inspecting the findings:

```text
/model reviewer
/code Fix all confirmed critical and high-severity findings from the previous
audit. Add regression tests for each fix. Do not weaken validation or remove
features merely to make tests pass. Run ruff, mypy, and pytest, then report any
remaining limitations.
```

Finally return to the normal pair for integration cleanup:

```text
/model architect
/editor-model programmer
/architect Review the completed implementation and tests against SPEC.md.
Resolve remaining medium-severity issues, simplify duplicated code, verify the
README installation instructions in a clean virtual environment, and leave the
repository in a releasable state.
```

## Acceptance criteria

The build is complete only when all of the following are demonstrated:

- A text-heavy PDF produces a non-empty, topic-balanced MCQ deck.
- A scanned or image-heavy PDF is understood by Claude without local OCR.
- Every card has exactly four choices and one validated answer.
- Every answer has a concise explanation and source-page reference.
- At least one fixture verifies an image-containing card and confirms that its media file exists inside the `.apkg`.
- Generated cards can be imported into Anki Desktop without missing-media warnings.
- User edits survive Streamlit reruns until the final download.
- API errors do not lose the uploaded local PDF or silently create a partial deck.
- The Anthropic remote file is deleted after success and after handled failure.
- Unit tests run without a live API key by mocking the Anthropic client.
- A separate opt-in integration test can call Claude using a small non-sensitive fixture PDF.

---

## References

1. [platform.claude.com › docs › enplatform.claude.com](https://platform.claude.com/docs/en/build-with-claude/files.md) - * **Maximum file size:** 500 MB per file
* **Total storage:** 1 TB per organization

2. [PDF support - Claude Platform Docs](https://platform.claude.com/docs/en/build-with-claude/pdf-support) - Provides complete visual analysis of PDFs · Can understand and analyze charts, graphs, images, and v...

3. [How to download a file in Streamlit?](https://docs.streamlit.io/knowledge-base/using-streamlit/how-download-file-streamlit) - Use the `st.download_button` widget that is natively built into Streamlit.

4. [st.file_uploader - Streamlit Docs](https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader) - By default, uploaded files are limited to 200 MB each.
You can

5. [Increase output consistency - Claude Platform Docs](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/increase-consistency) - If you need Claude to always output valid JSON that conforms to a specific schema, use Structured Ou...

6. [Messages - Claude API Reference - Claude Console](https://platform.claude.com/docs/en/api/python/messages) - A schema to specify Claude's output format in responses.
See structured outputs

7. [Structured outputs - Claude Platform Docs](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) - Structured outputs work by compiling your JSON schemas into a grammar that constrains Claude's outpu...

8. [Citations - Claude Platform Docs](https://platform.claude.com/docs/en/build-with-claude/citations) - APIPDF support Images and vision Skills. For PDFs: Citations include the page number range (1-indexe...

9. [pymupdf.readthedocs.io › _ › downloadsPyMuPDF Documentation](https://pymupdf.readthedocs.io/_/downloads/en/latest/pdf/) - 2. Extract the image with img = doc.extract_image(xref).

10. [Images - PyMuPDF documentation](https://pymupdf.readthedocs.io/en/latest/recipes-images.html) - pix = page.get_pixmap(matrix=mat, clip=clip)

11. [Exporting - Anki Docs](https://docs.ankiweb.net/manual/exporting) - `.apkg` stands for "Anki Deck Package".
You can use it to export a single deck

12. [Welcome to genanki’s documentation! - GitHub Pages](https://darigovresearch.github.io/genanki/build/html/index.html) - genanki allows you to programatically generate decks in Python 3 for Anki, a popular spaced-repetiti...

13. [Media Files | kerrickstaley/genanki | DeepWiki](https://deepwiki.com/kerrickstaley/genanki/4-media-files) - The `media_files` attribute should be a list of file paths (either relative or absolute) to each med...

14. [Overview — genanki 0.1 documentation - GitHub Pages](https://darigovresearch.github.io/genanki/build/html/overview.html) - my_package.media_files = ['sound.mp3', 'images/image.jpg']

15. [API and data retention - Claude Platform Docs](https://platform.claude.com/docs/en/manage-claude/api-and-data-retention) - When using structured outputs or tools with strict: true , the API compiles JSON schemas into gramma...

16. [Files - Claude API Reference](https://platform.claude.com/docs/en/api/files) - minLength1
maxLength500
...
Size of the file in bytes.

17. [Files API - Claude Platform](https://platform.claude.com/docs/en/build-with-claude/files?query=CTA&38d7aa68_page=2&fcdaa149_page=1) - - Files are inaccessible through the API shortly after deletion, but they may persist in active Mess...

