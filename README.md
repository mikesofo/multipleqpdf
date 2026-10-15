# pdf2anki-mcq

Generate ready-to-import Anki MCQ decks from a PDF using Anthropic's Claude.

## Overview

Upload a PDF, let Claude build a coverage plan and generate schema-validated
multiple-choice questions, preview and edit them, then export an `.apkg` deck
with embedded images.

This repository currently implements **Phase 1**:

- Project scaffolding
- Pydantic schemas
- Anthropic PDF upload/client abstraction
- Two-pass coverage + MCQ pipeline
- Validation
- Mocked unit tests
- Minimal Streamlit UI

PDF rendering and Anki export arrive in Phase 2.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Configuration

Copy `.env.example` to `.env` and set your key:

```bash
cp .env.example .env
```

```
ANTHROPIC_API_KEY=sk-ant-...
```

The key is read from the environment. You may also paste it into the masked
Streamlit input for local use.

## Running

```bash
streamlit run app.py
```

## Development

```bash
ruff check .
pytest
```

## Privacy

Do not commit your API key. Remove patient identifiers from clinical PDFs
before uploading. Remote Anthropic files are deleted after generation.
