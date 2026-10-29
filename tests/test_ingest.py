import sys
from pathlib import Path

import pytest
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from fixtures.readme_examples import PDF_CONTEXT_SUPERTECH  # noqa: E402


@pytest.fixture
def pdf_path():
    path = Path(__file__).resolve().parents[1] / "document.pdf"
    if not path.exists():
        pytest.skip("document.pdf não encontrado")
    return path


def test_pdf_contains_supertech_context(pdf_path):
    documents = PyPDFLoader(str(pdf_path)).load()
    text = "\n".join(document.page_content for document in documents)

    assert PDF_CONTEXT_SUPERTECH in text


def test_ingest_uses_required_chunk_settings(pdf_path):
    documents = PyPDFLoader(str(pdf_path)).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)

    assert len(chunks) > 0
    for chunk in chunks:
        assert len(chunk.page_content) <= 1000
