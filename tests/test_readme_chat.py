import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import search  # noqa: E402
from fixtures.readme_examples import (  # noqa: E402
    PDF_CONTEXT_SUPERTECH,
    README_ANSWER_IN_CONTEXT,
    README_QUESTION_IN_CONTEXT,
    README_QUESTION_OUT_OF_CONTEXT,
)


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    with patch.object(search, "build_vector_store", return_value=store):
        yield store


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    with patch.object(search, "get_llm", return_value=llm):
        yield llm


def test_readme_in_context_question(mock_vector_store, mock_llm):
    document = Document(page_content=PDF_CONTEXT_SUPERTECH)
    mock_vector_store.similarity_search_with_score.return_value = [(document, 0.05)]
    mock_llm.invoke.return_value = MagicMock(content=README_ANSWER_IN_CONTEXT)

    answer = search.search_prompt(README_QUESTION_IN_CONTEXT)

    assert answer == README_ANSWER_IN_CONTEXT
    mock_vector_store.similarity_search_with_score.assert_any_call(
        README_QUESTION_IN_CONTEXT,
        k=10,
    )
    prompt = mock_llm.invoke.call_args[0][0]
    assert PDF_CONTEXT_SUPERTECH in prompt
    assert README_QUESTION_IN_CONTEXT in prompt
    assert "CONTEXTO:" in prompt
    assert 'RESPONDA A "PERGUNTA DO USUÁRIO"' in prompt


def test_readme_out_of_context_question(mock_vector_store, mock_llm):
    document = Document(page_content=PDF_CONTEXT_SUPERTECH)
    mock_vector_store.similarity_search_with_score.return_value = [(document, 0.12)]
    mock_llm.invoke.return_value = MagicMock(content=search.OUT_OF_CONTEXT_ANSWER)

    answer = search.search_prompt(README_QUESTION_OUT_OF_CONTEXT)

    assert answer == search.OUT_OF_CONTEXT_ANSWER
    mock_vector_store.similarity_search_with_score.assert_any_call(
        README_QUESTION_OUT_OF_CONTEXT,
        k=10,
    )


def test_empty_search_returns_out_of_context_without_llm(mock_vector_store, mock_llm):
    mock_vector_store.similarity_search_with_score.return_value = []

    answer = search.search_prompt(README_QUESTION_OUT_OF_CONTEXT)

    assert answer == search.OUT_OF_CONTEXT_ANSWER
    mock_llm.invoke.assert_not_called()


def test_search_requires_question():
    with pytest.raises(ValueError, match="Informe uma pergunta"):
        search.search_prompt(None)

    with pytest.raises(ValueError, match="Informe uma pergunta"):
        search.search_prompt("")
