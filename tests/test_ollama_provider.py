import os
import sys
from pathlib import Path
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import search  # noqa: E402


def test_use_local_llm_selects_ollama_provider():
    env = {
        "USE_LOCAL_LLM": "true",
        "OPENAI_API_KEY": "sk-test",
        "LLM_PROVIDER": "",
        "EMBEDDINGS_PROVIDER": "",
    }
    with patch.dict(os.environ, env, clear=False):
        assert search._provider("LLM_PROVIDER") == "ollama"
        assert search._provider("EMBEDDINGS_PROVIDER") == "ollama"


def test_ollama_embeddings_do_not_require_openai_key():
    env = {
        "USE_LOCAL_LLM": "true",
        "OPENAI_API_KEY": "",
        "OLLAMA_EMBEDDING_MODEL": "nomic-embed-text",
    }
    with patch.dict(os.environ, env, clear=False):
        with patch("langchain_ollama.OllamaEmbeddings") as mock_embeddings:
            mock_embeddings.return_value = object()
            search.get_embeddings()
            mock_embeddings.assert_called_once()
