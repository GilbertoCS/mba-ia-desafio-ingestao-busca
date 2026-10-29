import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

OUT_OF_CONTEXT_ANSWER = "Não tenho informações necessárias para responder sua pergunta."

OLLAMA_CONTEXT_HINT = """
Nota sobre o CONTEXTO: linhas no formato "Empresa R$ valor ano" indicam o faturamento anual dessa empresa.
"""

DEFAULT_DATABASE_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/rag"
DEFAULT_COLLECTION_NAME = "documents"
DEFAULT_PDF_PATH = "document.pdf"
DEFAULT_OLLAMA_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_LLM_MODEL = "llama3.2"
DEFAULT_OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"


class GeminiBatchEmbeddings(Embeddings):
    def __init__(self, embeddings, delay_seconds):
        self.embeddings = embeddings
        self.delay_seconds = delay_seconds

    def embed_documents(self, texts):
        vectors = []
        for index, text in enumerate(texts):
            if index and self.delay_seconds:
                time.sleep(self.delay_seconds)
            vectors.append(self.embeddings.embed_query(text))
        return vectors

    def embed_query(self, text):
        return self.embeddings.embed_query(text)


def get_database_url():
    return os.getenv("DATABASE_URL") or DEFAULT_DATABASE_URL


def get_collection_name():
    return os.getenv("PG_VECTOR_COLLECTION_NAME") or DEFAULT_COLLECTION_NAME


def get_pdf_path():
    return os.getenv("PDF_PATH") or DEFAULT_PDF_PATH


def get_ollama_base_url():
    return os.getenv("OLLAMA_BASE_URL") or DEFAULT_OLLAMA_BASE_URL


def _use_local_llm():
    flag = (os.getenv("USE_LOCAL_LLM") or "").lower()
    return flag in {"1", "true", "yes", "on"}


def _default_provider():
    if _use_local_llm():
        return "ollama"
    if (os.getenv("LLM_PROVIDER") or os.getenv("EMBEDDINGS_PROVIDER") or "").lower() == "ollama":
        return "ollama"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("GOOGLE_API_KEY"):
        return "gemini"
    return "openai"


def _provider(env_name):
    return (os.getenv(env_name) or _default_provider()).lower()


def _require_env(name):
    if os.getenv(name):
        return
    raise ValueError(f"Configure a variável {name} no arquivo .env")


def get_embeddings():
    provider = _provider("EMBEDDINGS_PROVIDER")

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=os.getenv("OLLAMA_EMBEDDING_MODEL") or DEFAULT_OLLAMA_EMBEDDING_MODEL,
            base_url=get_ollama_base_url(),
        )

    if provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        _require_env("GOOGLE_API_KEY")
        embeddings = GoogleGenerativeAIEmbeddings(
            model=os.getenv("GOOGLE_EMBEDDING_MODEL") or "models/gemini-embedding-001"
        )
        return GeminiBatchEmbeddings(
            embeddings=embeddings,
            delay_seconds=float(os.getenv("GOOGLE_EMBEDDING_DELAY_SECONDS") or "1"),
        )

    _require_env("OPENAI_API_KEY")
    return OpenAIEmbeddings(
        model=os.getenv("OPENAI_EMBEDDING_MODEL") or "text-embedding-3-small"
    )


def get_llm():
    provider = _provider("LLM_PROVIDER")

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=os.getenv("OLLAMA_LLM_MODEL") or DEFAULT_OLLAMA_LLM_MODEL,
            base_url=get_ollama_base_url(),
            temperature=0,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        _require_env("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(
            model=os.getenv("GOOGLE_LLM_MODEL") or "gemini-2.5-flash-lite",
            temperature=0,
        )

    _require_env("OPENAI_API_KEY")
    return ChatOpenAI(
        model=os.getenv("OPENAI_LLM_MODEL") or "gpt-4o-mini",
        temperature=0,
    )


def build_vector_store():
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=get_collection_name(),
        connection=get_database_url(),
        use_jsonb=True,
    )


def _merge_retrieval_results(*result_lists):
    merged = {}
    for results in result_lists:
        for document, score in results:
            key = document.page_content
            if key not in merged or score < merged[key][1]:
                merged[key] = (document, score)
    return sorted(merged.values(), key=lambda item: item[1])


def _entity_tokens(question):
    import re

    tokens = re.findall(r"\b[A-Za-z0-9]{5,}\b", question)
    return [token for token in tokens if sum(character.isupper() for character in token) >= 2]


def retrieve_results(question):
    vector_store = build_vector_store()
    results = vector_store.similarity_search_with_score(question, k=10)

    pinned = []
    for token in _entity_tokens(question):
        extra = vector_store.similarity_search_with_score(token, k=10)
        for document, score in extra:
            if token in document.page_content:
                pinned.append((document, score))

    if not pinned:
        return results[:10]

    merged = _merge_retrieval_results(results, pinned)
    pinned_keys = {document.page_content for document, _score in pinned}
    pinned_items = [item for item in merged if item[0].page_content in pinned_keys]
    other_items = [item for item in merged if item[0].page_content not in pinned_keys]
    return (pinned_items + other_items)[:10]


def _select_context_results(question, results):
    tokens = _entity_tokens(question)
    if tokens:
        filtered = [
            item for item in results if any(token in item[0].page_content for token in tokens)
        ]
        if filtered:
            return filtered
    return results


def _format_context(results):
    return "\n\n".join(document.page_content for document, _score in results)


def _message_content(response):
    return getattr(response, "content", str(response)).strip()


def _ollama_retry_prompt(question, contexto):
    return f"""Responda somente com base no CONTEXTO abaixo.
Se a informação não estiver no CONTEXTO, responda exatamente:
"{OUT_OF_CONTEXT_ANSWER}"

CONTEXTO:
{contexto}

PERGUNTA: {question}
RESPOSTA:"""


def search_prompt(question=None):
    if not question:
        raise ValueError("Informe uma pergunta para realizar a busca.")

    results = retrieve_results(question)

    if not results:
        return OUT_OF_CONTEXT_ANSWER

    contexto = _format_context(_select_context_results(question, results))
    if _provider("LLM_PROVIDER") == "ollama":
        contexto = f"{contexto}\n{OLLAMA_CONTEXT_HINT}"

    prompt = PROMPT_TEMPLATE.format(
        contexto=contexto,
        pergunta=question,
    )

    response = get_llm().invoke(prompt)
    answer = _message_content(response)

    if (
        _provider("LLM_PROVIDER") == "ollama"
        and (answer or "") == OUT_OF_CONTEXT_ANSWER
        and _entity_tokens(question)
    ):
        retry = get_llm().invoke(_ollama_retry_prompt(question, contexto))
        answer = _message_content(retry)

    return answer or OUT_OF_CONTEXT_ANSWER
