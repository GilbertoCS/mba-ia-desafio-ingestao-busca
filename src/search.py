import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.embeddings import Embeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
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

DEFAULT_DATABASE_URL = "postgresql+psycopg://postgres:postgres@localhost:5432/rag"
DEFAULT_COLLECTION_NAME = "documents"
DEFAULT_PDF_PATH = "document.pdf"


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


def _default_provider():
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

    if provider == "gemini":
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

    if provider == "gemini":
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


def _format_context(results):
    return "\n\n".join(document.page_content for document, _score in results)


def _message_content(response):
    return getattr(response, "content", str(response)).strip()


def search_prompt(question=None):
    if not question:
        raise ValueError("Informe uma pergunta para realizar a busca.")

    vector_store = build_vector_store()
    results = vector_store.similarity_search_with_score(question, k=10)

    if not results:
        return OUT_OF_CONTEXT_ANSWER

    prompt = PROMPT_TEMPLATE.format(
        contexto=_format_context(results),
        pergunta=question,
    )

    response = get_llm().invoke(prompt)
    answer = _message_content(response)

    return answer or OUT_OF_CONTEXT_ANSWER
