from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

from search import get_collection_name, get_database_url, get_embeddings, get_pdf_path


def ingest_pdf():
    pdf_path = Path(get_pdf_path())

    if not pdf_path.exists():
        raise FileNotFoundError(f"Arquivo PDF não encontrado: {pdf_path}")

    print(f"Lendo PDF: {pdf_path}")
    documents = PyPDFLoader(str(pdf_path)).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(documents)

    print(f"Gravando {len(chunks)} chunks no PostgreSQL/pgVector...")
    PGVector.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name=get_collection_name(),
        connection=get_database_url(),
        pre_delete_collection=True,
        use_jsonb=True,
    )
    print("Ingestão concluída com sucesso.")


if __name__ == "__main__":
    ingest_pdf()
