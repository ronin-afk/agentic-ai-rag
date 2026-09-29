import os
from pathlib import Path
from uuid import uuid4

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from pinecone import Pinecone, ServerlessSpec

from src.config import (
    OPENAI_EMBEDDING_MODEL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)


PDF_PATH = Path("data/Ebook-Agentic-AI.pdf")


def create_pinecone_index():
    """
    Create the Pinecone index if it does not already exist.
    """

    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [index.name for index in pc.list_indexes()]

    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Creating Pinecone index: {PINECONE_INDEX_NAME}")

        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
        )

        print("Index created.")

    else:
        print(f"Using existing index: {PINECONE_INDEX_NAME}")

    return pc.Index(PINECONE_INDEX_NAME)


def load_and_chunk_pdf():
    """
    Load the PDF and split it into chunks.
    """

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found at {PDF_PATH}"
        )

    print("Loading PDF...")

    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    print(f"Loaded {len(documents)} pages.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    print(f"Created {len(chunks)} chunks.")

    return chunks


def ingest_documents():
    """
    Complete ingestion pipeline:

    PDF
      ↓
    chunks
      ↓
    embeddings
      ↓
    Pinecone
    """

    chunks = load_and_chunk_pdf()

    index = create_pinecone_index()

    embeddings = OpenAIEmbeddings(
        model=OPENAI_EMBEDDING_MODEL
    )

    print("Generating embeddings and uploading to Pinecone...")

    vectors = []

    for chunk in chunks:
        text = chunk.page_content.strip()

        if not text:
            continue

        vector = embeddings.embed_query(text)

        page_number = chunk.metadata.get("page", 0)

        vectors.append(
            {
                "id": str(uuid4()),
                "values": vector,
                "metadata": {
                    "text": text,
                    "page": page_number + 1,
                    "source": PDF_PATH.name,
                },
            }
        )

    # Upload in batches
    batch_size = 100

    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]

        index.upsert(vectors=batch)

        print(
            f"Uploaded {min(i + batch_size, len(vectors))}"
            f"/{len(vectors)} vectors"
        )

    print("Ingestion completed successfully.")


if __name__ == "__main__":
    ingest_documents()