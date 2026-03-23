"""
Vector store module for the RAG indexing pipeline.
Manages Pinecone index creation and document embedding storage.
"""

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from src.config.settings import (
    get_pinecone_api_key,
    PINECONE_INDEX_NAME,
    PINECONE_CLOUD,
    PINECONE_REGION,
    EMBEDDING_DIMENSIONS,
)


def _get_or_create_index(pc: Pinecone) -> None:
    """
    Creates the Pinecone index if it does not already exist.

    Args:
        pc (Pinecone): An authenticated Pinecone client instance.
    """
    existing = [idx.name for idx in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSIONS,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        print(f"Index '{PINECONE_INDEX_NAME}' created.")
    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists.")


def build_vector_store(
    chunks: list[Document],
    embeddings: OpenAIEmbeddings,
) -> PineconeVectorStore:
    """
    Embeds document chunks and stores them in the Pinecone vector store.

    Args:
        chunks (list[Document]): The document chunks to embed and store.
        embeddings (OpenAIEmbeddings): The embeddings model instance.

    Returns:
        PineconeVectorStore: The populated vector store.
    """
    pc = Pinecone(api_key=get_pinecone_api_key())
    _get_or_create_index(pc)

    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
    )
    print(f"Documents indexed in '{PINECONE_INDEX_NAME}'.")
    return vector_store


def load_vector_store(embeddings: OpenAIEmbeddings) -> PineconeVectorStore:
    """
    Loads an existing Pinecone vector store without re-indexing documents.

    Args:
        embeddings (OpenAIEmbeddings): The embeddings model instance.

    Returns:
        PineconeVectorStore: The existing vector store.
    """
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=get_pinecone_api_key(),
    )