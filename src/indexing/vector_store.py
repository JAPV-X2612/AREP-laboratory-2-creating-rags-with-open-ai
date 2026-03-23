"""
Vector store module for the RAG indexing pipeline.

Manages Pinecone index creation and document embedding storage
using the langchain-pinecone integration package.
"""

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
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
    if not pc.has_index(PINECONE_INDEX_NAME):
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
    embeddings: HuggingFaceEmbeddings,
) -> PineconeVectorStore:
    """
    Embeds document chunks and stores them in the Pinecone vector store.

    Args:
        chunks (list[Document]): The document chunks to embed and store.
        embeddings (HuggingFaceEmbeddings): The embeddings model instance.

    Returns:
        PineconeVectorStore: The populated vector store.
    """
    pc = Pinecone(api_key=get_pinecone_api_key())
    _get_or_create_index(pc)

    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME,
        pinecone_api_key=get_pinecone_api_key(),
    )
    print(f"Documents indexed in '{PINECONE_INDEX_NAME}'.")
    return vector_store


def load_vector_store(embeddings: HuggingFaceEmbeddings) -> PineconeVectorStore:
    """
    Loads an existing Pinecone vector store without re-indexing documents.

    Args:
        embeddings (HuggingFaceEmbeddings): The embeddings model instance.

    Returns:
        PineconeVectorStore: The existing vector store.
    """
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings,
        pinecone_api_key=get_pinecone_api_key(),
    )