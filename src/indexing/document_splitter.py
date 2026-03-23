"""
Document splitting module for the RAG indexing pipeline.
Splits large documents into smaller chunks suitable for embedding and retrieval.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from src.config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(docs: list[Document]) -> list[Document]:
    """
    Splits a list of documents into smaller chunks using recursive character splitting.

    Args:
        docs (list[Document]): The source documents to split.

    Returns:
        list[Document]: The resulting list of document chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} sub-documents.")
    return chunks