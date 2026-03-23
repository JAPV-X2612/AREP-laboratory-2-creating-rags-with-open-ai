"""
Retriever tool module for the RAG agent.

Wraps the Pinecone vector store similarity search as a LangChain tool,
exposing retrieved document content and metadata as artifacts.
"""

from langchain_core.tools import tool
from langchain_pinecone import PineconeVectorStore
from src.config.settings import RETRIEVAL_TOP_K


def build_retriever_tool(vector_store: PineconeVectorStore):
    """
    Builds a LangChain tool that retrieves relevant document chunks from Pinecone.

    Args:
        vector_store (PineconeVectorStore): The populated vector store to search against.

    Returns:
        Callable: A LangChain tool wrapping the similarity search.
    """

    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve information from the indexed blog post to help answer a query."""
        retrieved_docs = vector_store.similarity_search(query, k=RETRIEVAL_TOP_K)
        serialized = "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}"
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    return retrieve_context