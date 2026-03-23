"""
Application entry point for the RAG laboratory.

Demonstrates both a RAG agent (multi-step tool-based retrieval) and an LCEL
chain (single-pass retrieval and generation) over an indexed blog post.
"""

from src.config.settings import get_anthropic_api_key, get_huggingface_token, get_pinecone_api_key
from src.models.model_config import create_chat_model, create_embeddings
from src.indexing.document_loader import load_documents
from src.indexing.document_splitter import split_documents
from src.indexing.vector_store import build_vector_store
from src.retrieval.retriever_tool import build_retriever_tool
from src.agents.rag_agent import create_rag_agent, create_lcel_chain


def run_rag_agent(model, vector_store) -> None:
    """
    Executes two queries using the RAG agent, demonstrating iterative tool calling.

    Args:
        model: The configured language model instance.
        vector_store: The populated Pinecone vector store.
    """
    print("\n" + "=" * 60)
    print("RAG AGENT — Multi-Step Retrieval")
    print("=" * 60)

    retrieve_context = build_retriever_tool(vector_store)
    agent = create_rag_agent(model, [retrieve_context])

    query_1 = (
        "What is the standard method for Task Decomposition?\n\n"
        "Once you get the answer, look up common extensions of that method."
    )
    print(f"\n[Query 1]: {query_1}\n")
    for event in agent.stream(
        {"messages": [{"role": "user", "content": query_1}]},
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()

    query_2 = "What are the main challenges of using LLM as a planning component?"
    print(f"\n[Query 2]: {query_2}\n")
    for event in agent.stream(
        {"messages": [{"role": "user", "content": query_2}]},
        stream_mode="values",
    ):
        event["messages"][-1].pretty_print()


def run_lcel_chain(model, vector_store) -> None:
    """
    Executes a query using the LCEL chain, demonstrating single-pass retrieval.

    Args:
        model: The configured language model instance.
        vector_store: The populated Pinecone vector store.
    """
    print("\n" + "=" * 60)
    print("LCEL CHAIN — Single-Pass Retrieval")
    print("=" * 60)

    chain = create_lcel_chain(model, vector_store)
    query = "What is task decomposition?"
    print(f"\n[Query]: {query}\n")
    response = chain.invoke(query)
    print(f"[Response]: {response}")


def main() -> None:
    """
    Orchestrates the full RAG pipeline: indexing, agent execution, and chain execution.

    Raises:
        EnvironmentError: If any required API key is not configured.
    """
    get_anthropic_api_key()
    get_huggingface_token()
    get_pinecone_api_key()

    embeddings = create_embeddings()
    model = create_chat_model()

    # Indexing pipeline
    docs = load_documents()
    chunks = split_documents(docs)
    vector_store = build_vector_store(chunks, embeddings)

    # RAG agent
    run_rag_agent(model, vector_store)

    # LCEL chain
    run_lcel_chain(model, vector_store)


if __name__ == "__main__":
    main()