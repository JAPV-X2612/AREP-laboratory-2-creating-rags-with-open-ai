"""
Application settings and environment variable management.
Loads and validates all required environment variables for the RAG pipeline.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _require(var: str) -> str:
    """
    Retrieves a required environment variable.

    Args:
        var (str): The name of the environment variable.

    Returns:
        str: The value of the environment variable.

    Raises:
        EnvironmentError: If the variable is not set.
    """
    value = os.getenv(var)
    if not value:
        raise EnvironmentError(
            f"{var} environment variable is not set. "
            "Please configure it in your .env file."
        )
    return value


def get_anthropic_api_key() -> str:
    """Returns the Anthropic API key."""
    return _require("ANTHROPIC_API_KEY")


def get_huggingface_token() -> str:
    """Returns the HuggingFace API token."""
    return _require("HF_TOKEN")


def get_pinecone_api_key() -> str:
    """Returns the Pinecone API key."""
    return _require("PINECONE_API_KEY")


# LLM
ANTHROPIC_MODEL = "claude-sonnet-4-6"
AGENT_TEMPERATURE = 0.5
AGENT_MAX_TOKENS = 1000

# Embeddings
HUGGINGFACE_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSIONS = 384

# Pinecone
PINECONE_INDEX_NAME = "arep-lab-2-rag"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# Indexing
SOURCE_URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RETRIEVAL_TOP_K = 2