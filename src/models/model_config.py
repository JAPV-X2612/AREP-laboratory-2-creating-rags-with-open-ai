"""
Language model and embeddings initialization module.
Provides factory functions for the Anthropic LLM and OpenAI embeddings model.
"""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_huggingface import HuggingFaceEmbeddings
from src.config.settings import (
    ANTHROPIC_MODEL,
    AGENT_TEMPERATURE,
    AGENT_MAX_TOKENS,
    HUGGINGFACE_EMBEDDING_MODEL,
)


def create_chat_model() -> BaseChatModel:
    """
    Instantiates the Anthropic LLM via LangChain.

    Returns:
        BaseChatModel: A configured Claude Sonnet chat model instance.
    """
    return init_chat_model(
        ANTHROPIC_MODEL,
        model_provider="anthropic",
        temperature=AGENT_TEMPERATURE,
        max_tokens=AGENT_MAX_TOKENS,
    )


def create_embeddings() -> HuggingFaceEmbeddings:
    """
    Instantiates the HuggingFace embeddings model.

    Returns:
        HuggingFaceEmbeddings: A configured all-MiniLM-L6-v2 instance.
    """
    return HuggingFaceEmbeddings(model_name=HUGGINGFACE_EMBEDDING_MODEL)