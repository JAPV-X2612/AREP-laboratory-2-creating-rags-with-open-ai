"""
Language model and embeddings initialization module.
Provides factory functions for the Anthropic LLM and OpenAI embeddings model.
"""

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from langchain_openai import OpenAIEmbeddings
from src.config.settings import (
    ANTHROPIC_MODEL,
    AGENT_TEMPERATURE,
    AGENT_MAX_TOKENS,
    OPENAI_EMBEDDING_MODEL,
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


def create_embeddings() -> OpenAIEmbeddings:
    """
    Instantiates the OpenAI embeddings model.

    Returns:
        OpenAIEmbeddings: A configured text-embedding-3-small instance.
    """
    return OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)