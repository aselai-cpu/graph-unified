"""Embedding client factory for multiple providers."""

from typing import Union

from graphunified.config.settings import EmbeddingConfig
from graphunified.exceptions import ConfigurationError
from graphunified.utils.embedding import EmbeddingClient  # Voyage AI
from graphunified.utils.local_embedding import LocalEmbeddingClient
from graphunified.utils.logging import get_logger

logger = get_logger(__name__)


def create_embedding_client(
    config: EmbeddingConfig,
) -> Union[EmbeddingClient, LocalEmbeddingClient]:
    """Create embedding client based on provider configuration.

    Args:
        config: Embedding configuration

    Returns:
        Embedding client instance (Voyage AI or Local)

    Raises:
        ConfigurationError: If provider is not supported
    """
    provider = config.provider.lower()

    if provider == "voyage":
        logger.info("Using Voyage AI embedding client")
        return EmbeddingClient.from_config(config)

    elif provider in ["local", "sentence-transformers", "huggingface"]:
        logger.info(f"Using local sentence-transformers: {config.model}")
        return LocalEmbeddingClient.from_config(config)

    elif provider == "openai":
        # TODO: Implement OpenAI client
        raise ConfigurationError(
            "OpenAI embedding provider not yet implemented. "
            "Use 'voyage' or 'local' for now."
        )

    elif provider == "cohere":
        # TODO: Implement Cohere client
        raise ConfigurationError(
            "Cohere embedding provider not yet implemented. "
            "Use 'voyage' or 'local' for now."
        )

    else:
        raise ConfigurationError(
            f"Unsupported embedding provider: {provider}. "
            f"Supported providers: voyage, local (sentence-transformers), openai (TODO), cohere (TODO)"
        )


# Export factory as default
__all__ = ["create_embedding_client"]
