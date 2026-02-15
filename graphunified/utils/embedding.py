"""Embedding client with batching support."""

import asyncio
from typing import List

import numpy as np
import voyageai

from graphunified.config.settings import EmbeddingConfig
from graphunified.exceptions import EmbeddingError
from graphunified.utils.logging import get_logger

logger = get_logger(__name__)


class EmbeddingClient:
    """Async embedding client with batching support."""

    def __init__(
        self,
        api_key: str,
        model: str,
        dimension: int = 1024,
        batch_size: int = 128,
        normalize: bool = True,
    ):
        """Initialize embedding client.

        Args:
            api_key: Voyage AI API key
            model: Model identifier
            dimension: Expected embedding dimension
            batch_size: Batch size for API calls
            normalize: Whether to L2-normalize embeddings
        """
        self.client = voyageai.AsyncClient(api_key=api_key)
        self.model = model
        self.dimension = dimension
        self.batch_size = batch_size
        self.normalize = normalize

    @classmethod
    def from_config(cls, config: EmbeddingConfig) -> "EmbeddingClient":
        """Create client from configuration.

        Args:
            config: Embedding configuration

        Returns:
            Embedding client instance
        """
        return cls(
            api_key=config.api_key,
            model=config.model,
            dimension=config.dimension,
            batch_size=config.batch_size,
            normalize=config.normalize,
        )

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts with batching.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not texts:
            return []

        try:
            embeddings = []

            # Process in batches
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]

                # Call Voyage AI API
                result = await asyncio.to_thread(
                    self.client.embed, batch, model=self.model, input_type="document"
                )

                batch_embeddings = result.embeddings

                # Validate dimensions
                for emb in batch_embeddings:
                    if len(emb) != self.dimension:
                        raise EmbeddingError(
                            f"Expected embedding dimension {self.dimension}, got {len(emb)}"
                        )

                # Normalize if requested
                if self.normalize:
                    batch_embeddings = [
                        self._normalize_vector(emb) for emb in batch_embeddings
                    ]

                embeddings.extend(batch_embeddings)

                logger.debug(f"Generated embeddings for batch {i // self.batch_size + 1}")

            return embeddings

        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise EmbeddingError(f"Failed to generate embeddings: {e}")

    async def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a query text.

        Args:
            query: Query text

        Returns:
            Embedding vector

        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            result = await asyncio.to_thread(
                self.client.embed, [query], model=self.model, input_type="query"
            )

            embedding = result.embeddings[0]

            if len(embedding) != self.dimension:
                raise EmbeddingError(
                    f"Expected embedding dimension {self.dimension}, got {len(embedding)}"
                )

            if self.normalize:
                embedding = self._normalize_vector(embedding)

            return embedding

        except Exception as e:
            logger.error(f"Query embedding generation failed: {e}")
            raise EmbeddingError(f"Failed to generate query embedding: {e}")

    @staticmethod
    def _normalize_vector(vector: List[float]) -> List[float]:
        """L2-normalize a vector.

        Args:
            vector: Input vector

        Returns:
            Normalized vector
        """
        arr = np.array(vector)
        norm = np.linalg.norm(arr)
        if norm == 0:
            return vector
        return (arr / norm).tolist()
