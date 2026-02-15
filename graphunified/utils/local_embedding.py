"""Local embedding client using sentence-transformers."""

import asyncio
from typing import List

import numpy as np

from graphunified.config.settings import EmbeddingConfig
from graphunified.exceptions import EmbeddingError
from graphunified.utils.logging import get_logger

logger = get_logger(__name__)


class LocalEmbeddingClient:
    """Local embedding client using sentence-transformers."""

    def __init__(
        self,
        model: str = "BAAI/bge-large-en-v1.5",
        dimension: int = 1024,
        batch_size: int = 32,
        normalize: bool = True,
        device: str = "auto",
    ):
        """Initialize local embedding client.

        Args:
            model: HuggingFace model identifier
            dimension: Expected embedding dimension
            batch_size: Batch size for encoding
            normalize: Whether to L2-normalize embeddings
            device: Device to use ('auto', 'cuda', 'cpu')
        """
        self.model_name = model
        self.dimension = dimension
        self.batch_size = batch_size
        self.normalize = normalize
        self.device = device

        # Lazy load the model
        self._model = None

    def _load_model(self):
        """Lazy load the sentence-transformers model."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
            import torch

            logger.info(f"Loading sentence-transformers model: {self.model_name}")

            # Determine device
            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device

            # Load model
            self._model = SentenceTransformer(self.model_name, device=device)

            logger.info(f"Model loaded on device: {device}")

            # Verify dimension
            test_embedding = self._model.encode(["test"], show_progress_bar=False)
            actual_dim = test_embedding.shape[1]

            if actual_dim != self.dimension:
                logger.warning(
                    f"Model dimension {actual_dim} differs from configured {self.dimension}. "
                    f"Using actual dimension."
                )
                self.dimension = actual_dim

        except ImportError:
            raise EmbeddingError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
        except Exception as e:
            raise EmbeddingError(f"Failed to load sentence-transformers model: {e}")

    @classmethod
    def from_config(cls, config: EmbeddingConfig) -> "LocalEmbeddingClient":
        """Create client from configuration.

        Args:
            config: Embedding configuration

        Returns:
            Local embedding client instance
        """
        return cls(
            model=config.model,
            dimension=config.dimension,
            batch_size=config.batch_size,
            normalize=config.normalize,
        )

    @property
    def model(self) -> str:
        """Get model identifier."""
        return self.model_name

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not texts:
            return []

        # Load model if not already loaded
        self._load_model()

        try:
            # Run encoding in thread pool (blocking operation)
            embeddings = await asyncio.to_thread(
                self._encode_batch,
                texts,
            )

            return embeddings

        except Exception as e:
            logger.error(f"Local embedding generation failed: {e}")
            raise EmbeddingError(f"Failed to generate embeddings: {e}")

    def _encode_batch(self, texts: List[str]) -> List[List[float]]:
        """Encode a batch of texts.

        Args:
            texts: List of texts

        Returns:
            List of embeddings
        """
        # Generate embeddings
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=len(texts) > 100,  # Show progress for large batches
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )

        # Convert to list of lists
        return embeddings.tolist()

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
            embeddings = await self.embed([query])
            return embeddings[0]

        except Exception as e:
            logger.error(f"Query embedding generation failed: {e}")
            raise EmbeddingError(f"Failed to generate query embedding: {e}")


# Recommended models with their dimensions
RECOMMENDED_MODELS = {
    # Best quality (1024d)
    "bge-large": {
        "model": "BAAI/bge-large-en-v1.5",
        "dimension": 1024,
        "description": "Best quality, good for production",
    },
    # Good balance (768d)
    "bge-base": {
        "model": "BAAI/bge-base-en-v1.5",
        "dimension": 768,
        "description": "Good balance of quality and speed",
    },
    # Fast and lightweight (384d)
    "minilm": {
        "model": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "description": "Fast and lightweight, good for testing",
    },
    # E5 models (also excellent)
    "e5-large": {
        "model": "intfloat/e5-large-v2",
        "dimension": 1024,
        "description": "Alternative to BGE, also excellent quality",
    },
    "e5-base": {
        "model": "intfloat/e5-base-v2",
        "dimension": 768,
        "description": "E5 base model, good balance",
    },
}
