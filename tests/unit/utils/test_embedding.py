"""Tests for embedding client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from graphunified.utils.embedding import EmbeddingClient


@pytest.mark.asyncio
class TestEmbeddingClient:
    """Tests for EmbeddingClient."""

    def test_from_config(self, sample_embedding_config):
        """Test creating client from config."""
        client = EmbeddingClient.from_config(sample_embedding_config)

        assert client.model == sample_embedding_config.model
        assert client.dimension == sample_embedding_config.dimension
        assert client.batch_size == sample_embedding_config.batch_size

    def test_normalize_vector(self):
        """Test vector normalization."""
        vector = [3.0, 4.0]  # Length = 5
        normalized = EmbeddingClient._normalize_vector(vector)

        assert abs(normalized[0] - 0.6) < 0.001
        assert abs(normalized[1] - 0.8) < 0.001

        # Check L2 norm is 1
        import math
        norm = math.sqrt(sum(x**2 for x in normalized))
        assert abs(norm - 1.0) < 0.001

    @patch('voyageai.AsyncClient')
    async def test_embed_success(self, mock_voyage):
        """Test successful embedding generation."""
        # Mock response with 1024-dimensional embeddings
        mock_result = MagicMock()
        mock_result.embeddings = [[0.1] * 1024, [0.2] * 1024]

        mock_client = MagicMock()
        mock_client.embed = MagicMock(return_value=mock_result)
        mock_voyage.return_value = mock_client

        client = EmbeddingClient(
            api_key="test-key",
            model="voyage-3",
            dimension=1024,
            batch_size=128,
            normalize=False,
        )

        texts = ["Text 1", "Text 2"]
        embeddings = await client.embed(texts)

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1024
        assert len(embeddings[1]) == 1024

    async def test_embed_empty_list(self):
        """Test embedding empty list returns empty list."""
        client = EmbeddingClient(
            api_key="test-key",
            model="voyage-3",
            dimension=1024,
        )

        embeddings = await client.embed([])
        assert embeddings == []
