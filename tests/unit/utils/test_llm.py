"""Tests for LLM client."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from graphunified.utils.llm import ClaudeClient, RateLimiter
from graphunified.exceptions import RateLimitError, TokenLimitError


@pytest.mark.asyncio
class TestRateLimiter:
    """Tests for RateLimiter."""

    async def test_acquire_within_limits(self):
        """Test acquiring slot when within limits."""
        limiter = RateLimiter(requests_per_minute=10, tokens_per_minute=1000)

        # Should succeed without waiting
        await limiter.acquire(100)
        assert len(limiter.request_times) == 1
        assert len(limiter.token_usage) == 1

    async def test_token_limit_exceeded(self):
        """Test that exceeding token limit raises error."""
        limiter = RateLimiter(requests_per_minute=10, tokens_per_minute=1000)

        with pytest.raises(TokenLimitError):
            await limiter.acquire(2000)  # Exceeds per-minute limit


@pytest.mark.asyncio
class TestClaudeClient:
    """Tests for ClaudeClient."""

    def test_from_config(self, sample_llm_config):
        """Test creating client from config."""
        client = ClaudeClient.from_config(sample_llm_config)

        assert client.model == sample_llm_config.model
        assert client.temperature == sample_llm_config.temperature
        assert client.max_tokens == sample_llm_config.max_tokens

    @patch('graphunified.utils.llm.AsyncAnthropic')
    async def test_generate_success(self, mock_anthropic):
        """Test successful text generation."""
        # Mock response
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Generated response")]
        mock_response.usage = MagicMock(input_tokens=10, output_tokens=5)

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=mock_response)
        mock_anthropic.return_value = mock_client

        client = ClaudeClient(
            api_key="test-key",
            model="claude-3-haiku-20240307",
            requests_per_minute=100,
            tokens_per_minute=10000,
        )

        response, input_tokens, output_tokens = await client.generate(
            prompt="Test prompt"
        )

        assert response == "Generated response"
        assert input_tokens == 10
        assert output_tokens == 5
        assert mock_client.messages.create.called
