"""Claude LLM client with rate limiting and retry logic."""

import asyncio
import time
from typing import Tuple

from anthropic import AsyncAnthropic
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from graphunified.config.settings import LLMConfig
from graphunified.exceptions import APIError, RateLimitError, TokenLimitError
from graphunified.utils.logging import get_logger
from graphunified.utils.tokenizer import count_tokens

logger = get_logger(__name__)


class RateLimiter:
    """Dual rate limiter for requests and tokens using sliding window."""

    def __init__(self, requests_per_minute: int, tokens_per_minute: int):
        """Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests per minute
            tokens_per_minute: Maximum tokens per minute
        """
        self.requests_per_minute = requests_per_minute
        self.tokens_per_minute = tokens_per_minute

        self.request_times: list[float] = []
        self.token_usage: list[Tuple[float, int]] = []  # (timestamp, tokens)

        self.lock = asyncio.Lock()

    async def acquire(self, estimated_tokens: int) -> None:
        """Acquire rate limit slot, waiting if necessary.

        Args:
            estimated_tokens: Estimated tokens for this request

        Raises:
            RateLimitError: If estimated tokens exceed per-minute limit
        """
        # Check token limit before acquiring lock
        if estimated_tokens > self.tokens_per_minute:
            raise TokenLimitError(
                f"Estimated tokens {estimated_tokens} exceeds per-minute limit "
                f"{self.tokens_per_minute}"
            )

        while True:
            async with self.lock:
                now = time.time()
                minute_ago = now - 60.0

                # Clean old entries
                self.request_times = [t for t in self.request_times if t > minute_ago]
                self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > minute_ago]

                # Check if we'd exceed limits
                current_requests = len(self.request_times)
                current_tokens = sum(tokens for _, tokens in self.token_usage)

                # Calculate wait times
                request_wait_time = 0.0
                if current_requests >= self.requests_per_minute and self.request_times:
                    oldest_request = self.request_times[0]
                    request_wait_time = 60.0 - (now - oldest_request) + 0.1  # Add 100ms buffer

                token_wait_time = 0.0
                if current_tokens + estimated_tokens > self.tokens_per_minute and self.token_usage:
                    oldest_token_time = self.token_usage[0][0]
                    token_wait_time = 60.0 - (now - oldest_token_time) + 0.1  # Add 100ms buffer

                wait_time = max(request_wait_time, token_wait_time)

                # If no wait needed, record and return
                if wait_time <= 0:
                    self.request_times.append(now)
                    self.token_usage.append((now, estimated_tokens))
                    return

            # Release lock before sleeping to allow other coroutines to proceed
            if request_wait_time > 0:
                logger.debug(f"Request rate limit reached, waiting {request_wait_time:.2f}s")
            if token_wait_time > 0:
                logger.debug(f"Token rate limit reached, waiting {token_wait_time:.2f}s")

            await asyncio.sleep(wait_time)
            # Loop back to re-acquire lock and re-check (another coroutine may have taken the slot)


class ClaudeClient:
    """Async Claude API client with rate limiting and retry logic."""

    def __init__(
        self,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: int = 60,
        requests_per_minute: int = 50,
        tokens_per_minute: int = 40000,
        max_retry_attempts: int = 3,
        retry_backoff_factor: float = 2.0,
    ):
        """Initialize Claude client.

        Args:
            api_key: Anthropic API key
            model: Model identifier
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            timeout: Request timeout in seconds
            requests_per_minute: Rate limit for requests
            tokens_per_minute: Rate limit for tokens
            max_retry_attempts: Maximum retry attempts
            retry_backoff_factor: Exponential backoff factor
        """
        self.client = AsyncAnthropic(api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.rate_limiter = RateLimiter(requests_per_minute, tokens_per_minute)
        self.max_retry_attempts = max_retry_attempts
        self.retry_backoff_factor = retry_backoff_factor

    @classmethod
    def from_config(cls, config: LLMConfig) -> "ClaudeClient":
        """Create client from configuration.

        Args:
            config: LLM configuration

        Returns:
            Claude client instance
        """
        return cls(
            api_key=config.api_key,
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
            requests_per_minute=config.rate_limit.requests_per_minute,
            tokens_per_minute=config.rate_limit.tokens_per_minute,
            max_retry_attempts=config.retry.max_attempts,
            retry_backoff_factor=config.retry.backoff_factor,
        )

    async def generate(
        self,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system: str | None = None,
    ) -> Tuple[str, int, int]:
        """Generate text using Claude API with rate limiting and retry.

        Args:
            prompt: Input prompt
            temperature: Override temperature (optional)
            max_tokens: Override max tokens (optional)
            system: System prompt (optional)

        Returns:
            Tuple of (response_text, input_tokens, output_tokens)

        Raises:
            APIError: If API call fails after retries
            RateLimitError: If rate limits are exceeded
        """
        # Count input tokens BEFORE acquiring rate limit slot
        input_token_count = count_tokens(prompt)
        if system:
            input_token_count += count_tokens(system)

        # Estimate total tokens (input + output)
        estimated_tokens = input_token_count + (max_tokens or self.max_tokens)

        # Acquire rate limit slot (may wait)
        await self.rate_limiter.acquire(estimated_tokens)

        # Make API call with retry logic
        @retry(
            stop=stop_after_attempt(self.max_retry_attempts),
            wait=wait_exponential(multiplier=self.retry_backoff_factor, min=1, max=60),
            retry=retry_if_exception_type((APIError, RateLimitError)),
            reraise=True,
        )
        async def _call_api() -> Tuple[str, int, int]:
            try:
                messages = [{"role": "user", "content": prompt}]

                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature if temperature is not None else self.temperature,
                    "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
                }

                if system:
                    kwargs["system"] = system

                response = await self.client.messages.create(**kwargs)

                # Extract response text
                response_text = response.content[0].text

                # Get token usage
                actual_input_tokens = response.usage.input_tokens
                actual_output_tokens = response.usage.output_tokens

                return response_text, actual_input_tokens, actual_output_tokens

            except Exception as e:
                error_str = str(e).lower()
                if "rate_limit" in error_str or "429" in error_str:
                    logger.warning(f"Rate limit error: {e}")
                    raise RateLimitError(f"Rate limit exceeded: {e}")
                elif "overloaded" in error_str or "5" in error_str[:3]:
                    logger.warning(f"Server error: {e}")
                    raise APIError(f"Server error: {e}")
                else:
                    logger.error(f"API error: {e}")
                    raise APIError(f"API call failed: {e}")

        try:
            return await _call_api()
        except Exception as e:
            logger.error(f"Failed after {self.max_retry_attempts} attempts: {e}")
            raise
