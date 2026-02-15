"""Exception hierarchy for graph-unified."""


class GraphUnifiedError(Exception):
    """Base exception for all graph-unified errors."""
    pass


class ConfigurationError(GraphUnifiedError):
    """Raised when configuration is invalid or missing."""
    pass


class ValidationError(GraphUnifiedError):
    """Raised when data validation fails."""
    pass


class StorageError(GraphUnifiedError):
    """Raised when storage operations fail."""
    pass


class APIError(GraphUnifiedError):
    """Raised when external API calls fail."""
    pass


class RateLimitError(APIError):
    """Raised when rate limits are exceeded."""
    pass


class TokenLimitError(APIError):
    """Raised when token limits are exceeded."""
    pass


class EmbeddingError(APIError):
    """Raised when embedding generation fails."""
    pass
