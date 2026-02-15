"""Tokenization utilities using tiktoken."""

from functools import lru_cache
from typing import List

import tiktoken


@lru_cache(maxsize=1)
def get_encoding(encoding_name: str = "cl100k_base") -> tiktoken.Encoding:
    """Get tiktoken encoding with caching.

    Args:
        encoding_name: Name of encoding (cl100k_base, p50k_base, etc.)

    Returns:
        Tiktoken encoding instance
    """
    return tiktoken.get_encoding(encoding_name)


def count_tokens(text: str, encoding_name: str = "cl100k_base") -> int:
    """Count tokens in text using tiktoken.

    Args:
        text: Text to tokenize
        encoding_name: Encoding to use (default: cl100k_base for Claude/GPT-4)

    Returns:
        Number of tokens
    """
    encoding = get_encoding(encoding_name)
    return len(encoding.encode(text))


def tokenize(text: str, encoding_name: str = "cl100k_base") -> List[int]:
    """Tokenize text into token IDs.

    Args:
        text: Text to tokenize
        encoding_name: Encoding to use

    Returns:
        List of token IDs
    """
    encoding = get_encoding(encoding_name)
    return encoding.encode(text)


def decode_tokens(tokens: List[int], encoding_name: str = "cl100k_base") -> str:
    """Decode token IDs back to text.

    Args:
        tokens: List of token IDs
        encoding_name: Encoding to use

    Returns:
        Decoded text
    """
    encoding = get_encoding(encoding_name)
    return encoding.decode(tokens)
