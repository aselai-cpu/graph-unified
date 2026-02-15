"""Tests for tokenizer utilities."""

from graphunified.utils.tokenizer import count_tokens, decode_tokens, tokenize


class TestTokenizer:
    """Tests for tokenizer functions."""

    def test_count_tokens(self):
        """Test token counting."""
        text = "Hello, world!"
        token_count = count_tokens(text)
        assert token_count > 0
        assert isinstance(token_count, int)

    def test_count_tokens_empty_string(self):
        """Test token counting for empty string."""
        assert count_tokens("") == 0

    def test_tokenize_and_decode(self):
        """Test tokenization and decoding roundtrip."""
        text = "This is a test."
        tokens = tokenize(text)
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)

        decoded = decode_tokens(tokens)
        assert decoded == text

    def test_different_encodings(self):
        """Test different encoding schemes."""
        text = "Test text"

        # cl100k_base (default)
        tokens_cl100k = count_tokens(text, "cl100k_base")

        # p50k_base
        tokens_p50k = count_tokens(text, "p50k_base")

        # Both should return positive counts
        assert tokens_cl100k > 0
        assert tokens_p50k > 0
