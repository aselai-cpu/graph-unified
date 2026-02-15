"""Tests for configuration settings."""

import os
from pathlib import Path

import pytest

from graphunified.config.settings import ChunkingConfig, Settings
from graphunified.config.validation import substitute_env_vars
from graphunified.exceptions import ConfigurationError


class TestEnvironmentSubstitution:
    """Tests for environment variable substitution."""

    def test_substitute_required_var(self):
        """Test substituting a required environment variable."""
        os.environ["TEST_VAR"] = "test_value"
        result = substitute_env_vars("${TEST_VAR}")
        assert result == "test_value"
        del os.environ["TEST_VAR"]

    def test_substitute_with_default(self):
        """Test substituting with default value."""
        # Variable not set, should use default
        result = substitute_env_vars("${NONEXISTENT_VAR:-default_value}")
        assert result == "default_value"

    def test_missing_required_var_fails(self):
        """Test that missing required variable raises error."""
        with pytest.raises(ConfigurationError):
            substitute_env_vars("${NONEXISTENT_REQUIRED_VAR}")

    def test_substitute_in_dict(self):
        """Test substitution in nested dict."""
        os.environ["API_KEY"] = "secret123"
        config = {
            "llm": {
                "api_key": "${API_KEY}",
                "model": "claude-3",
            }
        }
        result = substitute_env_vars(config)
        assert result["llm"]["api_key"] == "secret123"
        del os.environ["API_KEY"]

    def test_substitute_in_list(self):
        """Test substitution in list."""
        os.environ["ITEM"] = "value"
        result = substitute_env_vars(["${ITEM}", "static"])
        assert result == ["value", "static"]
        del os.environ["ITEM"]


class TestChunkingConfig:
    """Tests for chunking configuration."""

    def test_valid_chunking_config(self):
        """Test creating valid chunking config."""
        config = ChunkingConfig(
            strategy="fixed",
            chunk_size=512,
            chunk_overlap=64,
        )
        assert config.chunk_size == 512
        assert config.chunk_overlap == 64

    def test_overlap_less_than_size_validation(self):
        """Test that chunk_overlap must be less than chunk_size."""
        with pytest.raises(ValueError):
            ChunkingConfig(
                chunk_size=100,
                chunk_overlap=150,  # Invalid: overlap > size
            )


class TestSettingsLoad:
    """Tests for loading settings from YAML."""

    def test_load_valid_config(self, sample_config_yaml):
        """Test loading a valid configuration file."""
        settings = Settings.load(sample_config_yaml)
        assert settings.version == "1.0"
        assert settings.llm.model == "claude-3-haiku-20240307"
        assert settings.embedding.dimension == 1024

    def test_load_nonexistent_file_fails(self):
        """Test that loading nonexistent file raises error."""
        with pytest.raises(ConfigurationError):
            Settings.load(Path("/nonexistent/config.yaml"))

    def test_load_with_env_vars(self, tmp_storage_dir):
        """Test loading config with environment variable substitution."""
        os.environ["TEST_API_KEY"] = "secret123"

        config_file = tmp_storage_dir / "test-env.yaml"
        config_file.write_text("""
version: "1.0"
llm:
  provider: "anthropic"
  model: "claude-3"
  api_key: "${TEST_API_KEY}"
embedding:
  provider: "voyage"
  model: "voyage-3"
  api_key: "test-key"
""")

        settings = Settings.load(config_file)
        assert settings.llm.api_key == "secret123"

        del os.environ["TEST_API_KEY"]

    def test_validate_completeness(self, sample_settings):
        """Test configuration completeness validation."""
        # Valid config should pass
        sample_settings.validate_completeness()

        # Invalid config with unresolved env var should fail
        sample_settings.llm.api_key = "${UNRESOLVED_VAR}"
        with pytest.raises(ConfigurationError):
            sample_settings.validate_completeness()
