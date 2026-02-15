"""Tests for .env file loading in configuration."""

import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from graphunified.config.settings import Settings
from graphunified.exceptions import ConfigurationError


@pytest.fixture(autouse=True)
def clean_test_env_vars(monkeypatch: pytest.MonkeyPatch):
    """Clean up test environment variables before each test."""
    # Remove any test env vars that might be left over
    test_vars = ["TEST_ANTHROPIC_KEY", "TEST_VOYAGE_KEY", "MISSING_API_KEY"]
    for var in test_vars:
        monkeypatch.delenv(var, raising=False)


def test_env_file_loading_from_config_dir(tmp_path: Path):
    """Test that .env file is loaded from config directory."""
    # Create .env file in config directory
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_ANTHROPIC_KEY=sk-ant-test-123\nTEST_VOYAGE_KEY=pa-test-456\n")

    # Create config file
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(
        """
version: "1.0"
llm:
  provider: anthropic
  model: claude-3-5-sonnet-20241022
  api_key: ${TEST_ANTHROPIC_KEY}
  temperature: 0.0
  max_tokens: 4096
  timeout: 60
embedding:
  provider: voyage
  model: voyage-3
  api_key: ${TEST_VOYAGE_KEY}
  dimension: 1024
  batch_size: 128
  normalize: true
"""
    )

    # Load settings (should auto-load .env)
    settings = Settings.load(config_file)

    assert settings.llm.api_key == "sk-ant-test-123"
    assert settings.embedding.api_key == "pa-test-456"


def test_env_file_loading_from_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test that .env file is loaded from current working directory."""
    # Create .env in CWD
    cwd_dir = tmp_path / "cwd"
    cwd_dir.mkdir()
    env_file = cwd_dir / ".env"
    env_file.write_text("TEST_ANTHROPIC_KEY=sk-ant-cwd-123\nTEST_VOYAGE_KEY=pa-cwd-456\n")

    # Create config in subdirectory (without .env)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_file = config_dir / "settings.yaml"
    config_file.write_text(
        """
version: "1.0"
llm:
  provider: anthropic
  model: claude-3-5-sonnet-20241022
  api_key: ${TEST_ANTHROPIC_KEY}
  temperature: 0.0
  max_tokens: 4096
  timeout: 60
embedding:
  provider: voyage
  model: voyage-3
  api_key: ${TEST_VOYAGE_KEY}
  dimension: 1024
  batch_size: 128
  normalize: true
"""
    )

    # Change to CWD
    monkeypatch.chdir(cwd_dir)

    # Load settings (should find .env in CWD)
    settings = Settings.load(config_file)

    assert settings.llm.api_key == "sk-ant-cwd-123"
    assert settings.embedding.api_key == "pa-cwd-456"


def test_existing_env_vars_not_overridden(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test that existing environment variables are not overridden by .env."""
    # Set environment variable
    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "sk-ant-env-original")

    # Create .env file with different value
    env_file = tmp_path / ".env"
    env_file.write_text(
        "TEST_ANTHROPIC_KEY=sk-ant-env-overwrite\nTEST_VOYAGE_KEY=pa-test-456\n"
    )

    # Create config file
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(
        """
version: "1.0"
llm:
  provider: anthropic
  model: claude-3-5-sonnet-20241022
  api_key: ${TEST_ANTHROPIC_KEY}
  temperature: 0.0
  max_tokens: 4096
  timeout: 60
embedding:
  provider: voyage
  model: voyage-3
  api_key: ${TEST_VOYAGE_KEY}
  dimension: 1024
  batch_size: 128
  normalize: true
"""
    )

    # Load settings
    settings = Settings.load(config_file)

    # Should use original env var, not .env value
    assert settings.llm.api_key == "sk-ant-env-original"
    # Should use .env value for var not in environment
    assert settings.embedding.api_key == "pa-test-456"


def test_missing_env_var_raises_error(tmp_path: Path):
    """Test that missing required environment variable raises error."""
    # Create config without .env file
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(
        """
version: "1.0"
llm:
  provider: anthropic
  model: claude-3-5-sonnet-20241022
  api_key: ${MISSING_API_KEY}
  temperature: 0.0
  max_tokens: 4096
  timeout: 60
embedding:
  provider: voyage
  model: voyage-3
  api_key: pa-test-123
  dimension: 1024
  batch_size: 128
  normalize: true
"""
    )

    with pytest.raises(ConfigurationError, match="MISSING_API_KEY.*required but not set"):
        Settings.load(config_file)


def test_env_file_with_default_values(tmp_path: Path):
    """Test environment variables with default values."""
    # Create .env with only one key
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_ANTHROPIC_KEY=sk-ant-test-123\n")

    # Create config with default for missing var
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(
        """
version: "1.0"
llm:
  provider: anthropic
  model: claude-3-5-sonnet-20241022
  api_key: ${TEST_ANTHROPIC_KEY}
  temperature: 0.0
  max_tokens: 4096
  timeout: 60
embedding:
  provider: voyage
  model: voyage-3
  api_key: ${TEST_VOYAGE_KEY:-pa-default-key}
  dimension: 1024
  batch_size: 128
  normalize: true
"""
    )

    settings = Settings.load(config_file)

    assert settings.llm.api_key == "sk-ant-test-123"
    assert settings.embedding.api_key == "pa-default-key"  # Uses default


def test_local_embeddings_no_api_key_needed(tmp_path: Path):
    """Test that local embeddings don't require API key."""
    # Create .env with only LLM key
    env_file = tmp_path / ".env"
    env_file.write_text("TEST_ANTHROPIC_KEY=sk-ant-test-123\n")

    # Create config for local embeddings
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(
        """
version: "1.0"
llm:
  provider: anthropic
  model: claude-3-5-sonnet-20241022
  api_key: ${TEST_ANTHROPIC_KEY}
  temperature: 0.0
  max_tokens: 4096
  timeout: 60
embedding:
  provider: local
  model: BAAI/bge-large-en-v1.5
  api_key: ""
  dimension: 1024
  batch_size: 32
  normalize: true
"""
    )

    settings = Settings.load(config_file)

    assert settings.llm.api_key == "sk-ant-test-123"
    assert settings.embedding.provider == "local"
    assert settings.embedding.api_key == ""  # Empty string is OK for local


def test_no_env_file_uses_system_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Test that system environment variables work without .env file."""
    # Set system env vars (no .env file)
    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "sk-ant-system-123")
    monkeypatch.setenv("TEST_VOYAGE_KEY", "pa-system-456")

    # Create config file
    config_file = tmp_path / "settings.yaml"
    config_file.write_text(
        """
version: "1.0"
llm:
  provider: anthropic
  model: claude-3-5-sonnet-20241022
  api_key: ${TEST_ANTHROPIC_KEY}
  temperature: 0.0
  max_tokens: 4096
  timeout: 60
embedding:
  provider: voyage
  model: voyage-3
  api_key: ${TEST_VOYAGE_KEY}
  dimension: 1024
  batch_size: 128
  normalize: true
"""
    )

    # Load settings (no .env file, should use system env vars)
    settings = Settings.load(config_file)

    assert settings.llm.api_key == "sk-ant-system-123"
    assert settings.embedding.api_key == "pa-system-456"
