"""Validation utilities for configuration."""

import os
import re
from typing import Any, Dict

from graphunified.exceptions import ConfigurationError


def substitute_env_vars(value: Any) -> Any:
    """Recursively substitute environment variables in configuration values.

    Supports syntax:
    - ${VAR_NAME} - Required variable, raises error if not found
    - ${VAR_NAME:-default} - Optional variable with default value

    Args:
        value: Configuration value (string, dict, list, or other)

    Returns:
        Value with environment variables substituted

    Raises:
        ConfigurationError: If required environment variable is not found
    """
    if isinstance(value, str):
        # Pattern: ${VAR_NAME} or ${VAR_NAME:-default_value}
        pattern = r"\$\{([A-Z_][A-Z0-9_]*?)(?::-([^}]*))?\}"

        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            default_value = match.group(2)

            env_value = os.environ.get(var_name)

            if env_value is not None:
                return env_value
            elif default_value is not None:
                return default_value
            else:
                raise ConfigurationError(
                    f"Environment variable ${{{var_name}}} is required but not set"
                )

        return re.sub(pattern, replacer, value)

    elif isinstance(value, dict):
        return {k: substitute_env_vars(v) for k, v in value.items()}

    elif isinstance(value, list):
        return [substitute_env_vars(item) for item in value]

    else:
        return value


def validate_api_key(api_key: str, provider: str) -> None:
    """Validate that an API key is non-empty and looks reasonable.

    Args:
        api_key: The API key to validate
        provider: The provider name (for error messages)

    Raises:
        ConfigurationError: If API key is invalid
    """
    if not api_key or not api_key.strip():
        raise ConfigurationError(f"{provider} API key cannot be empty")

    if api_key.startswith("${"):
        raise ConfigurationError(
            f"{provider} API key contains unresolved environment variable: {api_key}"
        )


def validate_path(path: str, name: str = "path") -> None:
    """Validate that a path is non-empty and doesn't contain suspicious patterns.

    Args:
        path: The path to validate
        name: Name of the path field (for error messages)

    Raises:
        ConfigurationError: If path is invalid
    """
    if not path or not path.strip():
        raise ConfigurationError(f"{name} cannot be empty")

    # Check for null bytes
    if "\0" in path:
        raise ConfigurationError(f"{name} contains null byte")
