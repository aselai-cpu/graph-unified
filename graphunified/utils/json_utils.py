"""Shared JSON parsing utilities.

Provides utilities for extracting and parsing JSON from LLM responses,
handling markdown code blocks and other formatting variations.
"""

import json
from typing import Any, Dict


def extract_json(text: str) -> str:
    """Extract JSON from text, handling markdown code blocks.

    Args:
        text: Text that may contain JSON, possibly wrapped in markdown ```json...```

    Returns:
        Clean JSON string ready for parsing

    Example:
        >>> text = "```json\\n{\"key\": \"value\"}\\n```"
        >>> extract_json(text)
        '{"key": "value"}'
    """
    text = text.strip()

    # Remove markdown code block markers
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```)
        lines = lines[1:]
        # Remove last line (```)
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    return text


def parse_json_response(text: str) -> Dict[str, Any]:
    """Parse JSON from LLM response text.

    Args:
        text: LLM response that may contain JSON

    Returns:
        Parsed JSON dictionary

    Raises:
        json.JSONDecodeError: If JSON cannot be parsed
    """
    json_text = extract_json(text)
    return json.loads(json_text)
