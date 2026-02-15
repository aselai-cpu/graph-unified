"""Logging configuration for graph-unified."""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

from graphunified.config.settings import LoggingConfig


class JSONFormatter(logging.Formatter):
    """Format logs as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_data)


def setup_logging(config: Optional[LoggingConfig] = None) -> logging.Logger:
    """Setup logging configuration.

    Args:
        config: Logging configuration (if None, uses defaults)

    Returns:
        Root logger instance
    """
    if config is None:
        config = LoggingConfig()

    # Create logger
    logger = logging.getLogger("graphunified")
    logger.setLevel(getattr(logging, config.level))

    # Remove existing handlers
    logger.handlers.clear()

    # Create formatter
    if config.format == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    # Setup handlers based on output config
    if config.output in ("stdout", "both"):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if config.output in ("file", "both"):
        # Create log directory if needed
        log_file = Path(config.file_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a specific module.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(f"graphunified.{name}")
