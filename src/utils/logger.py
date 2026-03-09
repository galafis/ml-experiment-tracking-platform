# -*- coding: utf-8 -*-
"""
Centralized logging configuration for ML Experiment Tracking Platform.

Provides structured logging with configurable levels, formatters,
and handlers for consistent observability across all platform modules.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(module)s:%(lineno)d | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_JSON_FORMAT = (
    '{{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
    '"logger": "%(name)s", "module": "%(module)s", '
    '"line": %(lineno)d, "message": "%(message)s"}}'
)


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    json_format: bool = False,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """Create or retrieve a configured logger instance.

    Parameters
    ----------
    name:
        Logger name, typically ``__name__`` of the calling module.
    level:
        Minimum logging level (default ``INFO``).
    log_file:
        Optional file path for rotating file handler.
    json_format:
        If ``True``, use JSON-structured log lines.
    max_bytes:
        Maximum size per log file before rotation (default 10 MB).
    backup_count:
        Number of rotated log files to keep (default 5).

    Returns
    -------
    logging.Logger
        Configured logger ready for use.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    fmt_string = _JSON_FORMAT if json_format else _LOG_FORMAT
    formatter = logging.Formatter(fmt=fmt_string, datefmt=_DATE_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Rotating file handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=str(log_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def configure_root_logger(
    level: int = logging.INFO,
    json_format: bool = False,
) -> None:
    """Configure the root logger for the entire platform.

    Call once at application startup to set consistent defaults.
    """
    root = logging.getLogger()
    root.setLevel(level)

    if root.handlers:
        root.handlers.clear()

    fmt_string = _JSON_FORMAT if json_format else _LOG_FORMAT
    formatter = logging.Formatter(fmt=fmt_string, datefmt=_DATE_FORMAT)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    root.addHandler(handler)


class LogContext:
    """Context manager that temporarily adjusts a logger's level."""

    def __init__(self, logger: logging.Logger, level: int) -> None:
        self._logger = logger
        self._new_level = level
        self._old_level = logger.level

    def __enter__(self) -> logging.Logger:
        self._logger.setLevel(self._new_level)
        return self._logger

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._logger.setLevel(self._old_level)
