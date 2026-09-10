"""
Logging configuration for PanoGS.
Provides clean, structured console output and optional file logging.
"""

import logging
import sys
from typing import Optional


class Formatter(logging.Formatter):
    """Clean formatter with color-aware prefixing when supported."""

    GREY = "\x1b[38;20m"
    CYAN = "\x1b[36;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    BOLD_RED = "\x1b[31;1m"
    RESET = "\x1b[0m"

    FORMAT_PREFIX = "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s"
    DATE_FORMAT = "%H:%M:%S"

    def format(self, record: logging.LogRecord) -> str:
        log_fmt = self.FORMAT_PREFIX
        formatter = logging.Formatter(log_fmt, datefmt=self.DATE_FORMAT)
        return formatter.format(record)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure root logger for PanoGS."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root_logger = logging.getLogger("panogs")
    root_logger.setLevel(numeric_level)

    # Avoid duplicate handlers if setup_logging is called multiple times
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(Formatter())
    root_logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] (%(filename)s:%(lineno)d): %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str = "panogs") -> logging.Logger:
    """Get a scoped logger under the 'panogs' namespace."""
    if not name.startswith("panogs"):
        return logging.getLogger(f"panogs.{name}")
    return logging.getLogger(name)
