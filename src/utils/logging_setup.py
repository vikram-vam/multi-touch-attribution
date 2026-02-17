"""
Structured logging configuration using loguru.
Provides consistent logging across all modules.
"""

import sys
from loguru import logger


def setup_logging(level: str = "INFO", log_file: str = None):
    """
    Configure loguru logger for the application.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file path for log output.
    """
    # Remove default handler
    logger.remove()

    # Console handler with structured format
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler if specified
    if log_file:
        logger.add(
            log_file,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            rotation="10 MB",
            retention="7 days",
        )

    return logger


# Module-level logger instance
log = setup_logging()
