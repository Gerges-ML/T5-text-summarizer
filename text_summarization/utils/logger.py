"""
utils/logger.py
---------------
Centralised logging factory for the Text Summarization project.

Call ``get_logger(__name__)`` at the top of every module instead of using
``print`` statements. Configuration is read from ``config.LoggingConfig`` so
the log level can be changed at runtime via the ``LOG_LEVEL`` environment
variable without touching source code.

Author: Gerges Emad
"""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

from config import LOGGING_CONFIG


_CONFIGURED: bool = False


def _configure_root_logger() -> None:
    """Set up the root logger once; subsequent calls are no-ops."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    cfg = LOGGING_CONFIG
    level = getattr(logging, cfg.level.upper(), logging.INFO)

    formatter = logging.Formatter(fmt=cfg.format, datefmt=cfg.date_format)

    # --- console handler ---------------------------------------------------
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console_handler)

    # --- optional file handler ---------------------------------------------
    if cfg.log_to_file:
        log_dir = os.path.dirname(cfg.log_file_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(
            cfg.log_file_path,
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Return a named logger with the project-wide configuration applied.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.  If ``None``, the
        root logger is returned.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Examples
    --------
    >>> from utils.logger import get_logger
    >>> logger = get_logger(__name__)
    >>> logger.info("Model loaded successfully.")
    """
    _configure_root_logger()
    return logging.getLogger(name)
