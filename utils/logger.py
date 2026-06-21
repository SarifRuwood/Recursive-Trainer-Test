"""
Logging utilities for consistent logging across the project.
"""

import logging
import os
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_dir: str = 'logs',
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Setup a logger with both console and file handlers.
    
    Args:
        name: Logger name (typically __name__)
        log_dir: Directory to save log files
        level: Logging level (logging.INFO, DEBUG, etc.)
    
    Returns:
        Configured logger instance
    """
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_format)
    
    # File handler
    log_file = os.path.join(log_dir, f'{name.replace(".", "_")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get an existing logger or create a new one.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        setup_logger(name)
    return logger
