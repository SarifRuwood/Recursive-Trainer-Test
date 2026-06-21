"""
Data utility functions for file and directory management.
"""

import os
from pathlib import Path
from typing import Optional


def ensure_dir(directory: str) -> str:
    """
    Ensure a directory exists, creating it if necessary.
    
    Args:
        directory: Path to directory
    
    Returns:
        Absolute path to directory
    """
    Path(directory).mkdir(parents=True, exist_ok=True)
    return os.path.abspath(directory)


def create_data_dir(base_dir: str = 'data') -> str:
    """
    Create the data directory structure.
    
    Args:
        base_dir: Base data directory
    
    Returns:
        Path to data directory
    """
    subdirs = ['historical', 'processed', 'cache']
    data_path = ensure_dir(base_dir)
    
    for subdir in subdirs:
        ensure_dir(os.path.join(data_path, subdir))
    
    return data_path


def get_cache_path(filename: str, cache_dir: str = 'data/cache') -> str:
    """
    Get the cache file path for a given filename.
    
    Args:
        filename: Name of the file
        cache_dir: Cache directory
    
    Returns:
        Full path to cache file
    """
    ensure_dir(cache_dir)
    return os.path.join(cache_dir, filename)
