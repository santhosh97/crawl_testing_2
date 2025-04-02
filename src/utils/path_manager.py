"""
Path management utilities for the GitHub Stars Crawler.

This module provides centralized path management for application directories
including logs, metrics, and cache directories.
"""
import os
from pathlib import Path
from typing import Optional


class PathManager:
    """Manages paths for logs, metrics, and other application directories.
    
    This class centralizes path management to avoid scattered directory
    creation and access across the codebase.
    """
    
    def __init__(self, base_dir: Optional[Path] = None):
        """Initialize the path manager.
        
        Args:
            base_dir: Optional base directory for all paths. If not provided,
                     defaults to the current working directory.
        """
        self.base_dir = base_dir or Path(".")
        self._logs_dir = None
        self._metrics_dir = None
        self._cache_dir = None
        
    def get_logs_dir(self) -> Path:
        """Get the logs directory path.
        
        Returns:
            Path object for the logs directory
        """
        if self._logs_dir is None:
            self._logs_dir = self.base_dir / "logs"
            self._logs_dir.mkdir(exist_ok=True)
        return self._logs_dir
    
    def get_metrics_dir(self) -> Path:
        """Get the metrics directory path.
        
        Returns:
            Path object for the metrics directory
        """
        if self._metrics_dir is None:
            self._metrics_dir = self.get_logs_dir() / "metrics"
            self._metrics_dir.mkdir(exist_ok=True)
        return self._metrics_dir
        
    def get_cache_dir(self) -> Path:
        """Get the cache directory path.
        
        Returns:
            Path object for the cache directory
        """
        if self._cache_dir is None:
            self._cache_dir = self.base_dir / "cache"
            self._cache_dir.mkdir(exist_ok=True)
        return self._cache_dir