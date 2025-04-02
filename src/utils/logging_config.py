"""Logging configuration for the GitHub Stars Crawler."""

import logging
import os
import sys
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any, Union

class LogManager:
    """Manages logging configuration without global state."""
    
    def __init__(self, log_level: int = logging.INFO, 
                 logs_dir: Optional[Path] = None,
                 metrics_dir: Optional[Path] = None,
                 console: bool = True, 
                 enable_debug_file: bool = True):
        """Initialize the log manager.
        
        Args:
            log_level: Logging level (e.g., logging.INFO)
            logs_dir: Path to logs directory
            metrics_dir: Path to metrics directory
            console: Whether to enable console logging
            enable_debug_file: Whether to enable debug file logging
        """
        self.log_level = log_level
        self.console_enabled = console
        self.debug_file_enabled = enable_debug_file
        
        # Set log directories
        self.logs_dir = logs_dir or Path("logs")
        self.metrics_dir = metrics_dir or self.logs_dir / "metrics"
            
        # Create log directories if they don't exist
        self._create_log_dirs()
        
        # Configure logging
        self._configure_logging()
        
        # Store loggers
        self._loggers = {}
        
        # Log debug info
        self.get_logger(__name__).debug(f"Log manager initialized with log_level={self.log_level}, "
                            f"logs_dir={self.logs_dir}, metrics_dir={self.metrics_dir}")
        
    def _create_log_dirs(self):
        """Create log directories if they don't exist."""
        self.logs_dir.mkdir(exist_ok=True)
        self.metrics_dir.mkdir(exist_ok=True, parents=True)
        
    def _configure_logging(self):
        """Configure logging with file and console handlers."""
        # Reset existing logging configuration
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
            
        # Configure root logger
        logging.root.setLevel(self.log_level)
        
        # Create formatters
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Add console handler if enabled
        if self.console_enabled:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(self.log_level)
            logging.root.addHandler(console_handler)
            
        # Add file handlers
        self._add_file_handler("github_stars.log", file_formatter, self.log_level)
        self._add_file_handler("github_api.log", file_formatter, self.log_level, 
                              ["src.api"])
        self._add_file_handler("database.log", file_formatter, self.log_level, 
                              ["src.db"])
        self._add_file_handler("bandit_learning.log", file_formatter, self.log_level, 
                              ["src.core.query_bandit"])
        self._add_file_handler("bandit_analytics.log", file_formatter, self.log_level,
                             ["src.core.query_evolution"])
        
        # Add debug file handler if enabled
        if self.debug_file_enabled:
            self._add_file_handler("github_stars_debug.log", file_formatter, logging.DEBUG)
            
    def _add_file_handler(self, filename: str, formatter: logging.Formatter, 
                         level: int, logger_names: Optional[List[str]] = None):
        """Add a file handler to specific loggers or root logger.
        
        Args:
            filename: Log filename
            formatter: Log formatter
            level: Log level
            logger_names: Optional list of logger names to add handler to
        """
        handler = logging.FileHandler(self.logs_dir / filename)
        handler.setFormatter(formatter)
        handler.setLevel(level)
        
        if logger_names:
            # Add to specific loggers
            for logger_name in logger_names:
                logger = logging.getLogger(logger_name)
                logger.addHandler(handler)
                logger.setLevel(min(level, logger.level))
        else:
            # Add to root logger
            logging.root.addHandler(handler)
            
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger with the specified name.
        
        Args:
            name: Logger name (usually __name__)
            
        Returns:
            Logger instance
        """
        if name not in self._loggers:
            self._loggers[name] = logging.getLogger(name)
            
        return self._loggers[name]
        
    def set_log_level(self, level: int):
        """Set the log level for all loggers.
        
        Args:
            level: New log level
        """
        self.log_level = level
        
        # Update root logger
        logging.root.setLevel(level)
        
        # Update all handlers
        for handler in logging.root.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(level)
                
        # Update all loggers
        for logger in self._loggers.values():
            logger.setLevel(level)