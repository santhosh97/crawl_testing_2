"""
Configuration module for GitHub Stars Crawler.

This module provides centralized configuration with environment variable
support and sensible defaults for all crawler components.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Union

# Default values
DEFAULT_CONFIG = {
    # Database settings
    "database": {
        "url": "sqlite:///github_stars.db",
        "backup_dir": "./",
        "backup_interval_hours": 6,
        "max_connections": 10,
        "timeout_seconds": 30,
    },
    
    # GitHub API settings
    "github_api": {
        "graphql_url": "https://api.github.com/graphql",
        "rest_base_url": "https://api.github.com",
        "max_workers": "auto",  # Auto-detect based on tokens
        "requests_per_hour": 4500,  # Conservative limit
        "batch_size": 100,  # Repositories per batch
    },
    
    # Cache settings
    "cache": {
        "enabled": True,
        "max_size": 10000,
        "default_ttl": 1800,  # 30 minutes
        "strategy": "hybrid",  # LRU, LFU, or hybrid
    },
    
    # Query settings
    "query": {
        "cooling_period": 1800,  # Seconds to wait before retrying exhausted queries (30 minutes)
        "exploration_weight": 1.0,  # Exploration weight for UCB algorithm
        "duplication_threshold": 0.75,  # Duplication rate threshold for query retirement
    },
    
    # Crawler settings
    "crawler": {
        "batch_size": 50,  # Batch size for database operations
        "total_count": 10000,  # Total repositories to crawl
        "max_retries": 3,  # Maximum retry attempts
        "db_workers": "auto",  # Auto-detect based on CPU
    },
    
    # Learning settings
    "learning": {
        "bandit_enabled": True,
        "bandit_exploration_weight": 2.0,  # Increased from 1.0 to encourage more exploration
        "bandit_cooling_factor": 0.997,    # Slower cooling to maintain exploration longer
        "similarity_threshold": 0.7,       # Reduced to consider more diverse queries
        "metrics_interval": 30,            # More frequent metrics logging (30 seconds)
        "evolve_frequency": 10,            # Evolve queries more frequently
        "evolve_count": 10,                # Generate more evolved queries
    },
    
    # Logging settings
    "logging": {
        "level": "INFO",
        "console_level": "INFO",
        "file_level": "DEBUG",
        "enable_debug_file": True,
        "enable_metrics": True,
    },
}



class Config:
    """Configuration manager for GitHub Stars Crawler."""
    
    def __init__(self, config_file: Optional[str] = None, environment=None, logger=None):
        """Initialize configuration from file and environment variables.
        
        Args:
            config_file: Optional path to configuration file
            environment: Environment instance for accessing environment variables
            logger: Logger instance
        """
        # Start with default configuration
        self._config = DEFAULT_CONFIG.copy()
        
        # Store dependencies
        self.environment = environment
        self.logger = logger
        
        # Load from config file if provided
        if config_file:
            self._load_from_file(config_file)
        
        # Override with environment variables (using injected environment if available)
        self._load_from_env()
        
        # Initialize derived settings
        self._init_derived_settings()
        
        if self.logger:
            self.logger.debug("Configuration initialized")
    
    def _load_from_file(self, config_file: str):
        """Load configuration from JSON file.
        
        Args:
            config_file: Path to configuration file
        """
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            
            # Update configuration with file values (nested update)
            self._update_nested_dict(self._config, file_config)
            
            if self.logger:
                self.logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Error loading configuration from {config_file}: {e}")
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # Get environment variables from injected environment
        if not self.environment:
            if self.logger:
                self.logger.warning("No environment instance provided, skipping environment variable loading")
            return
            
        get_env = self.environment.get
        
        # Database settings
        if db_url := get_env("DATABASE_URL"):
            self._config["database"]["url"] = db_url
        
        if backup_dir := get_env("DB_BACKUP_DIR"):
            self._config["database"]["backup_dir"] = backup_dir
        
        if backup_interval := get_env("DB_BACKUP_INTERVAL"):
            try:
                self._config["database"]["backup_interval_hours"] = int(backup_interval)
            except ValueError:
                pass
        
        # GitHub API settings
        if max_workers := get_env("MAX_WORKERS"):
            self._config["github_api"]["max_workers"] = max_workers
        
        if batch_size := get_env("API_BATCH_SIZE"):
            try:
                self._config["github_api"]["batch_size"] = int(batch_size)
            except ValueError:
                pass
        
        # Cache settings
        if cache_enabled := get_env("CACHE_ENABLED"):
            self._config["cache"]["enabled"] = cache_enabled.lower() in ("true", "1", "yes")
        
        if cache_ttl := get_env("CACHE_TTL"):
            try:
                self._config["cache"]["default_ttl"] = int(cache_ttl)
            except ValueError:
                pass
        
        # Crawler settings
        if total_count := get_env("TOTAL_COUNT"):
            try:
                self._config["crawler"]["total_count"] = int(total_count)
            except ValueError:
                pass
        
        if db_workers := get_env("MAX_DB_WORKERS"):
            self._config["crawler"]["db_workers"] = db_workers
        
        # Learning settings
        if bandit_enabled := get_env("BANDIT_ENABLED"):
            self._config["learning"]["bandit_enabled"] = bandit_enabled.lower() in ("true", "1", "yes")
        
        # Logging settings
        if log_level := get_env("LOG_LEVEL"):
            self._config["logging"]["level"] = log_level
        
        if self.logger:
            self.logger.debug("Loaded configuration from environment variables")
    
    def _update_nested_dict(self, target: Dict, source: Dict):
        """Update nested dictionary recursively.
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with new values
        """
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # Recursively update nested dictionaries
                self._update_nested_dict(target[key], value)
            else:
                # Direct update for non-dict values or new keys
                target[key] = value
    
    def _init_derived_settings(self):
        """Initialize settings derived from other configuration values."""
        # Create backup directory if it doesn't exist
        backup_dir = Path(self._config["database"]["backup_dir"])
        backup_dir.mkdir(exist_ok=True)
        
        # Convert string log levels to int if needed
        log_levels = {
            "DEBUG": 10,     # logging.DEBUG
            "INFO": 20,      # logging.INFO
            "WARNING": 30,   # logging.WARNING
            "ERROR": 40,     # logging.ERROR
            "CRITICAL": 50,  # logging.CRITICAL
        }
        
        for level_key in ["level", "console_level", "file_level"]:
            level = self._config["logging"][level_key]
            if isinstance(level, str) and level.upper() in log_levels:
                self._config["logging"][level_key] = log_levels[level.upper()]
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation path.
        
        Args:
            key_path: Dot notation path to configuration value (e.g., "database.url")
            default: Default value to return if path not found
            
        Returns:
            Configuration value or default
        """
        # Split path into parts
        parts = key_path.split('.')
        
        # Start at the root of the configuration
        value = self._config
        
        # Navigate through the path
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        
        return value
    
    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation path.
        
        Args:
            key_path: Dot notation path to configuration value (e.g., "database.url")
            value: Value to set
        """
        # Split path into parts
        parts = key_path.split('.')
        
        # Start at the root of the configuration
        config = self._config
        
        # Navigate through the path
        for i, part in enumerate(parts[:-1]):
            if part not in config:
                config[part] = {}
            elif not isinstance(config[part], dict):
                # Convert non-dict value to dict to allow nested keys
                config[part] = {}
            
            config = config[part]
        
        # Set the value at the final path
        config[parts[-1]] = value
    
    def get_all(self) -> Dict[str, Any]:
        """Get the entire configuration dictionary.
        
        Returns:
            Configuration dictionary
        """
        return self._config
    
    def save_to_file(self, file_path: str):
        """Save current configuration to a JSON file.
        
        Args:
            file_path: Path to save configuration file
        """
        try:
            with open(file_path, 'w') as f:
                json.dump(self._config, f, indent=2)
            
            if self.logger:
                self.logger.info(f"Saved configuration to {file_path}")
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error saving configuration to {file_path}: {e}")

def get_initial_queries():
    """Get initial set of queries for repository search.
    
    Returns:
        List of query configurations for starting the search.
    """
    return [
        # Popular repositories across different star ranges
        {"q": "stars:>50000", "sort": "stars", "order": "desc"},
        {"q": "stars:10000..50000", "sort": "stars", "order": "desc"},
        {"q": "stars:5000..10000", "sort": "stars", "order": "desc"},
        {"q": "stars:1000..5000", "sort": "stars", "order": "desc"},
        {"q": "stars:500..1000", "sort": "stars", "order": "desc"},
        {"q": "stars:100..500", "sort": "stars", "order": "desc"},
        
        # Popular repositories in different languages
        {"q": "language:javascript stars:>1000", "sort": "stars", "order": "desc"},
        {"q": "language:python stars:>1000", "sort": "stars", "order": "desc"},
        {"q": "language:java stars:>1000", "sort": "stars", "order": "desc"},
        {"q": "language:typescript stars:>1000", "sort": "stars", "order": "desc"},
        {"q": "language:go stars:>1000", "sort": "stars", "order": "desc"},
        {"q": "language:rust stars:>1000", "sort": "stars", "order": "desc"},
        {"q": "language:cpp stars:>1000", "sort": "stars", "order": "desc"},
        {"q": "language:php stars:>1000", "sort": "stars", "order": "desc"},
        {"q": "language:csharp stars:>1000", "sort": "stars", "order": "desc"},
        {"q": "language:ruby stars:>1000", "sort": "stars", "order": "desc"},
        
        # Recently updated repositories with stars
        {"q": "stars:>1000", "sort": "updated", "order": "desc"},
        {"q": "stars:>5000", "sort": "updated", "order": "desc"},
        {"q": "stars:>500", "sort": "updated", "order": "desc"},
        {"q": "stars:100..1000", "sort": "updated", "order": "desc"},
        
        # Time-specific queries with stars
        {"q": "stars:>10000 created:>2022-01-01", "sort": "stars", "order": "desc"},
        {"q": "stars:1000..10000 created:>2022-01-01", "sort": "stars", "order": "desc"},
        {"q": "stars:>500 created:2022-01-01..2023-01-01", "sort": "stars", "order": "desc"},
        {"q": "stars:>500 created:2021-01-01..2022-01-01", "sort": "stars", "order": "desc"},
        {"q": "stars:>500 created:2020-01-01..2021-01-01", "sort": "stars", "order": "desc"},
        
        # Combination queries (language + sort by updated)
        {"q": "stars:1000..10000 language:javascript", "sort": "updated", "order": "desc"},
        {"q": "stars:1000..10000 language:python", "sort": "updated", "order": "desc"},
        {"q": "stars:1000..10000 language:typescript", "sort": "updated", "order": "desc"},
        {"q": "stars:500..1000 language:javascript", "sort": "updated", "order": "desc"},
        {"q": "stars:500..1000 language:python", "sort": "updated", "order": "desc"},
        {"q": "stars:100..500 language:rust", "sort": "updated", "order": "desc"},
        {"q": "stars:100..500 language:go", "sort": "updated", "order": "desc"},
        
        # Topic-based queries
        {"q": "topic:machine-learning stars:>500", "sort": "stars", "order": "desc"},
        {"q": "topic:web-development stars:>500", "sort": "stars", "order": "desc"},
        {"q": "topic:data-science stars:>500", "sort": "stars", "order": "desc"},
        {"q": "topic:artificial-intelligence stars:>500", "sort": "stars", "order": "desc"},
        {"q": "topic:blockchain stars:>500", "sort": "stars", "order": "desc"},
        {"q": "topic:game-development stars:>500", "sort": "stars", "order": "desc"},
        {"q": "topic:mobile stars:>500", "sort": "stars", "order": "desc"},
        {"q": "topic:frontend stars:>500", "sort": "stars", "order": "desc"},
        {"q": "topic:backend stars:>500", "sort": "stars", "order": "desc"},
        {"q": "topic:devops stars:>500", "sort": "stars", "order": "desc"},
        
        # Ascending order to discover different repositories
        {"q": "stars:1000..5000", "sort": "stars", "order": "asc"},
        {"q": "stars:100..1000", "sort": "stars", "order": "asc"},
        {"q": "stars:>1000 language:javascript", "sort": "stars", "order": "asc"},
        {"q": "stars:>1000 language:python", "sort": "stars", "order": "asc"},
        
        # License-specific queries
        {"q": "license:mit stars:>1000", "sort": "stars", "order": "desc"},
        {"q": "license:apache-2.0 stars:>1000", "sort": "stars", "order": "desc"},
        {"q": "license:gpl-3.0 stars:>1000", "sort": "stars", "order": "desc"},
        
        # Fork-based queries
        {"q": "stars:>1000", "sort": "forks", "order": "desc"},
        {"q": "stars:>500 language:javascript", "sort": "forks", "order": "desc"},
        {"q": "stars:>500 language:python", "sort": "forks", "order": "desc"},
    ]