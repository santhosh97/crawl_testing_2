"""
Exceptions for GitHub Stars Crawler.

This module contains common exceptions used throughout the codebase.
All components should use these exception classes for consistency.
"""

# Base exceptions for different components
class GitHubException(Exception):
    """Base exception for all GitHub-related errors."""
    pass


class DatabaseException(Exception):
    """Base exception for all database-related errors."""
    pass


class CacheException(Exception):
    """Base exception for all cache-related errors."""
    pass


class ConfigException(Exception):
    """Base exception for all configuration-related errors."""
    pass


class ApplicationException(Exception):
    """Base exception for application-level errors."""
    pass


# GitHub API exceptions
class GitHubRateLimitError(GitHubException):
    """Exception raised when GitHub API rate limit is exceeded."""
    pass


class GitHubAPIError(GitHubException):
    """Exception raised when GitHub API returns an error."""
    pass


class GitHubAuthenticationError(GitHubException):
    """Exception raised when authentication to GitHub API fails."""
    pass


class GitHubServerError(GitHubException):
    """Exception raised when GitHub API returns a 5xx status code."""
    pass


class GitHubNetworkError(GitHubException):
    """Exception raised when network connection to GitHub API fails."""
    pass


class GitHubQueryError(GitHubException):
    """Exception raised when a query is invalid or fails."""
    pass


# Token management exceptions
class TokenManagementError(GitHubException):
    """Exception raised for token management errors."""
    pass


# Database exceptions
class DatabaseConnectionError(DatabaseException):
    """Exception raised when connection to database fails."""
    pass


class DatabaseQueryError(DatabaseException):
    """Exception raised when a database query fails."""
    pass


class DatabaseInitError(DatabaseException):
    """Exception raised when database initialization fails."""
    pass


# Cache exceptions
class CacheInitError(CacheException):
    """Exception raised when cache initialization fails."""
    pass


class CacheFullError(CacheException):
    """Exception raised when cache is full and can't evict items."""
    pass


# Config exceptions
class ConfigurationError(ConfigException):
    """Exception raised when there's an error in configuration."""
    pass


class MissingConfigError(ConfigException):
    """Exception raised when a required configuration value is missing."""
    pass


# Application exceptions
class InitializationError(ApplicationException):
    """Exception raised when component initialization fails."""
    pass


class ResourceCleanupError(ApplicationException):
    """Exception raised when resource cleanup fails."""
    pass


class TransientError(Exception):
    """Base exception for errors that are temporary and can be retried."""
    pass