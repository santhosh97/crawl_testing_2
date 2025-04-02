"""
Interfaces module for GitHub Stars Crawler.

This module defines protocol classes that represent interfaces for different
components of the application. Using these protocols promotes loose coupling
and helps avoid circular dependencies between modules.
"""

from typing import Dict, List, Optional, Any, Protocol, Set, Union, Iterator


class ITokenManager(Protocol):
    """Interface for token management components."""
    
    def get_token(self, high_priority: bool = False) -> str:
        """Get the best available token based on rate limits."""
        ...
    
    def update_token_usage(self, token: str, **kwargs) -> bool:
        """Update token usage statistics and parse rate limit headers."""
        ...
    
    def mark_token_depleted(self, token: str, **kwargs) -> bool:
        """Mark a token as depleted (0 remaining requests)."""
        ...
    
    def get_token_stats(self) -> Dict[str, Any]:
        """Get statistics about all tokens."""
        ...


class ICacheManager(Protocol):
    """Interface for cache management components."""
    
    def get_cached_query_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached query result if available and not expired."""
        ...
    
    def cache_query_result(self, cache_key: str, result: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """Cache a query result with optional TTL."""
        ...
    
    def is_duplicate_repository(self, repo_id: str) -> bool:
        """Check if repository has been seen before."""
        ...
    
    def mark_repository_seen(self, repo_id: str) -> None:
        """Mark repository as seen in the tracking set."""
        ...
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics."""
        ...


class IMetricsCollector(Protocol):
    """Interface for metrics collection components."""
    
    def update_api_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update API-related metrics."""
        ...
    
    def update_cache_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update cache-related metrics."""
        ...
    
    def update_repository_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update repository-related metrics."""
        ...
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all collected metrics."""
        ...
    
    def reset_stats(self) -> None:
        """Reset all metrics."""
        ...
    
    def set_global_target_count(self, count: int) -> None:
        """Set the global target count for repositories."""
        ...
    
    def get_global_target_count(self) -> int:
        """Get the global target count for repositories."""
        ...


class IQueryPool(Protocol):
    """Interface for query pool components."""
    
    def get_best_query(self, avoiding_queries=None, strategy="auto"):
        """Get the best query based on bandit algorithm."""
        ...
    
    def update_query_performance(self, query, results_count, unique_count, **kwargs):
        """Update performance metrics for a query."""
        ...
    
    def update_context(self, metrics_collector, target_count: int) -> None:
        """Update context features for contextual bandits."""
        ...
    
    def get_top_performing_queries(self, count: int = 5):
        """Get the top performing queries based on reward."""
        ...


class IConnectionManager(Protocol):
    """Interface for connection management components."""
    
    def get_session(self, identifier: str) -> Any:
        """Get a session for the given identifier."""
        ...
    
    def clear_session(self, identifier: str) -> None:
        """Clear a session for the given identifier."""
        ...
    
    def clear_all_sessions(self) -> None:
        """Clear all sessions."""
        ...


class IConfig(Protocol):
    """Interface for configuration components."""
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        ...
    
    def set(self, key: str, value: Any) -> None:
        """Set a configuration value."""
        ...
    
    def get_path(self, key: str, default: Optional[str] = None) -> str:
        """Get a configuration value as a path."""
        ...