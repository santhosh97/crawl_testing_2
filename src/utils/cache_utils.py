"""
Cache utilities for the GitHub Stars Crawler.

This module provides cache implementation and management classes
for efficient caching of GitHub API queries and other data with
TTL-based expiration and different eviction strategies.
"""
import time
import threading
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Set, Union, TypeVar, Generic, Callable
from collections import OrderedDict
from datetime import datetime
from abc import ABC, abstractmethod

from src.utils.path_manager import PathManager

# Type variable for generic cache value
T = TypeVar('T')


class Cache(Generic[T], ABC):
    """Abstract base class for a cache implementation."""
    
    @abstractmethod
    def get(self, key: str) -> Optional[T]:
        """Get a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            The cached value or None if not found or expired
        """
        pass
        
    @abstractmethod
    def put(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """Add or update a key-value pair in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
            ttl: Optional time-to-live in seconds
        """
        pass
        
    @abstractmethod
    def remove(self, key: str) -> bool:
        """Remove a specific key from the cache.
        
        Args:
            key: The key to remove
            
        Returns:
            True if the key was removed, False if it didn't exist
        """
        pass
        
    @abstractmethod
    def clear(self) -> None:
        """Clear all entries from the cache."""
        pass
        
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics.
        
        Returns:
            Dict[str, Any]: Dictionary containing cache metrics including
            hits, misses, hit_rate, size, capacity, utilization, and evictions
        """
        pass
        
    @abstractmethod
    def cleanup_expired(self) -> int:
        """Remove all expired entries from the cache.
        
        Returns:
            Number of entries removed
        """
        pass
        
    @abstractmethod
    def __len__(self) -> int:
        """Get the number of entries in the cache."""
        pass
        
class SmartCache(Cache[T]):
    """A smarter caching system with LRU and LFU eviction strategies.
    
    Features:
    - LRU (Least Recently Used) eviction
    - LFU (Least Frequently Used) eviction
    - Hybrid mode that combines both strategies
    - TTL-based expiration
    - Thread-safe operations
    - Cache statistics tracking
    """
    
    # Eviction strategy constants
    LRU = "lru"
    LFU = "lfu"
    HYBRID = "hybrid"
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 600, 
                 strategy: str = "hybrid", hybrid_weight: float = 0.5,
                 logger = None):
        """Initialize the smart cache.
        
        Args:
            max_size: Maximum number of entries in the cache
            default_ttl: Default time-to-live in seconds
            strategy: Eviction strategy ('lru', 'lfu', or 'hybrid')
            hybrid_weight: Weight for LRU vs LFU in hybrid mode (0.5 = equal weight)
            logger: Optional logger instance
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.logger = logger
        
        # Normalize strategy name
        strategy = strategy.lower()
        if strategy not in [self.LRU, self.LFU, self.HYBRID]:
            if self.logger:
                self.logger.warning(f"Unknown cache strategy '{strategy}', using 'hybrid'")
            strategy = self.HYBRID
            
        self.strategy = strategy
        self.hybrid_weight = hybrid_weight
        
        # Main cache storage (keeps insertion order for LRU)
        self._cache = OrderedDict()
        
        # Access frequency counter for LFU
        self._access_count = {}
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Cache metrics
        self._metrics = {
            "hits": 0,
            "misses": 0,
            "puts": 0,
            "evictions": 0,
            "expirations": 0,
            "strategy": strategy
        }
        
        if self.logger:
            self.logger.debug(f"Initialized SmartCache with size={max_size}, ttl={default_ttl}s, strategy='{strategy}'")
    
    def get(self, key: str) -> Optional[T]:
        """Get a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            The cached value or None if not found or expired
        """
        with self._lock:
            # Check if key exists
            if key not in self._cache:
                self._metrics["misses"] += 1
                return None
            
            # Get entry
            entry = self._cache[key]
            current_time = time.time()
            
            # Check if expired
            if entry["expires_at"] < current_time:
                # Remove expired entry
                del self._cache[key]
                if key in self._access_count:
                    del self._access_count[key]
                self._metrics["expirations"] += 1
                self._metrics["misses"] += 1
                return None
            
            # Update access count for LFU
            self._access_count[key] = self._access_count.get(key, 0) + 1
            
            # Move to end for LRU (most recently used)
            self._cache.move_to_end(key)
            
            # Record hit
            self._metrics["hits"] += 1
            
            # Return the value
            return entry["value"]
    
    def put(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """Add or update a key-value pair in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
            ttl: Optional time-to-live in seconds (defaults to cache default)
        """
        with self._lock:
            # Use default TTL if not specified
            if ttl is None:
                ttl = self.default_ttl
            
            # Evict if necessary before adding new entry
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_entries(1)
            
            # Add or update entry
            current_time = time.time()
            self._cache[key] = {
                "value": value,
                "added_at": current_time,
                "expires_at": current_time + ttl
            }
            
            # Initialize access count for new entries
            if key not in self._access_count:
                self._access_count[key] = 1
            
            # Move to end for LRU (most recently used)
            self._cache.move_to_end(key)
            
            # Record put
            self._metrics["puts"] += 1
    
    def _evict_entries(self, count: int = 1) -> None:
        """Evict entries from cache based on the selected strategy.
        
        Args:
            count: Number of entries to evict
        """
        if not self._cache:
            return
            
        if self.strategy == self.LRU:
            self._evict_lru(count)
        elif self.strategy == self.LFU:
            self._evict_lfu(count)
        else:  # Hybrid strategy
            # Divide evictions between LRU and LFU based on weight
            lru_count = int(count * self.hybrid_weight)
            lfu_count = count - lru_count
            
            # Perform evictions
            if lru_count > 0:
                self._evict_lru(lru_count)
            if lfu_count > 0:
                self._evict_lfu(lfu_count)
    
    def _evict_lru(self, count: int = 1) -> None:
        """Evict least recently used entries.
        
        Args:
            count: Number of entries to evict
        """
        # The OrderedDict maintains insertion order
        # We remove from the beginning (least recently used)
        for _ in range(min(count, len(self._cache))):
            # Get the first key (least recently used)
            key, _ = self._cache.popitem(last=False)
            
            # Also remove from access count
            if key in self._access_count:
                del self._access_count[key]
            
            # Record eviction
            self._metrics["evictions"] += 1
    
    def _evict_lfu(self, count: int = 1) -> None:
        """Evict least frequently used entries.
        
        Args:
            count: Number of entries to evict
        """
        if not self._access_count:
            return
            
        # Sort keys by access count (ascending)
        sorted_keys = sorted(self._access_count.items(), key=lambda x: x[1])
        
        # Evict the least frequently used
        for key, _ in sorted_keys[:min(count, len(sorted_keys))]:
            if key in self._cache:
                del self._cache[key]
            del self._access_count[key]
            
            # Record eviction
            self._metrics["evictions"] += 1
    
    def remove(self, key: str) -> bool:
        """Remove a specific key from the cache.
        
        Args:
            key: The key to remove
            
        Returns:
            True if the key was removed, False if it didn't exist
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._access_count:
                    del self._access_count[key]
                return True
            return False
    
    def clear(self) -> None:
        """Clear all entries from the cache."""
        with self._lock:
            self._cache.clear()
            self._access_count.clear()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics.
        
        Returns:
            Dict[str, Any]: Dictionary containing cache metrics including
            hits, misses, hit_rate, size, capacity, utilization, and evictions
        """
        with self._lock:
            metrics = dict(self._metrics)
            
            # Add derived metrics
            total_requests = metrics["hits"] + metrics["misses"]
            if total_requests > 0:
                metrics["hit_rate"] = metrics["hits"] / total_requests
            else:
                metrics["hit_rate"] = 0.0
                
            # Add current size
            metrics["size"] = len(self._cache)
            metrics["capacity"] = self.max_size
            metrics["utilization"] = len(self._cache) / self.max_size if self.max_size > 0 else 0
            
            return metrics
    
    def cleanup_expired(self) -> int:
        """Remove all expired entries from the cache.
        
        Returns:
            Number of entries removed
        """
        with self._lock:
            current_time = time.time()
            expired_keys = []
            
            # Find all expired keys
            for key, entry in self._cache.items():
                if entry["expires_at"] < current_time:
                    expired_keys.append(key)
            
            # Remove expired entries
            for key in expired_keys:
                del self._cache[key]
                if key in self._access_count:
                    del self._access_count[key]
                self._metrics["expirations"] += 1
            
            return len(expired_keys)
    
    def __len__(self) -> int:
        """Get the number of entries in the cache."""
        return len(self._cache)
    
    def cleanup(self) -> None:
        """Perform cleanup operations."""
        with self._lock:
            # Clear all data
            self._cache.clear()
            self._access_count.clear()
            
            # Reset metrics that are transient
            self._metrics["evictions"] = 0
            self._metrics["expirations"] = 0


class CacheManager:
    """Manager for caching GitHub API queries and results.
    
    Features:
    - Efficient cache key generation
    - TTL-based caching with automatic expiration
    - Deduplication of repositories
    - Cache statistics and monitoring
    - Thread-safe operations
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 600, strategy: str = "hybrid", 
                 metrics_collector = None, cache: Optional[Cache] = None, path_manager = None,
                 bloom_filter = None, logger = None):
        """Initialize the cache manager.
        
        Args:
            max_size: Maximum number of entries in the cache
            default_ttl: Default time-to-live in seconds
            strategy: Cache eviction strategy (lru, lfu, or hybrid)
            metrics_collector: Metrics collector instance
            cache: Optional custom cache implementation
            path_manager: Path manager instance
            bloom_filter: Optional pre-configured bloom filter
            logger: Logger instance
        """
        # Store dependencies
        self.metrics_collector = metrics_collector
        self.path_manager = path_manager
        self.logger = logger
        
        # Use provided cache or create a new SmartCache
        if cache is not None:
            self._cache = cache
        else:
            self._cache = SmartCache(max_size=max_size, default_ttl=default_ttl, 
                                    strategy=strategy, logger=self.logger)
        
        # Repository deduplication tracking - only using a set for efficiency and accuracy 
        self._seen_repos = set()
        self._seen_repos_lock = threading.RLock()
        
        # Last cleanup time tracking
        self._last_cleanup = time.time()
        
        # Periodic operations counter
        self._op_counter = 0
        
        # Cache metrics
        self._metrics = {
            "puts": 0,
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "cleanups": 0
        }
        
        if self.logger:
            self.logger.debug(f"Cache manager initialized with max_size={max_size}, ttl={default_ttl}s, strategy='{strategy}'")
            self.logger.debug("Using set for repository deduplication")
    
    def generate_cache_key(self, query_text: str, cursor: Optional[str] = None, limit: int = 100) -> str:
        """Generate a unique cache key for a query.
        
        Args:
            query_text: The query text
            cursor: Pagination cursor
            limit: Batch size limit
            
        Returns:
            A unique hash to use as cache key
        """
        # For very short queries, use a direct key
        if len(query_text) < 10 and not cursor:
            # For short queries with no cursor, a direct key is faster and still unique
            direct_key = f"q:{query_text}|l:{limit}"
            if len(direct_key) < 32:  # If short enough for efficient key
                return direct_key
        
        # Cache the hashlib import for better performance
        if not hasattr(self, '_md5_hasher'):
            import hashlib
            self._md5_hasher = hashlib.md5
        
        # Create key components as bytes directly to avoid extra encode step
        query_bytes = query_text.encode('utf-8')
        cursor_bytes = (cursor or 'None').encode('utf-8')
        limit_bytes = str(limit).encode('utf-8')
        
        # Create hasher and update with each component
        hasher = self._md5_hasher()
        hasher.update(query_bytes)
        hasher.update(b'|')
        hasher.update(cursor_bytes)
        hasher.update(b'|')
        hasher.update(limit_bytes)
        
        return hasher.hexdigest()
    
    def get_cached_query_result(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get cached query result if available and not expired.
        
        Args:
            cache_key: The cache key
            
        Returns:
            Cached data or None if not found or expired
        """
        # Increment operation counter
        self._op_counter += 1
        
        # Get from cache
        result = self._cache.get(cache_key)
        
        # Update hit/miss metrics
        if result is not None:
            self._metrics["hits"] += 1
            if self.metrics_collector:
                self.metrics_collector.update_cache_metrics({"hits": 1})
        else:
            self._metrics["misses"] += 1
            if self.metrics_collector:
                self.metrics_collector.update_cache_metrics({"misses": 1})
                
        # Calculate and update hit rate
        total_attempts = self._metrics["hits"] + self._metrics["misses"]
        if total_attempts > 0 and self.metrics_collector:
            hit_rate = self._metrics["hits"] / total_attempts
            self.metrics_collector.update_cache_metrics({"hit_rate": hit_rate})
        
        # Periodically run cleanup
        if self._op_counter % 100 == 0:
            self._cleanup_expired()
        
        return result
        
    def cache_query_result(self, cache_key: str, result: Dict[str, Any], ttl: Optional[int] = None) -> None:
        """Cache a query result with optional TTL.
        
        Args:
            cache_key: The cache key
            result: The query result to cache
            ttl: Optional TTL in seconds (uses default if not specified)
        """
        # Increment operation counter
        self._op_counter += 1
        
        # Add to cache
        self._cache.put(cache_key, result, ttl)
        
        # Update metrics
        self._metrics["puts"] += 1
        
        # Periodically run cleanup
        if self._op_counter % 100 == 0:
            self._cleanup_expired()
    
    def is_duplicate_repository(self, repo_id: str) -> bool:
        """Check if repository has been seen before.
        
        Uses an efficient set-based approach for repository deduplication.
        
        Args:
            repo_id: Repository ID
            
        Returns:
            True if repository is a duplicate, False if it's new
        """
        with self._seen_repos_lock:
            # Check the repository set
            is_duplicate = repo_id in self._seen_repos
            
            # Record cache hit or miss metrics
            if self.metrics_collector:
                if is_duplicate:
                    self.metrics_collector.update_cache_metrics({"hits": 1})
                else:
                    self.metrics_collector.update_cache_metrics({"misses": 1})
                    
            return is_duplicate
    
    def mark_repository_seen(self, repo_id: str) -> None:
        """Mark repository as seen in the tracking set.
        
        Args:
            repo_id: Repository ID to mark as seen
        """
        with self._seen_repos_lock:
            self._seen_repos.add(repo_id)
    
    def _cleanup_expired(self) -> None:
        """Clean up expired cache entries."""
        current_time = time.time()
        
        # Only run cleanup once per minute at most
        if current_time - self._last_cleanup < 60:
            return
            
        self._last_cleanup = current_time
        
        # Clean up expired entries
        removed_count = self._cache.cleanup_expired()
        if removed_count > 0:
            self._metrics["cleanups"] += removed_count
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics.
        
        Returns:
            Dict[str, Any]: Dictionary containing cache metrics including
            hits, misses, hit_rate, size, capacity, utilization, and evictions
        """
        # Get cache metrics
        metrics = self._cache.get_metrics()
        
        # Add repository tracking metrics
        metrics["unique_repositories"] = len(self._seen_repos)
        
        # Add our own metrics
        metrics.update(self._metrics)
        
        return metrics
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        
    def cleanup_expired(self) -> int:
        """Remove all expired entries from the cache.
        
        Returns:
            Number of entries removed
        """
        return self._cache.cleanup_expired()
        
    def __len__(self) -> int:
        """Get the number of entries in the cache."""
        return len(self._cache)
        
    def cleanup(self) -> None:
        """Clean up resources used by the cache manager."""
        # Clear cache
        self._cache.cleanup()
        
        # Clear repository tracking
        with self._seen_repos_lock:
            self._seen_repos.clear()
                
        if self.logger:
            self.logger.debug("Cache manager resources cleaned up")