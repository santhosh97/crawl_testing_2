"""
Metrics collection and tracking for the GitHub Stars Crawler.

This module provides tools for collecting and reporting metrics about
the crawler's performance, including API usage, caching, database operations,
resource utilization, and crawler effectiveness. It supports both real-time
metrics tracking and persistent storage for later analysis.

Key components:
- MetricsCollector: Core class for collecting and managing metrics
- MetricsExporter: Handles exporting metrics to various formats (CSV, JSON)
- Metrics types tracked:
  * API: requests, errors, rate limits, etc.
  * Cache: hit rate, size, evictions
  * Repositories: counts, uniqueness, star distribution
  * Database: operations, performance
  * System: memory usage, CPU usage
  * Bandit algorithm: exploration rates, rewards
"""

import os
import csv
import json
import time
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Set, Tuple, Union

# Configure logging
logger = logging.getLogger(__name__)

class MetricsExporter:
    """Handles exporting metrics to various formats."""
    
    def __init__(self, metrics_dir: Optional[Union[str, Path]] = None):
        """Initialize the metrics exporter.
        
        Args:
            metrics_dir: Directory to store metrics files (optional)
        """
        # Use provided metrics_dir or default to logs/metrics
        self.metrics_dir = Path(metrics_dir) if metrics_dir else Path("logs/metrics")
        # Ensure directory exists
        self.metrics_dir.mkdir(exist_ok=True, parents=True)
        
        # Store the last export times
        self._last_csv_export = time.time()
        self._last_json_export = time.time()
        
    def get_metrics_filepath(self, extension: str, timestamp: Optional[datetime] = None) -> Path:
        """Get filepath for metrics with timestamp.
        
        Args:
            extension: File extension (without dot)
            timestamp: Optional timestamp to use (defaults to now)
            
        Returns:
            Path object for the metrics file
        """
        ts = timestamp or datetime.now()
        filename = f"metrics_{ts.strftime('%Y%m%d_%H%M%S')}.{extension}"
        return self.metrics_dir / filename
        
    def export_to_csv(self, metrics: Dict[str, Any], timestamp: Optional[datetime] = None) -> str:
        """Export metrics to CSV file.
        
        Args:
            metrics: Metrics dictionary to export
            timestamp: Optional timestamp to use in filename
            
        Returns:
            Path to the CSV file
        """
        filepath = self.get_metrics_filepath("csv", timestamp)
        
        # Extract metrics for CSV format
        if isinstance(metrics.get("api"), dict):
            api_metrics = metrics["api"]
        else:
            api_metrics = {}
            
        if isinstance(metrics.get("cache"), dict):
            cache_metrics = metrics["cache"]
        else:
            cache_metrics = {}
            
        if isinstance(metrics.get("repositories"), dict):
            repo_metrics = metrics["repositories"]
        else:
            repo_metrics = {}
            
        if isinstance(metrics.get("database"), dict):
            db_metrics = metrics["database"]
        else:
            db_metrics = {}
            
        if isinstance(metrics.get("performance"), dict):
            perf_metrics = metrics["performance"]
        else:
            perf_metrics = {}
            
        if isinstance(metrics.get("bandit"), dict):
            bandit_metrics = metrics["bandit"]
        else:
            bandit_metrics = {}
        
        # Determine if this is a new file or update
        file_exists = filepath.exists()
        
        with open(filepath, 'w' if not file_exists else 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Write header if new file
            if not file_exists:
                writer.writerow([
                    'timestamp',
                    'api_requests',
                    'api_errors',
                    'api_rate_limit_hits',
                    'cache_hit_rate',
                    'repositories_fetched',
                    'repositories_unique',
                    'duplicate_rate',
                    'db_writes',
                    'db_errors',
                    'run_time',
                    'memory_usage_mb',
                    'cpu_usage_percent',
                    'bandit_exploration_ratio',
                ])
            
            # Write data row
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                api_metrics.get('requests', 0),
                api_metrics.get('errors', 0),
                api_metrics.get('rate_limit_hits', 0),
                cache_metrics.get('hit_rate', 0.0),
                repo_metrics.get('fetched', 0),
                repo_metrics.get('unique', 0),
                repo_metrics.get('duplicate_rate', 0.0),
                db_metrics.get('writes', 0),
                db_metrics.get('errors', 0),
                perf_metrics.get('run_time', 0.0),
                perf_metrics.get('memory_usage_mb', 0.0),
                perf_metrics.get('cpu_usage_percent', 0.0),
                bandit_metrics.get('exploration_ratio', 0.0),
            ])
            
        self._last_csv_export = time.time()
        logger.debug(f"Exported metrics to CSV: {filepath}")
        return str(filepath)
        
    def export_to_json(self, metrics: Dict[str, Any], timestamp: Optional[datetime] = None) -> str:
        """Export metrics to JSON file.
        
        Args:
            metrics: Metrics dictionary to export
            timestamp: Optional timestamp to use in filename
            
        Returns:
            Path to the JSON file
        """
        filepath = self.get_metrics_filepath("json", timestamp)
        
        with open(filepath, 'w') as f:
            # Use custom serializer to handle non-serializable types
            json.dump(metrics, f, indent=2, default=self._json_serialize)
            
        self._last_json_export = time.time()
        logger.debug(f"Exported metrics to JSON: {filepath}")
        return str(filepath)
        
    def _json_serialize(self, obj):
        """Custom JSON serializer for handling non-serializable objects.
        
        Args:
            obj: Object to serialize
            
        Returns:
            JSON serializable version of the object
        """
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, threading.RLock):
            return "RLock (not serializable)"
        if isinstance(obj, datetime):
            return obj.isoformat()
        
        # For objects with no serializer, convert to string
        return str(obj)


class MetricsCollector:
    """Collects and reports metrics for the GitHub Stars Crawler."""
    
    def __init__(self, metrics_dir: Optional[Union[str, Path]] = None, 
                 exporter: Optional[MetricsExporter] = None,
                 path_manager: Optional[Any] = None):
        """Initialize the metrics collector.
        
        Args:
            metrics_dir: Directory to store metrics files (optional)
            exporter: Optional metrics exporter instance
            path_manager: Optional PathManager instance for directory management
        
        Returns:
            None
        """
        # Store dependencies
        self.path_manager = path_manager
        
        # Flag for initialization tracking (instance-level, not class-level)
        self._is_initialized = False
        
        # Determine metrics directory
        if metrics_dir:
            self._metrics_dir = Path(metrics_dir)
        elif path_manager:
            self._metrics_dir = path_manager.get_metrics_dir()
        else:
            # Fallback to default path
            self._metrics_dir = Path("logs/metrics")
            self._metrics_dir.mkdir(exist_ok=True, parents=True)
            
        # Create or use provided exporter
        self.exporter = exporter or MetricsExporter(metrics_dir=self._metrics_dir)
        
        # Initialize metrics storage
        self._metrics = {
            "api": {
                "requests": 0,
                "errors": 0,
                "rate_limit_hits": 0,
                "total_time": 0.0,
                "requests_per_second": 0.0,
                "successful_calls": 0,
                "failed_calls": 0,
            },
            "cache": {
                "hits": 0,
                "misses": 0,
                "hit_rate": 0.0,
                "size": 0,
                "evictions": 0,
            },
            "repositories": {
                "fetched": 0,
                "unique": 0,
                "duplicate_rate": 0.0,
                "star_ranges": {},
                "language_distribution": {},
                "auto_completed": False,
            },
            "database": {
                "writes": 0,
                "reads": 0,
                "errors": 0,
                "total_time": 0.0,
                "writes_per_second": 0.0,
            },
            "performance": {
                "start_time": time.time(),
                "run_time": 0.0,
                "memory_usage_mb": 0.0,
                "cpu_usage_percent": 0.0,
            },
            "bandit": {
                "total_queries": 0,
                "total_reward": 0.0,
                "exploration_ratio": 0.0,
            },
            "worker_stats": {
                "standard": {
                    "api_calls": 0,
                    "unique_results": 0,
                    "high_star_repos": 0,
                },
                "high_star_hunter": {
                    "api_calls": 0,
                    "unique_results": 0,
                    "high_star_repos": 0,
                },
            },
            "active_queries": set(),  # For tracking active queries
        }
        
        # Lock for thread safety
        self._lock = threading.RLock()
        
        # Track global target count
        self.global_target_count = 5000  # Default, will be overridden
        
        # Historical data tracking
        self._metrics_history = {
            "timestamps": [],
            "total_runs": [],
            "unique_repos_found": [],
            "api_efficiency": [],
            "exploration_ratio": [],
            "duplication_rate": [],
            "pareto_optimal_queries": [],
            "top_heavy_ratio": [],
            "diversity_score": [],
            "high_star_percentage": [],
            "query_evolution_generations": [],
        }
        
        # Duplication history tracking
        self._duplication_history = []
        
        # Load resource monitoring
        self._has_resource_monitoring = False
        self._psutil = None
        try:
            import psutil
            self._psutil = psutil
            self._has_resource_monitoring = True
        except ImportError:
            logger.warning("psutil not available. Resource monitoring disabled.")
            
        # Mark as initialized and not shutting down
        self._is_initialized = True
        self._is_shutting_down = False
    
    def update_cache_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update cache metrics with data from the cache system.
        
        Updates cache-related metrics like hit rate, cache size, and evictions
        from the provided metrics dictionary.
        
        Args:
            metrics: Dictionary of cache metrics containing keys like
                     "hits", "misses", "hit_rate", "size", "evictions"
                     
        Returns:
            None
        """
        with self._lock:
            for key, value in metrics.items():
                if key in self._metrics["cache"]:
                    self._metrics["cache"][key] = value
                    
    def update_bandit_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update metrics related to the multi-armed bandit algorithm.
        
        Updates bandit algorithm metrics such as exploration ratio, 
        rewards, and query statistics from the provided metrics dictionary.
        
        Args:
            metrics: Dictionary of bandit metrics containing keys like
                    "exploration_ratio", "total_reward", "total_queries"
                    
        Returns:
            None
        """
        with self._lock:
            # Update bandit section with provided metrics
            for key, value in metrics.items():
                if key != "timestamp":  # Skip timestamp as it's not part of our structure
                    self._metrics["bandit"][key] = value
    
    def record_api_request(self, success: bool = True, time_taken: float = 0.0, rate_limited: bool = False):
        """Record API request metrics.
        
        Args:
            success: Whether the request was successful
            time_taken: Time taken for the request in seconds
            rate_limited: Whether the request was rate limited
        """
        with self._lock:
            self._metrics["api"]["requests"] += 1
            self._metrics["api"]["total_time"] += time_taken
            
            # Track success/failure counts
            if success:
                self._metrics["api"]["successful_calls"] += 1
            else:
                self._metrics["api"]["errors"] += 1
                self._metrics["api"]["failed_calls"] += 1
            
            if rate_limited:
                self._metrics["api"]["rate_limit_hits"] += 1
            
            # Calculate requests per second
            total_requests = self._metrics["api"]["requests"]
            total_time = self._metrics["api"]["total_time"]
            if total_time > 0:
                self._metrics["api"]["requests_per_second"] = total_requests / total_time
    
    def record_database_operation(self, operation_type: str, success: bool = True, time_taken: float = 0.0) -> None:
        """Record database operation metrics.
        
        Args:
            operation_type: Type of operation ('write' or 'read')
            success: Whether the operation was successful
            time_taken: Time taken for the operation in seconds
            
        Returns:
            None
        """
        with self._lock:
            # Update counters based on operation type
            if operation_type == 'write':
                self._metrics["database"]["writes"] += 1
            elif operation_type == 'read':
                self._metrics["database"]["reads"] += 1
            
            self._metrics["database"]["total_time"] += time_taken
            
            if not success:
                self._metrics["database"]["errors"] += 1
            
            # Calculate writes per second
            total_writes = self._metrics["database"]["writes"]
            total_time = self._metrics["database"]["total_time"]
            if total_time > 0:
                self._metrics["database"]["writes_per_second"] = total_writes / total_time
    
    def update_performance_metrics(self) -> None:
        """Update performance metrics like run time, memory and CPU usage.
        
        Returns:
            None
        """
        with self._lock:
            # Update run time
            start_time = self._metrics["performance"]["start_time"]
            self._metrics["performance"]["run_time"] = time.time() - start_time
            
            # Update resource metrics if available
            if self._has_resource_monitoring:
                # Get current process
                process = self._psutil.Process(os.getpid())
                
                # Memory usage
                memory_info = process.memory_info()
                self._metrics["performance"]["memory_usage_mb"] = memory_info.rss / (1024 * 1024)
                
                # CPU usage
                self._metrics["performance"]["cpu_usage_percent"] = process.cpu_percent(interval=0.1)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all current metrics.
        
        Returns:
            Dict[str, Any]: Dictionary of current metrics
        """
        with self._lock:
            # Make a deep copy to avoid modification during access
            return json.loads(json.dumps(self._metrics, default=self.exporter._json_serialize))
    
    def set_global_target_count(self, count: int):
        """Set the global target count for repositories to fetch.
        
        Args:
            count: Target number of repositories
        """
        with self._lock:
            self.global_target_count = count
    
    def get_global_target_count(self) -> int:
        """Get the global target count for repositories to fetch.
        
        Returns:
            Target number of repositories
        """
        with self._lock:
            return self.global_target_count
    
    def mark_auto_completed(self, completed: bool = True):
        """Mark the current run as auto-completed.
        
        Args:
            completed: Whether the run has been auto-completed
        """
        with self._lock:
            self._metrics["repositories"]["auto_completed"] = completed
    
    def is_auto_completed(self) -> bool:
        """Check if the current run has been auto-completed.
        
        Returns:
            True if the run has been auto-completed
        """
        with self._lock:
            return self._metrics["repositories"]["auto_completed"]
    
    def add_active_query(self, query_text: str):
        """Add a query to the active queries set.
        
        Args:
            query_text: The query text to add
        """
        # Use _metrics active_queries with proper locking
        with self._lock:
            if "active_queries" not in self._metrics:
                self._metrics["active_queries"] = set()
            self._metrics["active_queries"].add(query_text)
    
    def remove_active_query(self, query_text: str):
        """Remove a query from the active queries set.
        
        Args:
            query_text: The query text to remove
        """
        # Use _metrics active_queries with proper locking
        with self._lock:
            if "active_queries" in self._metrics and query_text in self._metrics["active_queries"]:
                self._metrics["active_queries"].remove(query_text)
            
    def get_active_queries(self) -> Set[str]:
        """Get the set of active queries.
        
        Returns:
            Set of active query texts
        """
        # Use _metrics active_queries with proper locking
        with self._lock:
            # Return a copy to avoid concurrent modification
            if "active_queries" not in self._metrics:
                return set()
            return set(self._metrics["active_queries"])
    
    def update_worker_stats(self, worker_type: str, api_calls: int = 0, unique_results: int = 0, high_star_repos: int = 0):
        """Update worker-specific statistics.
        
        Args:
            worker_type: Type of worker ("standard" or "high_star_hunter")
            api_calls: Number of API calls to add
            unique_results: Number of unique results to add
            high_star_repos: Number of high-star repositories to add
        """
        with self._lock:
            if worker_type not in self._metrics["worker_stats"]:
                logger.warning(f"Invalid worker type: {worker_type}")
                return
                
            stats = self._metrics["worker_stats"][worker_type]
            stats["api_calls"] += api_calls
            stats["unique_results"] += unique_results
            stats["high_star_repos"] += high_star_repos
    
    
    def add_duplication_rate(self, rate: float):
        """Add a duplication rate to the history.
        
        Args:
            rate: Duplication rate to add
        """
        with self._lock:
            self._duplication_history.append(rate)
    
    def reset_stats(self):
        """Reset all statistics for a new run."""
        with self._lock:
            # Reset API stats
            self._metrics["api"]["requests"] = 0
            self._metrics["api"]["errors"] = 0
            self._metrics["api"]["rate_limit_hits"] = 0
            self._metrics["api"]["successful_calls"] = 0
            self._metrics["api"]["failed_calls"] = 0
            
            # Reset repository stats
            self._metrics["repositories"]["fetched"] = 0
            self._metrics["repositories"]["unique"] = 0
            self._metrics["repositories"]["duplicate_rate"] = 0.0
            self._metrics["repositories"]["star_ranges"] = {}
            self._metrics["repositories"]["language_distribution"] = {}
            self._metrics["repositories"]["auto_completed"] = False
            
            # Reset performance metrics
            self._metrics["performance"]["start_time"] = time.time()
            
            # Reset worker stats
            for worker_type in self._metrics["worker_stats"]:
                worker_stats = self._metrics["worker_stats"][worker_type]
                worker_stats["api_calls"] = 0
                worker_stats["unique_results"] = 0
                worker_stats["high_star_repos"] = 0
            
            # Clear active queries - already inside _lock context
            if "active_queries" in self._metrics:
                self._metrics["active_queries"].clear()
            else:
                self._metrics["active_queries"] = set()
            
            logger.info("Metrics collector reset for new run")
    
    def export_metrics_json(self, file_path: Optional[str] = None) -> str:
        """Export metrics to a JSON file.
        
        Args:
            file_path: Optional custom file path
            
        Returns:
            Path to the written JSON file
        """
        # Update performance metrics before exporting
        self.update_performance_metrics()
        
        with self._lock:
            # Create a complete metrics object including history
            full_metrics = {
                "current": self._metrics,
                "history": {
                    "metrics_history": self._metrics_history,
                    "duplication_history": self._duplication_history
                }
            }
            
            # Let the exporter handle the file operations
            return self.exporter.export_to_json(full_metrics)
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate a summary report of metrics.
        
        Returns:
            Dictionary with summary metrics
        """
        with self._lock:
            # Update performance metrics
            self.update_performance_metrics()
            
            # Extract key metrics for summary
            summary = {
                "runtime_seconds": self._metrics["performance"]["run_time"],
                "repositories": {
                    "fetched": self._metrics["repositories"]["fetched"],
                    "unique": self._metrics["repositories"]["unique"],
                    "duplicate_rate": self._metrics["repositories"]["duplicate_rate"],
                    "star_distribution": self._metrics["repositories"]["star_ranges"],
                },
                "api": {
                    "requests": self._metrics["api"]["requests"],
                    "errors": self._metrics["api"]["errors"],
                    "rate_limit_hits": self._metrics["api"]["rate_limit_hits"],
                    "requests_per_second": self._metrics["api"]["requests_per_second"],
                },
                "cache": {
                    "hit_rate": self._metrics["cache"]["hit_rate"],
                },
                "database": {
                    "writes": self._metrics["database"]["writes"],
                    "errors": self._metrics["database"]["errors"],
                    "writes_per_second": self._metrics["database"]["writes_per_second"],
                },
                "resources": {
                    "memory_usage_mb": self._metrics["performance"]["memory_usage_mb"],
                    "cpu_usage_percent": self._metrics["performance"]["cpu_usage_percent"],
                },
                "bandit": {
                    "total_queries": self._metrics["bandit"]["total_queries"],
                    "total_reward": self._metrics["bandit"]["total_reward"],
                },
            }
            
            # Calculate overall efficiency
            if self._metrics["api"]["requests"] > 0:
                summary["efficiency"] = {
                    "unique_repos_per_request": self._metrics["repositories"]["unique"] / self._metrics["api"]["requests"],
                }
            
            return summary
    
    def log_summary(self):
        """Log a summary of metrics to the logger."""
        # Update performance metrics
        self.update_performance_metrics()
        
        # Generate summary report
        summary = self.generate_summary_report()
        
        # Format and log summary
        runtime_min = summary["runtime_seconds"] / 60.0
        unique_repos = summary["repositories"]["unique"]
        dup_rate = summary["repositories"]["duplicate_rate"] * 100
        
        logger.info(f"======= Crawler Metrics Summary =======")
        logger.info(f"Runtime: {runtime_min:.2f} minutes")
        logger.info(f"Repositories fetched: {summary['repositories']['fetched']}")
        logger.info(f"Unique repositories: {unique_repos} (duplicate rate: {dup_rate:.1f}%)")
        
        # Log star distribution
        logger.info(f"Star distribution:")
        for range_key, count in sorted(summary["repositories"]["star_distribution"].items()):
            pct = count / unique_repos * 100 if unique_repos > 0 else 0
            logger.info(f"  {range_key}: {count} repositories ({pct:.1f}%)")
        
        # Log API metrics
        logger.info(f"API requests: {summary['api']['requests']} ({summary['api']['requests_per_second']:.2f} req/s)")
        logger.info(f"API errors: {summary['api']['errors']} ({summary['api']['rate_limit_hits']} rate limit hits)")
        
        # Log cache metrics
        logger.info(f"Cache hit rate: {summary['cache']['hit_rate'] * 100:.1f}%")
        
        # Log database metrics
        logger.info(f"Database writes: {summary['database']['writes']} ({summary['database']['writes_per_second']:.2f} writes/s)")
        logger.info(f"Database errors: {summary['database']['errors']}")
        
        # Log resource usage
        logger.info(f"Memory usage: {summary['resources']['memory_usage_mb']:.1f} MB")
        logger.info(f"CPU usage: {summary['resources']['cpu_usage_percent']:.1f}%")
        
        if "efficiency" in summary:
            logger.info(f"Efficiency: {summary['efficiency']['unique_repos_per_request']:.2f} unique repos per API request")
        
        logger.info(f"========================================")
        
    def cleanup(self):
        """Clean up resources used by the metrics collector."""
        # Signal that we're shutting down (for the metrics updater thread)
        self._is_shutting_down = True
        
        # Export final metrics if needed
        self.export_metrics_json()
        
        # Clear metrics data
        with self._lock:
            self._metrics.clear()
            self._metrics_history.clear()
            self._duplication_history.clear()