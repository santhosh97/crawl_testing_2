#!/usr/bin/env python3
"""
Parallel GitHub repository fetching module.

This module provides functionality for fetching GitHub repositories in parallel using
a multi-threaded approach. It efficiently manages multiple workers, token rotation,
query optimization, and rate limit handling to maximize throughput while respecting
GitHub API constraints.

Key components:
- ParallelFetcher: Main class that coordinates parallel fetching
- Worker management: Adaptive scaling of worker threads based on token health
- Query optimization: Integration with multi-armed bandit for optimal queries
- Rate limit handling: Proper handling of GitHub API rate limits
- Connection reuse: Optimized HTTP connection pooling
"""
import os
import logging
import time
import threading
import functools
import concurrent.futures
from datetime import datetime
from typing import Dict, List, Optional, Any, Iterator, Tuple, Set, Union

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Import connection management
from src.utils.connection_manager import ConnectionManager

# Use centralized logging configuration
logger = logging.getLogger(__name__)

# GitHub GraphQL API endpoint
GITHUB_API_URL = "https://api.github.com/graphql"

# Import the unified TokenManager from token_management
from src.api.token_management import TokenManager

# Import required functions from github_client.py and github_api_utils.py
from src.api.github_client import process_repository_data, fetch_repositories_with_query
from src.core.query_pool import QueryPool
from src.api.github_api_utils import execute_graphql_query
from src.utils.cache_utils import CacheManager

# Import the unified Worker implementation
from src.api.worker import Worker

# GitHub API rate limits
# For GraphQL, the limit is 5000 points per hour per token
# Each request costs 1 point, so we set a conservative rate limit
CALLS_PER_HOUR = 4500  # Slightly conservative to avoid hitting the limit
ONE_HOUR = 3600  # seconds

# Import exceptions from central location
from src.api.github_exceptions import (
    GitHubRateLimitError,
    GitHubAPIError
)


class ParallelFetcher:
    """
    A class to manage parallel fetching of GitHub repositories.
    
    This class encapsulates the parallel processing state and coordinates
    worker threads for fetching repositories using a query pool strategy.
    """
    def __init__(self, token_manager, query_pool, cache_mgr, metrics_collector, connection_manager=None):
        """
        Initialize the ParallelFetcher with required dependencies.
        
        Args:
            token_manager: TokenManager instance for managing GitHub API tokens
            query_pool: QueryPool instance for query selection
            cache_mgr: CacheManager instance for caching
            metrics_collector: MetricsCollector instance for metrics tracking
            connection_manager: ConnectionManager instance for HTTP connection management
        """
        # Validate all required dependencies
        if token_manager is None:
            raise ValueError("token_manager must be provided")
        if query_pool is None:
            raise ValueError("query_pool must be provided")
        if cache_mgr is None:
            raise ValueError("cache_mgr must be provided")
        if metrics_collector is None:
            raise ValueError("metrics_collector must be provided")
            
        # Store dependencies
        self.token_manager = token_manager
        self.query_pool = query_pool
        self.cache_mgr = cache_mgr
        self.metrics_collector = metrics_collector
        self.connection_manager = connection_manager
        
        # Initialize state variables
        self.work_queue = []
        self.work_queue_lock = threading.Lock()
        self.idle_workers = set()
        self.active_workers = 0
        self.last_worker_adjustment = time.time()
        self.lock = threading.RLock()
        
        # Set default global target count in metrics collector
        self.metrics_collector.set_global_target_count(5000)  # Default, will be overridden
    
    def worker_fetch(self, worker_id: int, target_count: int, token: str, 
                   diverse_config: Dict[str, Any] = None, 
                   worker_type: str = "standard") -> List[Dict[str, Any]]:
        """Worker function to fetch repositories with adaptive query selection.
        
        Args:
            worker_id: Worker ID
            target_count: Target number of repositories to fetch
            token: GitHub API token to use
            diverse_config: Optional configuration for diverse starting queries
            worker_type: Type of worker - "standard" or "high_star_hunter"
            
        Returns:
            List of repository data
        """
        # Create a Worker instance using our unified implementation
        worker = Worker(
            worker_id=worker_id,
            token=token,
            query_pool=self.query_pool,
            token_manager=self.token_manager,
            cache_mgr=self.cache_mgr,
            metrics_collector=self.metrics_collector,
            worker_type=worker_type,
            connection_manager=self.connection_manager
        )
        
        # Use the worker to fetch repositories
        result = worker.fetch(target_count, diverse_config)
        
        # Add work stealing implementation if needed in the future
        return result
    
    def stats_monitor(self):
        """
        Thread function to monitor and report statistics and dynamically adjust worker count.
        Uses non-blocking approaches for improved performance.
        """
        token_manager = self.token_manager  # Ensure token_manager is accessible
        
        # Use shorter initial sleep time for more responsive adjustments
        initial_sleep = 30  # Start with 30 seconds
        
        while True:
            try:
                # Adaptive sleep pattern based on collection progress
                time.sleep(initial_sleep)
                
                # Get metrics from collector
                metrics = self.metrics_collector.get_metrics()
                global_target = self.metrics_collector.get_global_target_count()
                unique_results = metrics["repositories"]["unique"]
                
                # Don't run if we're done
                if unique_results >= global_target:
                    break
                
                # Calculate progress percentage
                progress = unique_results / global_target if global_target > 0 else 0
                
                # Adjust sleep interval based on progress - more frequent updates as we get closer to target
                if progress > 0.8:
                    # Almost done - check more frequently (15 seconds)
                    initial_sleep = 15
                elif progress > 0.5:
                    # More than halfway - check every 30 seconds 
                    initial_sleep = 30
                else:
                    # Early stages - check every minute
                    initial_sleep = 60
                
                # Calculate elapsed time
                start_time = metrics["performance"]["start_time"]
                elapsed = time.time() - start_time
                
                # Calculate performance metrics
                fetched = metrics["repositories"]["fetched"]
                duplication_rate = metrics["repositories"]["duplicate_rate"]
                
                api_requests = metrics["api"]["requests"]
                successful_calls = metrics["api"]["successful_calls"]
                
                if api_requests > 0:
                    success_rate = successful_calls / api_requests
                    api_efficiency = unique_results / api_requests
                else:
                    success_rate = 0.0
                    api_efficiency = 0.0
                
                # Print stats
                logger.info(f"Progress stats: {elapsed:.1f}s elapsed, {unique_results}/{global_target} repositories "
                          f"({unique_results/global_target*100:.1f}%)")
                logger.info(f"Efficiency: {api_efficiency:.2f} unique repos per API call, "
                          f"duplication rate: {duplication_rate:.1%}, success rate: {success_rate:.1%}")
                
                # Log worker and work queue status
                worker_status = f"Workers: {self.active_workers} active, {len(self.idle_workers)} idle, {len(self.work_queue)} tasks in queue"
                logger.info(worker_status)
                
                # Log worker statistics
                try:
                    standard_stats = metrics["worker_stats"].get("standard", {})
                    standard_api_calls = standard_stats.get("api_calls", 0)
                    standard_unique = standard_stats.get("unique_results", 0)
                    standard_high_star_repos = standard_stats.get("high_star_repos", 0)
                    
                    # Calculate efficiency
                    standard_efficiency = standard_unique / max(1, standard_api_calls)
                    
                    # High-star focus rate - percentage of results that are high-star repos
                    standard_focus_rate = standard_high_star_repos / max(1, standard_unique) * 100
                    
                    logger.info(f"Worker stats: {standard_unique} repos ({standard_efficiency:.2f}/call), "
                              f"{standard_high_star_repos} high-star repos ({standard_focus_rate:.1f}% of finds)")
                except Exception as e:
                    logger.warning(f"Error calculating worker stats: {e}")
                
                # Update the query pool's context with current repository statistics
                # This allows contextual bandits to adapt to the current collection state
                try:
                    if self.query_pool:
                        # Pass metrics_collector directly 
                        self.query_pool.update_context(self.metrics_collector, global_target)
                except Exception as e:
                    logger.warning(f"Error updating query context: {e}")
                
                # Clean up exhausted queries older than the cooling period
                if self.query_pool:
                    # Use the query_pool's method to clean up expired queries
                    expired_queries = self.query_pool.cleanup_exhausted_queries()
                
                # Get top queries if we have enough data
                if self.query_pool and self.query_pool.total_runs >= 10:
                    top_queries = self.query_pool.get_top_performing_queries(count=3)
                    if top_queries:
                        logger.info("Top performing queries:")
                        for i, (query, score) in enumerate(top_queries[:3], 1):
                            logger.info(f"  {i}. {query['query_text']} (score: {score:.2f})")
                            
                    # Also log contextual and diversity stats occasionally
                    if elapsed % 180 < 60:  # Every 3 minutes or so
                        diversity = self.query_pool.context_features.get("diversity_score", 0)
                        collection_stage = self.query_pool.context_features.get("collection_stage", 0)
                        top_heavy = self.query_pool.context_features.get("top_heavy_ratio", 0)
                        
                        logger.info(f"Collection context: stage={collection_stage:.2f}, diversity={diversity:.2f}, top-heavy={top_heavy:.2f}")
                
                # Dynamic worker scaling logic - check token health and adjust workers
                current_time = time.time()
                worker_adjustment_interval = 20  # Reduced interval for more responsive scaling
                min_workers = max(1, self.active_workers // 4)  # Minimum workers to maintain
                max_workers_per_token = 5  # Increased max workers per token for better throughput
                
                if current_time - self.last_worker_adjustment > worker_adjustment_interval:
                    try:
                        # Get token stats to determine available capacity
                        token_stats = token_manager.get_token_stats()
                        
                        # Count healthy tokens (not depleted or at risk)
                        healthy_tokens = token_stats.get("_summary", {}).get("healthy_tokens", 0)
                        depleted_tokens = token_stats.get("_summary", {}).get("depleted_tokens", 0)
                        
                        # Calculate optimal worker count based on token health and progress
                        max_workers = 16  # Increased from 10 to 16 for better parallelism
                        
                        # Scale max_workers based on progress - use more workers when we're starting
                        if progress < 0.3:
                            # Early stages - use more workers to quickly get results
                            max_workers = 20
                        elif progress > 0.8:
                            # Almost done - reduce worker count to avoid wasted API calls
                            max_workers = 8
                        
                        optimal_workers = min(max_workers, healthy_tokens * max_workers_per_token)
                        
                        # Adjust worker count gradually (up to 2 workers at a time)
                        if optimal_workers > self.active_workers + 2:
                            # Room to scale up significantly
                            new_workers = self.active_workers + 2
                            logger.info(f"Scaling up workers: {self.active_workers} → {new_workers} (optimal: {optimal_workers})")
                            self.active_workers = new_workers
                        elif optimal_workers > self.active_workers:
                            # Small scale up
                            new_workers = self.active_workers + 1
                            logger.info(f"Scaling up workers: {self.active_workers} → {new_workers} (optimal: {optimal_workers})")
                            self.active_workers = new_workers
                        elif optimal_workers < self.active_workers - 2 and self.active_workers > min_workers:
                            # Need to scale down significantly
                            new_workers = max(min_workers, self.active_workers - 2)
                            logger.info(f"Scaling down workers: {self.active_workers} → {new_workers} (optimal: {optimal_workers})")
                            self.active_workers = new_workers
                        elif optimal_workers < self.active_workers and self.active_workers > min_workers:
                            # Small scale down
                            new_workers = max(min_workers, self.active_workers - 1)
                            logger.info(f"Scaling down workers: {self.active_workers} → {new_workers} (optimal: {optimal_workers})")
                            self.active_workers = new_workers
                            
                        # Update timestamp
                        self.last_worker_adjustment = current_time
                        
                    except Exception as e:
                        logger.warning(f"Error during worker scaling: {e}")
                
                # Auto-exit if we've exceeded our target (indicating we have more than enough)
                # This prevents unnecessary API calls and processing
                if unique_results >= global_target:
                    logger.info(f"Reached target of {global_target} repositories. Auto-terminating to save API usage.")
                    logger.info(f"Currently have {unique_results} repositories (exceeds target by {unique_results - global_target}).")
                    
                    # Signal all worker threads to terminate by setting auto_completed flag
                    self.metrics_collector.mark_auto_completed(True)
                    break
                
            except Exception as e:
                logger.error(f"Error in stats monitor: {e}")
    
    def fetch_repositories_with_query_pool(self, total_count: int = 5000, max_workers: int = 10, 
                                          yield_chunks: bool = False) -> Union[List[Dict[str, Any]], Iterator[List[Dict[str, Any]]]]:
        """Fetch repositories using a diverse pool of queries with multi-armed bandit optimization.
        
        This is the primary entry point for fetching repositories in parallel
        using the most efficient query strategy. It uses a multi-armed bandit
        approach to adaptively select the best performing queries.
        
        Args:
            total_count: Total number of repositories to fetch
            max_workers: Maximum number of parallel workers
            yield_chunks: If True, yield chunks of repositories as they are fetched; 
                         if False, return all at once
            
        Returns:
            If yield_chunks is False: List of repository data
            If yield_chunks is True: Iterator yielding chunks of repository data
        
        Raises:
            ValueError: If required components are missing
        """
        # Use class attributes for token_manager and cache_mgr
        token_manager = self.token_manager
        cache_mgr = self.cache_mgr
        
        # Validate inputs
        if token_manager is None:
            raise ValueError("token_manager must be provided")
            
        # Import necessary libraries
        import concurrent.futures
        import threading
        import numpy as np
        from datetime import datetime
        
        # Import tqdm for progress bar
        try:
            from tqdm import tqdm
        except ImportError:
            # Fallback to a simple function if tqdm is not available
            def tqdm(iterable=None, total=None, desc=None, **kwargs):
                return iterable
        
        # Use the query pool provided during initialization
        query_pool = self.query_pool
        
        # Initialize result storage
        repositories = []
        
        # Update global target count in metrics collector
        self.metrics_collector.set_global_target_count(total_count)
        
        # Reset work queue and other state variables
        self.work_queue = []
        self.idle_workers = set()
        
        # Create a progress bar
        progress_bar = tqdm(total=total_count, desc="Fetching repositories", 
                           unit="repo", ncols=100, 
                           bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")
        
        # Calculate initial workers - we'll dynamically scale later
        num_tokens = len(token_manager.tokens)
        initial_workers = min(max_workers, num_tokens)
        
        # All workers are standard workers
        standard_worker_count = initial_workers
        
        logger.info(f"Allocating {standard_worker_count} standard workers")
        
        # Dynamic scaling parameters
        min_workers = max(1, initial_workers // 4)  # Minimum workers to maintain
        max_workers_per_token = 4  # Maximum workers per healthy token
        
        # Update instance variables
        self.active_workers = initial_workers
        self.last_worker_adjustment = time.time()
        
        # All workers are standard type
        worker_types = {}
        for i in range(self.active_workers):
            worker_types[i] = "standard"
        
        # Calculate initial target per worker
        # We aim for 125% of the target to account for duplicates
        adjusted_target = int(total_count * 1.25)
        repos_per_worker = max(100, (adjusted_target + self.active_workers - 1) // self.active_workers)
        
        logger.info(f"Starting repository fetching with optimized query pool (bandit algorithm), target: {total_count}")
        logger.info(f"Using {self.active_workers} workers, each fetching ~{repos_per_worker} repositories")
        
        # Reset metrics for this run
        self.metrics_collector.reset_stats()
        
        # Map tokens to workers
        token_worker_map = {}
        for i in range(self.active_workers):
            if i < num_tokens:
                token_worker_map[i] = token_manager.tokens[i]
            else:
                # If more workers than tokens, reuse tokens
                token_worker_map[i] = token_manager.tokens[i % num_tokens]
        
        # Start a monitoring thread to periodically print stats and adjust worker count
        stats_thread = threading.Thread(target=self.stats_monitor, daemon=True)
        stats_thread.start()
        
        # Create the thread pool executor with adaptive size
        # Use a ThreadPoolExecutor with an initial size but allow it to grow as needed
        import os
        
        # Determine thread count based on system resources
        cpu_count = os.cpu_count() or 4
        memory_available = True  # Assume sufficient memory, adjust if needed
        
        # Limit the thread pool to a reasonable size based on available CPUs
        # but allow it to expand based on I/O-bound nature of the work
        base_workers = min(max_workers, cpu_count * 2)
        max_thread_limit = max(base_workers, max_workers)
        
        logger.info(f"Using adaptive thread pool with base={base_workers}, max={max_thread_limit}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_thread_limit) as executor:
            # Submit all workers to the executor
            future_to_worker = {}
            for i in range(self.active_workers):
                # Each worker gets a different token
                token = token_worker_map[i]
                
                # All workers are standard
                worker_type = "standard"
                
                # Launch worker
                future = executor.submit(
                    self.worker_fetch,
                    worker_id=i,
                    target_count=repos_per_worker, 
                    token=token,
                    worker_type=worker_type
                )
                future_to_worker[future] = i
            
            # Process results as they complete
            chunks = []
            current_chunk = []
            chunk_size = 50  # Default chunk size, can be adjusted
            
            for future in concurrent.futures.as_completed(future_to_worker):
                worker_id = future_to_worker[future]
                try:
                    worker_repos = future.result()
                    
                    # Process the worker's results
                    if worker_repos:
                        # Update main result list and progress bar
                        repositories.extend(worker_repos)
                        progress_bar.update(len(worker_repos))
                        
                        # Add to current chunk for yielding
                        current_chunk.extend(worker_repos)
                        
                        # Yield a chunk if it's big enough
                        if yield_chunks and len(current_chunk) >= chunk_size:
                            yield current_chunk
                            chunks.append(current_chunk)  # Keep track for potential "return" later
                            current_chunk = []  # Reset chunk
                    
                    # Mark worker as idle
                    self.idle_workers.add(worker_id)
                    
                except Exception as e:
                    logger.error(f"Worker {worker_id} failed with error: {e}")
                    
                # Check if we've reached the target
                if len(repositories) >= total_count:
                    # Signal other workers to terminate early
                    self.metrics_collector.mark_auto_completed(True)
                    
                    # If there's a partial chunk, yield it if yield_chunks is True
                    if yield_chunks and current_chunk:
                        yield current_chunk
                        chunks.append(current_chunk)
                    
                    # Close the progress bar
                    progress_bar.close()
                    
                    # We're done - return full results if not yielding chunks
                    if not yield_chunks:
                        return repositories[:total_count]  # Trim to exactly the requested amount
                    
                    # For yield_chunks=True, we've already yielded everything so just return
                    return
        
        # Close the progress bar when all workers are done
        progress_bar.close()
        
        # Yield any final partial chunk if we're yielding
        if yield_chunks and current_chunk:
            yield current_chunk
        
        # If we're not yielding or we need to return something
        if not yield_chunks:
            # Trim to exactly the requested total
            if len(repositories) > total_count:
                return repositories[:total_count]
            return repositories

# Import the unified implementation from github_client
from src.api.github_client import execute_graphql_query