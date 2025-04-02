"""
GitHub worker coordination for the Stars Crawler.

This module provides functionality for coordinating multiple worker threads
for fetching GitHub repositories in parallel, with proper resource management,
metrics tracking, and error handling.
"""

import logging
import time
import threading
import queue
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

from src.api.token_management import TokenManager
from src.api.worker import Worker
from src.api.api_client import GitHubApiClient
from src.api.repository_fetcher import RepositoryFetcher
from src.utils.cache_utils import CacheManager
from src.metrics.collector.collector import MetricsCollector
from src.core.query_pool import QueryPool

# Configure logging
logger = logging.getLogger(__name__)

class WorkerCoordinator:
    """Coordinates multiple worker threads for parallel repository fetching."""
    
    def __init__(self, token_manager: TokenManager, query_pool: Optional[QueryPool] = None,
                cache_manager: Optional[CacheManager] = None, metrics_collector: Optional[MetricsCollector] = None,
                api_client: Optional[GitHubApiClient] = None, repository_fetcher: Optional[RepositoryFetcher] = None):
        """Initialize the worker coordinator.
        
        Args:
            token_manager: Manager for GitHub API tokens
            query_pool: Optional query pool for selecting queries
            cache_manager: Optional cache manager for caching results
            metrics_collector: Optional metrics collector for tracking metrics
            api_client: Optional GitHubApiClient instance
            repository_fetcher: Optional RepositoryFetcher instance
        """
        self.token_manager = token_manager
        self.query_pool = query_pool
        self.cache_manager = cache_manager
        self.metrics_collector = metrics_collector
        
        # Create API client and repository fetcher if not provided
        self.api_client = api_client or GitHubApiClient(token_manager)
        self.repository_fetcher = repository_fetcher or RepositoryFetcher(
            token_manager, self.api_client, cache_manager, metrics_collector
        )
        
        # Initialize coordination components
        self._worker_lock = threading.RLock()
        self._workers = {}
        self._stop_event = threading.Event()
        self._repository_queue = queue.Queue()
        
        # Results tracking
        self._total_repositories = 0
        self._total_unique = 0
        self._active_tokens = set()
        self._worker_stats = {}
        
    def start_workers(self, num_workers: int, target_repos: int, worker_type: str = "standard") -> None:
        """Start worker threads for parallel fetching.
        
        Args:
            num_workers: Number of worker threads to start
            target_repos: Target number of repositories to fetch
            worker_type: Type of worker threads to start
        """
        logger.info(f"Starting {num_workers} {worker_type} workers with target of {target_repos} repositories")
        
        # Initialize shared components for workers
        work_queue = queue.Queue()
        work_queue_lock = threading.RLock()
        result_queue = queue.Queue()
        
        # Distribute repositories per worker
        repositories_per_worker = max(100, target_repos // num_workers)
        
        # Start workers with available tokens
        for worker_id in range(num_workers):
            # Get a token for this worker
            token = self.token_manager.get_token()
            if not token:
                logger.warning(f"No available token for worker {worker_id}, not starting")
                continue
                
            # Track active tokens
            with self._worker_lock:
                self._active_tokens.add(token)
                
            # Create a worker
            worker = Worker(
                worker_id=worker_id,
                token=token,
                query_pool=self.query_pool,
                token_manager=self.token_manager,
                cache_mgr=self.cache_manager,
                metrics_collector=self.metrics_collector,
                worker_type=worker_type
            )
            
            # Store the worker
            with self._worker_lock:
                self._workers[worker_id] = worker
                self._worker_stats[worker_id] = {
                    "repositories": 0,
                    "unique": 0,
                    "api_calls": 0,
                    "start_time": time.time()
                }
                
            # Start the worker in a thread
            worker_thread = threading.Thread(
                target=self._worker_thread,
                args=(worker, work_queue, work_queue_lock, result_queue, repositories_per_worker),
                daemon=True
            )
            worker_thread.start()
            
    def _worker_thread(self, worker: Worker, work_queue: queue.Queue, work_queue_lock: threading.RLock,
                     result_queue: queue.Queue, target_repos: int) -> None:
        """Worker thread function for fetching repositories.
        
        Args:
            worker: Worker instance
            work_queue: Queue for work items
            work_queue_lock: Lock for the work queue
            result_queue: Queue for results
            target_repos: Target number of repositories to fetch
        """
        worker_id = worker.worker_id
        logger.info(f"Worker {worker_id} started with token {worker.token[:4]}...{worker.token[-4:]}, target: {target_repos} repos")
        
        try:
            # Initialize worker state
            repositories_fetched = 0
            unique_repos = 0
            
            # Start fetching repositories
            while repositories_fetched < target_repos and not self._stop_event.is_set():
                # Get a query from the query pool
                active_queries = set()
                if self.metrics_collector:
                    active_queries = self.metrics_collector.get_active_queries()
                
                query = self.query_pool.get_best_query_avoiding(active_queries)
                if not query:
                    logger.warning(f"Worker {worker_id}: No queries available, sleeping")
                    time.sleep(2)
                    continue
                
                # Mark query as active
                if self.metrics_collector:
                    self.metrics_collector.add_active_query(query["query_text"])
                
                try:
                    # Fetch repositories with this query
                    logger.info(f"Worker {worker_id} using query: {query['query_text'][:50]}... (strategy: {query.get('strategy', 'unknown')})")
                    
                    # Track start time for metrics
                    start_time = time.time()
                    
                    # Use repository fetcher to fetch repositories
                    repo_generator = self.repository_fetcher.fetch_repositories_paginated(
                        query=query,
                        max_results=target_repos - repositories_fetched,
                        stop_on_duplicates=True,
                        assigned_token=worker.token
                    )
                    
                    # Process fetched repositories
                    batch_repositories = 0
                    batch_unique = 0
                    
                    for repository in repo_generator:
                        # Add to result queue
                        result_queue.put(repository)
                        
                        # Update counts
                        repositories_fetched += 1
                        batch_repositories += 1
                        
                        if not self.repository_fetcher._is_duplicate(repository):
                            unique_repos += 1
                            batch_unique += 1
                            
                            # Update worker stats
                            with self._worker_lock:
                                self._worker_stats[worker_id]["repositories"] = repositories_fetched
                                self._worker_stats[worker_id]["unique"] = unique_repos
                                self._total_repositories += 1
                                self._total_unique += 1
                        
                        # Check if we've hit our target
                        if repositories_fetched >= target_repos or self._stop_event.is_set():
                            break
                    
                    # Update query performance
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    # Calculate query performance metrics
                    duplication_rate = 0.0
                    if batch_repositories > 0:
                        duplication_rate = 1.0 - (batch_unique / batch_repositories)
                        
                    # Update query performance in the query pool
                    self.query_pool.update_query_performance(
                        query=query,
                        results_count=batch_repositories,
                        unique_count=batch_unique,
                        api_calls=1,  # We count this as one logical API call
                        success=True,
                        quality_score=0.5,  # Default quality score
                        duplication_rate=duplication_rate
                    )
                    
                    # Log query performance
                    logger.info(f"Worker {worker_id} query performance: {query['query_text'][:50]}... - "
                               f"{batch_repositories} results, {batch_repositories - batch_unique} duplicates "
                               f"({duplication_rate:.1%} duplication rate), "
                               f"quality: {0.5:.2f}")
                    
                except Exception as e:
                    logger.error(f"Worker {worker_id} error fetching repositories: {str(e)}")
                    
                finally:
                    # Mark query as inactive
                    if self.metrics_collector:
                        self.metrics_collector.remove_active_query(query["query_text"])
            
            # Log worker completion
            logger.info(f"Worker {worker_id} completed with {repositories_fetched}/{target_repos} repositories "
                       f"using {worker._api_calls} API calls ({worker._api_calls / max(1, repositories_fetched):.2f} calls per repo)")
                       
        except Exception as e:
            logger.error(f"Worker {worker_id} thread error: {str(e)}")
            
    def wait_for_completion(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """Wait for all workers to complete.
        
        Args:
            timeout: Optional timeout in seconds
            
        Returns:
            Dictionary with completion statistics
        """
        start_time = time.time()
        end_time = None if timeout is None else start_time + timeout
        
        # Check worker status periodically
        while end_time is None or time.time() < end_time:
            with self._worker_lock:
                active_workers = sum(1 for stats in self._worker_stats.values() 
                                   if stats["repositories"] < stats.get("target", 0))
                
            if active_workers == 0:
                break
                
            # Wait a bit before checking again
            time.sleep(0.5)
            
        # Collect statistics
        with self._worker_lock:
            stats = {
                "total_repositories": self._total_repositories,
                "total_unique": self._total_unique,
                "worker_count": len(self._workers),
                "active_tokens": len(self._active_tokens),
                "elapsed_time": time.time() - start_time
            }
            
        return stats
        
    def stop_workers(self) -> None:
        """Stop all worker threads."""
        # Set stop event
        self._stop_event.set()
        
        # Wait for workers to finish
        logger.info("Waiting for workers to stop...")
        time.sleep(2)
        
        # Log final statistics
        with self._worker_lock:
            logger.info(f"Workers stopped. Fetched {self._total_repositories} repositories "
                       f"({self._total_unique} unique) with {len(self._workers)} workers.")
            
        # Clear worker state
        with self._worker_lock:
            self._workers.clear()
            self._worker_stats.clear()
            self._active_tokens.clear()
            
        # Reset stop event
        self._stop_event.clear()