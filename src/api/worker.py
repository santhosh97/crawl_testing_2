#!/usr/bin/env python3
"""
GitHub Worker implementation for fetching repositories.

This module provides a unified Worker implementation that handles repository
fetching with adaptive query selection, optimized pagination, and metrics tracking.
It's used by both the sequential and parallel GitHub clients.
"""

import logging
import time
import random
from datetime import datetime
from typing import Dict, List, Optional, Any, Set, Tuple

import requests

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
from src.utils.cache_utils import CacheManager


class Worker:
    """Worker class for fetching repositories from GitHub.
    
    This class encapsulates the logic for a worker that fetches
    repositories from GitHub using the search API with optimized queries.
    It handles query selection, adaptive pagination, and metrics tracking.
    
    Can be used in both standalone and parallel execution modes.
    """
    
    def __init__(self, worker_id: int, token: str, query_pool, token_manager, 
                 cache_mgr, metrics_collector, worker_type: str = "standard",
                 connection_manager: Optional[ConnectionManager] = None):
        """Initialize worker with required dependencies.
        
        Args:
            worker_id: Unique ID for this worker
            token: GitHub API token to use
            query_pool: QueryPool for selecting queries
            token_manager: TokenManager for token operations
            cache_mgr: CacheManager for caching
            metrics_collector: MetricsCollector for metrics
            worker_type: Worker specialization type
            connection_manager: ConnectionManager for HTTP sessions
        """
        self.worker_id = worker_id
        self.token = token
        self.query_pool = query_pool
        self.token_manager = token_manager
        self.cache_mgr = cache_mgr
        self.metrics_collector = metrics_collector
        self.worker_type = worker_type
        self.connection_manager = connection_manager
        
        # State tracking
        self.repositories = []
        self.api_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0
        self.is_idle = False
        self.last_active_time = time.time()
        self.last_query_text = None
        
        # Strategy parameters
        self.strategy = "ucb"  # Start with UCB for exploration
        
        # Use centralized star weights from query_utils module
        self.star_weights = None  # Will use defaults from query_utils
        
        logger.info(f"Worker {worker_id} started with token {token[:4]}...{token[-4:]}")
    
    def select_query(self, avoiding_queries: Set[str] = None) -> Dict[str, Any]:
        """Select the best query based on current strategy.
        
        Args:
            avoiding_queries: Set of queries to avoid
            
        Returns:
            Selected query
        """
        avoiding = avoiding_queries or set()
        
        # Try to get a query that's not being used by another worker
        query = self.query_pool.get_best_query_avoiding(
            avoid_texts=avoiding,
            strategy=self.strategy,
            exclude_used=False
        )
        
        # If no query available, try again without constraints
        if query is None:
            query = self.query_pool.get_best_query(
                strategy=self.strategy,
                exclude_used=False
            )
            
        return query
    
    def get_pagination_thresholds(self, query_text: str) -> Dict[str, int]:
        """Get adaptive pagination thresholds based on query characteristics.
        
        Args:
            query_text: Query text to analyze
            
        Returns:
            Dictionary of threshold parameters
        """
        # Use the centralized pagination configuration
        from src.utils.query_utils import get_smart_pagination_config
        return get_smart_pagination_config(query_text)

    def fetch(self, target_count: int, diverse_config: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Fetch repositories with adaptive query selection.
        
        This is the main worker method that fetches repositories using
        multi-armed bandit optimization for query selection.
        
        Args:
            target_count: Target number of repositories to fetch
            diverse_config: Optional configuration for diverse starting queries
            
        Returns:
            List of repository data
        """
        worker_repos = []
        batch_size = 100  # GitHub API maximum per request
        
        # Track API usage and results
        api_calls = 0
        successful_calls = 0
        failed_calls = 0
        
        # Worker status tracking
        is_idle = False
        last_active_time = time.time()
        
        logger.info(f"Worker {self.worker_id} started with token {self.token[:4]}...{self.token[-4:]}, target: {target_count} repos")
        
        # Worker strategy adapts over time:
        # - Initial phase: Pure exploration (try diverse queries)
        # - Middle phase: Mix of exploration and exploitation 
        # - Final phase: Pure exploitation (use best queries)
        
        # Start with UCB, then switch to Thompson sampling after we have more data
        strategy = "ucb"
        strategy_switch_point = min(50, target_count // 4)
        exploration_phase_threshold = min(20, target_count // 10)
        
        # Last query used - for path tracking
        last_query_text = None
        
        while len(worker_repos) < target_count:
            # Check if we've been signaled to terminate early (auto_completed)
            unique_results = self.metrics_collector._metrics["repositories"]["unique"]
            global_target = self.metrics_collector.get_global_target_count()
            if self.metrics_collector.is_auto_completed() or unique_results >= global_target:
                logger.info(f"Worker {self.worker_id} terminating early due to reaching total target of {global_target} repositories")
                break
            
            # Check if we should switch to exploitation strategy
            if len(worker_repos) > strategy_switch_point:
                strategy = "thompson"  # More exploitative
            
            # Check if we should simply use the best queries
            if len(worker_repos) > target_count * 0.8:
                strategy = "best"  # Pure exploitation

            # Build set of queries to avoid
            avoid_texts = set()
            
            # 1. Avoid queries currently in use by other workers
            avoid_texts.update(self.metrics_collector.get_active_queries())
            
            # 2. Avoid exhausted queries that are still in cooling period
            if self.query_pool:
                # Get all exhausted queries from the QueryCoolingManager
                exhausted_queries = self.query_pool.query_cooling_manager.get_exhausted_queries()
                
                # Check each query to see if it's still exhausted and add to avoid list
                for q_text in exhausted_queries.keys():
                    if self.query_pool.is_query_exhausted(q_text):
                        avoid_texts.add(q_text)
            
            # Determine query to use
            # Try to get a query that avoids both active and exhausted queries
            query = self.query_pool.get_best_query_avoiding(
                avoid_texts=avoid_texts,
                strategy=strategy,
                exclude_used=len(worker_repos) < exploration_phase_threshold
            )
            
            # If no query available with all constraints, fall back to just avoiding active ones
            if query is None:
                active_only = self.metrics_collector.get_active_queries()
                
                query = self.query_pool.get_best_query_avoiding(
                    avoid_texts=active_only,
                    strategy=strategy,
                    exclude_used=False
                )
                
                # If still no query available, use the regular selection method as fallback
                if query is None:
                    query = self.query_pool.get_best_query(
                        strategy=strategy,
                        exclude_used=False
                    )
            
            query_text = query["query_text"]
            cursor = None  # Start from the beginning for this query
            
            # Add the query to active_queries to prevent other workers from using it
            self.metrics_collector.add_active_query(query_text)
            
            # Track parent-child query relationship for path rewards
            if last_query_text is not None:
                # This will be updated with actual rewards after execution
                if self.query_pool:
                    self.query_pool.track_query_path(
                        parent_query=last_query_text, 
                        child_query=query_text,
                        success=True, 
                        reward=0.0
                    )
            
            # Save current query for next iteration's path tracking
            last_query_text = query_text
            
            # Track this query's results
            query_results = 0
            query_duplicates = 0
            api_calls_for_query = 0
            
            # Use a try/finally block to ensure query is removed from active queries when done
            try:
                logger.info(f"Worker {self.worker_id} using query: {query_text} (strategy: {strategy})")
                
                # Track star counts of repos found with this query
                query_star_distribution = {}
                
                # Fetch repositories with this query (with pagination)
                # Track pagination metrics for adaptive skipping
                page_number = 0
                duplication_rates = []  # Track duplication rate for each page
                duplication_gradient = 0  # Rate of increase in duplication
                pages_since_useful_content = 0  # Count of pages with limited new content
                skip_factor = 1  # Initialize to not skip (1 = normal, 2 = skip one page, etc.)
                
                while len(worker_repos) < target_count:
                    try:
                        # Update page counter
                        page_number += skip_factor
                        
                        # Fetch a batch
                        api_calls += 1
                        api_calls_for_query += 1
                        
                        # Update metrics
                        self.metrics_collector._metrics["api"]["requests"] += 1
                        self.metrics_collector.update_worker_stats(
                            worker_type=self.worker_type,
                            api_calls=1
                        )
                        
                        # Fetch repositories
                        result = fetch_repositories_with_query(
                            query_text, 
                            batch_size, 
                            cursor, 
                            assigned_token=self.token, 
                            token_manager=self.token_manager, 
                            cache_mgr=self.cache_mgr, 
                            query_pool=self.query_pool, 
                            connection_manager=self.connection_manager
                        )
                        
                        # Get page info
                        page_info = result["data"]["search"]["pageInfo"]
                        has_next_page = page_info["hasNextPage"]
                        cursor = page_info["endCursor"]
                        
                        # Process the repositories
                        batch_repos = process_repository_data(result, cache_mgr=self.cache_mgr)
                        
                        # Track success
                        successful_calls += 1
                        self.metrics_collector.record_api_request(success=True)
                        
                        # No results? Move to next query
                        if not batch_repos:
                            logger.warning(f"Worker {self.worker_id} received empty batch for query: {query_text}")
                            break
                    
                        # Track query results
                        current_page_count = len(batch_repos)
                        query_results += current_page_count
                        self.metrics_collector._metrics["repositories"]["fetched"] += current_page_count
                    
                        # Track star distribution for quality score
                        for repo in batch_repos:
                            star_count = repo["star_count"]
                            # Determine star range
                            for min_stars, max_stars, range_name in self.query_pool.star_ranges:
                                if (min_stars is None or star_count >= min_stars) and \
                                   (max_stars is None or star_count < max_stars):
                                    query_star_distribution[range_name] = query_star_distribution.get(range_name, 0) + 1
                                    break
                    
                        # Filter out duplicates using cache_mgr
                        new_repos = []
                        for repo in batch_repos:
                            repo_id = repo["github_id"]
                            if not self.cache_mgr.is_duplicate_repository(repo_id):
                                new_repos.append(repo)
                                
                        current_page_duplicates = current_page_count - len(new_repos)
                        
                        # Track duplicates for this query
                        query_duplicates += current_page_duplicates
                        
                        # Update metrics
                        with self.metrics_collector._lock:
                            self.metrics_collector._metrics["repositories"]["unique"] += len(new_repos)
                            
                            # Calculate and store current duplicate rate
                            fetched = self.metrics_collector._metrics["repositories"]["fetched"]
                            unique = self.metrics_collector._metrics["repositories"]["unique"]
                            if fetched > 0:
                                dup_rate = 1.0 - (unique / fetched)
                                self.metrics_collector._metrics["repositories"]["duplicate_rate"] = dup_rate
                                
                                # Add to duplication history
                                self.metrics_collector.add_duplication_rate(dup_rate)
                        
                        # Track worker-specific statistics
                        high_star_count = sum(1 for repo in new_repos if repo["star_count"] >= 20000)
                        self.metrics_collector.update_worker_stats(
                            worker_type=self.worker_type,
                            unique_results=len(new_repos),
                            high_star_repos=high_star_count
                        )
                        
                        # Track distributions in metrics
                        with self.metrics_collector._lock:
                            for repo in new_repos:
                                # Star distribution
                                star_count = repo["star_count"]
                                for min_stars, max_stars, range_name in self.query_pool.star_ranges:
                                    if (min_stars is None or star_count >= min_stars) and \
                                      (max_stars is None or star_count < max_stars):
                                        star_ranges = self.metrics_collector._metrics["repositories"]["star_ranges"]
                                        star_ranges[range_name] = star_ranges.get(range_name, 0) + 1
                                        break
                                
                                # Language distribution
                                language = query.get("language", "any")
                                if language != "any":
                                    lang_dist = self.metrics_collector._metrics["repositories"]["language_distribution"]
                                    lang_dist[language] = lang_dist.get(language, 0) + 1
                        
                            # Mark repos as seen using cache_mgr
                            for repo in new_repos:
                                repo_id = repo["github_id"]
                                self.cache_mgr.mark_repository_seen(repo_id)
                        
                        # Add new repositories to worker result
                        worker_repos.extend(new_repos)
                    
                        # Calculate current page duplication rate
                        if current_page_count > 0:
                            current_duplication_rate = current_page_duplicates / current_page_count
                            duplication_rates.append(current_duplication_rate)
                        else:
                            current_duplication_rate = 0
                            
                        # Import the centralized pagination utilities
                        from src.utils.query_utils import get_pagination_thresholds, calculate_adaptive_skip_factor, adjust_skip_factor_for_query_type
                        
                        # Get pagination thresholds for current query
                        duplication_threshold, max_pages_threshold, max_skip_factor = get_pagination_thresholds(query_text)
                        
                        # Smart pagination logic: detect duplication patterns and adjust skipping
                        if len(duplication_rates) >= 3:
                            # Count consecutive low-yield pages using query-specific threshold
                            if current_duplication_rate > duplication_threshold:
                                pages_since_useful_content += 1
                            else:
                                pages_since_useful_content = 0
                            
                            # Calculate adaptive skip factor based on duplication patterns
                            query_specific_skip_factor = calculate_adaptive_skip_factor(
                                duplication_rates=duplication_rates,
                                pages_since_useful_content=pages_since_useful_content,
                                skip_factor=skip_factor,
                                max_skip_factor=max_skip_factor
                            )
                            
                            # Apply query-specific adjustments (e.g., slower for high-star queries)
                            query_specific_skip_factor = adjust_skip_factor_for_query_type(
                                query_text=query_text,
                                skip_factor=query_specific_skip_factor
                            )
                            
                            # Update the skip factor
                            skip_factor = query_specific_skip_factor
                        
                            # Log significant pagination adjustments
                            if skip_factor > 1 and current_duplication_rate > 0.5:
                                logger.info(f"Worker {self.worker_id}: Adaptive pagination - skipping {skip_factor-1} pages due to {current_duplication_rate:.1%} duplication rate")
                        
                        # Update active time
                        last_active_time = time.time()
                        is_idle = False
                        
                        # Check if we've reached target count
                        if len(worker_repos) >= target_count:
                            break
                            
                        # Use centralized pagination termination logic
                        from src.utils.query_utils import should_terminate_pagination
                        
                        # Create pagination config structure with current thresholds
                        pagination_config = {
                            "duplication_threshold": duplication_threshold,
                            "early_stop_threshold": max_pages_threshold,
                            "max_skip_factor": max_skip_factor
                        }
                        
                        # Check if we should terminate pagination
                        should_terminate, reason = should_terminate_pagination(
                            query_text=query_text,
                            query_results=query_results,
                            query_duplicates=query_duplicates,
                            api_calls_for_query=api_calls_for_query,
                            pages_since_useful_content=pages_since_useful_content,
                            has_next_page=has_next_page,
                            pagination_config=pagination_config
                        )
                        
                        if should_terminate:
                            logger.info(f"Worker {self.worker_id} stopping query: {reason}")
                            break
                        elif reason:  # Log any non-empty reasons even if not terminating
                            logger.debug(f"Worker {self.worker_id}: {reason}")
                
                    except Exception as e:
                        logger.error(f"Worker {self.worker_id} error with query {query_text}: {e}")
                        failed_calls += 1
                        self.metrics_collector.record_api_request(success=False)
                        break
            
                # Update query stats in the pool
                if query_results > 0:
                    duplication_rate = query_duplicates / query_results
                    
                    # Calculate quality score based on star distribution
                    from src.utils.query_utils import calculate_quality_score
                    quality_score = calculate_quality_score(
                        star_distribution=query_star_distribution,
                        star_weights=self.star_weights
                    )
                
                    # Collect star counts from all repositories found
                    query_star_counts = []
                    for repo in worker_repos[-query_results:]:
                        query_star_counts.append(repo["star_count"])
                    
                    # Update query performance using comprehensive metrics including star counts
                    self.query_pool.update_query_performance(
                        query=query,
                        results_count=query_results,
                        unique_count=query_results - query_duplicates,
                        api_calls=api_calls_for_query,
                        success=successful_calls > 0,
                        quality_score=quality_score,
                        duplication_rate=duplication_rate,
                        star_counts=query_star_counts
                    )
                
                    logger.info(f"Query performance: {query_text} - {query_results} results, "
                              f"{query_duplicates} duplicates ({duplication_rate:.1%} duplication rate), "
                              f"quality: {quality_score:.2f}")
                
                    # Update path rewards with actual reward value
                    if last_query_text and last_query_text != query_text:
                        current_reward = 0.0
                        if api_calls_for_query > 0:
                            current_reward = (query_results - query_duplicates) / api_calls_for_query * (1.0 - duplication_rate) * quality_score
                        
                        # Update the path relationship with actual data
                        if self.query_pool:
                            self.query_pool.track_query_path(
                                parent_query=last_query_text,
                                child_query=query_text,
                                success=successful_calls > 0,
                                reward=current_reward
                            )
                    
                    # Check if this query has high duplication rate and should be marked as exhausted
                    # Get appropriate exhaustion threshold based on star range in query
                    query_lower = query_text.lower()
                    
                    # Import the query exhaustion parameter function
                    from src.utils.query_utils import get_query_exhaustion_parameters
                    
                    # Get collection stage (0.0-1.0)
                    collection_stage = getattr(self.query_pool, 'context_features', {}).get("collection_stage", 0.0) if self.query_pool else 0.0
                    
                    # Get query-specific exhaustion parameters using the shared utility
                    params = get_query_exhaustion_parameters(query_text, collection_stage)
                    min_results = params["min_results"]
                    exhaustion_threshold = params["exhaustion_threshold"]
                    cooling_multiplier = params["cooling_multiplier"]
                    
                    # Apply the star-range-specific exhaustion threshold
                    if query_results > min_results and duplication_rate > exhaustion_threshold:
                            
                        # Store with adjusted cooling period info using the query_pool instance
                        if self.query_pool:
                            self.query_pool.mark_query_exhausted(query_text, self.worker_id, cooling_multiplier)
                            logger.info(f"Worker {self.worker_id} marked query as exhausted due to high duplication rate ({duplication_rate:.1%} > {exhaustion_threshold:.1%}): {query_text[:50]}...")
                            
                            # Trigger aggressive local mutation when a query is exhausted
                            # This allows immediate adaptation without waiting for central evolution
                            try:
                                # Choose more aggressive mutation types based on duplication rate
                                if duplication_rate > 0.95:
                                    # Very high duplication - use the most aggressive mutations
                                    mutation_types = ["extreme_stars", "compound"]
                                elif duplication_rate > 0.9:
                                    # High duplication - use aggressive mutations
                                    mutation_types = ["compound", "crossover", "stars"]
                                else:
                                    # Moderate duplication - use standard mutations
                                    mutation_types = ["stars", "language", "sort", "creation", "compound"]
                                
                                # Select a random aggressive mutation type
                                mutation_type = random.choice(mutation_types)
                                
                                # Create a locally evolved query based on the exhausted one
                                logger.info(f"Worker {self.worker_id} triggering local aggressive {mutation_type.upper()} mutation due to query exhaustion")
                                
                                # Get the query object from the pool
                                parent_query = None
                                for q in self.query_pool.queries:
                                    if q["query_text"] == query_text:
                                        parent_query = q
                                        break
                                
                                if parent_query:
                                    try:
                                        # Apply the mutation directly using the query pool's mutation logic
                                        # We're doing this locally rather than waiting for the central evolution
                                        
                                        # Set more aggressive local mutation count based on duplication rate
                                        local_mutation_count = 1  # Default
                                        if duplication_rate > 0.95:
                                            local_mutation_count = 3  # Try 3 alternatives for very high duplication
                                        elif duplication_rate > 0.9:
                                            local_mutation_count = 2  # Try 2 alternatives for high duplication
                                            
                                        logger.info(f"Generating {local_mutation_count} local mutations to address {duplication_rate:.1%} duplication rate")
                                        
                                        new_queries = self.query_pool._evolve_queries(
                                            count=local_mutation_count,  # Generate multiple alternatives based on duplication rate
                                            mutation_types=[mutation_type],  # Force our chosen mutation type
                                            parent_queries=[parent_query],  # Use the exhausted query as parent
                                            local_mutation=True,  # Flag to indicate this is a local mutation
                                            cache_manager=self.cache_mgr  # Pass cache manager
                                        )
                                    except Exception as e:
                                        logger.warning(f"Error in first mutation attempt: {e}")
                                        # Fallback to simpler evolution method if the more complex one fails
                                        try:
                                            # Try with a more stable mutation
                                            logger.info(f"Falling back to simpler STARS mutation for local evolution")
                                            new_queries = self.query_pool._evolve_queries(
                                                count=local_mutation_count,  # Use the same count from above
                                                mutation_types=["stars"],
                                                parent_queries=[parent_query],
                                                local_mutation=True,
                                                cache_manager=self.cache_mgr  # Pass cache manager
                                            )
                                        except Exception as e2:
                                            logger.warning(f"Error in fallback mutation attempt: {e2}")
                                            raise
                                    
                                    if new_queries:
                                        # Log all created mutations
                                        for i, new_query in enumerate(new_queries):
                                            logger.info(f"Worker {self.worker_id} created aggressive local mutation #{i+1}: {new_query['query_text'][:50]}... (from: {query_text[:30]}...)")
                            except Exception as e:
                                logger.warning(f"Error creating local mutation for exhausted query: {e}")
            finally:
                # Always remove the query from active queries when done with it
                self.metrics_collector.remove_active_query(query_text)
        
        # Log worker completion
        logger.info(f"Worker {self.worker_id} completed with {len(worker_repos)}/{target_count} repositories "
                  f"using {api_calls} API calls ({api_calls/max(1, len(worker_repos)):.2f} calls per repo)")
        
        return worker_repos