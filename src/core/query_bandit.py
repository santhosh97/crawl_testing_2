"""
Multi-armed bandit algorithm for optimizing GitHub search queries.

This module implements various bandit algorithms for selecting and evaluating
queries to optimize repository discovery and minimize duplication.
"""

import os
import logging
import time
import threading
import random
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union, TypeVar, Generic, Callable
from pathlib import Path
from collections import defaultdict, deque, OrderedDict

import numpy as np

# Use centralized logging configuration instead of redundant setup
logger = logging.getLogger(__name__)

# Get loggers for bandit algorithm and analytics
bandit_logger = logging.getLogger('bandit_algorithm') 
bandit_analytics_logger = logging.getLogger('bandit_analytics')

# We don't need to import utilities for directory paths
# These should be passed through DI or configuration

# Metrics paths will be provided via dependency injection through metrics_collector
# We should not be creating paths here based on imports from other modules

# Default cooling period if not provided when instantiating
DEFAULT_QUERY_COOLING_PERIOD = 1800  # 30 minutes in seconds


def compute_collection_progress(found_repos, target_repos, min_progress=0.0, max_progress=1.0):
    """
    Compute collection progress as a value between 0.0 and 1.0
    
    Args:
        found_repos: Number of unique repositories found
        target_repos: Target number of repositories to find
        min_progress: Minimum progress value (default: 0.0)
        max_progress: Maximum progress value (default: 1.0)
    
    Returns:
        Progress value between min_progress and max_progress
    """
    if target_repos <= 0:
        return min_progress
    
    progress = min(found_repos / target_repos, 1.0)
    # Scale to the specified range
    scaled_progress = min_progress + progress * (max_progress - min_progress)
    return scaled_progress


def compute_novelty_alpha(collection_progress):
    """
    Compute alpha value for blending novelty with standard reward
    
    Args:
        collection_progress: Collection progress (0.0 to 1.0)
    
    Returns:
        Alpha value between 0.8 (early) and 0.2 (late)
    """
    # Early phase (0-0.3): α = 0.8 (favor reward)
    # Mid phase (0.3-0.7): Linear transition from 0.8 to 0.2
    # Late phase (0.7-1.0): α = 0.2 (favor novelty)
    
    if collection_progress < 0.3:
        return 0.8
    elif collection_progress > 0.7:
        return 0.2
    else:
        # Linear interpolation between 0.8 and 0.2
        progress_in_mid_phase = (collection_progress - 0.3) / 0.4
        return 0.8 - progress_in_mid_phase * 0.6


def compute_novelty_threshold(collection_progress):
    """
    Compute novelty threshold for filtering queries (Option 1)
    
    Args:
        collection_progress: Collection progress (0.0 to 1.0)
    
    Returns:
        Novelty threshold (0.0 = no filtering, higher values = stricter filtering)
    """
    if collection_progress < 0.7:
        # First 70%: No filtering
        return 0.0
    else:
        # Last 30%: Gradually increase threshold from 0.3 to 0.7
        phase_progress = (collection_progress - 0.7) / 0.3
        return 0.3 + phase_progress * 0.4


def calculate_query_cooling(query_text, avg_duplication_rate=0.0, 
                           high_duplication_threshold=0.8, 
                           super_high_duplication_threshold=0.9):
    """Calculate cooling factor for a query based on performance.
    
    Args:
        query_text: The query text to check
        avg_duplication_rate: Average duplication rate across all queries
        high_duplication_threshold: Threshold for high duplication rate
        super_high_duplication_threshold: Threshold for very high duplication rate
        
    Returns:
        Cooling multiplier (1.0 = no cooling, higher values = longer timeout)
    """
    # Adjust cooling factor based on global duplication rates
    # This helps prevent all queries from being marked exhausted simultaneously
    base_cooling = 1.0
    
    # Apply global state adjustments
    if avg_duplication_rate > super_high_duplication_threshold:
        # System-wide high duplication - reduce cooling to keep queries flowing
        base_cooling = 0.6
    elif avg_duplication_rate > high_duplication_threshold:
        # Moderate system-wide duplication - slightly reduce cooling
        base_cooling = 0.8
    
    # Check query string characteristics to adjust cooling
    # Queries targeting high-star repos shouldn't be cooled as much
    high_star_indicator = any(term in query_text for term in 
                             ["stars:>10000", "stars:>50000", "stars:>30000"])
    
    if high_star_indicator:
        # Reduce cooling for high-star queries to ensure they're revisited
        base_cooling *= 0.7
    
    # Check for specific languages that might have more repos
    common_languages = ["javascript", "typescript", "python", "java"]
    has_common_language = any(f"language:{lang}" in query_text for lang in common_languages)
    
    if not has_common_language:
        # Uncommon languages might have fewer repos, so cool them more
        base_cooling *= 1.2
    
    return base_cooling


class QueryCoolingManager:
    """
    Manager for tracking and cooling exhausted queries.
    
    This class manages the cooling period for queries that have been exhausted
    (returned too many duplicate results). It tracks when queries were marked as
    exhausted, applies appropriate cooling periods based on query characteristics,
    and determines when queries are ready to be used again.
    """
    
    def __init__(self, cooling_period=1800):
        """
        Initialize the QueryCoolingManager.
        
        Args:
            cooling_period: Base cooling period in seconds (default: 30 minutes)
        """
        # Dictionary of exhausted queries: {query_text: (timestamp, worker_id, cooling_multiplier)}
        self.exhausted_queries = {}
        # Lock for thread-safe access to exhausted_queries
        self.exhausted_queries_lock = threading.RLock()
        # Base cooling period in seconds
        self.base_cooling_period = cooling_period
    
    def cool_queries(self):
        """
        Cool down exhausted queries, gradually allowing them to be used again.
        
        This function:
        1. Reduces cooling for queries that have been cool for a while
        2. Removes queries from exhausted list if they've cooled enough
        3. Prioritizes cooling high-star queries faster
        
        Returns:
            Number of queries that were cooled and removed from the exhausted list
        """
        current_time = time.time()
        queries_to_remove = []
        
        with self.exhausted_queries_lock:
            for query_text, values in list(self.exhausted_queries.items()):
                # We always expect a tuple of (timestamp, worker_id, cooling_multiplier)
                timestamp, worker_id, cooling_multiplier = values
                
                # Calculate time in the cooling period
                time_cooling = current_time - timestamp
                
                # Update cooling multiplier for long-cooling queries
                # Every 10 minutes, reduce multiplier by 10%
                if time_cooling > 600:  # 10 minutes
                    cooling_reduction_cycles = int(time_cooling / 600)
                    new_multiplier = cooling_multiplier * (0.9 ** cooling_reduction_cycles)
                    new_multiplier = max(0.5, new_multiplier)  # Don't go below 0.5
                    
                    # Update the cooling multiplier in the exhausted queries dict
                    self.exhausted_queries[query_text] = (timestamp, worker_id, new_multiplier)
                
                # Check if the query is still exhausted using the same logic as is_query_exhausted
                if not self.is_query_exhausted(query_text):
                    queries_to_remove.append(query_text)
                    logger.info(f"Query cooled and ready: {query_text[:50]}... (cooled for {time_cooling/60:.1f}m)")
            
            # Remove cooled queries
            for query_text in queries_to_remove:
                self.exhausted_queries.pop(query_text, None)
                
        # Return the number of queries that were cooled
        return len(queries_to_remove)
    
    def mark_query_exhausted(self, query_text, worker_id, cooling_multiplier=1.0):
        """
        Mark a query as exhausted with a specific cooling multiplier.
        
        Args:
            query_text: The query text to mark as exhausted
            worker_id: ID of the worker that found the query exhausted
            cooling_multiplier: Multiplier for the cooling period (default: 1.0)
        """
        if not query_text:
            return
            
        with self.exhausted_queries_lock:
            current_time = time.time()
            self.exhausted_queries[query_text] = (current_time, worker_id, cooling_multiplier)
    
    def _calculate_adjusted_cooling_period(self, query_text, cooling_multiplier):
        """
        Calculate the adjusted cooling period for a query.
        
        Args:
            query_text: The query text
            cooling_multiplier: The cooling multiplier to apply
            
        Returns:
            The adjusted cooling period in seconds
        """
        adjusted_cooling_period = self.base_cooling_period * cooling_multiplier
        
        # Check for high-star queries to cool faster
        if any(term in query_text for term in ["stars:>10000", "stars:>50000", "stars:>30000"]):
            adjusted_cooling_period /= 2.0
            
        return adjusted_cooling_period
    
    def is_query_exhausted(self, query_text):
        """
        Check if a query is currently exhausted (in cooling period).
        
        Args:
            query_text: The query text to check
            
        Returns:
            Boolean indicating whether the query is currently exhausted
        """
        current_time = time.time()
        
        with self.exhausted_queries_lock:
            if query_text not in self.exhausted_queries:
                return False
                
            # Get the values and calculate if still in cooling period
            timestamp, _, cooling_multiplier = self.exhausted_queries[query_text]
            
            # Calculate time in cooling period
            time_cooling = current_time - timestamp
            
            # Use helper method to calculate adjusted cooling period
            adjusted_cooling_period = self._calculate_adjusted_cooling_period(query_text, cooling_multiplier)
                
            # If cooling period has expired, return False
            return time_cooling <= adjusted_cooling_period
    
    def get_exhausted_queries(self):
        """
        Get a copy of the current exhausted queries dictionary.
        
        Returns:
            Dictionary of exhausted queries
        """
        with self.exhausted_queries_lock:
            return dict(self.exhausted_queries)


class BanditAlgorithm:
    """
    Multi-armed bandit algorithm for optimizing GitHub search queries.
    
    Implements UCB (Upper Confidence Bound) and Thompson Sampling strategies
    for balancing exploration and exploitation in query selection.
    
    This class uses various strategies to select the most promising queries for
    repository discovery while balancing exploration of new query patterns with
    exploitation of known good patterns. It adapts its strategy as the collection
    progresses to optimize for different criteria at different stages.
    
    Key features:
    - UCB (Upper Confidence Bound) for early exploration
    - Thompson Sampling for later exploitation
    - Novelty-guided exploration to diversify results
    - Component-level learning for meta-optimization
    - Adaptive exploration-exploitation balance
    - Query cooling mechanism to avoid query exhaustion
    """
    
    def __init__(self, queries=None, target_repos=10000, 
                similarity_engine=None, metrics_collector=None, cooling_period=1800):
        """
        Initialize the bandit algorithm with query data and parameters.
        
        Args:
            queries: List of query dictionaries with performance metrics
            target_repos: Target number of repositories to collect
            similarity_engine: Optional QuerySimilarityEngine for novelty-guided selection
            metrics_collector: Optional instance of MetricsCollector for tracking metrics
            cooling_period: Optional base cooling period in seconds for exhausted queries
        """
        self.queries = queries or []
        self.similarity_engine = similarity_engine
        self.target_repos = target_repos
        self.found_repos = 0
        self.metrics_collector = metrics_collector
        
        # Initialize the query cooling manager to track exhausted queries
        self.query_cooling_manager = QueryCoolingManager(cooling_period=cooling_period)
        
        # UCB parameters
        self.exploration_weight = 1.0  # Default UCB exploration parameter
        self.cooling_factor = 0.995    # Base cooling factor for gradually reducing exploration
        
        # Thompson sampling parameters
        self.alpha_prior = 1.0  # Prior for beta distribution (success)
        self.beta_prior = 1.0   # Prior for beta distribution (failure)
        
        # Runtime metrics
        self.total_runs = 0
        self.context_features = {}
        
        # Bayesian priors for components
        self.component_prior_alpha = {
            "language": 1.0,
            "stars": 1.0,
            "sort": 1.0,
            "api_per_page": 1.0,
            "creation": 1.0,
            "topic": 1.0
        }
        self.component_prior_beta = {
            "language": 1.0,
            "stars": 1.0,
            "sort": 1.0,
            "api_per_page": 1.0,
            "creation": 1.0,
            "topic": 1.0
        }
        
        # Track success rates for query components
        self.component_success_rates = {
            "language": defaultdict(float),
            "stars": defaultdict(float),
            "sort": defaultdict(float),
            "creation": defaultdict(float),
            "topic": defaultdict(float),
            "api_per_page": defaultdict(float)
        }
        
        # Track component usage count
        self.component_counts = {
            "language": defaultdict(int),
            "stars": defaultdict(int),
            "sort": defaultdict(int),
            "creation": defaultdict(int),
            "topic": defaultdict(int),
            "api_per_page": defaultdict(int)
        }
        
        # Track component metrics
        self.component_metrics = {
            "language": defaultdict(lambda: {"reward": 0.0, "count": 0}),
            "stars": defaultdict(lambda: {"reward": 0.0, "count": 0}),
            "sort": defaultdict(lambda: {"reward": 0.0, "count": 0}),
            "creation": defaultdict(lambda: {"reward": 0.0, "count": 0}),
            "topic": defaultdict(lambda: {"reward": 0.0, "count": 0}),
            "api_per_page": defaultdict(lambda: {"reward": 0.0, "count": 0})
        }
        
        # Enable novelty-guided exploration after sufficient data
        self.enable_novelty = True
        
        # Star range definitions for categorizing repositories
        self.star_ranges = [
            (100000, None, "100K+"),
            (50000, 100000, "50K-100K"),
            (30000, 50000, "30K-50K"),
            (20000, 30000, "20K-30K"),
            (10000, 20000, "10K-20K"),
            (7500, 10000, "7.5K-10K"),
            (5000, 7500, "5K-7.5K"),
            (2500, 5000, "2.5K-5K"),
            (1000, 2500, "1K-2.5K"),
            (500, 1000, "500-1K"),
            (100, 500, "100-500"),
            (50, 100, "50-100"),
            (10, 50, "10-50"),
            (0, 10, "<10")
        ]
    
    
    def get_best_query(self, avoiding_queries=None, strategy="auto"):
        """Get the best query based on bandit algorithm.
        
        Uses UCB (Upper Confidence Bound) for early exploration and 
        Thompson Sampling for later exploitation.
        
        Args:
            avoiding_queries: Optional set of queries to avoid
            strategy: Selection strategy: "ucb", "thompson", or "auto"
            
        Returns:
            The selected query dictionary
        """
        # Filter out avoided queries
        if avoiding_queries:
            available_queries = [q for q in self.queries if q["query_text"] not in avoiding_queries]
        else:
            available_queries = self.queries.copy()
            
        # If there are no available queries, return None
        if not available_queries:
            return None
        
        # Apply novelty-based filtering if enabled
        if self.enable_novelty and self.similarity_engine:
            # Calculate collection progress for adaptive filtering
            try:
                # Get metrics from MetricsCollector instance
                if self.metrics_collector:
                    metrics = self.metrics_collector.get_metrics()
                    self.found_repos = metrics["repositories"]["unique"]
                else:
                    logger.warning("No metrics_collector provided to BanditAlgorithm")
            except Exception as e:
                # If there's an error, use a default value
                logger.warning(f"Could not access metrics collector for unique repository count: {e}")
                
            collection_progress = compute_collection_progress(
                self.found_repos, 
                self.target_repos
            )
            
            # Compute novelty threshold based on collection progress
            novelty_threshold = compute_novelty_threshold(collection_progress)
            
            # Only filter if threshold is above zero (happens in late phase)
            if novelty_threshold > 0:
                # Log the filtering operation
                bandit_logger.info(f"Applying novelty filtering with threshold {novelty_threshold:.2f} "
                                  f"at collection progress {collection_progress:.2f}")
                
                # Count queries before filtering
                before_count = len(available_queries)
                
                # Filter queries by novelty score
                available_queries = [q for q in available_queries 
                                    if q.get("novelty_score", 1.0) >= novelty_threshold]
                
                # Count queries after filtering
                after_count = len(available_queries)
                
                # Log the effect of filtering
                bandit_logger.info(f"Novelty filtering: {before_count} → {after_count} queries "
                                  f"(removed {before_count - after_count} queries)")
                
                # If filtering left too few queries, revert to all queries
                if after_count < 3 and before_count > after_count:
                    logger.warning(f"Novelty filtering left only {after_count} queries. Using all available queries.")
                    available_queries = self.queries
        
        # Track exploration vs exploitation
        exploration_ratio = 0.0
        
        # Determine selection strategy
        if strategy == "ucb" or (strategy == "auto" and self.total_runs < 100):
            # Update UCB scores first
            self._update_ucb_scores()
            
            # Log UCB score distribution for insight into algorithm behavior
            if self.total_runs % 20 == 0:
                ucb_scores = [(q["query_text"][:30], q["ucb_score"], q["reward"]/max(1, q["usage_count"])) 
                              for q in self.queries if q["usage_count"] > 0]
                ucb_scores.sort(key=lambda x: x[1], reverse=True)
                
                bandit_logger.info(f"Top 5 UCB scores: {ucb_scores[:5]}")
                
                # Calculate exploration component
                if ucb_scores:
                    avg_base_reward = sum(score[2] for score in ucb_scores) / len(ucb_scores)
                    avg_ucb = sum(score[1] for score in ucb_scores) / len(ucb_scores)
                    if avg_ucb > 0:
                        exploration_ratio = (avg_ucb - avg_base_reward) / avg_ucb
            
            # Find the query with the highest UCB score
            best_query = max(available_queries, key=lambda q: q["ucb_score"])
            
        elif strategy == "thompson" or strategy == "auto":
            # Use Thompson Sampling
            thompson_scores = self._calculate_thompson_scores()
            
            # Log Thompson sampling distribution occasionally
            if self.total_runs % 20 == 0:
                # Get top scoring queries for insight
                thompson_data = [(i, s, self.queries[i]["query_text"][:30]) 
                                 for i, s in thompson_scores]
                top_thompson = sorted(thompson_data, key=lambda x: x[1], reverse=True)[:5]
                bandit_logger.info(f"Top 5 Thompson scores: {top_thompson}")
                
                # Calculate exploration ratio (variance in scores reflects exploration)
                scores = [s for _, s, _ in thompson_data]
                if scores:
                    import numpy as np
                    std_dev = np.std(scores)
                    mean_score = np.mean(scores)
                    if mean_score > 0:
                        exploration_ratio = std_dev / mean_score
            
            # Filter to only available queries
            available_indices = [i for i, q in enumerate(self.queries) if q in available_queries]
            filtered_scores = [(i, s) for i, s in thompson_scores if i in available_indices]
            
            # Get the query with the highest Thompson score
            best_index, _ = max(filtered_scores, key=lambda x: x[1])
            best_query = self.queries[best_index]
            
        else:
            # Fallback to random with weights based on duplication rate
            weights = [1.0 / (1.0 + q.get("duplication_rate", 0.0)) for q in available_queries]
            best_query = random.choices(available_queries, weights=weights, k=1)[0]
            exploration_ratio = 1.0  # Random selection is pure exploration
        
        # Log the selected query to CLI and bandit file
        query_log_msg = (
            f"Selected query: {best_query['query_text'][:50]}... | "
            f"UCB Score: {best_query.get('ucb_score', 0):.4f} | "
            f"Usage: {best_query['usage_count']} | "
            f"Reward: {best_query['reward']/max(1, best_query['usage_count']):.4f} | "
            f"API Efficiency: {best_query.get('api_efficiency', 0):.2f} | "
            f"Uniq Rate: {best_query.get('unique_rate', 0):.2f} | "
            f"Duplication: {best_query.get('duplication_rate', 0):.2f}"
        )
        
        # Log to both CLI and bandit log file
        logger.info(query_log_msg)
        bandit_logger.debug(query_log_msg)
        
        # Increment usage count and total runs
        best_query["usage_count"] += 1
        self.total_runs += 1
        
        # Cool down exploration weight after a sufficient number of runs
        if self.total_runs > 100:
            old_weight = self.exploration_weight
            self.exploration_weight *= self.cooling_factor
            
            if self.total_runs % 50 == 0:
                bandit_logger.info(f"Cooling exploration weight: {old_weight:.4f} → {self.exploration_weight:.4f}")
        
        # Log performance metrics more frequently for better visibility
        # Every run for the first 5, then every 5 runs
        if self.total_runs <= 5 or self.total_runs % 5 == 0:
            # Force a test log of analytics
            bandit_analytics_logger.info(f"Logging metrics for run {self.total_runs}")
            self._log_learning_metrics(exploration_ratio)
        
        return best_query

    def _update_ucb_scores(self):
        """Update UCB scores for all queries based on performance and exploration weight."""
        for query in self.queries:
            query["ucb_score"] = self._compute_ucb(query)

    def _compute_ucb(self, query):
        """Compute the UCB score for a query.
        
        Args:
            query: Query to compute UCB score for
            
        Returns:
            UCB score for the query
        """
        # If the query has never been used, give it a high UCB score to encourage exploration
        if query.get("usage_count", 0) == 0:
            return float('inf')
        
        # Calculate success rate
        if query.get("usage_count", 0) > 0:
            success_rate = query.get("success_count", 0) / query["usage_count"]
        else:
            success_rate = 0
        
        # Calculate the exploitation component
        # Average reward per usage
        exploitation = query.get("reward", 0) / max(1, query["usage_count"])
        
        # Calculate the exploration component
        # UCB formula: exploration term increases for less-used queries
        # and decreases as total runs increases
        exploration = self.exploration_weight * math.sqrt(math.log(self.total_runs) / query["usage_count"])
        
        # Combine the components
        # Higher weight given to exploitation as we gather more data
        exploration_weight = max(0.1, min(1.0, 100 / max(1, self.total_runs)))
        
        # Dynamic exploration-exploitation balance
        # Early on, explore more. Later, exploit more.
        # Also, if the query has been very successful, reduce exploration
        if query.get("reward", 0) > 0:
            reward_factor = math.log10(query["reward"] + 1)
            exploration_weight = max(0.05, exploration_weight / (1 + reward_factor))
        
        # Calculate final UCB score
        ucb_score = exploitation + exploration_weight * exploration
        
        # Apply duplication penalty - queries with high duplication rates get lower scores
        duplication_penalty = 1.0 - min(0.9, query.get("duplication_rate", 0.0))
        ucb_score = ucb_score * duplication_penalty
        
        return ucb_score
    
    def _calculate_thompson_scores(self):
        """Calculate Thompson sampling scores for all queries.
        
        Returns:
            List of (query_index, score) tuples
        """
        scores = []
        for i, query in enumerate(self.queries):
            # Skip queries that haven't been used yet
            if query.get("usage_count", 0) == 0:
                scores.append((i, 0.0))
                continue
            
            # Get the success and failure counts
            success = query.get("success_count", 0)
            failure = query.get("usage_count", 0) - success
            
            # Adjust for API efficiency and unique rate
            # These are continuous metrics, so we need to convert to "virtual" successes/failures
            if "api_efficiency" in query and query["api_efficiency"] > 0:
                # Scale success by API efficiency (0.0-1.0)
                virtual_success = success * (query["api_efficiency"] / 10.0)  # Normalized to reasonable range
                # Also factor in unique rate
                if "unique_rate" in query:
                    virtual_success *= query["unique_rate"]
                success += virtual_success
            
            # Apply duplication penalty
            duplication_factor = 1.0
            if "duplication_rate" in query:
                duplication_factor = 1.0 - min(0.9, query["duplication_rate"])
            
            # Calculate a score from the beta distribution
            # Add the priors to the empirical counts
            alpha = self.alpha_prior + success * duplication_factor
            beta = self.beta_prior + failure
            
            # Sample from the beta distribution
            score = np.random.beta(alpha, beta)
            scores.append((i, score))
        
        return scores
    
    def update_query_performance(self, query, results_count, unique_count, api_calls=1, 
                                success=True, quality_score=0.0, duplication_rate=0.0,
                                star_counts=None):
        """Update performance metrics for a query with enhanced tracking of historical performance.
        
        Now includes novelty calculation and combined novelty-reward scoring for better exploration.
        
        Args:
            query: The query to update
            results_count: Total number of repositories returned
            unique_count: Number of unique repositories discovered
            api_calls: Number of API calls made for this query
            success: Whether the query execution was successful
            quality_score: Quality score for the repositories (0.0-1.0)
            duplication_rate: Rate of duplication with existing repos (0.0-1.0)
            star_counts: Optional list of star counts for the repositories found by this query
        """
        # Find the query in our pool
        for q in self.queries:
            if q["query_text"] == query["query_text"]:
                # Update basic stats and metrics
                self._update_basic_metrics(q, results_count, unique_count, api_calls, success, quality_score, duplication_rate)
                
                # Calculate collection context and stage
                collection_stage = self.context_features.get("collection_stage", 0.0)
                collection_progress = compute_collection_progress(self.found_repos, self.target_repos)
                
                # Calculate star rewards
                star_reward, star_bucket_distribution = self._calculate_star_reward(
                    q, star_counts, collection_stage
                )
                
                # Calculate base reward components
                base_reward = self._calculate_base_reward(
                    q, unique_count, api_calls, success, quality_score, duplication_rate
                )
                
                # Calculate combined reward with novelty factor
                novelty_score = q.get("novelty_score", 0.5)  # Default to neutral if not available
                alpha = compute_novelty_alpha(collection_progress)
                
                combined_reward = (
                    alpha * base_reward +
                    (1.0 - alpha) * novelty_score +
                    star_reward
                )
                
                # Update query history and rewards
                self._update_query_history(q, combined_reward)
                
                # Update component stats for meta-learning
                self._update_component_stats(
                    q, success, unique_count, results_count, 
                    api_calls, duplication_rate, quality_score
                )
                break
    
    def _update_basic_metrics(self, query, results_count, unique_count, api_calls, success, quality_score, duplication_rate):
        """Update basic metrics for a query.
        
        Args:
            query: The query to update
            results_count: Total number of repositories returned
            unique_count: Number of unique repositories discovered
            api_calls: Number of API calls made
            success: Whether the query execution was successful
            quality_score: Quality score for the repositories
            duplication_rate: Rate of duplication with existing repos
        """
        # Update execution stats
        if success:
            query["success_count"] = query.get("success_count", 0) + 1
        else:
            query["error_count"] = query.get("error_count", 0) + 1
        
        # Update result counts
        query["total_results"] = query.get("total_results", 0) + results_count
        query["unique_results"] = query.get("unique_results", 0) + unique_count
        
        # Update derived metrics
        if results_count > 0:
            query["unique_rate"] = unique_count / results_count
        
        if api_calls > 0:
            # Exponential moving average for API efficiency
            new_efficiency = unique_count / api_calls
            if query.get("api_efficiency", 0.0) == 0.0:
                query["api_efficiency"] = new_efficiency
            else:
                query["api_efficiency"] = 0.8 * query["api_efficiency"] + 0.2 * new_efficiency
        
        # Update quality score with exponential moving average
        if query.get("quality_score", 0.0) == 0.0:
            query["quality_score"] = quality_score
        else:
            query["quality_score"] = 0.8 * query["quality_score"] + 0.2 * quality_score
        
        # Update duplication rate with exponential moving average
        if "duplication_rate" not in query:
            query["duplication_rate"] = 0.0
        query["duplication_rate"] = 0.7 * query["duplication_rate"] + 0.3 * duplication_rate
    
    def _calculate_star_reward(self, query, star_counts, collection_stage):
        """Calculate reward based on star counts of repositories.
        
        Args:
            query: The query being evaluated
            star_counts: List of star counts for repositories found
            collection_stage: Current collection stage (0.0-1.0)
            
        Returns:
            Tuple of (star_reward, star_bucket_distribution)
        """
        star_reward = 0.0
        star_bucket_distribution = {}
        
        # Configure star reward parameters
        star_weight_coefficient = 0.5  # Increased coefficient for star rewards
        
        # Reduce star weight coefficient as collection progresses to favor more exploration
        adjusted_star_weight = star_weight_coefficient * (1.0 - (collection_stage * 0.3))
        
        # Initialize star range multiplier to prioritize high-star queries
        star_range_multiplier = 1.0
        
        if not star_counts or len(star_counts) == 0:
            return star_reward, star_bucket_distribution
        
        # Calculate average star count
        avg_star_count = sum(star_counts) / len(star_counts)
        
        # Apply logarithmic transformation to avoid over-prioritizing very high-star repos
        # Normalize to a 0-1 range (assuming most repos have <100K stars)
        star_reward = star_weight_coefficient * math.log10(avg_star_count + 1) / 5.0
        
        # Track star distribution in buckets
        if "star_bucket_counts" not in query:
            query["star_bucket_counts"] = {}
        
        # Process star counts into buckets
        for star_count in star_counts:
            bucket = self._get_star_bucket(star_count)
            
            # Update bucket counts in query metadata
            if bucket:
                if bucket not in query["star_bucket_counts"]:
                    query["star_bucket_counts"][bucket] = 0
                query["star_bucket_counts"][bucket] += 1
                
                # Also track for this specific update
                if bucket not in star_bucket_distribution:
                    star_bucket_distribution[bucket] = 0
                star_bucket_distribution[bucket] += 1
        
        # Apply the star range multiplier to the star reward, but scale down with collection progress
        # As we collect more, we care less about high stars and more about unique repos
        scaled_multiplier = star_range_multiplier * (1.0 - (collection_stage * 0.4))
        star_reward = star_reward * scaled_multiplier
        
        # Store star-related metrics in query metadata
        query["avg_star_count"] = avg_star_count
        query["last_star_reward"] = star_reward
        query["star_bucket_distribution"] = star_bucket_distribution
        query["star_range_multiplier"] = star_range_multiplier
        
        return star_reward, star_bucket_distribution
    
    def _get_star_bucket(self, star_count):
        """Get the appropriate star bucket for a given star count.
        
        Args:
            star_count: Number of stars
            
        Returns:
            String bucket name
        """
        if star_count >= 100000:
            return "100K+"
        elif star_count >= 50000:
            return "50K-100K"
        elif star_count >= 10000:
            return "10K-50K"
        elif star_count >= 5000:
            return "5K-10K"
        elif star_count >= 1000:
            return "1K-5K"
        elif star_count >= 100:
            return "100-1K"
        else:
            return "<100"
    
    def _calculate_base_reward(self, query, unique_count, api_calls, success, quality_score, duplication_rate):
        """Calculate the base reward for a query execution.
        
        Args:
            query: The query being evaluated
            unique_count: Number of unique repositories discovered
            api_calls: Number of API calls made
            success: Whether the query execution was successful
            quality_score: Quality score for the repositories
            duplication_rate: Rate of duplication with existing repos
            
        Returns:
            Base reward value
        """
        # Define reward component weights
        weights = {
            "api_efficiency": 0.30,    # Finding many unique repos per API call
            "unique_rate": 0.25,       # Proportion of returned results that are unique
            "duplication": 0.20,       # Finding repos not similar to ones we already have
            "information_density": 0.15, # Finding unique repos with fewer total results
            "quality": 0.10            # Quality of repos based on external metrics
        }
        
        # Normalize api_efficiency to 0-1 range (assuming max efficiency of 100)
        # Use a higher ceiling to reward exceptionally efficient queries more
        normalized_efficiency = min(1.0, query.get("api_efficiency", 0.0) / 50.0)
        
        # Information density measure (related to, but distinct from unique_rate)
        # This specifically rewards queries that find unique repos with fewer total results
        information_density = unique_count / max(1, api_calls * 100)  # 100 items per API call maximum
        normalized_density = min(1.0, information_density)
        
        # Success factor: heavily penalize failed queries
        success_factor = 1.0 if success else 0.1
        
        # Calculate the weighted reward
        base_reward = (
            weights["api_efficiency"] * normalized_efficiency +
            weights["unique_rate"] * query.get("unique_rate", 0.0) +
            weights["duplication"] * (1.0 - duplication_rate) +
            weights["information_density"] * normalized_density +
            weights["quality"] * quality_score
        ) * success_factor
        
        return base_reward
    
    def _update_query_history(self, query, combined_reward):
        """Update query history with the new reward.
        
        Args:
            query: The query to update
            combined_reward: The combined reward value to add
        """
        # Increment reward by the new reward amount
        if "reward" not in query:
            query["reward"] = 0.0
        query["reward"] += combined_reward
        
        # Record timestamp and reward in history
        current_time = time.time()
        if "timestamp_history" not in query:
            query["timestamp_history"] = []
        if "reward_history" not in query:
            query["reward_history"] = []
            
        query["timestamp_history"].append(current_time)
        query["reward_history"].append(combined_reward)
        
        # Keep only the most recent 20 history entries to limit memory usage
        if len(query["timestamp_history"]) > 20:
            query["timestamp_history"] = query["timestamp_history"][-20:]
            query["reward_history"] = query["reward_history"][-20:]
    
    def _update_component_stats(self, query, success, unique_count, results_count, 
                               api_calls, duplication_rate, quality_score):
        """Update component statistics for meta-learning.
        
        Args:
            query: The query being updated
            success: Whether the query execution was successful
            unique_count: Number of unique repositories discovered
            results_count: Total number of repositories returned
            api_calls: Number of API calls made for this query
            duplication_rate: Rate of duplication with existing repos (0.0-1.0)
            quality_score: Quality score for the repositories (0.0-1.0)
        """
        # Extract components from query text/config
        components = self._extract_query_components(query)
        
        # Skip if no components were found
        if not components:
            return
        
        # Compute component reward
        base_reward = 0.0
        if success and api_calls > 0:
            # Compute the reward for this query run
            # Favor high unique count, low duplication, high quality, and few API calls
            reward_per_unique = unique_count / api_calls  # Efficiency measure
            base_reward = (
                0.5 * reward_per_unique +  # More unique repos per API call
                0.3 * (1.0 - duplication_rate) +  # Low duplication
                0.2 * quality_score  # High quality repos
            )
        
        # Update component metrics
        for component_type, component_value in components.items():
            # Skip if component type is not tracked
            if component_type not in self.component_metrics:
                continue
            
            # Update usage count
            self.component_counts[component_type][component_value] += 1
            
            # Update success rate
            current_success_rate = self.component_success_rates[component_type][component_value]
            usage_count = self.component_counts[component_type][component_value]
            
            if usage_count == 1:
                # First usage
                self.component_success_rates[component_type][component_value] = 1.0 if success else 0.0
            else:
                # Running average
                new_success_rate = current_success_rate + (1.0 if success else 0.0 - current_success_rate) / usage_count
                self.component_success_rates[component_type][component_value] = new_success_rate
            
            # Update reward metrics
            component_data = self.component_metrics[component_type][component_value]
            component_data["reward"] += base_reward
            component_data["count"] += 1
    
    def _extract_query_components(self, query):
        """Extract components from a query for meta-learning.
        
        Args:
            query: Query to extract components from
            
        Returns:
            Dictionary of component types to component values
        """
        import re
        
        components = {}
        query_text = query.get("query_text", "")
        
        # Use regex patterns to extract components
        patterns = {
            "language": r"language:(\S+)",
            "stars": r"stars:(\S+)",
            "sort": r"sort:(\S+)"
        }
        
        for component, pattern in patterns.items():
            match = re.search(pattern, query_text, re.IGNORECASE)
            if match:
                components[component] = match.group(1)
        
        # Extract per_page from config if present
        if isinstance(query, dict) and "api_per_page" in query:
            components["api_per_page"] = str(query["api_per_page"])
        
        return components
    
    def _log_learning_metrics(self, exploration_ratio):
        """Log learning metrics for bandit algorithm.
        
        Args:
            exploration_ratio: Ratio of exploration vs exploitation
        """
        # Log detailed metrics to the analytics log file
        bandit_analytics_logger.info("==================== BANDIT LEARNING METRICS ====================")
        
        # Log algorithm state
        collection_progress = compute_collection_progress(self.found_repos, self.target_repos)
        
        # Log core metrics
        self._log_collection_metrics(collection_progress, exploration_ratio)
        self._log_performance_metrics()
        self._log_query_diversity_metrics()
        self._log_strategy_metrics()
    
    def _log_collection_metrics(self, collection_progress, exploration_ratio):
        """Log collection progress metrics"""
        bandit_analytics_logger.info(f"Collection progress: {collection_progress:.2f} ({self.found_repos}/{self.target_repos} repositories)")
        bandit_analytics_logger.info(f"Exploration ratio: {exploration_ratio:.3f}, Exploration weight: {self.exploration_weight:.4f}")
    
    def _log_performance_metrics(self):
        """Log performance metrics for the bandit algorithm"""
        # Performance metrics
        total_reward = sum(q.get("reward", 0.0) for q in self.queries)
        avg_reward_per_query = total_reward / max(1, len(self.queries))
        avg_duplication_rate = sum(q.get("duplication_rate", 0.0) for q in self.queries) / max(1, len(self.queries))
        
        bandit_analytics_logger.info(f"Performance: Total reward: {total_reward:.2f}, Avg reward/query: {avg_reward_per_query:.4f}")
        bandit_analytics_logger.info(f"Duplication metrics: Avg rate: {avg_duplication_rate:.3f}")
    
        
        # Top performing queries
        queries_with_usage = [q for q in self.queries if q.get("usage_count", 0) > 5]
        if queries_with_usage:
            # Sort by average reward per use
            queries_with_usage.sort(key=lambda q: q.get("reward", 0) / max(1, q.get("usage_count", 1)), reverse=True)
            
            bandit_analytics_logger.info("Top 5 performing queries:")
            for i, q in enumerate(queries_with_usage[:5], 1):
                avg_reward = q.get("reward", 0) / max(1, q.get("usage_count", 1))
                efficiency = q.get("api_efficiency", 0)
                unique_rate = q.get("unique_rate", 0)
                duplication = q.get("duplication_rate", 0)
                
                bandit_analytics_logger.info(f"  {i}. {q['query_text'][:50]}...")
                bandit_analytics_logger.info(f"     Reward: {avg_reward:.4f}, Efficiency: {efficiency:.2f}, Uniq: {unique_rate:.2f}, Dup: {duplication:.2f}")
        
        # Log the most successful query components
        bandit_analytics_logger.info("Component performance by type:")
        for component_type in self.component_success_rates:
            # Get the components with the highest success rates
            components = [(component, rate) for component, rate in 
                         self.component_success_rates[component_type].items() 
                         if self.component_counts[component_type][component] >= 5]
            
            if components:
                # Sort by success rate (highest first)
                components.sort(key=lambda x: x[1], reverse=True)
                
                # Log the top components
                bandit_analytics_logger.info(f"  Top {component_type} components by success rate:")
                for component, rate in components[:5]:
                    count = self.component_counts[component_type][component]
                    reward = 0
                    if component in self.component_metrics[component_type]:
                        metrics = self.component_metrics[component_type][component]
                        if metrics["count"] > 0:
                            reward = metrics["reward"] / metrics["count"]
                    
                    bandit_analytics_logger.info(f"    {component}: Success={rate:.3f}, Reward={reward:.3f}, Used={count} times")
        
        # Strategy analytics
        # Calculate UCB score distribution to understand exploration patterns
        ucb_scores = [(q["query_text"][:30], q.get("ucb_score", 0), q.get("reward", 0)/max(1, q.get("usage_count", 1)))
                     for q in self.queries if q.get("usage_count", 0) > 0]
        if ucb_scores:
            # Calculate statistics
            ucb_values = [score for _, score, _ in ucb_scores if score != float('inf')]
            if ucb_values:
                avg_ucb = sum(ucb_values) / len(ucb_values)
                min_ucb = min(ucb_values)
                max_ucb = max(ucb_values)
                
                # Log UCB metrics
                bandit_analytics_logger.info("UCB Strategy Metrics:")
                bandit_analytics_logger.info(f"  Score range: {min_ucb:.4f} - {max_ucb:.4f}, Average: {avg_ucb:.4f}")
                bandit_analytics_logger.info(f"  Exploration spread: {max_ucb - min_ucb:.4f}")
                
                # Sort by UCB score to see exploration priorities
                ucb_scores.sort(key=lambda x: x[1], reverse=True)
                bandit_analytics_logger.info(f"  Top UCB scores: {ucb_scores[:3]}")
        
        # Log algorithm parameters
        bandit_analytics_logger.info(f"Algorithm parameters:")
        bandit_analytics_logger.info(f"  Exploration weight: {self.exploration_weight:.4f}, Cooling factor: {self.cooling_factor:.4f}")
        bandit_analytics_logger.info(f"  Thompson priors: alpha={self.alpha_prior:.2f}, beta={self.beta_prior:.2f}")
        bandit_analytics_logger.info("=================================================================")
        
        # Use metrics collector for CSV export if available
        if hasattr(self, 'metrics_collector') and self.metrics_collector:
            try:
                # Prepare metrics data
                collection_progress = compute_collection_progress(self.found_repos, self.target_repos)
                
                # Bundle metrics data for export
                bandit_metrics = {
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "total_runs": self.total_runs,
                    "total_reward": round(total_reward, 2),
                    "avg_reward_per_query": round(avg_reward_per_query, 4),
                    "avg_duplication_rate": round(avg_duplication_rate, 3),
                    "exploration_ratio": round(exploration_ratio, 3),
                    "exploration_weight": round(self.exploration_weight, 4),
                    "found_repos": self.found_repos,
                    "target_repos": self.target_repos,
                    "collection_progress": round(collection_progress, 3)
                }
                
                # Update metrics via the collector
                if hasattr(self.metrics_collector, "update_bandit_metrics"):
                    self.metrics_collector.update_bandit_metrics(bandit_metrics)
                    logger.debug("Bandit metrics passed to metrics collector")
            except Exception as e:
                logger.warning(f"Failed to export bandit metrics: {e}")
        else:
            logger.debug("Metrics collector not available, skipping metrics export")