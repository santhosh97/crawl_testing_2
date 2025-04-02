"""
Metrics and logging utilities for the GitHub Stars Crawler.

This module provides helper functions for tracking and recording performance metrics,
logging statistics, and monitoring the crawler's operation.
"""
import logging
from typing import Dict, Any, Optional
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Set up a dedicated logger for the bandit algorithm
bandit_logger = logging.getLogger('bandit_algorithm')

def update_bandit_metrics(metrics_collector: Any, metrics: Dict[str, Any]) -> None:
    """Update bandit metrics in the centralized metrics collector.
    
    Args:
        metrics_collector: The metrics collector instance
        metrics: Dictionary of bandit metrics to update
    """
    if not metrics_collector:
        # Log a warning and return if no metrics collector available
        logger.warning("No metrics collector provided for bandit metrics update")
        return
        
    # Update bandit metrics in the central collector
    with metrics_collector._lock:
        for key, value in metrics.items():
            if key in metrics_collector._metrics["bandit"]:
                metrics_collector._metrics["bandit"][key] = value
            
    # Log a summary of key metrics to the bandit logger
    bandit_logger.info(
        f"Bandit Metrics | Runs: {metrics.get('total_runs', 0)} | "
        f"API Efficiency: {metrics.get('api_efficiency', 0):.2f} | "
        f"Reward: {metrics.get('average_reward', 0):.4f} | "
        f"Explore/Exploit: {metrics.get('exploration_exploitation_ratio', 0):.2f} | "
        f"Best Query: {metrics.get('best_query_types', [])} | "
        f"Path Rewards: {metrics.get('path_rewards', 0.0):.4f}"
    )

def update_cache_metrics(metrics_collector: Any, cache_metrics: Dict[str, Any]) -> None:
    """Update cache metrics in the centralized metrics collector.
    
    Args:
        metrics_collector: The metrics collector instance
        cache_metrics: Dictionary of cache metrics to update
    """
    if not metrics_collector:
        # Log a warning and return if no metrics collector available
        logger.warning("No metrics collector provided for cache metrics update")
        return
        
    # Add timestamp to metrics
    cache_metrics['timestamp'] = datetime.now().isoformat()
    
    # Update cache metrics directly in the collector for efficiency
    with metrics_collector._lock:
        for key, value in cache_metrics.items():
            if key in metrics_collector._metrics["cache"]:
                metrics_collector._metrics["cache"][key] = value
    
    # Log a summary of key metrics
    logger.info(
        f"Cache Metrics | Size: {cache_metrics.get('size', 0)}/{cache_metrics.get('capacity', 0)} | "
        f"Hit Rate: {cache_metrics.get('hit_rate', 0):.2f} | "
        f"Evictions: {cache_metrics.get('evictions', 0)} | "
        f"Unique Repos: {cache_metrics.get('unique_repositories', 0)}"
    )

def get_query_effectiveness_score(query_stats: Dict[str, Any]) -> float:
    """Calculate effectiveness score for a query based on historical performance.
    
    Args:
        query_stats: Statistics about query's past performance
        
    Returns:
        Effectiveness score between 0.0 and 1.0
    """
    # Extract statistics
    runs = query_stats.get("runs", 0)
    if runs == 0:
        return 0.0
        
    unique_results = query_stats.get("unique_results", 0)
    total_results = query_stats.get("total_results", 0)
    api_calls = query_stats.get("api_calls", 0)
    high_star_repos = query_stats.get("high_star_repos", 0)
    
    # Calculate metrics
    novelty_rate = unique_results / total_results if total_results > 0 else 0
    api_efficiency = unique_results / api_calls if api_calls > 0 else 0
    high_star_rate = high_star_repos / unique_results if unique_results > 0 else 0
    
    # Calculate overall effectiveness (weighted combination)
    effectiveness = (
        0.4 * novelty_rate +
        0.4 * api_efficiency +
        0.2 * high_star_rate
    )
    
    return effectiveness

def calculate_exploration_weight(collection_progress: float) -> float:
    """Calculate an appropriate exploration weight based on collection progress.
    
    Args:
        collection_progress: Current collection progress (0.0 to 1.0)
        
    Returns:
        Appropriate exploration weight value
    """
    # Calculate exploration weight based on progress
    if collection_progress < 0.3:
        # Early phase: Moderate exploration
        exploration_weight = 2.0
    elif collection_progress < 0.7:
        # Mid phase: High exploration
        exploration_weight = 3.0
    else:
        # Late phase: Focus on exploitation of best patterns
        exploration_weight = 1.0
    
    logger.debug(f"Calculated exploration weight {exploration_weight} based on progress {collection_progress:.2f}")
    return exploration_weight