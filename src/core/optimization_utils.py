"""
Optimization utilities for GitHub Stars Crawler.

This module provides optimization algorithms for query selection and evaluation,
including Pareto-optimal query identification.
"""
import math
from typing import Dict, List, Any, Optional, Callable


def is_pareto_optimal(query: Dict[str, Any], query_pool: List[Dict[str, Any]], query_pool_instance=None) -> bool:
    """Determine if a query is on the Pareto frontier of multiple objectives.
    
    A query is Pareto optimal if no other query dominates it across all objectives.
    This implements multi-objective optimization to balance competing goals.
    
    Args:
        query: The query to evaluate
        query_pool: Pool of queries to compare against
        query_pool_instance: Optional QueryPool instance for path reward calculation
        
    Returns:
        True if the query is Pareto optimal, False otherwise
    """
    # Define objective functions to maximize (higher is better)
    objectives = [
        lambda q: q.get("api_efficiency", 0.0),           # API efficiency
        lambda q: q.get("unique_rate", 0.0),              # Uniqueness rate
        lambda q: q.get("quality_score", 0.0),            # Quality score
        lambda q: 1.0 - q.get("duplication_rate", 0.0),   # Inverse duplication rate (higher is better)
        lambda q: query_pool_instance.get_path_reward(q.get("query_text", "")) if query_pool_instance else 0.0,  # Path reward
    ]
    
    # Check if any other query dominates this one
    for other_query in query_pool:
        if other_query == query:
            continue
            
        # Check if other query is at least as good in all objectives
        at_least_as_good = all(
            obj_func(other_query) >= obj_func(query) 
            for obj_func in objectives
        )
        
        # Check if other query is strictly better in at least one objective
        strictly_better = any(
            obj_func(other_query) > obj_func(query) 
            for obj_func in objectives
        )
        
        # If other query dominates this one, it's not Pareto optimal
        if at_least_as_good and strictly_better:
            return False
    
    # If no other query dominates this one, it's Pareto optimal
    return True


def get_pareto_optimal_queries(query_pool: List[Dict[str, Any]], query_pool_instance=None) -> List[Dict[str, Any]]:
    """Get the set of Pareto optimal queries from the pool.
    
    This identifies queries that represent optimal tradeoffs between competing objectives.
    
    Args:
        query_pool: Pool of queries to evaluate
        query_pool_instance: Optional QueryPool instance for path reward calculation
        
    Returns:
        List of Pareto optimal queries
    """
    return [query for query in query_pool if is_pareto_optimal(query, query_pool, query_pool_instance)]


def calculate_dynamic_reward_weights(collection_progress: float) -> Dict[str, float]:
    """Calculate dynamic reward weights based on collection progress.
    
    Args:
        collection_progress: Progress value from 0.0 (just started) to 1.0 (complete)
        
    Returns:
        Dictionary of weights for different reward components
    """
    # Base weights - these will be adjusted dynamically
    weights = {
        "unique_rate": 0.0,      # Uniqueness of results
        "api_efficiency": 0.0,   # API calls efficiency 
        "quality_score": 0.0,    # Repository quality
        "path_reward": 0.0,      # Long-term value of query paths
        "diversity": 0.0,        # Collection diversity
    }
    
    # Simple focus only on number of results and duplicates across all stages
    # Set high weights for unique_rate and diversity (which corresponds to 1.0 - duplication_rate)
    weights["unique_rate"] = 0.33     # High focus on unique repos
    weights["api_efficiency"] = 0.33   # Not considering API efficiency
    weights["quality_score"] = 0.0    # Not considering quality score
    weights["path_reward"] = 0.0      # Not considering path reward
    weights["diversity"] = 0.33        # High focus on avoiding duplicates (1.0 - duplication_rate)
    
    return weights