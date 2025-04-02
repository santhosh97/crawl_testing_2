"""
Thompson Sampling strategy for query selection.

This module implements the Thompson Sampling strategy for query selection,
which selects queries based on probability matching from Beta distributions.
"""
import random
from typing import Dict, Any, List, Optional, Set

from .base import QuerySelectionStrategy


class ThompsonSamplingStrategy(QuerySelectionStrategy):
    """
    Thompson Sampling strategy for query selection.
    
    This strategy selects queries by sampling from Beta distributions that
    represent our beliefs about each query's performance.
    """
    
    def __init__(self, exploration_weight=1.0, **kwargs):
        """
        Initialize the Thompson Sampling strategy.
        
        Args:
            exploration_weight: Weight for exploration vs. exploitation
            **kwargs: Additional keyword arguments (ignored)
        """
        # Thompson sampling doesn't use exploration_weight directly
        # but we accept it for API compatibility
        pass
    
    def _calculate_thompson_score(self, query_key, query_stats, query_priors, path_reward=0.0):
        """
        Calculate Thompson Sampling score for a query.
        
        Args:
            query_key: Key identifying the query
            query_stats: Dictionary of query statistics
            query_priors: Dictionary of prior parameters (alpha, beta) for each query
            path_reward: Additional reward from query paths
            
        Returns:
            Thompson score for the query
        """
        # Skip if no stats or prior
        if query_key not in query_stats or query_key not in query_priors:
            return 0.0
            
        # Get prior
        prior = query_priors.get(query_key, {"alpha": 1.0, "beta": 1.0})
        alpha = prior["alpha"]
        beta = prior["beta"]
        
        # Sample from Beta distribution
        try:
            thompson_score = random.betavariate(alpha, beta)
            
            # Blend with actual observed value for more stability
            value = query_stats[query_key].get("value", 0.5)
            blended_score = 0.7 * thompson_score + 0.3 * value
            
            # Add path reward
            blended_score += 0.2 * path_reward
            
            return blended_score
        except ValueError:
            # Handle case where alpha or beta are invalid
            return 0.0
    
    def select_query(self, query_pool, query_stats, active_queries, context_features=None, query_priors=None, get_path_reward=None):
        """
        Select the best query using Thompson Sampling.
        
        Args:
            query_pool: List of query configurations to select from
            query_stats: Dictionary of query statistics for each query
            active_queries: Set of query keys to avoid (currently in use)
            context_features: Optional dictionary of context features
            query_priors: Dictionary of prior parameters for each query
            get_path_reward: Optional function to calculate path rewards
            
        Returns:
            The selected query or None if no suitable query is found
        """
        # Make a copy of query pool to work with
        available_queries = list(query_pool)
        
        # Shuffle to randomize ties
        random.shuffle(available_queries)
        
        # Filter out active queries
        filtered_queries = [q for q in available_queries if str(q) not in active_queries]
        
        # If no queries left after filtering, return None
        if not filtered_queries:
            return None
        
        # Default priors if not provided
        if query_priors is None:
            query_priors = {str(q): {"alpha": 1.0, "beta": 1.0} for q in filtered_queries}
        
        # Calculate Thompson scores
        best_query = None
        best_score = float('-inf')
        
        for query in filtered_queries:
            query_key = str(query)
            
            # Calculate path reward if a function is provided
            path_reward = 0.0
            if get_path_reward:
                path_reward = get_path_reward(query)
            
            # Calculate Thompson score
            score = self._calculate_thompson_score(query_key, query_stats, query_priors, path_reward)
            
            # Update best if this score is higher
            if score > best_score:
                best_score = score
                best_query = query
                
        return best_query