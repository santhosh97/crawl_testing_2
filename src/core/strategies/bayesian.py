"""
Bayesian strategy for query selection.

This module implements a Bayesian strategy for query selection,
which uses priors to inform query selection.
"""
from typing import Dict, Any, List, Optional, Set
import random

from .base import QuerySelectionStrategy


class BayesianStrategy(QuerySelectionStrategy):
    """
    Bayesian strategy for query selection.
    
    This strategy selects queries based on Bayesian inference using
    prior information about query performance.
    """
    
    def __init__(self, exploration_weight=1.0, **kwargs):
        """
        Initialize the Bayesian strategy.
        
        Args:
            exploration_weight: Weight for exploration vs. exploitation
            **kwargs: Additional keyword arguments (ignored)
        """
        # Bayesian strategy doesn't use exploration_weight directly
        # but we accept it for API compatibility
        pass
    
    def _calculate_bayesian_score(self, query_key, query_priors, path_reward=0.0):
        """
        Calculate Bayesian score for a query.
        
        Args:
            query_key: Key identifying the query
            query_priors: Dictionary of prior parameters for each query
            path_reward: Additional reward from query paths
            
        Returns:
            Bayesian score for the query
        """
        if query_key not in query_priors:
            return 0.5  # Default score
            
        prior = query_priors[query_key]
        alpha = prior.get("alpha", 1.0)
        beta = prior.get("beta", 1.0)
        
        # Expected value of Beta distribution
        expected_value = alpha / (alpha + beta)
        
        # Add path reward
        score = expected_value + 0.2 * path_reward
        
        return score
    
    def select_query(self, query_pool, query_stats, active_queries, context_features=None, query_priors=None, get_path_reward=None):
        """
        Select the best query using the Bayesian strategy.
        
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
            
        # Calculate Bayesian scores
        best_query = None
        best_score = float('-inf')
        
        for query in filtered_queries:
            query_key = str(query)
            
            # Calculate path reward if a function is provided
            path_reward = 0.0
            if get_path_reward:
                path_reward = get_path_reward(query)
            
            # Calculate Bayesian score
            score = self._calculate_bayesian_score(query_key, query_priors, path_reward)
            
            # Update best if this score is higher
            if score > best_score:
                best_score = score
                best_query = query
                
        return best_query