"""
Hybrid strategy for query selection.

This module implements a hybrid strategy that combines multiple
selection strategies to balance their strengths and weaknesses.
"""
import random
from typing import Dict, Any, List, Optional, Set

from .base import QuerySelectionStrategy
from .ucb import UCBStrategy
from .thompson import ThompsonSamplingStrategy
from .bayesian import BayesianStrategy


class HybridStrategy(QuerySelectionStrategy):
    """
    Hybrid strategy for query selection.
    
    This strategy combines multiple selection strategies and
    dynamically chooses between them based on context.
    """
    
    def __init__(self, exploration_weight=1.0, **kwargs):
        """
        Initialize the hybrid strategy with sub-strategies.
        
        Args:
            exploration_weight: Weight for exploration vs. exploitation
            **kwargs: Additional keyword arguments
        """
        self.strategies = {
            "thompson": ThompsonSamplingStrategy(exploration_weight=exploration_weight),
            "ucb": UCBStrategy(exploration_weight=exploration_weight),
            "bayesian": BayesianStrategy(exploration_weight=exploration_weight)
        }
        
        # Default weights for strategies (higher = more likely to be chosen)
        self.weights = {
            "thompson": 0.5,  # Favor Thompson sampling
            "ucb": 0.3,
            "bayesian": 0.2
        }
        
    def select_query(self, query_pool, query_stats, active_queries, context_features=None, query_priors=None, get_path_reward=None):
        """
        Select the best query using a hybrid approach.
        
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
        # Choose a strategy based on weights
        strategy_names = list(self.strategies.keys())
        strategy_weights = [self.weights[name] for name in strategy_names]
        
        chosen_strategy_name = random.choices(strategy_names, weights=strategy_weights, k=1)[0]
        chosen_strategy = self.strategies[chosen_strategy_name]
        
        # Call the chosen strategy
        if chosen_strategy_name in ["thompson", "bayesian"]:
            return chosen_strategy.select_query(
                query_pool=query_pool,
                query_stats=query_stats,
                active_queries=active_queries,
                context_features=context_features,
                query_priors=query_priors,
                get_path_reward=get_path_reward
            )
        else:
            return chosen_strategy.select_query(
                query_pool=query_pool,
                query_stats=query_stats,
                active_queries=active_queries,
                context_features=context_features
            )