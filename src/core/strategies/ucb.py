"""
Upper Confidence Bound (UCB) strategy for query selection.

This module implements the UCB strategy for query selection, which balances
exploration and exploitation based on the upper confidence bound formula.
"""
import math
import random
from typing import Dict, Any, List, Optional, Set

from .base import QuerySelectionStrategy
from src2.metrics.utils import calculate_exploration_weight


class UCBStrategy(QuerySelectionStrategy):
    """
    Upper Confidence Bound (UCB) strategy for query selection.
    
    This strategy selects queries based on their UCB score, which balances
    the query's observed value (exploitation) with the uncertainty in that
    value (exploration).
    """
    
    def __init__(self, exploration_weight=1.0):
        """
        Initialize the UCB strategy.
        
        Args:
            exploration_weight: Weight for exploration vs. exploitation
        """
        self.exploration_weight = exploration_weight
        self._exploration_weight = exploration_weight
        
    def _calculate_contextual_factor(self, context_features=None):
        """
        Calculate contextual weight factor.
        
        Args:
            context_features: Dictionary of context features
            
        Returns:
            Adjustment factor for UCB calculation
        """
        if not context_features:
            return 1.0
            
        # Extract relevant features
        collection_progress = context_features.get("collection_progress", 0.0)
        recent_efficiency = context_features.get("recent_efficiency", 0.5)
        
        # Adjust based on collection progress
        # Early: High exploration, Late: Focus on exploitation
        progress_factor = 1.5 - collection_progress
        
        # Adjust based on recent efficiency
        # High efficiency: Reduce exploration, Low efficiency: Increase exploration
        efficiency_factor = 1.0 + (1.0 - recent_efficiency)
        
        # Combine factors
        return progress_factor * efficiency_factor
        
    def _calculate_ucb_score(self, query_key, stats, total_runs, path_reward=0.0, context_features=None):
        """
        Calculate the UCB score for a query.
        
        Args:
            query_key: Key identifying the query
            stats: Statistics for the query
            total_runs: Total number of runs across all queries
            path_reward: Additional reward from query paths
            context_features: Optional context features for adjustment
            
        Returns:
            UCB score for the query
        """
        # Extract values
        value = stats.get("value", 0.5)
        runs = stats.get("runs", 0)
        
        # Calculate contextual factor for exploration
        contextual_factor = self._calculate_contextual_factor(context_features)
        
        # Enhanced UCB with diminishing returns for pure exploration
        log_term = min(10, math.log(total_runs + 1))  # Cap to prevent explosion
        exploration_term = self._exploration_weight * math.sqrt(log_term / (runs + 1))
        
        # Apply contextual factor
        exploration_term *= contextual_factor
        
        # Calculate UCB score
        ucb_score = value + exploration_term
        
        # Add path reward
        ucb_score += 0.2 * path_reward  # Weight path reward as 20% of score
        
        return ucb_score
    
    def _update_exploration_weight(self, context_features=None):
        """
        Update exploration weight based on context features.
        
        Args:
            context_features: Dictionary of features about the current context
                (e.g., collection progress, duplication rate, etc.)
        """
        if not context_features:
            return
            
        # Get collection progress
        progress = context_features.get("collection_progress", 0)
        
        # Get the base target weight from the metrics utility function
        target_weight = calculate_exploration_weight(progress)
        
        # Factor in recent efficiency
        recent_efficiency = context_features.get("recent_efficiency", 0.5)
        # If efficiency is poor, increase exploration
        efficiency_factor = 1.0 + (1.0 - recent_efficiency) * 0.5
        
        # Factor in duplication rate
        duplication_rate = context_features.get("duplication_rate", 0)
        # If duplication is high, increase exploration
        dup_factor = 1.0 + duplication_rate * 0.5
        
        # Combine factors
        adjusted_weight = target_weight * efficiency_factor * dup_factor
        
        # Apply soft limit to prevent extreme values
        max_weight = self.exploration_weight * 5
        adjusted_weight = min(adjusted_weight, max_weight)
        
        # Smooth the transition (50% old weight, 50% new weight)
        self._exploration_weight = 0.5 * self._exploration_weight + 0.5 * adjusted_weight
    
    def select_query(self, query_pool, query_stats, active_queries, context_features=None):
        """
        Select the best query using the UCB strategy.
        
        Args:
            query_pool: List of query configurations to select from
            query_stats: Dictionary of query statistics for each query
            active_queries: Set of query keys to avoid (currently in use)
            context_features: Optional dictionary of context features
            
        Returns:
            The selected query or None if no suitable query is found
        """
        # Update exploration weight based on context
        self._update_exploration_weight(context_features)
        
        # Make a copy of query pool to work with
        available_queries = list(query_pool)
        
        # Shuffle to randomize ties
        random.shuffle(available_queries)
        
        # Filter out active queries
        filtered_queries = [q for q in available_queries if str(q) not in active_queries]
        
        # If no queries left after filtering, return None
        if not filtered_queries:
            return None
            
        # Calculate total runs for UCB formula
        total_runs = sum(stats.get("runs", 0) for stats in query_stats.values())
        total_runs = max(1, total_runs)  # Avoid division by zero
        
        # Find best UCB score
        best_query = None
        best_score = float('-inf')
        
        for query in filtered_queries:
            query_key = str(query)
            stats = query_stats.get(query_key, {"value": 0.5, "runs": 0})
            
            # Calculate path reward if a method is provided
            path_reward = 0.0
            
            # Calculate UCB score
            score = self._calculate_ucb_score(query_key, stats, total_runs, path_reward, context_features)
            
            # Update best if this score is higher
            if score > best_score:
                best_score = score
                best_query = query
                
        return best_query