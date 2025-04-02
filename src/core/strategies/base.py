"""
Base strategy interface for query selection algorithms.

This module defines the base strategy interface that all query selection
strategies must implement. It follows the Strategy pattern to allow
QueryPool to use different query selection algorithms interchangeably.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set, Union


class QuerySelectionStrategy(ABC):
    """Base class for query selection strategies.
    
    All query selection strategies must inherit from this class and implement
    the select_query method.
    """
    
    @abstractmethod
    def select_query(self, 
                     query_pool: List[Dict[str, Any]],
                     query_stats: Dict[str, Dict[str, Any]],
                     active_queries: Set[str],
                     context_features: Optional[Dict[str, float]] = None) -> Optional[Dict[str, Any]]:
        """
        Select the best query from the pool based on the strategy.
        
        Args:
            query_pool: List of query configurations to select from
            query_stats: Dictionary of query statistics for each query
            active_queries: Set of query keys to avoid (currently in use)
            context_features: Optional dictionary of context features
                (e.g., collection_progress, recent_efficiency, etc.)
                
        Returns:
            The selected query configuration or None if no suitable query is found
        """
        pass