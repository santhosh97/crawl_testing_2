"""
Metrics utilities for GitHub Stars Crawler.

This module provides utility functions for metrics logging, calculation,
and analysis across the crawler system.
"""

from .utils import (
    update_bandit_metrics,
    update_cache_metrics,
    get_query_effectiveness_score,
    calculate_exploration_weight
)