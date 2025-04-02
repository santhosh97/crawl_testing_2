"""
Metrics package for GitHub Stars Crawler.

This package provides unified metrics collection, tracking, and reporting
for the GitHub Stars Crawler system.
"""

from .collector.collector import MetricsCollector
from .utils.utils import (
    get_query_effectiveness_score,
    calculate_exploration_weight
)

# Helper functions for creating or accessing metrics collectors
# will be provided in application.py instead via dependency injection