"""
Metrics collection module for GitHub Stars Crawler.

This module provides the core metrics collection and storage
capabilities used throughout the system.
"""

from .collector import MetricsCollector

# Import only the class - no global singleton instance
# Components should receive MetricsCollector instances via dependency injection