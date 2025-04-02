"""
Query selection strategies for the GitHub Stars Crawler.

This module provides a collection of strategies for selecting queries in the
QueryPool based on various algorithms and techniques.
"""

from .base import QuerySelectionStrategy
from .ucb import UCBStrategy
from .thompson import ThompsonSamplingStrategy
from .bayesian import BayesianStrategy
from .hybrid import HybridStrategy

# Import all strategy classes for direct instantiation by callers