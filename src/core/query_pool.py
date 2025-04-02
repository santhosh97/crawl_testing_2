
import os
import logging
import time
import threading
import random
import itertools
import hashlib
import json
import csv
import math
import re
import zlib
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Union, Iterator, Tuple
from pathlib import Path
from collections import defaultdict

# Import centralized configuration early
from typing import TYPE_CHECKING

# Import interfaces instead of concrete classes
from src.interfaces import IConfig, IMetricsCollector
from src.core.query_evolution import evolve_queries, QuerySimilarityEngine, compute_mutation_intensity, compute_novelty_threshold, compute_collection_progress, compute_novelty_alpha

# Use TYPE_CHECKING for imports only needed for type checking
if TYPE_CHECKING:
    from src.core.config import Config, get_initial_queries
    from src.metrics.collector.collector import MetricsCollector

import requests
from requests.adapters import HTTPAdapter


class StatsUtils:
    """Utility class for statistical calculations and metrics.
    
    This class provides helper methods for common calculations used in the system,
    such as exponential moving averages and normalization functions.
    """
    
    @staticmethod
    def exponential_moving_average(current: float, new_value: float, weight: float = 0.8) -> float:
        """Calculate exponential moving average.
        
        Args:
            current: Current value
            new_value: New value to incorporate
            weight: Weight for current value (0.0-1.0)
            
        Returns:
            Updated EMA value
        """
        return current * weight + new_value * (1.0 - weight)
    
    @staticmethod
    def normalize_probabilities(values: Dict[str, float]) -> Dict[str, float]:
        """Normalize a dictionary of values to sum to 1.0.
        
        Args:
            values: Dictionary of values to normalize
            
        Returns:
            Dictionary with normalized values
        """
        if not values:
            return {}
            
        total = sum(values.values())
        if total <= 0:
            return {k: 1.0 / len(values) for k in values}
            
        return {k: v / total for k, v in values.items()}
    
    @staticmethod
    def calculate_success_rate(success_count: int, total_count: int, min_rate: float = 0.0, default: float = 0.5) -> float:
        """Calculate success rate with safeguards.
        
        Args:
            success_count: Number of successes
            total_count: Total number of attempts
            min_rate: Minimum allowed rate
            default: Default rate if total_count is 0
            
        Returns:
            Success rate between min_rate and 1.0
        """
        if total_count == 0:
            return default
            
        rate = success_count / total_count
        return max(min_rate, min(1.0, rate))


class QueryMutator:
    """Class responsible for mutating queries to generate variations.
    
    This class provides methods to mutate different aspects of GitHub search queries,
    such as star ranges, date ranges, languages, and sorting options.
    """
    
    # Define language options for mutation
    LANGUAGES = [
        "javascript", "python", "java", "typescript", "go", "c", "cpp", "csharp", 
        "php", "ruby", "swift", "kotlin", "rust", "scala", "dart", "shell", "elixir"
    ]
    
    # Define sort options for mutation
    SORT_OPTIONS = ["stars", "forks", "updated", "help-wanted-issues"]
    
    # Define order options for mutation
    ORDER_OPTIONS = ["desc", "asc"]
    
    @staticmethod
    def mutate_star_range(star_range: str) -> str:
        """Mutate a star range component.
        
        Args:
            star_range: Original star range string
            
        Returns:
            Mutated star range string
        """
        # Parse the original range
        min_stars, max_stars = QueryBuilder.parse_star_range(star_range)
        
        if min_stars is None:
            # Default range if parsing failed
            return "10..1000" 
            
        # Apply a mutation strategy
        mutation_type = random.choice([
            "narrow", "widen", "shift_up", "shift_down", "radically_change"
        ])
        
        if mutation_type == "narrow":
            # Narrow the range
            if max_stars is None:
                # For unbounded ranges, add an upper bound
                return f"{min_stars}..{min_stars*10}"
            else:
                # For bounded ranges, narrow the window
                new_min = min_stars + int((max_stars - min_stars) * 0.2)
                new_max = max_stars - int((max_stars - min_stars) * 0.2)
                return f"{new_min}..{new_max}"
                
        elif mutation_type == "widen":
            # Widen the range
            new_min = max(0, min_stars - int(min_stars * 0.3))
            if max_stars is None:
                return f"{new_min}.."
            else:
                new_max = max_stars + int(max_stars * 0.3)
                return f"{new_min}..{new_max}"
                
        elif mutation_type == "shift_up":
            # Shift range up
            shift = int(min_stars * 0.5) + 10
            new_min = min_stars + shift
            new_max = None if max_stars is None else max_stars + shift
            return f"{new_min}..{new_max}" if new_max else f"{new_min}.."
            
        elif mutation_type == "shift_down":
            # Shift range down (ensure min_stars doesn't go below 0)
            shift = min(int(min_stars * 0.5), min_stars - 1) if min_stars > 1 else 0
            new_min = max(0, min_stars - shift)
            new_max = None if max_stars is None else max(0, max_stars - shift)
            return f"{new_min}..{new_max}" if new_max else f"{new_min}.."
            
        else:  # radically_change
            # Pick a completely new star range category
            categories = [
                "0..10", "10..50", "50..100", "100..500", "500..1000", 
                "1000..2500", "2500..5000", "5000..7500", "7500..10000", 
                "10000..20000", "20000..30000", "30000..50000", "50000..100000"
            ]
            return random.choice(categories)
    
    @staticmethod
    def mutate_language(language: Optional[str] = None) -> str:
        """Mutate a language component.
        
        Args:
            language: Original language string or None
            
        Returns:
            Mutated language string
        """
        if language is None or random.random() < 0.3:
            # 30% chance to pick a completely new language
            return random.choice(QueryMutator.LANGUAGES)
        else:
            # 70% chance to pick a related language
            language_groups = {
                "javascript": ["typescript", "jsx", "nodejs"],
                "python": ["jupyter-notebook", "python3"],
                "java": ["kotlin", "scala", "groovy"],
                "typescript": ["javascript", "tsx"],
                "go": ["golang"],
                "c": ["cpp", "c-plus-plus"],
                "cpp": ["c", "c-plus-plus"],
                "csharp": ["c-sharp", "f-sharp"],
                "php": ["hack"],
                "ruby": ["ruby-on-rails"],
                "swift": ["objective-c"],
                "kotlin": ["java"],
                "rust": ["c", "cpp"],
                "scala": ["java"],
                "dart": ["flutter"],
                "shell": ["bash", "zsh", "powershell"],
                "elixir": ["erlang"]
            }
            
            if language in language_groups:
                related = language_groups[language]
                return random.choice(related + QueryMutator.LANGUAGES)
            else:
                return random.choice(QueryMutator.LANGUAGES)
    
    @staticmethod
    def mutate_date_range(date_range: Optional[str] = None) -> str:
        """Mutate a date range component.
        
        Args:
            date_range: Original date range string or None
            
        Returns:
            Mutated date range string
        """
        # Define time periods
        periods = [
            (">2023-01-01", "Recent"),
            ("2022-01-01..2023-01-01", "Last 1-2 years"),
            ("2020-01-01..2022-01-01", "2-3 years ago"),
            ("2015-01-01..2020-01-01", "3-8 years ago"),
            ("2010-01-01..2015-01-01", "8-13 years ago"),
            ("2005-01-01..2010-01-01", "13-18 years ago"),
            ("<2005-01-01", "Very old")
        ]
        
        # Default behavior is to select a random period
        return random.choice([p[0] for p in periods])
    
    @staticmethod
    def mutate_sort_option(sort: Optional[str] = None) -> str:
        """Mutate a sort option.
        
        Args:
            sort: Original sort option or None
            
        Returns:
            Mutated sort option
        """
        if sort is None or random.random() < 0.3:
            # 30% chance to pick a random sort option
            return random.choice(QueryMutator.SORT_OPTIONS)
        else:
            # 70% chance to pick a different sort option than current
            options = [s for s in QueryMutator.SORT_OPTIONS if s != sort]
            if options:
                return random.choice(options)
            else:
                return random.choice(QueryMutator.SORT_OPTIONS)
    
    @staticmethod
    def mutate_query(query_text: str, mutation_type: Optional[str] = None) -> str:
        """Apply a comprehensive mutation to a query.
        
        Args:
            query_text: Original query text
            mutation_type: Optional specific mutation type to apply
            
        Returns:
            Mutated query text
        """
        # Parse the original query
        components = QueryBuilder.parse_query(query_text)
        
        # If no specific mutation type, pick one randomly
        if mutation_type is None:
            mutation_type = random.choice([
                "star_range", "language", "date_range", "sort_option", "mixed"
            ])
        
        # Apply the mutation
        if mutation_type == "star_range" or mutation_type == "mixed":
            if "stars" in components:
                components["stars"] = QueryMutator.mutate_star_range(components["stars"])
            else:
                components["stars"] = QueryMutator.mutate_star_range("10..1000")
                
        if mutation_type == "language" or mutation_type == "mixed":
            components["language"] = QueryMutator.mutate_language(components.get("language"))
                
        if mutation_type == "date_range" or mutation_type == "mixed":
            components["creation"] = QueryMutator.mutate_date_range(components.get("creation"))
                
        if mutation_type == "sort_option" or mutation_type == "mixed":
            components["sort"] = QueryMutator.mutate_sort_option(components.get("sort"))
            components["order"] = random.choice(QueryMutator.ORDER_OPTIONS)
        
        # Build the new query
        return QueryBuilder.build_query(components)


class QueryBuilder:
    """Helper class for parsing and building GitHub search queries.
    
    This class provides utility methods to parse query components, build query strings,
    and manipulate query parameters in a consistent way.
    """
    
    @staticmethod
    def parse_query(query_text: str) -> Dict[str, str]:
        """Parse a query string into its components.
        
        Args:
            query_text: Query string to parse
            
        Returns:
            Dictionary of component types and values
        """
        components = {}
        
        # Extract language
        language_match = re.search(r'language:([^\s]+)', query_text.lower())
        if language_match:
            components['language'] = language_match.group(1)
        
        # Extract stars
        stars_match = re.search(r'stars:([^\s]+)', query_text.lower())
        if stars_match:
            components['stars'] = stars_match.group(1)
        
        # Extract sort
        sort_match = re.search(r'sort:([^\s]+)', query_text.lower())
        if sort_match:
            components['sort'] = sort_match.group(1)
            
        # Extract creation date range
        created_match = re.search(r'created:([^\s]+)', query_text.lower())
        if created_match:
            components['creation'] = created_match.group(1)
            
        # Extract topic
        topic_match = re.search(r'topic:([^\s]+)', query_text.lower())
        if topic_match:
            components['topic'] = topic_match.group(1)
        
        return components
    
    @staticmethod
    def build_query(components: Dict[str, str]) -> str:
        """Build a query string from its components.
        
        Args:
            components: Dictionary of component types and values
            
        Returns:
            Complete query string
        """
        query_parts = []
        
        # Add language component if present
        if 'language' in components:
            query_parts.append(f"language:{components['language']}")
        
        # Add stars component if present
        if 'stars' in components:
            query_parts.append(f"stars:{components['stars']}")
        
        # Add creation date component if present
        if 'creation' in components:
            query_parts.append(f"created:{components['creation']}")
            
        # Add topic component if present
        if 'topic' in components:
            query_parts.append(f"topic:{components['topic']}")
        
        # Build base query
        query = " ".join(query_parts)
        
        # Add sort parameter if present
        if 'sort' in components:
            query += f" sort:{components['sort']}"
            
        # Add order parameter if present
        if 'order' in components:
            query += f" order:{components['order']}"
            
        return query
    
    @staticmethod
    def parse_star_range(star_component: str) -> Tuple[Optional[int], Optional[int]]:
        """Parse a star range component into min and max values.
        
        Args:
            star_component: Star range string (e.g. ">100", "100..500")
            
        Returns:
            Tuple of (min_stars, max_stars)
        """
        if not star_component:
            return None, None
            
        # Handle greater than format (e.g. ">100")
        if star_component.startswith('>'):
            min_value = int(star_component[1:])
            return min_value, None
            
        # Handle range format (e.g. "100..500")
        if '..' in star_component:
            parts = star_component.split('..')
            min_value = int(parts[0]) if parts[0] else 0
            max_value = int(parts[1]) if parts[1] else None
            return min_value, max_value
            
        # Handle equals format (e.g. "100")
        try:
            exact_value = int(star_component)
            return exact_value, exact_value
        except ValueError:
            return None, None
from urllib3.util.retry import Retry
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Use bandit logger configured in the centralized logging_config
bandit_logger = logging.getLogger('bandit_algorithm')

class QueryPool:
    """A pool of diverse queries with multi-armed bandit optimization for GitHub repositories."""
    
    def __init__(self, metrics_collector=None, config=None):
        """Initialize the query pool with diverse parameters and bandit algorithm.
        
        Args:
            metrics_collector: Optional metrics collector for tracking metrics
            config: Required configuration object for settings
        """
        # Store dependencies through explicit injection
        self.metrics_collector = metrics_collector
        
        # Config is now required
        if config is None:
            raise ValueError("Config object must be provided to QueryPool")
            
        self.config = config
        
        # Get parameters from config
        self.exploration_weight = config.get("query.exploration_weight", 1.0)
        self.target_repos = config.get("crawler.total_count", 1000000)
        self.enable_novelty = config.get("learning.bandit_enabled", True)
        
        # Query statistics tracking - helps prioritize effective queries
        # Structure: {query_pattern: {"success_rate": float, "unique_rate": float, "last_used": timestamp}}
        self.query_stats = {}
        self.query_stats_lock = threading.RLock()
        
        # Path tracking for queries - helps track which queries lead to other successful queries
        # Structure: {parent_query: {child_query: {"success_count": int, "reward": float}}}
        self.query_paths = {}
        self.query_paths_lock = threading.RLock()
        
        # Get cooling period from config
        self.query_cooling_period = self.config.get("query.cooling_period", 1800)
        
        # Initialize the query cooling manager to track exhausted queries
        from src.core.query_bandit import QueryCoolingManager
        self.query_cooling_manager = QueryCoolingManager(cooling_period=self.query_cooling_period)
        
        # Query stats are regenerated each time - no persistence needed
        # Starting fresh with each run to avoid using stale data
        
        # Bandit algorithm parameters
        self.total_runs = 0  # Total number of query executions
        self.cooling_factor = 0.995  # Base cooling factor, will be adjusted dynamically
        
        # Novelty-driven exploration parameters
        self.found_repos = 0  # Number of unique repositories found so far
        self.total_unique_repositories = 0  # Track total unique repositories
        
        # Initialize the query similarity engine for novelty tracking
        self.similarity_engine = QuerySimilarityEngine(
            num_permutations=128, 
            num_bands=10, 
            n_gram_size=3
        ) if self.enable_novelty else None
        
        # Query string to similarity ID mapping for fast lookups
        self.query_to_similarity_id = {}
        
        # Component tracking for success rates
        self.component_success_rates = {
            "stars": {},      # Performance by star range
            "language": {},   # Performance by language
            "sort": {},       # Performance by sort option
            "creation": {},   # Performance by creation date
            "topic": {},      # Performance by topic
        }
        
        # Track usage count for each component value
        self.component_usage_count = {
            "stars": {},      # Usage count by star range
            "language": {},   # Usage count by language
            "sort": {},       # Usage count by sort option 
            "creation": {},   # Usage count by creation date
            "topic": {},      # Usage count by topic
        }
        
        # Track detailed metrics for each component value
        self.component_metrics = {
            "stars": {},      # Metrics by star range (unique results, API calls, etc.)
            "language": {},   # Metrics by language
            "sort": {},       # Metrics by sort option
            "creation": {},   # Metrics by creation date
            "topic": {},      # Metrics by topic
        }
        
        # Enhanced contextual bandit parameters
        self.context_features = {
            # Basic metadata features
            "star_ranges": {},  # Distribution of star counts
            "languages": {},    # Distribution of languages
            "topics": {},       # Distribution of topics
            "ages": {},         # Distribution of repository ages
            "diversity_score": 0.0,  # How diverse the collected repositories are
            "top_heavy_ratio": 0.0,  # Ratio of high-star to low-star repositories
            "collection_stage": 0.0,  # Early (0.0) to late (1.0) stage of collection
            
            # Advanced features - repository activity
            "activity_levels": {
                "high": 0.0,     # Repositories with recent commits/PRs
                "medium": 0.0,   # Repositories with moderate activity
                "low": 0.0,      # Repositories with little recent activity
                "dormant": 0.0   # Repositories with no recent activity
            },
            
            # Contributor metrics
            "contributor_distribution": {
                "solo": 0.0,           # Single contributor
                "small_team": 0.0,     # 2-5 contributors
                "medium_team": 0.0,    # 6-20 contributors
                "large_team": 0.0,     # 21-100 contributors
                "community": 0.0       # 100+ contributors
            },
            
            # Issue/PR velocity
            "engagement_metrics": {
                "issue_velocity": 0.0,    # Average issues per month
                "pr_velocity": 0.0,       # Average PRs per month
                "response_time": 0.0,     # Average time to first response on issues
                "resolution_rate": 0.0,   # Percentage of closed issues/PRs
            },
            
            # Time-based features
            "temporal_patterns": {
                "weekly_activity": [0.0] * 7,  # Activity distribution by day of week
                "yearly_trend": 0.0,           # Growth trend over past year
                "consistency": 0.0             # Consistency of activity (low variance = high consistency)
            }
        }
        
        # Enhanced Bayesian prior parameters - modeling repository distribution
        self.bayesian_priors = {
            # Model repository star count as log-normal distribution
            "star_distribution": {
                "mean": 2.5,     # Log-scale mean (approximately 12k stars)
                "variance": 2.0,  # Wide variance to account for uncertainty
                "sample_count": 5  # Equivalent to 5 samples (low confidence)
            },
            # Language distribution priors
            "language_distribution": {
                "javascript": 0.25,  # Most common language
                "python": 0.20,
                "java": 0.15,
                "typescript": 0.12,
                "other": 0.28,      # All other languages
                "sample_count": 10  # Equivalent to 10 samples
            },
            # Activity distribution priors
            "activity_distribution": {
                "high": 0.20,      # High activity repositories
                "medium": 0.30,    # Medium activity repositories
                "low": 0.30,       # Low activity repositories
                "dormant": 0.20,   # Dormant repositories
                "sample_count": 8  # Equivalent to 8 samples
            },
            # Contributor distribution priors
            "contributor_distribution": {
                "solo": 0.40,         # Single contributor
                "small_team": 0.30,   # 2-5 contributors
                "medium_team": 0.20,  # 6-20 contributors
                "large_team": 0.08,   # 21-100 contributors
                "community": 0.02,    # 100+ contributors
                "sample_count": 8     # Equivalent to 8 samples
            }
        }
        
        # Decay factors for exploration
        self.min_exploration_weight = 0.1  # Minimum exploration factor
        self.collection_completion_decay = 1.1  # Decay when collection is near complete
        self.diversity_decay = 1.1  # Decay when diversity is high (increased from 0.9 to 0.95 for more aggressive warming)
        
        # Star ranges (can't overlap to avoid duplicates)
        self.star_ranges = [
            (100000, 500000, "100K-500K"),
            (50000, 100000, "50K-100K"),
            (30000, 50000, "30K-50K"),
            (20000, 30000, "20K-30K"),
            (10000, 20000, "10K-20K"),
            (7500, 10000, "7.5K-10K"),
            (5000, 7500, "5K-7.5K"),
            (2500, 5000, "2.5K-5K"),
            (1000, 2500, "1K-2.5K"),
            (500, 1000, "500-1K"),
            (100, 500, "100-500"),
            (50, 100, "50-100"),
            (10, 50, "10-50"),
            (0, 10, "<10")
        ]
        
        # Programming languages (most popular on GitHub)
        self.languages = [
            "javascript",
            "python",
            "java",
            "typescript",
            "csharp",
            "cpp",
            "php",
            "ruby",
            "go",
            "rust",
            "kotlin",
            "swift",
            "scala",
            "dart",
            "html",  # Not a true language but used for categorization
            ""  # Empty means no language filter
        ]
        
        # Sort options
        self.sort_options = [
            "stars",     # Most stars first
            "forks",     # Most forks first
            "updated",   # Recently updated first
            "stars-asc", # Fewest stars first (good for lower ranges)
            "forks-asc"  # Fewest forks first
        ]
        
        # Creation date ranges with more granular periods
        self.creation_periods = [
            "created:>2023-01-01",      # Very recent
            "created:2022-01-01..2023-01-01",  # Last year
            "created:2020-01-01..2022-01-01",  # Last few years
            "created:2015-01-01..2020-01-01",  # Older
            "created:2010-01-01..2015-01-01",  # Much older
            "created:2005-01-01..2010-01-01",  # Very old
            "created:<2005-01-01",      # Ancient
            ""                          # No creation filter
        ]
        
        # Topic filters to enrich queries
        self.topics = [
            "topic:machine-learning",
            "topic:deep-learning",
            "topic:web-development",
            "topic:devops",
            "topic:data-science",
            "topic:blockchain",
            "topic:game-development",
            "topic:security",
            "topic:mobile",
            ""  # No topic filter
        ]
        
        # Generate all query combinations and load performance data
        self._generate_query_combinations()
        logger.info(f"Query pool initialized with {len(self.queries)} unique query combinations")
        logger.info(f"Multi-armed bandit algorithm: UCB with exploration weight {self.exploration_weight}")
        
    def mark_query_exhausted(self, query_text, worker_id=None, cooling_multiplier=1.0):
        """Mark a query as exhausted due to high duplication rate.
        
        Args:
            query_text: The query text to mark as exhausted
            worker_id: The ID of the worker that marked the query as exhausted
            cooling_multiplier: Multiplier for the cooling period (1.0 = standard, <1.0 = shorter)
        """
        self.query_cooling_manager.mark_query_exhausted(query_text, worker_id, cooling_multiplier)
            
    def is_query_exhausted(self, query_text):
        """Check if a query is currently exhausted.
        
        Args:
            query_text: The query text to check
            
        Returns:
            True if the query is exhausted, False otherwise
        """
        return self.query_cooling_manager.is_query_exhausted(query_text)
                
    def cleanup_exhausted_queries(self):
        """Clean up exhausted queries that have cooled down."""
        # Use the QueryCoolingManager's cool_queries method
        num_cooled = self.query_cooling_manager.cool_queries()
        
        # Get the current set of exhausted queries for logging
        exhausted_queries = self.query_cooling_manager.get_exhausted_queries()
        
        # Count high-star queries that are still cooling
        high_star_queries_cooling = 0
        for query_text, values in exhausted_queries.items():
            # High-star queries typically have a cooling_multiplier < 1.0
            cooling_multiplier = values[2]
            if cooling_multiplier < 1.0:
                high_star_queries_cooling += 1
        
        # Log stats about exhausted queries
        if exhausted_queries:
            logger.info(f"Exhausted queries: {len(exhausted_queries)} remaining, {num_cooled} cleaned up")
            if high_star_queries_cooling > 0:
                logger.info(f"High-star queries with reduced cooling periods: {high_star_queries_cooling}")
        
        # Return the number of queries that were cooled and removed
        return num_cooled
        
    def update_query_stats(self, query_text: str, success: bool, unique_rate: float) -> None:
        """Update query performance statistics.
        
        Args:
            query_text: The query text
            success: Whether the query was successful
            unique_rate: Rate of unique repositories returned (0.0-1.0)
        """
        # Extract query pattern by removing specific values like dates, numbers
        # This helps group similar queries for better statistics
        import re
        pattern = re.sub(r'\d{4}-\d{2}-\d{2}', 'DATE', query_text)
        pattern = re.sub(r'\b\d+\b', 'N', pattern)
        
        with self.query_stats_lock:
            if pattern not in self.query_stats:
                self.query_stats[pattern] = {
                    "success_count": 0,
                    "total_count": 0,
                    "unique_rate": 0.0,
                    "last_used": time.time()
                }
            
            stats = self.query_stats[pattern]
            stats["total_count"] += 1
            if success:
                stats["success_count"] += 1
            
            # Update unique rate with exponential moving average (0.2 weight for new value)
            stats["unique_rate"] = 0.8 * stats["unique_rate"] + 0.2 * unique_rate
            stats["last_used"] = time.time()
            
            # Periodically save stats to disk
            if random.random() < 0.01:  # ~1% chance each update
                try:
                    with open(self.database_path, "w") as f:
                        json.dump(self.query_stats, f)
                except Exception as e:
                    logger.warning(f"Could not save query statistics: {e}")
                    
    def track_query_path(self, parent_query: str, child_query: str, success: bool = True, reward: float = 0.0) -> None:
        """Track when one query leads to another successful query.
        
        This function records relationships between queries to identify effective query paths.
        
        Args:
            parent_query: The original query that led to the child query
            child_query: The subsequent query that was executed
            success: Whether the child query was successful
            reward: The reward value associated with the child query execution
        """
        with self.query_paths_lock:
            # Initialize parent entry if not exists
            if parent_query not in self.query_paths:
                self.query_paths[parent_query] = {}
                
            # Initialize child entry if not exists
            if child_query not in self.query_paths[parent_query]:
                self.query_paths[parent_query][child_query] = {
                    "success_count": 0,
                    "total_count": 0,
                    "reward": 0.0
                }
                
            # Update statistics
            path_stats = self.query_paths[parent_query][child_query]
            path_stats["total_count"] = path_stats.get("total_count", 0) + 1
            
            if success:
                path_stats["success_count"] = path_stats.get("success_count", 0) + 1
            
            # Update reward with exponential moving average (0.2 weight for new value)
            path_stats["reward"] = 0.8 * path_stats.get("reward", 0.0) + 0.2 * reward
            
            # Log path discovery to help with debugging and visualization
            if path_stats["success_count"] > 3 and path_stats["reward"] > 0.5:
                bandit_logger.info(f"High-value query path discovered: {parent_query} → {child_query} "
                              f"(reward: {path_stats['reward']:.2f}, success: {path_stats['success_count']})")
                              
    def get_path_reward(self, query_text: str, depth: int = 2) -> float:
        """Calculate downstream reward for a query based on paths originating from it.
        
        This implements enhanced long-term credit assignment by rewarding queries that lead to 
        valuable future discoveries, even if they themselves don't directly find many repos.
        Uses a more sophisticated weighting system that considers success rates, reward magnitude,
        and path diversity.
        
        Args:
            query_text: The query to evaluate
            depth: How many steps forward to consider in the path (default: 2)
            
        Returns:
            The calculated path reward value
        """
        # Base case: if depth is 0 or no paths found, return 0
        if depth <= 0 or query_text not in self.query_paths:
            return 0.0
        
        total_path_reward = 0.0
        path_count = 0
        unique_paths = set()
        
        # Calculate immediate children path rewards
        with self.query_paths_lock:
            # First pass: collect all child queries for diversity calculation
            for child_query in self.query_paths[query_text].keys():
                if self.query_paths[query_text][child_query]["success_count"] > 0:
                    unique_paths.add(child_query)
            
            # Diversity bonus: queries that lead to more diverse paths are more valuable
            diversity_factor = min(1.0, len(unique_paths) / 5.0)  # Cap at 1.0 when there are 5+ unique paths
            
            # Second pass: calculate rewards with diversity bonus
            for child_query, stats in self.query_paths[query_text].items():
                # Only consider successful paths
                if stats["success_count"] > 0:
                    # Calculate success rate with confidence adjustment
                    # More confidence in paths that have been tried more times
                    confidence = min(1.0, stats["total_count"] / 10.0)  # Cap at 1.0 when tried 10+ times
                    success_rate = stats["success_count"] / max(1, stats["total_count"])
                    adjusted_success_rate = success_rate * confidence
                    
                    # Direct reward from this path, now considers the magnitude of the reward
                    # Higher rewards get higher weight
                    reward_magnitude = min(1.0, stats["reward"] / 5.0)  # Normalize, cap at 1.0
                    direct_reward = stats["reward"] * adjusted_success_rate * (1.0 + reward_magnitude)
                    
                    # Recursive reward from downstream paths (with adaptive decay factor based on depth)
                    # Deeper paths get more aggressive decay
                    decay_factor = 0.8 / (1.0 + 0.1 * (depth - 1))  # Starts at 0.8 and decreases with depth
                    downstream_reward = decay_factor * self.get_path_reward(child_query, depth - 1)
                    
                    # Apply diversity bonus to both direct and downstream rewards
                    diversity_bonus = 1.0 + 0.5 * diversity_factor
                    
                    # Combine rewards with diversity bonus
                    path_reward = (direct_reward + downstream_reward) * diversity_bonus
                    total_path_reward += path_reward
                    path_count += 1
        
        # Return weighted path reward or 0 if no paths
        if path_count == 0:
            return 0.0
        
        # Weight more by the total reward than just the average
        # This prefers queries that lead to many good paths over those with few good paths
        path_count_factor = min(2.0, 1.0 + (path_count / 10.0))  # Bonus caps at 2x when there are 10+ paths
        
        return (total_path_reward / path_count) * path_count_factor
    
    def _generate_query_combinations(self):
        """Generate all possible query combinations."""
        self.queries = []
        
        # Keep track of queries we've generated to avoid exact duplicates
        seen_queries = set()
        
        # Build queries from all combinations (using a subset to avoid combinatorial explosion)
        # We'll add more combinations through evolutionary methods over time
        for star_range, language, sort, creation in itertools.product(
            self.star_ranges, 
            self.languages, 
            self.sort_options[:3],  # Limit to top 3 sort options initially
            self.creation_periods[:6]  # Use top 6 creation periods for more coverage
        ):
            # Parse star range
            min_stars, max_stars, range_name = star_range
            
            # Build star part of query
            star_query = []
            if min_stars is not None:
                star_query.append(f"stars:>={min_stars}")
            if max_stars is not None:
                star_query.append(f"stars:<{max_stars}")
            
            # Combine star conditions
            star_part = " ".join(star_query)
            
            # Add language if provided
            lang_part = f"language:{language}" if language else ""
            
            # Add sort option
            sort_part = f"sort:{sort}"
            
            # Add creation date filter
            creation_part = creation
            
            # Combine parts, removing empty strings
            parts = [part for part in [star_part, lang_part, creation_part, sort_part] if part]
            query_text = " ".join(parts)
            
            # Only add queries we haven't seen before (avoid exact duplication)
            if query_text not in seen_queries:
                seen_queries.add(query_text)
                self.queries.append({
                    "query_text": query_text,
                    "star_range": range_name,
                    "language": language or "any",
                    "sort": sort,
                    "creation": creation or "any",
                    "topic": "",  # Will be added for evolved queries
                    
                    # Query execution stats
                    "usage_count": 0,  # Track how many times this query has been used
                    "success_count": 0,  # Successful API calls (no errors)
                    "error_count": 0,  # Failed API calls
                    
                    # Performance metrics
                    "total_results": 0,  # Total repositories returned by this query
                    "unique_results": 0,  # Number of unique repositories discovered
                    "unique_rate": 0.0,  # Unique repos / total repos
                    "api_efficiency": 0.0,  # Unique repos per API call
                    "quality_score": 0.0,  # Quality score based on star counts
                    "reward": 0.0,  # Combined reward signal for bandit algorithm
                    "ucb_score": 0.0,  # Upper confidence bound score
                    
                    # Evolutionary parameters
                    "generation": 0,  # Generation count (0 = initial)
                    "parent": None,   # Parent query if evolved
                    "mutation_type": None,  # Type of mutation if evolved
                    "duplication_rate": 0.0,  # Duplication rate with existing repos
                    
                    # Thompson sampling parameters
                    "alpha": 1.0,  # Success parameter for beta distribution
                    "beta": 1.0,   # Failure parameter for beta distribution
                })
        
        # Add some topic-specific queries for popular repositories
        for star_range in self.star_ranges[:8]:  # Focus on higher-star repos for topics
            min_stars, max_stars, range_name = star_range
            
            for topic in self.topics:
                if not topic:  # Skip empty topic
                    continue
                    
                # Build query with topic
                star_query = []
                if min_stars is not None:
                    star_query.append(f"stars:>={min_stars}")
                if max_stars is not None:
                    star_query.append(f"stars:<{max_stars}")
                
                star_part = " ".join(star_query)
                
                # Combine parts with topic
                parts = [part for part in [star_part, topic, "sort:stars"] if part]
                query_text = " ".join(parts)
                
                if query_text not in seen_queries:
                    seen_queries.add(query_text)
                    self.queries.append({
                        "query_text": query_text,
                        "star_range": range_name,
                        "language": "any",
                        "sort": "stars",
                        "creation": "any",
                        "topic": topic.replace("topic:", ""),
                        
                        # Query execution stats
                        "usage_count": 0,
                        "success_count": 0,
                        "error_count": 0,
                        
                        # Performance metrics
                        "total_results": 0,
                        "unique_results": 0,
                        "unique_rate": 0.0,
                        "api_efficiency": 0.0,
                        "quality_score": 0.0,
                        "reward": 0.0,
                        "ucb_score": 0.0,
                        
                        # Evolutionary parameters
                        "generation": 0,
                        "parent": None,
                        "mutation_type": None,
                        "duplication_rate": 0.0,
                        
                        # Thompson sampling parameters
                        "alpha": 1.0,
                        "beta": 1.0,
                    })
    
   
    def _update_exploration_weight(self):
        """Update exploration weight based on context features."""
        # Start with base exploration weight
        base_weight = self.exploration_weight
        
        # Add a sanity check for unreasonable starting values
        MAX_EXPLORATION_WEIGHT = 100.0  # Define a reasonable maximum
        if not (0.01 <= base_weight <= MAX_EXPLORATION_WEIGHT):
            logger.warning(f"Detected unreasonable base exploration weight: {base_weight}. Resetting to default.")
            base_weight = 1.0
        
        # Apply collection stage boost - increase exploration as we collect more
        collection_stage = self.context_features["collection_stage"]
        stage_factor = 1.0 + (collection_stage * 0.5)  # Boost exploration up to 50% as collection progresses
        
        # Apply diversity-based adjustment - reduce exploration if we have high diversity
        # but make this effect weaker as we collect more
        diversity = self.context_features["diversity_score"]
        diversity_impact = max(0.1, 1.0 - collection_stage)  # Diversity matters less as collection grows
        diversity_factor = 1.0 - (diversity * (1.0 - self.diversity_decay) * diversity_impact)
        
        # Combine factors and apply both minimum and maximum bounds
        adjusted_weight = base_weight * stage_factor * diversity_factor
        
        # Apply a warming factor that increases exploration over time
        # This makes exploration grow stronger as we collect more repositories (contrary to traditional cooling)
        # The intensity of warming is tunable through the diversity_decay parameter:
        # - Higher diversity_decay (0.8-0.95) = more aggressive warming
        # - Lower diversity_decay (0.3-0.5) = more conservative warming
        warming_intensity = self.diversity_decay  # Use diversity_decay as a tuning parameter
        warming_factor = 1.0 + (collection_stage * 0.01 * warming_intensity * 5.0)  # Scaled by warming intensity
        adjusted_weight *= warming_factor
        
        self.exploration_weight = max(self.min_exploration_weight, min(MAX_EXPLORATION_WEIGHT, adjusted_weight))
        
        if abs(adjusted_weight - base_weight) > 0.05:
            logger.info(f"Adjusted exploration weight: {base_weight:.2f} → {self.exploration_weight:.2f} " 
                      f"(stage: {collection_stage:.2f}, diversity: {diversity:.2f}, warming: {warming_factor:.3f}, intensity: {warming_intensity:.2f})")
    
    
    def _update_activity_metrics(self, activity_metadata):
        """Update repository activity metrics based on query results.
        
        Args:
            activity_metadata: Dictionary containing activity data for repositories
        """
        # Get current activity distribution
        activity_levels = self.context_features.get("activity_levels", {
            "high": 0.0,
            "medium": 0.0,
            "low": 0.0,
            "dormant": 0.0
        })
        
        # Get counts of each activity level from metadata
        if isinstance(activity_metadata, dict):
            high_count = activity_metadata.get("high", 0)
            medium_count = activity_metadata.get("medium", 0)
            low_count = activity_metadata.get("low", 0)
            dormant_count = activity_metadata.get("dormant", 0)
            total_count = high_count + medium_count + low_count + dormant_count
            
            if total_count > 0:
                # Calculate new activity distribution with exponential moving average
                decay_factor = 0.9  # 90% from old data, 10% from new data
                
                # Update each level
                activity_levels["high"] = decay_factor * activity_levels["high"] + (1 - decay_factor) * (high_count / total_count)
                activity_levels["medium"] = decay_factor * activity_levels["medium"] + (1 - decay_factor) * (medium_count / total_count)
                activity_levels["low"] = decay_factor * activity_levels["low"] + (1 - decay_factor) * (low_count / total_count)
                activity_levels["dormant"] = decay_factor * activity_levels["dormant"] + (1 - decay_factor) * (dormant_count / total_count)
                
                # Normalize to ensure sum equals 1.0
                total = sum(activity_levels.values())
                if total > 0:
                    for key in activity_levels:
                        activity_levels[key] /= total
                
                # Update context features
                self.context_features["activity_levels"] = activity_levels
                
                # Update Bayesian prior with activity distribution
                self._update_bayesian_prior("activity_distribution", {
                    "high": high_count / total_count,
                    "medium": medium_count / total_count,
                    "low": low_count / total_count,
                    "dormant": dormant_count / total_count
                })
                
    def _update_contributor_metrics(self, contributor_metadata):
        """Update repository contributor metrics based on query results.
        
        Args:
            contributor_metadata: Dictionary containing contributor data for repositories
        """
        # Get current contributor distribution
        contributor_distribution = self.context_features.get("contributor_distribution", {
            "solo": 0.0,
            "small_team": 0.0,
            "medium_team": 0.0,
            "large_team": 0.0,
            "community": 0.0
        })
        
        # Get counts of each contributor level from metadata
        if isinstance(contributor_metadata, dict):
            solo_count = contributor_metadata.get("solo", 0)
            small_team_count = contributor_metadata.get("small_team", 0)
            medium_team_count = contributor_metadata.get("medium_team", 0)
            large_team_count = contributor_metadata.get("large_team", 0)
            community_count = contributor_metadata.get("community", 0)
            total_count = solo_count + small_team_count + medium_team_count + large_team_count + community_count
            
            if total_count > 0:
                # Calculate new contributor distribution with exponential moving average
                decay_factor = 0.9  # 90% from old data, 10% from new data
                
                # Update each level
                contributor_distribution["solo"] = decay_factor * contributor_distribution["solo"] + (1 - decay_factor) * (solo_count / total_count)
                contributor_distribution["small_team"] = decay_factor * contributor_distribution["small_team"] + (1 - decay_factor) * (small_team_count / total_count)
                contributor_distribution["medium_team"] = decay_factor * contributor_distribution["medium_team"] + (1 - decay_factor) * (medium_team_count / total_count)
                contributor_distribution["large_team"] = decay_factor * contributor_distribution["large_team"] + (1 - decay_factor) * (large_team_count / total_count)
                contributor_distribution["community"] = decay_factor * contributor_distribution["community"] + (1 - decay_factor) * (community_count / total_count)
                
                # Normalize to ensure sum equals 1.0
                total = sum(contributor_distribution.values())
                if total > 0:
                    for key in contributor_distribution:
                        contributor_distribution[key] /= total
                
                # Update context features
                self.context_features["contributor_distribution"] = contributor_distribution
                
                # Update Bayesian prior with contributor distribution
                self._update_bayesian_prior("contributor_distribution", {
                    "solo": solo_count / total_count,
                    "small_team": small_team_count / total_count,
                    "medium_team": medium_team_count / total_count,
                    "large_team": large_team_count / total_count,
                    "community": community_count / total_count
                })
    
    def _update_engagement_metrics(self, engagement_metadata):
        """Update repository engagement metrics based on query results.
        
        Args:
            engagement_metadata: Dictionary containing engagement data for repositories
        """
        # Get current engagement metrics
        engagement_metrics = self.context_features.get("engagement_metrics", {
            "issue_velocity": 0.0,
            "pr_velocity": 0.0,
            "response_time": 0.0,
            "resolution_rate": 0.0
        })
        
        # Update engagement metrics using exponential moving average
        if isinstance(engagement_metadata, dict):
            decay_factor = 0.9  # 90% from old data, 10% from new data
            
            # Update each metric if available
            # Update engagement metrics using exponential weighted moving average
            metrics_to_update = ["issue_velocity", "pr_velocity", "response_time", "resolution_rate"]
            
            for metric in metrics_to_update:
                if metric in engagement_metadata:
                    current_value = engagement_metrics[metric]
                    new_value = engagement_metadata[metric]
                    engagement_metrics[metric] = decay_factor * current_value + (1 - decay_factor) * new_value
            
            # Update temporal patterns if available
            if "weekly_activity" in engagement_metadata:
                weekly_activity = engagement_metadata["weekly_activity"]
                if isinstance(weekly_activity, list) and len(weekly_activity) == 7:
                    current_weekly = self.context_features.get("temporal_patterns", {}).get("weekly_activity", [0.0] * 7)
                    updated_weekly = [decay_factor * curr + (1 - decay_factor) * new for curr, new in zip(current_weekly, weekly_activity)]
                    
                    # Normalize to ensure sum equals 1.0
                    total = sum(updated_weekly)
                    if total > 0:
                        updated_weekly = [val / total for val in updated_weekly]
                    
                    # Update temporal patterns
                    if "temporal_patterns" not in self.context_features:
                        self.context_features["temporal_patterns"] = {}
                    self.context_features["temporal_patterns"]["weekly_activity"] = updated_weekly
            
            # Update yearly trend if available
            if "yearly_trend" in engagement_metadata:
                current_trend = self.context_features.get("temporal_patterns", {}).get("yearly_trend", 0.0)
                new_trend = decay_factor * current_trend + (1 - decay_factor) * engagement_metadata["yearly_trend"]
                
                if "temporal_patterns" not in self.context_features:
                    self.context_features["temporal_patterns"] = {}
                self.context_features["temporal_patterns"]["yearly_trend"] = new_trend
            
            # Update consistency if available
            if "consistency" in engagement_metadata:
                current_consistency = self.context_features.get("temporal_patterns", {}).get("consistency", 0.0)
                new_consistency = decay_factor * current_consistency + (1 - decay_factor) * engagement_metadata["consistency"]
                
                if "temporal_patterns" not in self.context_features:
                    self.context_features["temporal_patterns"] = {}
                self.context_features["temporal_patterns"]["consistency"] = new_consistency
            
            # Update context features
            self.context_features["engagement_metrics"] = engagement_metrics
        
    def _update_component_stats(self, query, success, unique_count, results_count, api_calls, duplication_rate, quality_score):
        """Update component success rates based on query performance.
        
        Args:
            query: The query dictionary
            success: Whether the query execution was successful
            unique_count: Number of unique repositories discovered
            results_count: Total number of repositories returned
            api_calls: Number of API calls made for this query
            duplication_rate: Rate of duplication with existing repos (0.0-1.0)
            quality_score: Quality score for the repositories (0.0-1.0)
        """
        # Skip if the query was not successful
        if not success:
            return
            
        # Calculate performance metrics
        api_efficiency = unique_count / max(1, api_calls)
        unique_rate = unique_count / max(1, results_count)
        novelty_score = 1.0 - duplication_rate
        
        # Combined score - weighted average of different metrics
        combined_score = (
            0.4 * api_efficiency +    # 40% weight on API efficiency
            0.3 * unique_rate +       # 30% weight on unique rate
            0.2 * novelty_score +     # 20% weight on novelty
            0.1 * quality_score       # 10% weight on quality
        )
        
        # Normalize score to 0-1 range (assuming max API efficiency of 10)
        normalized_score = min(1.0, combined_score / 4.0)
        
        # Update component usage counts and success rates
        components = {
            "stars": query["star_range"],
            "language": query["language"],
            "sort": query["sort"],
            "creation": query["creation"],
            "topic": query["topic"] if query["topic"] else "none"
        }
        
        # Update each component's statistics
        for component_type, component_value in components.items():
            # Initialize if this component value is new
            if component_value not in self.component_usage_count[component_type]:
                self.component_usage_count[component_type][component_value] = 0
                self.component_success_rates[component_type][component_value] = 0.0
                self.component_metrics[component_type][component_value] = {
                    "api_calls": 0,
                    "unique_results": 0,
                    "total_results": 0,
                    "duplication_rate": 0.0,
                    "quality_score": 0.0
                }
            
            # Update usage count
            self.component_usage_count[component_type][component_value] += 1
            
            # Update success rate using exponential moving average (more weight to recent performance)
            current_rate = self.component_success_rates[component_type][component_value]
            # Alpha determines how much weight to give to the new data vs. historical data
            # Higher alpha = more weight to recent performance
            alpha = 0.2  # 20% weight to new performance, 80% to historical
            self.component_success_rates[component_type][component_value] = (
                (1 - alpha) * current_rate + alpha * normalized_score
            )
            
            # Update detailed metrics
            metrics = self.component_metrics[component_type][component_value]
            metrics["api_calls"] += api_calls
            metrics["unique_results"] += unique_count
            metrics["total_results"] += results_count
            
            # Update average duplication rate and quality score
            metrics["duplication_rate"] = (
                (metrics["duplication_rate"] * (metrics["api_calls"] - api_calls) + 
                 duplication_rate * api_calls) / max(1, metrics["api_calls"])
            )
            metrics["quality_score"] = (
                (metrics["quality_score"] * (metrics["unique_results"] - unique_count) + 
                 quality_score * unique_count) / max(1, metrics["unique_results"])
            )
    
    def _update_bayesian_prior(self, prior_name, new_distribution):
        """Update a Bayesian prior with new observation data.
        
        Args:
            prior_name: Name of the prior to update
            new_distribution: New distribution values to incorporate
        """
        if prior_name in self.bayesian_priors:
            prior = self.bayesian_priors[prior_name]
            
            # Extract sample count (represents confidence in prior)
            sample_count = prior.get("sample_count", 10)
            
            # For each key in the new distribution that also exists in the prior
            for key in new_distribution:
                if key in prior and key != "sample_count":
                    # Bayesian update formula: new = (prior * prior_weight + new * 1) / (prior_weight + 1)
                    prior[key] = (prior[key] * sample_count + new_distribution[key]) / (sample_count + 1)
            
            # Slowly increase sample count to represent increasing confidence
            # but cap at 50 to allow for adaptation to changing trends
            prior["sample_count"] = min(50, sample_count + 0.1)
            
            # Renormalize distribution if needed
            distribution_keys = [k for k in prior if k != "sample_count"]
            total = sum(prior[k] for k in distribution_keys)
            if total > 0 and abs(total - 1.0) > 0.01:  # Only renormalize if significantly off
                for key in distribution_keys:
                    prior[key] /= total

    def _analyze_component_success_rates(self, queries):
        """Analyze which query components lead to success across multiple queries.
        
        Args:
            queries: List of query dictionaries to analyze
            
        Returns:
            Dictionary of component success probabilities
        """
        # Track component occurrences and successes
        components = {
            "stars": {},
            "language": {},
            "sort": {},
            "creation": {},
            "topic": {}
        }
        
        # Count occurrences and success rates for each component
        for query in queries:
            # Skip queries with too little usage
            if query["usage_count"] < 3:
                continue
                
            # Calculate success metrics for this query
            reward_ratio = query["reward"] / max(1, query["usage_count"])
            unique_rate = query["unique_rate"]
            api_efficiency = query["api_efficiency"]
            
            # Combine metrics into a single success score
            success_score = 0.5 * reward_ratio + 0.3 * unique_rate + 0.2 * api_efficiency
            
            # Extract components from the query
            query_parts = query["query_text"].split()
            
            # Track stars
            star_parts = []
            for part in query_parts:
                if part.startswith("stars:"):
                    star_parts.append(part)
            star_key = "|".join(sorted(star_parts)) if star_parts else "any"
            
            if star_key not in components["stars"]:
                components["stars"][star_key] = {"count": 0, "score_sum": 0.0}
            components["stars"][star_key]["count"] += 1
            components["stars"][star_key]["score_sum"] += success_score
            
            # Track language
            language = query["language"]
            if language not in components["language"]:
                components["language"][language] = {"count": 0, "score_sum": 0.0}
            components["language"][language]["count"] += 1
            components["language"][language]["score_sum"] += success_score
            
            # Track sort
            sort_option = query["sort"]
            if sort_option not in components["sort"]:
                components["sort"][sort_option] = {"count": 0, "score_sum": 0.0}
            components["sort"][sort_option]["count"] += 1
            components["sort"][sort_option]["score_sum"] += success_score
            
            # Track creation date
            creation = query["creation"]
            if creation not in components["creation"]:
                components["creation"][creation] = {"count": 0, "score_sum": 0.0}
            components["creation"][creation]["count"] += 1
            components["creation"][creation]["score_sum"] += success_score
            
            # Track topic
            topic = query.get("topic", "")
            if topic not in components["topic"]:
                components["topic"][topic] = {"count": 0, "score_sum": 0.0}
            components["topic"][topic]["count"] += 1
            components["topic"][topic]["score_sum"] += success_score
        
        # Calculate success probabilities
        component_success_rates = {}
        for component_type, values in components.items():
            component_success_rates[component_type] = {}
            for key, data in values.items():
                if data["count"] > 0:
                    avg_score = data["score_sum"] / data["count"]
                    component_success_rates[component_type][key] = {
                        "count": data["count"],
                        "success_rate": avg_score
                    }
        
        return component_success_rates

    def _identify_query_patterns(self, queries):
        """Identify successful patterns across multiple queries.
        
        Args:
            queries: List of query dictionaries to analyze
            
        Returns:
            List of pattern dictionaries
        """
        if not queries or len(queries) < 5:
            return []
            
        # Count co-occurrences of components
        component_pairs = {}
        
        for query in queries:
            # Skip queries with too little usage
            if query["usage_count"] < 3:
                continue
                
            # Calculate success score
            reward_ratio = query["reward"] / max(1, query["usage_count"])
            
            # Extract components
            components = []
            query_parts = query["query_text"].split()
            
            # Extract stars, language, sort, etc.
            star_range = query["star_range"]
            language = query["language"]
            sort_option = query["sort"]
            creation = query["creation"]
            topic = query.get("topic", "")
            
            # Map component types to their values and conditions
            component_mapping = [
                ("stars", star_range, star_range != "any"),
                ("language", language, language != "any"),
                ("sort", sort_option, bool(sort_option)),
                ("creation", creation, creation != "any"),
                ("topic", topic, bool(topic))
            ]
            
            # Add components that meet their conditions
            for prefix, value, condition in component_mapping:
                if condition:
                    components.append(f"{prefix}:{value}")
            
            # Record all pairs of components
            for i in range(len(components)):
                for j in range(i+1, len(components)):
                    pair = (components[i], components[j])
                    
                    if pair not in component_pairs:
                        component_pairs[pair] = {"count": 0, "score_sum": 0.0}
                    
                    component_pairs[pair]["count"] += 1
                    component_pairs[pair]["score_sum"] += reward_ratio
        
        # Find frequently co-occurring component pairs with high success
        patterns = []
        for pair, data in component_pairs.items():
            if data["count"] >= 3:  # At least 3 queries have this pattern
                avg_score = data["score_sum"] / data["count"]
                
                if avg_score > 0.7:  # High success rate
                    patterns.append({
                        "components": pair,
                        "count": data["count"],
                        "success_rate": avg_score
                    })
        
        # Sort by success rate
        patterns.sort(key=lambda x: x["success_rate"], reverse=True)
        
        return patterns[:5]  # Return top 5 patterns

    def _save_performance_data(self):
        """Save query performance data to file with enhanced metrics.
        
        Note: This function is disabled to start fresh with each run.
        """
        # Function disabled - we want each run to start fresh
        # No logging needed - silently return
        return
    
    def _update_ucb_scores(self):
        """Update Upper Confidence Bound scores for all queries with contextual factors."""
        # Calculate UCB scores for the multi-armed bandit algorithm
        for query in self.queries:
            if query["usage_count"] == 0:
                # For unused queries, assign a high UCB to encourage exploration
                query["ucb_score"] = 100.0  # High initial score
            else:
                # Base UCB calculation
                avg_reward = query["reward"] / query["usage_count"]
                
                # Exploration bonus decreases as we use the query more
                exploration = self.exploration_weight * math.sqrt(
                    math.log(max(1, self.total_runs)) / query["usage_count"]
                )
                
                # Apply contextual modifications based on repository statistics and collection state
                contextual_factor = self._calculate_contextual_factor(query)
                
                # Combine with Bayesian prior knowledge
                bayesian_factor = self._calculate_bayesian_factor(query)
                
                # Calculate the final UCB score with all factors
                query["ucb_score"] = (avg_reward * contextual_factor * bayesian_factor) + exploration
                
                # Log detailed UCB calculation for significant updates (every 10 runs or high reward queries)
                if self.total_runs % 10 == 0 or query.get("reward", 0) > 0.75:
                    logger.info(f"UCB calculation: Query '{query['query_text']}' - "
                               f"Score {query['ucb_score']:.4f} = "
                               f"(Reward {avg_reward:.4f} × Context {contextual_factor:.2f} × Bayes {bayesian_factor:.2f}) + "
                               f"Exploration {exploration:.4f}")
    
    def _calculate_contextual_factor(self, query: Dict[str, Any]) -> float:
        """Calculate contextual weight factor based on repository state and query characteristics.
        
        This incorporates the current collection context to prioritize queries
        that target underrepresented areas of the repository space.
        
        Args:
            query: The query to evaluate
            
        Returns:
            A factor to multiply the base reward (>1 = boost, <1 = penalize)
        """
        # Start with neutral factor
        factor = 1.0
        
        # Get query characteristics
        star_range = query.get("star_range", "any")
        language = query.get("language", "any")
        creation = query.get("creation", "any")
        topic = query.get("topic", "")
        sort = query.get("sort", "stars")
        
        # 1. Star Range Balance: Check if we need more of this star range
        star_ranges = self.context_features.get("star_ranges", {})
        
        # First, apply a base boost for high-star ranges to address bottom-heavy distribution
        is_high_star_range = star_range in ["100K+", "50K-100K", "30K-50K", "20K-30K", "10K-20K", "10K-50K"]
        is_very_high_star_range = star_range in ["100K+", "50K-100K"]
        
        # Apply aggressive boost for high-star ranges
        if is_very_high_star_range:
            factor *= 2.0  # 100% boost for very high star ranges (>50K)
        elif is_high_star_range:
            factor *= 1.5  # 50% boost for high star ranges (>10K)
            
        # Then apply the original balance-based adjustments
        if star_ranges and star_range != "any":
            # Calculate what percentage of our collection is in this range
            total_repos = sum(star_ranges.values())
            if total_repos > 0:
                current_range_count = star_ranges.get(star_range, 0)
                range_percentage = current_range_count / total_repos
                
                # If we have very few in this range, boost its score
                if range_percentage < 0.05:  # Less than 5% of collection
                    factor *= 1.5  # 50% boost (increased from 30%)
                elif range_percentage < 0.10:  # Less than 10% of collection
                    factor *= 1.3  # 30% boost (increased from 20%)
                # If we have many in this range, slightly reduce its score,
                # but with less penalty for high-star ranges
                elif range_percentage > 0.30:  # More than 30% of collection
                    if is_high_star_range:
                        factor *= 0.95  # Only 5% penalty for high-star ranges
                    else:
                        factor *= 0.9   # 10% penalty for other ranges
                elif range_percentage > 0.50:  # More than 50% of collection
                    if is_high_star_range:
                        factor *= 0.9   # Only 10% penalty for high-star ranges
                    else:
                        factor *= 0.8   # 20% penalty for other ranges
        
        # 2. Language Balance: Check if we need more of this language
        languages = self.context_features.get("languages", {})
        if languages and language != "any":
            total_langs = sum(languages.values())
            if total_langs > 0:
                current_lang_count = languages.get(language, 0)
                lang_percentage = current_lang_count / total_langs
                
                # Similar logic to star ranges - boost underrepresented languages
                if lang_percentage < 0.05:
                    factor *= 1.25
                elif lang_percentage < 0.10:
                    factor *= 1.15
                elif lang_percentage > 0.40:
                    factor *= 0.9
                elif lang_percentage > 0.60:
                    factor *= 0.8
        
        # 3. Collection Stage: Adjust based on how far along we are
        collection_stage = self.context_features.get("collection_stage", 0.0)
        if collection_stage > 0.8:  # Late stage collection
            # In late stages, focus more on high-efficiency queries
            if query["api_efficiency"] > 30:  # Very efficient
                factor *= 1.2
            elif query["api_efficiency"] < 5:  # Very inefficient
                factor *= 0.8
        
        # 4. Top-Heavy Balance: Ensure we have both high and low star repos
        top_heavy_ratio = self.context_features.get("top_heavy_ratio", 1.0)
        high_star_query = star_range in ["100K-500K", "50K-100K", "30K-50K", "20K-30K"]
        low_star_query = star_range in ["1K-2.5K", "500-1K", "100-500", "50-100", "10-50", "<10"]
        
        if top_heavy_ratio > 2.0 and low_star_query:
            # Too many high-star repos, boost low-star queries
            factor *= 1.2
        elif top_heavy_ratio < 0.5 and high_star_query:
            # Too many low-star repos, boost high-star queries
            factor *= 1.2
        
        # 5. Activity Level Balance: Ensure we have repositories with different activity levels
        activity_levels = self.context_features.get("activity_levels", {})
        total_activity = sum(activity_levels.values()) or 1.0
        
        # Determine if query characteristics correlate with certain activity levels
        # Recent repositories are more likely to be active
        is_recent_query = "2022" in creation or "2023" in creation
        # High-star repos are more likely to be active
        is_high_activity_query = high_star_query or is_recent_query or sort == "updated"
        # Low-star old repos are more likely to be dormant
        is_low_activity_query = low_star_query and ("<2015" in creation)
        
        # Calculate activity imbalances
        high_activity_percentage = (activity_levels.get("high", 0.0) + activity_levels.get("medium", 0.0)) / total_activity
        low_activity_percentage = (activity_levels.get("low", 0.0) + activity_levels.get("dormant", 0.0)) / total_activity
        
        # Adjust based on activity distribution
        if high_activity_percentage < 0.3 and is_high_activity_query:
            # Need more high-activity repos
            factor *= 1.15
        elif low_activity_percentage < 0.3 and is_low_activity_query:
            # Need more low-activity repos
            factor *= 1.1
        elif high_activity_percentage > 0.7 and is_high_activity_query:
            # Too many high-activity repos
            factor *= 0.9
        elif low_activity_percentage > 0.7 and is_low_activity_query:
            # Too many low-activity repos
            factor *= 0.9
        
        # 6. Contributor Diversity: Ensure we have repositories with different team sizes
        contributor_distribution = self.context_features.get("contributor_distribution", {})
        total_contributors = sum(contributor_distribution.values()) or 1.0
        
        # Small team repos are more common in niche topics
        is_small_team_query = topic in ["blockchain", "game-development"] or low_star_query
        # Large team/community repos are more common in popular topics with high stars
        is_large_team_query = topic in ["machine-learning", "web-development"] or high_star_query
        
        # Calculate contributor imbalances
        small_team_percentage = (contributor_distribution.get("solo", 0.0) + 
                                 contributor_distribution.get("small_team", 0.0)) / total_contributors
        large_team_percentage = (contributor_distribution.get("large_team", 0.0) + 
                                 contributor_distribution.get("community", 0.0)) / total_contributors
        
        # Adjust based on contributor distribution
        if small_team_percentage < 0.3 and is_small_team_query:
            # Need more small team repos
            factor *= 1.1
        elif large_team_percentage < 0.2 and is_large_team_query:
            # Need more large team repos
            factor *= 1.15
        elif small_team_percentage > 0.7 and is_small_team_query:
            # Too many small team repos
            factor *= 0.9
        elif large_team_percentage > 0.5 and is_large_team_query:
            # Too many large team repos
            factor *= 0.9
        
        # 7. Temporal Pattern Awareness: Favor queries that match current temporal patterns
        temporal_patterns = self.context_features.get("temporal_patterns", {})
        weekly_activity = temporal_patterns.get("weekly_activity", [])
        
        if weekly_activity and len(weekly_activity) == 7:
            # Get current day of week (0 = Monday, 6 = Sunday)
            current_day = datetime.now().weekday()
            
            # Check if today is a high-activity day
            today_activity = weekly_activity[current_day]
            if today_activity > 0.2:  # High activity day
                # Boost active queries on high activity days
                if is_high_activity_query:
                    factor *= 1.05
        
        # 8. Topic Diversity: Ensure we have repositories from different topics
        topics = self.context_features.get("topics", {})
        if topics and topic:
            total_topics = sum(topics.values())
            if total_topics > 0:
                current_topic_count = topics.get(topic, 0)
                topic_percentage = current_topic_count / total_topics
                
                # Boost underrepresented topics
                if topic_percentage < 0.05:
                    factor *= 1.2
                elif topic_percentage < 0.10:
                    factor *= 1.1
                # Penalize overrepresented topics
                elif topic_percentage > 0.30:
                    factor *= 0.9
                elif topic_percentage > 0.50:
                    factor *= 0.8
        
        # 9. Engagement Metrics Correlation: Prefer queries that find highly engaged repositories
        engagement_metrics = self.context_features.get("engagement_metrics", {})
        
        # High issue/PR velocity indicates active repositories
        issue_velocity = engagement_metrics.get("issue_velocity", 0.0)
        pr_velocity = engagement_metrics.get("pr_velocity", 0.0)
        
        # If current metrics show low engagement, boost queries likely to find engaged repositories
        if (issue_velocity + pr_velocity) < 5.0 and is_high_activity_query:
            factor *= 1.1
        
        # 10. Collection Diversity Score: As diversity increases, focus more on underrepresented areas
        diversity_score = self.context_features.get("diversity_score", 0.0)
        if diversity_score > 0.7:  # High diversity already achieved
            # In high diversity situation, optimize for efficiency rather than exploration
            if query["api_efficiency"] > 20:
                factor *= 1.1
        else:  # Low diversity
            # Boost queries that could increase diversity
            if star_range not in star_ranges or language not in languages:
                factor *= 1.15
        
        return factor
    
    def _calculate_bayesian_factor(self, query: Dict[str, Any]) -> float:
        """Calculate Bayesian adjustment factor using prior knowledge.
        
        This incorporates our prior beliefs about repository distributions
        to guide query selection toward likely high-yield areas with emphasis on stars.
        
        Args:
            query: The query to evaluate
            
        Returns:
            A factor to multiply the base reward (>1 = boost, <1 = penalize)
        """
        # Start with neutral factor
        factor = 1.0
        
        # Get query characteristics
        star_range = query.get("star_range", "any")
        language = query.get("language", "any")
        
        # 1. Apply star distribution prior
        if star_range != "any":
            # Get the star range boundaries
            star_values = {
                "100K+": math.log(300000),  # Increased from 200K to prioritize higher stars
                "50K-100K": math.log(70000),
                "30K-50K": math.log(40000),
                "20K-30K": math.log(25000),
                "10K-20K": math.log(15000),
                "7.5K-10K": math.log(8500),
                "5K-7.5K": math.log(6000),
                "2.5K-5K": math.log(3500),
                "1K-2.5K": math.log(1500),
                "500-1K": math.log(750),
                "100-500": math.log(250),
                "50-100": math.log(75),
                "10-50": math.log(25),
                "<10": math.log(5)
            }
            
            # Apply an immediate boost for high-star ranges
            if star_range in ["100K+", "50K-100K"]:
                factor *= 2.5  # 150% boost for very high star ranges (>50K)
            elif star_range in ["30K-50K", "20K-30K", "10K-20K", "10K-50K"]:
                factor *= 2.0  # 100% boost for high star ranges (>10K)
            elif star_range in ["7.5K-10K", "5K-10K", "5K-7.5K"]:
                factor *= 1.5  # 50% boost for medium-high star ranges (>5K)
            
            if star_range in star_values:
                # Get our current best estimate of the log-normal distribution
                prior = self.bayesian_priors.get("star_distribution", {})
                # Adjust mean to favor higher stars (increase from default 2.5 to 3.5)
                mean = prior.get("mean", 3.5)  # Default is now around 33K stars instead of 12K
                variance = prior.get("variance", 2.0)
                
                # Calculate log-normal PDF value for this star range
                log_star = star_values[star_range]
                if variance > 0:
                    # Calculate normalized score from log-normal PDF
                    exponent = -((log_star - mean) ** 2) / (2 * variance)
                    pdf_value = math.exp(exponent) / (log_star * math.sqrt(2 * math.pi * variance))
                    
                    # Normalize to a reasonable range (0.5 to 2.5) - increased upper bound
                    # This is a bit of a hack, but it gives reasonable values
                    if pdf_value > 0:
                        norm_factor = 1.0 + math.log(pdf_value + 1) / 2.5  # Adjusted from /3 to /2.5
                        # Increased upper bound from 2.0 to 2.5
                        norm_factor = max(0.5, min(2.5, norm_factor))
                        factor *= norm_factor
        
        # 2. Apply language distribution prior
        if language != "any":
            prior = self.bayesian_priors.get("language_distribution", {})
            lang_prob = prior.get(language, prior.get("other", 0.1))
            
            # Convert probability to factor (0.5 to 1.5 range)
            # Higher probability languages get higher factors
            if lang_prob > 0:
                norm_factor = 0.5 + lang_prob
                norm_factor = max(0.5, min(1.5, norm_factor))
                factor *= norm_factor
        
        return factor
    
    def _calculate_thompson_scores(self):
        """Calculate Thompson Sampling scores using advanced probabilistic models and particle filtering."""
        thompson_scores = []
        
        # Time decay and context-based adjustment factors
        collection_stage = self.context_features.get("collection_stage", 0.0)
        current_time = time.time()
        max_age = 3600 * 24 * 7  # One week in seconds
        
        # Number of particles for particle filtering
        n_particles = 20
        
        for i, query in enumerate(self.queries):
            # Use different approaches based on query usage
            if query["usage_count"] == 0:
                # For unused queries, use a more exploratory approach with prior knowledge
                base_score = np.random.random() * 0.8 + 0.2  # Random value between 0.2 and 1.0
                
                # Apply a more sophisticated boost based on context
                contextual_factor = self._calculate_contextual_factor(query)
                bayesian_factor = self._calculate_bayesian_factor(query)
                
                # Calculate optimism bonus for untried queries (UCB-like)
                # Log scaling to avoid too much exploration
                base_exploration_bonus = 0.3 * (math.log(self.total_runs + 1) / (1 + math.log(self.total_runs + 1)))
                
                # Scale exploration bonus by collection stage to increase exploration as we collect more
                collection_stage = self.context_features.get("collection_stage", 0.0)
                exploration_multiplier = 1.0 + (collection_stage * 0.7)  # Up to 70% more exploration as we collect more
                exploration_bonus = base_exploration_bonus * exploration_multiplier
                
                # Use query metadata to inform prior (helps with cold start)
                prior_boost = self._calculate_prior_score(query)
                
                # Combine with weighted factors - more weight on priors for unused queries
                # As collection progresses, increase the weight of exploration
                exploration_weight = 0.1 + (0.1 * collection_stage)  # Increase from 0.1 to 0.2 as collection completes
                
                # Adjust other weights to maintain sum = 1.0
                base_weight = max(0.3, 0.4 - (0.1 * collection_stage))  # Reduce from 0.4 to 0.3
                
                score = min(1.0, base_weight * base_score + 
                                 0.2 * (contextual_factor - 1.0) + 
                                 0.2 * (bayesian_factor - 1.0) + 
                                 exploration_weight * exploration_bonus +
                                 0.1 * prior_boost)
            else:
                # For used queries, implement a particle filtering approach for Thompson sampling
                # This better handles multimodal reward distributions
                
                # Get query history and calculate time decay weights
                # Check if the history exists as a key in the dictionary
                timestamps = []
                # Use dictionary access for consistency - query should always be a dict
                if "timestamp_history" in query and query["timestamp_history"]:
                    timestamps = query["timestamp_history"]
                
                if timestamps:
                    # Calculate time decay weights from timestamps
                    time_weights = [math.exp(-0.1 * min(max_age, current_time - ts) / 86400) for ts in timestamps]
                else:
                    # Default recency weight if no history
                    time_weights = [0.8]
                
                # Normalize weights
                total_weight = sum(time_weights) or 1.0
                time_weights = [w / total_weight for w in time_weights]
                
                # Generate particles for posterior distribution
                # Each particle represents a possible reward value
                particles = []
                
                # Get basic parameters
                alpha = max(1.0, query.get("alpha", 1.0))
                beta = max(1.0, query.get("beta", 1.0))
                
                # Generate candidate particles from multiple distributions
                
                # 1. Beta distribution for main Thompson sampling
                beta_particles = np.random.beta(alpha, beta, n_particles // 2)
                
                # 2. Gaussian particles centered around our best estimate
                # This helps model non-Beta-shaped distributions
                mean_estimate = alpha / (alpha + beta)
                variance = (alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))
                gaussian_particles = np.random.normal(mean_estimate, math.sqrt(variance), n_particles // 4)
                
                # 3. Historical performance particles - resampling from past performance
                # Get recent reward history or create synthetic ones
                reward_history = []
                
                # Try to get reward history from dictionary key - query is always a dict
                if "reward_history" in query and query["reward_history"]:
                    reward_history = query["reward_history"]
                
                # If no history available, create synthetic history from overall reward
                if not reward_history:
                    default_reward = query.get("reward", 0.0) / max(1, query.get("usage_count", 1))
                    reward_history = [default_reward]
                
                # Sample with time decay weights
                if len(reward_history) > 1:
                    # If we have history, sample from it
                    # Make sure time_weights and reward_history have the same length
                    weights = time_weights
                    if len(weights) != len(reward_history):
                        # Truncate or pad weights to match reward_history length
                        if len(weights) > len(reward_history):
                            weights = weights[:len(reward_history)]
                        else:
                            # Pad with average weight
                            avg_weight = sum(weights) / len(weights) if weights else 0.5
                            weights = weights + [avg_weight] * (len(reward_history) - len(weights))
                        
                        # Renormalize weights to sum to 1
                        total = sum(weights) or 1.0
                        weights = [w / total for w in weights]
                    
                    # Sample using normalized weights
                    try:
                        indices = np.random.choice(len(reward_history), n_particles // 4, p=weights)
                        historical_particles = [reward_history[idx] for idx in indices]
                    except ValueError:
                        # Fallback in case of any sampling error
                        historical_mean = sum(reward_history) / len(reward_history)
                        historical_particles = np.random.normal(historical_mean, 0.1, n_particles // 4)
                else:
                    # Otherwise create synthetic samples around the mean
                    historical_mean = reward_history[0]
                    historical_particles = np.random.normal(historical_mean, 0.1, n_particles // 4)
                
                # Combine all particles for our distribution estimate
                particles = list(beta_particles) + list(gaussian_particles) + list(historical_particles)
                
                # Add small amount of noise to prevent identical particles
                particles = [max(0.0, min(1.0, p + np.random.normal(0, 0.01))) for p in particles]
                
                # Sample one particle as our raw score
                raw_score = np.random.choice(particles)
                
                # Apply modifiers based on query performance metrics
                # We use sigmoid functions to smooth the transitions
                
                # 1. API efficiency (strongly favor efficient queries)
                # Sigmoid function: 1 / (1 + exp(-k*(x - x0)))
                efficiency = min(1.0, query["api_efficiency"] / 50.0)
                efficiency_bonus = 0.2 * (1 / (1 + math.exp(-10 * (efficiency - 0.5))))
                
                # 2. Quality score with weighted importance
                quality_score = query["quality_score"]
                quality_importance = self.context_features.get("top_heavy_ratio", 1.0)
                quality_bonus = 0.1 * quality_score * min(1.5, quality_importance)
                
                # 3. Duplication rate penalty with exponential scaling
                # Higher duplication rates get penalized exponentially more
                duplication_rate = query["duplication_rate"]
                duplication_penalty = 15 * (duplication_rate ** 1.5)
                
                # 4. New factor: Reward stability/consistency bonus
                # Queries with more consistent rewards are favored
                stability_bonus = 0.0
                
                # Get reward history for consistency calculation
                reward_hist = []
                # Use dictionary access for consistency - query is always a dict
                if "reward_history" in query and isinstance(query["reward_history"], (list, tuple)) and len(query["reward_history"]) > 3:
                    reward_hist = query["reward_history"]
                
                if reward_hist:
                    try:
                        # Calculate coefficient of variation (lower = more consistent)
                        rewards = np.array(reward_hist)
                        mean_reward = np.mean(rewards)
                        if mean_reward > 1e-6:  # Avoid division by near-zero
                            cv = np.std(rewards) / (mean_reward + 1e-6)
                            stability_bonus = 0.05 * (1 / (1 + math.exp(5 * (cv - 0.3))))
                    except (ValueError, TypeError) as e:
                        # Fallback in case of calculation error
                        logger.debug(f"Error calculating reward stability: {e}")
                        stability_bonus = 0.0
                
                # Apply contextual and Bayesian modifiers
                contextual_factor = self._calculate_contextual_factor(query)
                bayesian_factor = self._calculate_bayesian_factor(query)
                
                # Dynamic weighting based on collection stage and confidence
                context_weight = min(0.5, 0.3 + collection_stage * 0.4)
                confidence = min(1.0, query["usage_count"] / 20.0)
                raw_weight = 0.6 * confidence
                
                # Combine all factors with proportional weighting
                combined_score = (
                    raw_score * raw_weight +                       # Particle-filtered estimate
                    (contextual_factor - 1.0) * context_weight +   # Context adjustment
                    (bayesian_factor - 1.0) * (context_weight/2) + # Bayesian adjustment
                    efficiency_bonus +                             # Efficiency bonus
                    quality_bonus +                                # Quality bonus
                    stability_bonus -                              # Reward stability bonus
                    duplication_penalty                            # Duplication penalty
                )
                
                # Ensure score is in valid range
                score = max(0.0, min(1.0, combined_score))
            
            thompson_scores.append((i, score))
        
        return thompson_scores
        
    def _calculate_prior_score(self, query):
        """Calculate a prior score for a query based on its metadata.
        
        This helps with cold start for new queries by using metadata to predict performance.
        
        Args:
            query: The query to evaluate
            
        Returns:
            Prior score between 0.0 and 1.0
        """
        score = 0.5  # Start with neutral score
        
        # 1. Star range prior - high stars but not too high
        star_range = query.get("star_range", "any")
        if star_range in ["10K-20K", "20K-30K", "5K-7.5K", "7.5K-10K"]:
            score += 0.1  # Medium-high star ranges tend to perform well
        elif star_range in ["30K-50K", "50K-100K"]:
            score += 0.05  # Very high star ranges are good but can be sparse
        elif star_range in ["1K-2.5K", "2.5K-5K"]:
            score += 0.02  # Lower but still respectable star counts
        elif star_range in ["<10", "10-50"]:
            score -= 0.1  # Very low star ranges often have more noise
            
        # 2. Language prior - based on popularity and richness of ecosystem
        language = query.get("language", "any")
        if language in ["javascript", "typescript", "python"]:
            score += 0.08  # These languages tend to have good repositories
        elif language in ["go", "rust", "kotlin"]:
            score += 0.05  # Newer languages with active ecosystems
        elif language in ["java", "csharp", "cpp"]:
            score += 0.03  # Established languages
        elif language == "any":
            score += 0.01  # No language filter can be good for discovery
            
        # 3. Sort order prior
        sort = query.get("sort", "stars")
        if sort == "stars":
            score += 0.05  # Sorting by stars is reliable
        elif sort == "updated":
            score += 0.08  # Recently updated repos often more relevant
        elif sort == "forks":
            score += 0.02  # Forks can be a good signal, but less reliable
            
        # 4. Creation date prior
        creation = query.get("creation", "any")
        if "2022" in creation or "2023" in creation:
            score += 0.07  # Very recent repos
        elif "2020" in creation or "2021" in creation:
            score += 0.04  # Recent but established repos
        elif "<2015" in creation:
            score -= 0.02  # Much older repos might be less maintained
            
        # 5. Topic prior
        topic = query.get("topic", "")
        if topic in ["machine-learning", "deep-learning", "data-science"]:
            score += 0.06  # These topics tend to have high-quality repos
        elif topic in ["web-development", "devops", "mobile"]:
            score += 0.04  # Common topics with many good repos
        elif topic in ["blockchain", "game-development"]:
            score += 0.02  # More niche topics
            
        # Ensure score is in range [0.0, 1.0]
        return max(0.0, min(1.0, score))
    
    def update_context(self, metrics_collector, target_count: int) -> None:
        """Update context features for contextual bandits.
        
        This method updates the context features used by the bandit algorithm
        based on the current state of repository collection. These features
        influence query selection to target underrepresented areas.
        
        Args:
            metrics_collector: Metrics collector providing statistics about collected repositories
            target_count: Target number of repositories to collect
        """
        # Get metrics from collector
        if metrics_collector is None:
            logger.warning("No metrics collector provided to update_context, skipping update")
            return
            
        metrics = metrics_collector.get_metrics()
        if not metrics:
            logger.warning("Could not get metrics from collector")
            return
            
        try:
            # Get repository statistics
            repo_stats = metrics.get("repositories", {})
            unique_repos = repo_stats.get("unique", 0)
            collection_progress = min(1.0, unique_repos / max(1, target_count))
            
            # Update collection stage context feature
            self.context_features["collection_stage"] = collection_progress
            
            # Update repository distribution statistics
            if "distribution" in repo_stats:
                distribution = repo_stats.get("distribution", {})
                
                # Update star range distribution
                if "star_ranges" in distribution:
                    self.context_features["star_ranges"] = distribution["star_ranges"]
                    
                    # Calculate top-heavy ratio (high-star to low-star repos)
                    high_star_count = sum(distribution["star_ranges"].get(range_name, 0)
                                        for range_name in ["100K+", "50K-100K", "30K-50K", "20K-30K", "10K-20K"])
                    low_star_count = sum(distribution["star_ranges"].get(range_name, 0)
                                        for range_name in ["1K-2.5K", "500-1K", "100-500", "50-100", "10-50", "<10"])
                    
                    if low_star_count > 0:
                        self.context_features["top_heavy_ratio"] = high_star_count / low_star_count
                    else:
                        self.context_features["top_heavy_ratio"] = 1.0
                
                # Update language distribution
                if "languages" in distribution:
                    self.context_features["languages"] = distribution["languages"]
                
                # Update topic distribution
                if "topics" in distribution:
                    self.context_features["topics"] = distribution["topics"]
                    
                # Update repository age distribution
                if "ages" in distribution:
                    self.context_features["ages"] = distribution["ages"]
            
            # Update activity metrics
            if "activity" in metrics:
                activity_metrics = metrics.get("activity", {})
                
                # Update activity levels
                if "levels" in activity_metrics:
                    self.context_features["activity_levels"] = activity_metrics["levels"]
                
                # Update contributor distribution
                if "contributors" in activity_metrics:
                    self.context_features["contributor_distribution"] = activity_metrics["contributors"]
                
                # Update engagement metrics
                if "engagement" in activity_metrics:
                    self.context_features["engagement_metrics"] = activity_metrics["engagement"]
                
                # Update temporal patterns
                if "temporal" in activity_metrics:
                    self.context_features["temporal_patterns"] = activity_metrics["temporal"]
            
            # Update diversity score based on distributions
            diversity_components = []
            
            # Calculate star range diversity
            if "star_ranges" in self.context_features and self.context_features["star_ranges"]:
                star_values = list(self.context_features["star_ranges"].values())
                if sum(star_values) > 0:
                    star_diversity = min(1.0, 1.0 - (max(star_values) / sum(star_values)))
                    diversity_components.append(star_diversity)
            
            # Calculate language diversity
            if "languages" in self.context_features and self.context_features["languages"]:
                lang_values = list(self.context_features["languages"].values())
                if sum(lang_values) > 0:
                    lang_diversity = min(1.0, 1.0 - (max(lang_values) / sum(lang_values)))
                    diversity_components.append(lang_diversity)
            
            # Calculate topic diversity
            if "topics" in self.context_features and self.context_features["topics"]:
                topic_values = list(self.context_features["topics"].values())
                if sum(topic_values) > 0:
                    topic_diversity = min(1.0, 1.0 - (max(topic_values) / sum(topic_values)))
                    diversity_components.append(topic_diversity)
            
            # Calculate overall diversity score
            if diversity_components:
                self.context_features["diversity_score"] = sum(diversity_components) / len(diversity_components)
            else:
                self.context_features["diversity_score"] = 0.0
            
            # Update exploration weight based on new context features
            self._update_exploration_weight()
            
            # Log important context features for debugging
            bandit_logger.info(
                f"Updated context features - Progress: {collection_progress:.2f}, "
                f"Unique repos: {unique_repos}, Target: {target_count}, "
                f"Diversity: {self.context_features['diversity_score']:.2f}, "
                f"Exploration weight: {self.exploration_weight:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Error updating context features: {e}")
    
    def get_best_query_avoiding(self, avoid_texts: Set[str], strategy="ucb", exclude_used=True, max_usage=10) -> Optional[Dict[str, Any]]:
        """Get the best query while avoiding specified queries.
        
        Args:
            avoid_texts: Set of query texts to avoid
            strategy: Selection strategy - "ucb" or "thompson"
            exclude_used: Whether to exclude heavily used queries
            max_usage: Maximum usage count for a query to be considered
            
        Returns:
            The selected query dictionary or None if no suitable query found
        """
        # Log the beginning of query selection
        bandit_logger.debug(f"Selecting query using {strategy} strategy (total runs: {self.total_runs}), avoiding {len(avoid_texts)} queries")
        
        # Filter out avoided queries and heavily used queries
        available_queries = [q for q in self.queries if q["query_text"] not in avoid_texts]
        
        if exclude_used:
            available_queries = [q for q in available_queries if q["usage_count"] < max_usage]
        
        # If no queries available, return None
        if not available_queries:
            return None
        
        # Determine selection strategy (rest of logic is same as get_best_query)
        if strategy == "ucb" or (strategy == "auto" and self.total_runs < 100):
            # Update UCB scores first
            self._update_ucb_scores()
            
            # Find the query with the highest UCB score plus small random jitter to prevent ties
            best_query = max(available_queries, key=lambda q: q["ucb_score"] + random.uniform(-0.05, 0.05))
            
        elif strategy == "thompson" or strategy == "auto":
            # Use Thompson Sampling
            thompson_scores = self._calculate_thompson_scores()
            
            # Filter to only available queries
            available_indices = [i for i, q in enumerate(self.queries) if q in available_queries]
            if not available_indices:
                return None
                
            filtered_scores = [(i, s + random.uniform(-0.05, 0.05)) for i, s in thompson_scores if i in available_indices]
            
            # Get the query with the highest Thompson score
            best_index, _ = max(filtered_scores, key=lambda x: x[1])
            best_query = self.queries[best_index]
            
        else:
            # Fallback to random with weights based on duplication rate
            weights = [1.0 / (1.0 + q["duplication_rate"]) for q in available_queries]
            best_query = random.choices(available_queries, weights=weights, k=1)[0]
        
        return best_query
    
    def update_query_performance(self, query: Dict[str, Any], 
                                results_count: int, 
                                unique_count: int, 
                                api_calls: int = 1, 
                                success: bool = True, 
                                quality_score: float = 0.0, 
                                duplication_rate: float = 0.0,
                                star_counts: Optional[List[int]] = None):
        """Update performance metrics for a query with enhanced tracking of historical performance.
        
        Now includes novelty calculation and combined novelty-reward scoring for better exploration.
        
        Args:
            query: The query to update
            results_count: Total number of repositories returned
            unique_count: Number of unique repositories discovered
            api_calls: Number of API calls made for this query
            success: Whether the query execution was successful
            quality_score: Quality score for the repositories (0.0-1.0)
            duplication_rate: Rate of duplication with existing repos (0.0-1.0)
            star_counts: Optional list of star counts for the repositories found by this query
        """
        # Find the query in our pool
        for q in self.queries:
            if q["query_text"] == query["query_text"]:
                # Update execution stats
                if success:
                    q["success_count"] += 1
                else:
                    q["error_count"] += 1
                
                # Update result counts
                q["total_results"] += results_count
                q["unique_results"] += unique_count
                
                # Update derived metrics
                if results_count > 0:
                    q["unique_rate"] = unique_count / results_count
                
                if api_calls > 0:
                    # Exponential moving average for API efficiency
                    new_efficiency = unique_count / api_calls
                    if q["api_efficiency"] == 0.0:
                        q["api_efficiency"] = new_efficiency
                    else:
                        q["api_efficiency"] = 0.8 * q["api_efficiency"] + 0.2 * new_efficiency
                
                # Update quality score with exponential moving average
                if q["quality_score"] == 0.0:
                    q["quality_score"] = quality_score
                else:
                    q["quality_score"] = 0.8 * q["quality_score"] + 0.2 * quality_score
                
                # Update duplication rate with exponential moving average
                q["duplication_rate"] = 0.7 * q["duplication_rate"] + 0.3 * duplication_rate
                
                # Calculate star-based rewards if star counts are provided
                star_reward = 0.0
                star_weight_coefficient = 0.5  # Increased coefficient for star rewards (from 0.3 to 0.5)
                star_bucket_distribution = {}
                
                # Get collection stage (0.0-1.0) to scale star rewards
                collection_stage = self.context_features.get("collection_stage", 0.0)
                
                # Reduce star weight coefficient as collection progresses to favor more exploration
                adjusted_star_weight = star_weight_coefficient * (1.0 - (collection_stage * 0.3))
                
                # Initialize star range multiplier to prioritize high-star queries
                star_range_multiplier = 1.0
                
                if star_counts and len(star_counts) > 0:
                    # Calculate average star count
                    avg_star_count = sum(star_counts) / len(star_counts)
                    
                    # Apply logarithmic transformation to avoid over-prioritizing very high-star repos
                    # Normalize to a 0-1 range (assuming most repos have <100K stars)
                    star_reward = star_weight_coefficient * math.log10(avg_star_count + 1) / 5.0
                    
                    # Track star distribution in buckets
                    if "star_bucket_counts" not in q:
                        q["star_bucket_counts"] = {}
                    
                    # Create star buckets for analysis
                    # Define star bucket thresholds
                    star_buckets = [
                        (100000, float('inf'), "100K+"),
                        (50000, 100000, "50K-100K"),
                        (10000, 50000, "10K-50K"),
                        (5000, 10000, "5K-10K"),
                        (1000, 5000, "1K-5K"),
                        (100, 1000, "100-1K"),
                        (0, 100, "<100")
                    ]
                    
                    for star_count in star_counts:
                        # Find the appropriate bucket for this star count
                        bucket = next((name for min_val, max_val, name in star_buckets 
                                      if min_val <= star_count < max_val), None)
                        
                        if bucket:
                            # Update bucket counts in query metadata
                            if bucket not in q["star_bucket_counts"]:
                                q["star_bucket_counts"][bucket] = 0
                            q["star_bucket_counts"][bucket] += 1
                            
                            # Also track for this specific update
                            if bucket not in star_bucket_distribution:
                                star_bucket_distribution[bucket] = 0
                            star_bucket_distribution[bucket] += 1
                    
                    # Track the star distribution in the overall collection
                    collection_star_distribution = self.context_features.get("star_distribution", {})
                    
                    # Calculate weighted score based on collection needs
                    # Apply diminishing returns - higher reward for star ranges we have fewer of
                    diminishing_returns_factor = 0.0
                    if collection_star_distribution and star_bucket_distribution:
                        total_collection_repos = sum(collection_star_distribution.values()) or 1
                        
                        # Determine which star buckets are underrepresented
                        weighted_score = 0.0
                        total_weight = 0.0
                        
                        for bucket, count in star_bucket_distribution.items():
                            # Calculate what percentage of our collection is in this bucket
                            bucket_pct_in_collection = collection_star_distribution.get(bucket, 0) / total_collection_repos
                            
                            # Give higher weight to underrepresented buckets
                            # Use an inverse relationship with a smoothing factor
                            if bucket_pct_in_collection > 0:
                                weight = 1.0 / (bucket_pct_in_collection + 0.05)
                            else:
                                weight = 10.0  # High weight for completely missing buckets
                                
                            # Additional bonus for high-star repos
                            if bucket in ["100K+", "50K-100K", "10K-50K"]:
                                weight *= 1.5
                                
                            weighted_score += count * weight
                            total_weight += count
                        
                        if total_weight > 0:
                            diminishing_returns_factor = weighted_score / total_weight
                            # Normalize the factor to a reasonable range (0.0-0.5)
                            diminishing_returns_factor = min(0.5, diminishing_returns_factor / 10.0)
                    
                    # Apply the diminishing returns adjustment to star reward
                    star_reward = star_reward * (1.0 + diminishing_returns_factor)
                    
                    # Set star range multiplier based on query's target star range
                    star_range = q.get("star_range", "any")
                    
                    # Define multiplier ranges for cleaner code
                    star_range_multipliers = {
                        "very_high": {"ranges": ["100K+", "50K-100K"], "multiplier": 5.0},
                        "high": {"ranges": ["30K-50K", "20K-30K", "10K-20K", "10K-50K"], "multiplier": 3.0},
                        "medium_high": {"ranges": ["7.5K-10K", "5K-10K", "5K-7.5K"], "multiplier": 2.0}
                    }
                    
                    # Default multiplier (for lower star ranges)
                    star_range_multiplier = 1.0
                    
                    # Find the appropriate multiplier based on the star range
                    for category in star_range_multipliers:
                        if star_range in star_range_multipliers[category]["ranges"]:
                            star_range_multiplier = star_range_multipliers[category]["multiplier"]
                            break
                    
                    # Apply the star range multiplier to the star reward, but scale down with collection progress
                    # As we collect more, we care less about high stars and more about unique repos
                    scaled_multiplier = star_range_multiplier * (1.0 - (collection_stage * 0.4))
                    star_reward = star_reward * scaled_multiplier
                    
                    # Store star-related metrics in query metadata
                    q["avg_star_count"] = avg_star_count
                    q["last_star_reward"] = star_reward
                    q["star_bucket_distribution"] = star_bucket_distribution
                    q["star_range_multiplier"] = star_range_multiplier
                
                # Calculate combined reward metric
                # Weight factors adjusted to prioritize API efficiency and unique discovery
                # - unique_rate: higher is better (0.0-1.0)
                # - api_efficiency: higher is better (0.0-100.0 typically)
                # - quality_score: higher is better (0.0-1.0)
                # - success: 1.0 if successful, 0.0 if not
                # - duplication_rate: lower is better (0.0-1.0)
                # - information_density: unique repos per result (higher is better)
                # - star_reward: higher is better (0.0-0.3 typically)
                
                # Normalize api_efficiency to 0-1 range (assuming max efficiency of 100)
                # Use a higher ceiling to reward exceptionally efficient queries more
                normalized_efficiency = min(1.0, q["api_efficiency"] / 50.0)
                
                # Information density measure (related to, but distinct from unique_rate)
                # This specifically rewards queries that find unique repos with fewer total results
                information_density = unique_count / max(1, api_calls * 100)  # 100 items per API call maximum
                normalized_density = min(1.0, information_density)
                
                # Success factor: heavily penalize failed queries
                success_factor = 1.0 if success else 0.1
                
                # Calculate reward with enhanced weighting based on collection context
                # Get the collection stage to dynamically adjust weights
                collection_stage = self.context_features.get("collection_stage", 0.0)
                
                # Early stage: Focus on discovery (unique rate)
                # Calculate collection progress for adaptive scoring
                try:
                    # Use metrics collector passed through dependency injection
                    if self.metrics_collector:
                        metrics = self.metrics_collector.get_metrics()
                        self.found_repos = metrics["repositories"]["unique"]
                except:
                    # If there's an error, use a default value
                    logger.warning("Could not access metrics collector for unique repository count")
                    
                collection_progress = compute_collection_progress(
                    self.found_repos, 
                    self.target_repos
                )
                
                # Calculate query novelty score if novelty-driven exploration is enabled
                novelty_score = 0.0
                if self.enable_novelty and self.similarity_engine is not None:
                    # Convert query configuration to string representation for similarity comparison
                    query_str = self._query_to_string(q)
                    
                    # Check if we've already computed the novelty for this query
                    if query_str not in self.query_to_similarity_id:
                        # Compute novelty score and add to index
                        novelty_score = self.similarity_engine.compute_novelty_score(query_str)
                        # Add to index for future comparisons
                        similarity_id = self.similarity_engine.add_query(query_str)
                        self.query_to_similarity_id[query_str] = similarity_id
                    
                    # Store the novelty score in the query object
                    q["novelty_score"] = novelty_score
                
                # Get adaptive alpha value for blending reward and novelty
                novelty_alpha = compute_novelty_alpha(collection_progress) if self.enable_novelty else 0.0
                    
                # Mid stage: Balanced approach
                # Late stage: Focus on efficiency and quality
                if collection_stage < 0.3:  # Early stage
                    # Early stage prioritizes discovery and high-star repos
                    standard_reward = (
                        0.25 * q["unique_rate"] +                # Uniqueness (25%) - decreased to make room for star reward
                        0.20 * normalized_efficiency +           # Efficiency (20%) - decreased
                        0.15 * q["quality_score"] +              # Quality (15%)
                        0.05 * success_factor +                  # Success (5%)
                        0.10 * (1.0 - q["duplication_rate"]) +   # Novelty (10%)
                        0.10 * normalized_density +              # Information density (10%)
                        0.15 * star_reward                       # Star reward (15%) - NEW
                    )
                else: 
                    # Balanced approach with significant star consideration
                    standard_reward = (
                        0.5 * q["unique_rate"] +                # Uniqueness (50%)
                        0.5 * (1.0 - q["duplication_rate"])     # Novelty (50%)
                    )
                
                # Blend standard reward with novelty score based on collection progress
                if self.enable_novelty:
                    reward = (novelty_alpha * standard_reward) + ((1.0 - novelty_alpha) * novelty_score)
                    
                    # Log the impact of novelty-driven exploration
                    if self.total_runs % 100 == 0:  # Log periodically to avoid too much output
                        bandit_logger.info(
                            f"Novelty impact - Progress: {collection_progress:.2f}, "
                            f"Alpha: {novelty_alpha:.2f}, Standard: {standard_reward:.4f}, "
                            f"Novelty: {novelty_score:.4f}, Combined: {reward:.4f}"
                        )
                else:
                    reward = standard_reward
                
                # Add star efficiency tracking - how efficiently this query finds high-star repos
                if star_counts and len(star_counts) > 0 and api_calls > 0:
                    # Calculate the sum of star counts found per API call
                    star_efficiency = sum(star_counts) / api_calls
                    
                    # Initialize or update the star efficiency metric
                    if "star_efficiency" not in q or q["star_efficiency"] == 0.0:
                        q["star_efficiency"] = star_efficiency
                    else:
                        # Use exponential moving average
                        q["star_efficiency"] = 0.8 * q["star_efficiency"] + 0.2 * star_efficiency
                
                # Accumulate reward for UCB calculation
                q["reward"] += reward
                
                # Store timestamp and reward for this update (for time-weighted sampling)
                # Initialize history tracking if it doesn't exist
                if "timestamp_history" not in q:
                    q["timestamp_history"] = []
                if "reward_history" not in q:
                    q["reward_history"] = []
                
                # Add current timestamp and reward
                current_time = time.time()
                q["timestamp_history"].append(current_time)
                q["reward_history"].append(reward)
                
                # Keep history limited to last 50 entries to prevent unbounded growth
                max_history = 50
                if len(q["timestamp_history"]) > max_history:
                    q["timestamp_history"] = q["timestamp_history"][-max_history:]
                if len(q["reward_history"]) > max_history:
                    q["reward_history"] = q["reward_history"][-max_history:]
                
                # Update Thompson sampling parameters with more sophisticated approach
                if success:
                    if unique_count > 0:
                        # Success with unique results: increase alpha proportionally to success factors
                        
                        # Base success contribution - unique repositories found
                        unique_contribution = unique_count * q["unique_rate"] 
                        
                        # Efficiency bonus - reward efficient queries more
                        efficiency_factor = min(2.0, normalized_efficiency * 3.0)
                        
                        # Quality bonus - slightly reward higher quality repositories
                        quality_factor = 1.0 + (0.5 * q["quality_score"])
                        
                        # Star bonus - significantly reward finding high-star repos
                        star_factor = 1.0
                        if star_counts and len(star_counts) > 0:
                            avg_stars = sum(star_counts) / len(star_counts)
                            # Log scale to prevent extreme values dominating
                            star_factor = 1.0 + min(3.0, math.log10(avg_stars + 1) / 2.0)
                        
                        # Combined success update with star factor
                        success_update = unique_contribution * efficiency_factor * quality_factor * star_factor
                        
                        # Update alpha with combined factors
                        q["alpha"] += max(0.5, success_update)  # Minimum success reward of 0.5
                    else:
                        # Success but no unique results: smaller alpha increase
                        q["alpha"] += 0.2
                        # With some beta increase for duplication
                        q["beta"] += 0.5 * duplication_rate
                else:
                    # API or other failure: significant beta increase
                    q["beta"] += 2.0
                    
                # Additional beta adjustment based on duplication rate
                # Higher duplication = higher beta (failure weight)
                if duplication_rate > 0.7 and api_calls > 1:
                    # Penalize queries with extreme duplication
                    q["beta"] += 10 * api_calls * (duplication_rate - 0.5)
                
                # Enhanced contextual data tracking
                # Update repository activity based on result metadata if available
                # These fields would be populated by the caller if available
                # Use dictionary access for consistency - query is always a dict
                if "activity_metadata" in query and query["activity_metadata"]:
                    self._update_activity_metrics(query["activity_metadata"])
                
                # Update contributor distribution if available
                if "contributor_metadata" in query and query["contributor_metadata"]:
                    self._update_contributor_metrics(query["contributor_metadata"])
                
                # Update engagement metrics if available
                if "engagement_metadata" in query and query["engagement_metadata"]:
                    self._update_engagement_metrics(query["engagement_metadata"])
                
                # Update component success rates
                self._update_component_stats(
                    q,
                    success=success,
                    unique_count=unique_count,
                    results_count=results_count,
                    api_calls=api_calls,
                    duplication_rate=duplication_rate,
                    quality_score=q["quality_score"]
                )
                
                # Log component success rates every 10 runs
                if self.total_runs % 10 == 0:
                    # Get the top 3 performing values for each component type
                    top_components = {}
                    for component_type, rates in self.component_success_rates.items():
                        # Skip if no data yet
                        if not rates:
                            continue
                            
                        # Sort by success rate and get top 3
                        sorted_items = sorted(rates.items(), key=lambda x: x[1], reverse=True)
                        top_3 = sorted_items[:3]
                        
                        # Format for logging
                        top_components[component_type] = {k: round(v, 2) for k, v in top_3}
                    
                    # Save performance data periodically
                if self.total_runs % 10 == 0:
                    self._save_performance_data()
                
                break 
    
    def _query_to_string(self, query):
        """Convert a query configuration to a string for similarity comparison.
        
        Args:
            query: Query configuration dictionary
            
        Returns:
            String representation of the query
        """
        # For simplicity, we'll use the query_text field if available
        if "query_text" in query:
            return query["query_text"]
            
        # Otherwise, build a string representation from key components
        components = []
        
        # Extract key components in a specific order
        for key in ["stars", "language", "sort", "creation", "topic"]:
            if key in query and query[key]:
                components.append(f"{key}:{query[key]}")
        
        # Join components with spaces for a consistent string representation
        return " ".join(components)
    
    def _get_collection_progress(self):
        """Get the current collection progress for adaptive mutation.
        
        Returns:
            float: Collection progress as a value between 0.0 and 1.0
        """
        try:
            # Use metrics collector passed through dependency injection
            if self.metrics_collector:
                metrics = self.metrics_collector.get_metrics()
                self.found_repos = metrics["repositories"]["unique"]
        except:
            # If there's an error, use a default value
            logger.warning("Could not access metrics collector for unique repository count")
            
        return compute_collection_progress(self.found_repos, self.target_repos)
        
    def _determine_mutation_strategy(self, star_focused, provided_mutation_types, local_mutation):
        """Determine mutation types to use based on context.
        
        Args:
            star_focused: Whether to focus on star-optimized mutations
            provided_mutation_types: Optional specific mutation types to use
            local_mutation: Whether this is a local mutation
            
        Returns:
            tuple: (mutation_types_to_use, exploration_boost, generate_random_probability)
        """
        # Base mutation types from original implementation
        base_mutation_types = ["language", "sort", "stars", "creation", "topic", "crossover"]
        
        # Add radical mutation types for enhanced evolution
        radical_mutation_types = [
            "extreme_stars",       # Make dramatic jumps in star ranges
            "compound",            # Change 2-3 parameters at once
            "temporal_partition",  # Split date ranges into finer increments
            "parameter_inversion", # Invert sort order and flip parameters
            "micro_date_range",    # Split date ranges into days or weeks
            "language_combination", # Combine multiple languages
            "niche_finder",        # Target repositories with specific file types
            "topic_explorer"       # Generate queries with multiple topic combinations
        ]
        
        # Add new star-focused mutation types
        star_mutation_types = [
            "star_bracket_shift",        # Strategically shift star ranges up or down
            "viral_repo_finder",        # Target repositories with rapid star growth rates
            "star_topic_correlation",   # Combine high-performing star ranges with successful topics
            "star_threshold_exploration", # Create very specific star thresholds based on success
            "star_band_narrowing",     # Create very narrow star range queries
            "language_star_correlation", # Target languages with unusually high star counts
            "creation_star_correlation"  # Find relationships between creation dates and star counts
        ]
        
        # Check if we need exploration boost
        unique_count = self.metrics_collector.get_metrics().get("repositories", {}).get("unique", 0)
            
        # Use provided mutation types if specified, otherwise select based on context
        if provided_mutation_types:
            # Use the specifically provided mutation types (for local aggressive mutations)
            mutation_types_to_use = provided_mutation_types
        else:
            # Base exploration boost decision on repository count
            exploration_boost = unique_count >= 25000
            
            # Determine if we need to focus on star optimization
            if star_focused:
                # When star-focused, heavily weight star-specific mutations
                mutation_types_to_use = base_mutation_types + radical_mutation_types + star_mutation_types * 3
                bandit_logger.info("Applying star-focused evolution - prioritizing star-based mutations")
            # Combine mutation types, giving more weight to radical mutations when we need more exploration
            elif exploration_boost:
                # When we have 25K+ repos, use more radical mutations
                mutation_types_to_use = base_mutation_types + radical_mutation_types * 2 + star_mutation_types
                bandit_logger.info("Applying exploration boost - using more radical mutations")
            else:
                # Normal operation - balanced mutation types
                mutation_types_to_use = base_mutation_types + radical_mutation_types + star_mutation_types
        
        # Set a flag to represent exploration boost (needed for later in the function)
        # For local mutations, we always want to boost exploration
        exploration_boost = local_mutation or (unique_count >= 25000)
        
        # With exploration boost or star focus, sometimes generate completely random queries
        generate_random_probability = 0.25 if (exploration_boost or star_focused) else 0.05
        
        return mutation_types_to_use, exploration_boost, generate_random_probability
    
    def _select_parent_queries(self, parent_queries, star_focused):
        """Select parent queries for evolution.
        
        Args:
            parent_queries: Provided parent queries or None to select from top performers
            star_focused: Whether to focus on star-optimized evolution
            
        Returns:
            tuple: (parent_candidates, elite_parents, top_parents)
        """
        # Use provided parent queries if specified, otherwise select from top performers
        if parent_queries:
            parent_candidates = parent_queries
        else:
            # Create a top performer pool from the top 20% of queries
            parent_candidates = [q for q in self.queries if q["usage_count"] >= 3]
            
        if not parent_candidates:
            return [], [], []
        
        # Create weighted selection based on both standard reward and star-finding ability
        scored_parents = []
        for q in parent_candidates:
            # Calculate standard reward per usage
            base_reward = q["reward"] / max(1, q["usage_count"])
            
            # Add star-based weighting for parent selection
            star_weight = 0.0
            if "avg_star_count" in q and q["avg_star_count"] > 0:
                # Give higher probability to queries that found high-star repositories
                # Use logarithmic scaling to prevent extreme domination
                star_weight = math.log10(q["avg_star_count"] + 1) / 5.0
            
            # Combined selection weight with star bonus
            selection_weight = base_reward * (1.0 + star_weight)
            scored_parents.append((q, selection_weight))
            
        # Sort by combined score (descending)
        scored_parents.sort(key=lambda x: x[1], reverse=True)
        
        # Get the top 20% (or at least 10) of parents
        top_percent = max(10, int(len(scored_parents) * 0.2))
        top_parents_scored = scored_parents[:top_percent]
        
        # Always include the top 5 performers (elite preservation)
        elite_parents = [p[0] for p in scored_parents[:min(5, len(scored_parents))]]
        
        # Create a slightly larger pool for the rest
        top_parents = [p[0] for p in top_parents_scored]
        
        # Track individual query components that lead to success
        component_success_rates = self._analyze_component_success_rates(top_parents)
        
        # Identify successful patterns across queries
        patterns = self._identify_query_patterns(top_parents)
        if patterns:
            bandit_logger.info(f"Identified successful patterns: {patterns}")
            
        return parent_candidates, elite_parents, top_parents
    
    def _adjust_mutation_count_for_plateaus(self, count):
        """Adjust mutation count based on repository count to overcome known plateaus.
        
        Args:
            count: Base number of mutations to generate
            
        Returns:
            int: Adjusted mutation count
        """
        try:
            # Use metrics collector passed through dependency injection
            if self.metrics_collector:
                metrics = self.metrics_collector.get_metrics()
                current_repos = metrics["repositories"]["unique"]
            
            # Local minima ranges where we want to increase mutations
            # Increase mutations if we're in known plateau regions
            if 65000 <= current_repos <= 75000:
                # We're in the 70k region where we often get stuck
                count = max(count, 15)  # Triple the mutations
                logger.info(f"Increasing mutation count to {count} to overcome 70k repository plateau")
            elif 40000 <= current_repos <= 45000:
                # Another potential plateau region
                count = max(count, 12)  # More than double mutations
                logger.info(f"Increasing mutation count to {count} to overcome 40k repository plateau")
            elif current_repos >= 10000:
                # General scaling as we collect more repositories
                # Start increasing gradually after 10k repos
                scaling_factor = min(3.0, 1.0 + (current_repos / 50000))  # Cap at 3x
                new_count = max(count, int(count * scaling_factor))
                if new_count > count:
                    count = new_count
                    logger.info(f"Scaling mutation count to {count} based on collection progress ({current_repos} repos)")
        except Exception as e:
            logger.warning(f"Error adjusting mutation count: {e}")
            
        return count
    
    def _apply_novelty_guided_evolution(self, parent_queries, count, collection_progress):
        """Apply novelty-guided evolution to generate new queries.
        
        Args:
            parent_queries: Provided parent queries or None to use top performers
            count: Number of queries to evolve
            collection_progress: Current collection progress value
            
        Returns:
            list: Evolved query objects
        """
        # Get the best performing queries as parents if not provided
        if parent_queries is None:
            # Select top performing queries as parents
            parent_queries = sorted(
                [q for q in self.queries if q.get("success_count", 0) > 0],
                key=lambda q: q.get("reward", 0.0),
                reverse=True
            )[:25]  # Use top 25 as parents for more diversity
        
        # Generate mutations using our novelty-guided adaptive strategy
        novelty_mutations = evolve_queries(
            collection_progress=collection_progress,
            parent_queries=parent_queries,
            num_mutations=count * 6,  # Tripled the base number of mutations
            max_mutations=count * 12,  # Up to 6x the original count
            similarity_engine=self.similarity_engine  # Pass similarity engine for novelty guidance
        )
        
        # Convert these mutations to our query format
        evolved_queries = []
        for mutation in novelty_mutations:
            # Create a basic query dictionary
            query = {
                "query_text": "", # Will be filled in later during query building
                "stars": mutation.get("stars", "10..1000"),
                "language": mutation.get("language", ""),
                "sort": mutation.get("sort", "stars"),
                "creation": mutation.get("created", ""),
                "topic": mutation.get("topic", ""),
                "generation": 1,
                # Initialize performance metrics
                "usage_count": 0,
                "success_count": 0,
                "error_count": 0,
                "total_results": 0,
                "unique_results": 0,
                "unique_rate": 0.0,
                "api_efficiency": 0.0,
                "quality_score": 0.5,  # Neutral starting point
                "duplication_rate": 0.0,
                "reward": 0.1,  # Small initial reward to encourage trying
                "star_bucket_counts": {},
                "timestamp_history": [],
                "reward_history": []
            }
            
            # Get novelty score and add it to the query
            if self.similarity_engine is not None:
                query_str = self._query_to_string(query)
                novelty_score = self.similarity_engine.compute_novelty_score(query_str)
                query["novelty_score"] = novelty_score
            
            evolved_queries.append(query)
            
        return evolved_queries
        
    def _evolve_queries(self, count=5, star_focused=False, mutation_types=None, parent_queries=None, local_mutation=False, cache_manager=None):
        """Evolve new queries based on high-performing ones.
        
        Args:
            count: Number of new queries to evolve
            star_focused: Whether to focus on star-optimized mutations
            mutation_types: Optional list of specific mutation types to use
            parent_queries: Optional list of specific parent queries to use
            local_mutation: Whether this is a local mutation triggered by query exhaustion
        
        Returns:
            List of new query objects created by evolution
        """
        # Calculate collection progress for adaptive mutation
        collection_progress = self._get_collection_progress()
        
        # If novelty-driven evolution is enabled, use our more sophisticated evolution approach
        if self.enable_novelty and not local_mutation:
            evolved_queries = self._apply_novelty_guided_evolution(parent_queries, count, collection_progress)
            if len(evolved_queries) >= count:
                # If we have enough, just return these
                return evolved_queries[:count]
        
        # If novelty mutations are disabled or we need more, fall through to original evolution
        # Adjust mutation count if this is a regular evolution (not a local mutation)
        if not local_mutation and not parent_queries:
            count = self._adjust_mutation_count_for_plateaus(count)
        
        # If this is a local aggressive mutation (parent_queries provided), skip the 100 run check
        # This allows local mutations even early in the crawl
        if parent_queries is None and self.total_runs < 100:
            # Not enough data to evolve yet
            return []
            
        # Select parent queries for evolution
        parent_candidates, elite_parents, top_parents = self._select_parent_queries(parent_queries, star_focused)
        if not parent_candidates:
            return []
            
        # Give higher weight to elite parents
        weighted_parents = elite_parents * 3 + top_parents
        
        # Set of existing query texts to avoid duplicates
        existing_queries = set(q["query_text"] for q in self.queries)
        
        # Evolve new queries through different mutation types
        new_queries = []
        
        # Determine mutation types and exploration settings
        mutation_types_to_use, exploration_boost, generate_random_probability = self._determine_mutation_strategy(
            star_focused, 
            mutation_types, 
            local_mutation
        )
            
        for _ in range(count):
            # Determine if we should generate a completely random query
            if random.random() < generate_random_probability:
                # Generate completely random query to explore new areas
                new_query_text = self._generate_random_query()
                
                if new_query_text and new_query_text not in existing_queries:
                    # Extract components for metadata
                    star_range = "any"
                    language = "any"
                    sort_option = random.choice(self.sort_options)
                    creation_period = "any"
                    topic = ""
                    
                    # Extract from query parts
                    for part in new_query_text.split():
                        if part.startswith("stars:"):
                            # Find matching star range
                            for min_stars, max_stars, range_name in self.star_ranges:
                                if (min_stars is None or f"stars:>={min_stars}" in part) and \
                                   (max_stars is None or f"stars:<{max_stars}" in part):
                                    star_range = range_name
                                    break
                        elif part.startswith("language:"):
                            language = part.replace("language:", "")
                        elif part.startswith("sort:"):
                            sort_option = part.replace("sort:", "")
                        elif part.startswith("created:"):
                            creation_period = part
                        elif part.startswith("topic:"):
                            topic = part.replace("topic:", "")
                    
                    # Create the random query
                    new_query = {
                        "query_text": new_query_text,
                        "star_range": star_range,
                        "language": language,
                        "sort": sort_option,
                        "creation": creation_period,
                        "topic": topic,
                        
                        # Query execution stats
                        "usage_count": 0,
                        "success_count": 0,
                        "error_count": 0,
                        
                        # Performance metrics
                        "total_results": 0,
                        "unique_results": 0,
                        "unique_rate": 0.0,
                        "api_efficiency": 0.5,
                        "quality_score": 0.5,
                        "reward": 0.0,
                        "ucb_score": 150.0,  # Very high initial UCB for random queries
                        
                        # Evolutionary parameters
                        "generation": 1,
                        "parent": "random",
                        "mutation_type": "random",
                        "duplication_rate": 0.3,
                        
                        # Thompson sampling parameters
                        "alpha": 2.0,  # Optimistic prior
                        "beta": 1.0,
                    }
                    
                    # Add to queries and tracking
                    self.queries.append(new_query)
                    existing_queries.add(new_query_text)
                    new_queries.append(new_query)
                    
                    # Skip to next iteration
                    continue
            
            # Regular evolution path - select parent and mutation type
            # Use weighted parents to favor elite performers
            
            # If specific parent queries were provided (for local aggressive mutation),
            # select from those instead of weighted parents
            if parent_queries:
                parent = random.choice(parent_queries)
            
            # Enhanced parent selection based on star success for star-focused mutations
            elif star_focused and len(weighted_parents) > 5:
                # When doing star-focused evolution, prioritize parents with high star counts
                star_weighted_parents = []
                for p in weighted_parents:
                    star_weight = 1.0
                    if "avg_star_count" in p and p["avg_star_count"] > 0:
                        # Higher stars = higher selection probability
                        star_weight = 1.0 + min(5.0, math.log10(p["avg_star_count"] + 1))
                    # Add multiple copies based on star weight
                    copies = max(1, int(star_weight * 2))
                    star_weighted_parents.extend([p] * copies)
                    
                # Use the star-weighted parent selection
                parent = random.choice(star_weighted_parents if star_weighted_parents else weighted_parents)
            else:
                # Standard parent selection
                parent = random.choice(weighted_parents)
            
            # If specific mutation types were provided (for local aggressive mutation),
            # select from those instead of standard selection logic
            if mutation_types:
                mutation_type = random.choice(mutation_types)
            else:
                # Adjust mutation type probabilities based on collection phase
                # cache_manager is required - no fallbacks
                unique_count = cache_manager.get_metrics().get("unique_repositories", 0)
                    
                target_count = 100000  # Default target
                collection_phase = min(1.0, unique_count / max(1, target_count))
                
                # Calculate "desperation level" that increases as more queries are exhausted
                current_time = time.time()
                recent_exhausted = 0
                # Use the instance variables since this is a method of the QueryPool class
                with self.exhausted_queries_lock:
                    for query_text, values in self.exhausted_queries.items():
                        # Handle both formats - older format with just (timestamp, worker_id) and newer format with cooling_multiplier
                        timestamp = values[0] if isinstance(values, tuple) and len(values) > 0 else 0
                        if current_time - timestamp < 600:  # 10 minutes
                            recent_exhausted += 1
                            
                # Calculate desperation level (0.0 to 1.0)
                desperation_level = min(1.0, recent_exhausted / 10.0)
                
                # Use component success rates to intelligently select mutation type
                if self.component_success_rates and all(rates for rates in self.component_success_rates.values()):
                    # We have enough data to make intelligent choices
                    
                    # Map mutation types to component categories
                    component_to_mutation = {
                        "stars": ["stars", "star_bracket_shift", "extreme_stars", "star_band_narrowing"],
                        "language": ["language", "language_combination", "language_star_correlation"],
                        "sort": ["sort", "parameter_inversion"],
                        "creation": ["creation", "temporal_partition", "micro_date_range", "creation_star_correlation"],
                        "topic": ["topic", "topic_explorer", "star_topic_correlation"]
                    }
                    
                    # Find the worst-performing component to prioritize for mutation
                    component_avg_scores = {}
                    for component_type, rates in self.component_success_rates.items():
                        if rates:
                            avg_score = sum(rates.values()) / len(rates)
                            component_avg_scores[component_type] = avg_score
                    
                    # Sort components by performance (worst first)
                    sorted_components = sorted(component_avg_scores.items(), key=lambda x: x[1])
                    
                    # Log the component performance
                    logger.info(f"Component performance ranking (worst to best): {sorted_components}")
                    
                    # Weighted random selection - worse components have higher chance of mutation
                    # With some randomness to avoid getting stuck
                    mutation_weights = []
                    component_types = []
                    
                    for i, (component_type, score) in enumerate(sorted_components):
                        # Inverse weight - lower scores get higher weights
                        # Position penalty - earlier items get higher weights
                        weight = (1.0 - score) * (len(sorted_components) - i)
                        mutation_weights.append(weight)
                        component_types.append(component_type)
                    
                    # Normalize weights
                    total_weight = sum(mutation_weights)
                    if total_weight > 0:
                        mutation_weights = [w / total_weight for w in mutation_weights]
                    
                    # Add randomness factor - 30% chance of completely random choice
                    if random.random() < 0.3:
                        # Choose random component type
                        component_type = random.choice(component_types)
                    else:
                        # Weighted choice based on component performance
                        component_type = random.choices(
                            component_types, 
                            weights=mutation_weights, 
                            k=1
                        )[0]
                    
                    # Get mutation types for the selected component
                    available_mutations = component_to_mutation.get(component_type, mutation_types_to_use)
                    
                    # In late stage or high desperation, include radical mutations
                    if collection_phase > 0.75 or desperation_level > 0.6:
                        available_mutations = available_mutations + radical_mutation_types
                    
                    # Final mutation selection
                    mutation_type = random.choice(available_mutations)
                    
                    logger.info(f"Selected mutation type {mutation_type} to improve {component_type} component")
                else:
                    # Not enough data yet - use original logic
                    # In late stage or with high desperation, favor more radical mutations
                    if collection_phase > 0.75 or desperation_level > 0.6:
                        # Favor radical mutations
                        if random.random() < 0.7:  # 70% chance for radical mutations
                            mutation_type = random.choice(radical_mutation_types)
                        else:
                            mutation_type = random.choice(mutation_types_to_use)
                    else:
                        # Normal mutation type selection
                        mutation_type = random.choice(mutation_types_to_use)
            
            if mutation_type == "language":
                # Change language
                new_language = random.choice([l for l in self.languages if l != parent["language"]])
                
                # Create new query text
                parts = []
                for part in parent["query_text"].split():
                    if part.startswith("language:"):
                        if new_language:
                            parts.append(f"language:{new_language}")
                    else:
                        parts.append(part)
                
                if not new_language and "language:" not in parent["query_text"]:
                    parts.append("")
                    
                new_query_text = " ".join(parts)
                
                # Store language in new query metadata
                new_language_metadata = new_language if new_language else "any"
                
            elif mutation_type == "sort":
                # Change sort order
                new_sort = random.choice([s for s in self.sort_options if f"sort:{s}" not in parent["query_text"]])
                
                # Create new query text
                parts = []
                for part in parent["query_text"].split():
                    if part.startswith("sort:"):
                        parts.append(f"sort:{new_sort}")
                    else:
                        parts.append(part)
                
                if "sort:" not in parent["query_text"]:
                    parts.append(f"sort:{new_sort}")
                    
                new_query_text = " ".join(parts)
                
                # Store sort in new query metadata
                new_sort_metadata = new_sort
                
            elif mutation_type == "stars":
                # Adjust star range
                # Find current star range
                star_parts = []
                non_star_parts = []
                
                for part in parent["query_text"].split():
                    if part.startswith("stars:"):
                        star_parts.append(part)
                    else:
                        non_star_parts.append(part)
                
                # Choose a new adjacent star range
                idx = None
                for i, (min_stars, max_stars, range_name) in enumerate(self.star_ranges):
                    if range_name == parent["star_range"]:
                        idx = i
                        break
                
                if idx is not None:
                    # Choose an adjacent index (or random if at boundary)
                    if idx == 0:
                        new_idx = 1
                    elif idx == len(self.star_ranges) - 1:
                        new_idx = idx - 1
                    else:
                        new_idx = idx + random.choice([-1, 1])
                        
                    # Get new star range
                    min_stars, max_stars, range_name = self.star_ranges[new_idx]
                    
                    # Build new star query
                    new_star_parts = []
                    if min_stars is not None:
                        new_star_parts.append(f"stars:>={min_stars}")
                    if max_stars is not None:
                        new_star_parts.append(f"stars:<{max_stars}")
                    
                    # Combine with non-star parts
                    new_query_text = " ".join(new_star_parts + non_star_parts)
                    
                    # Store star range in new query metadata
                    new_star_range_metadata = range_name
                else:
                    # Couldn't find current star range, keep the query the same
                    new_query_text = parent["query_text"]
                    new_star_range_metadata = parent["star_range"]
                    
            elif mutation_type == "creation":
                # Change creation date range
                # Find current creation range
                creation_part = None
                other_parts = []
                
                for part in parent["query_text"].split():
                    if part.startswith("created:"):
                        creation_part = part
                    else:
                        other_parts.append(part)
                
                # Choose a new creation range
                new_creation = random.choice([
                    c for c in self.creation_periods 
                    if c and c != creation_part
                ])
                
                if creation_part:
                    # Replace existing creation part
                    new_query_text = " ".join([p for p in parent["query_text"].split() if not p.startswith("created:")])
                    if new_creation:
                        new_query_text = f"{new_query_text} {new_creation}"
                else:
                    # Add new creation part
                    if new_creation:
                        new_query_text = f"{parent['query_text']} {new_creation}"
                    else:
                        new_query_text = parent["query_text"]
                
                # Store creation range in new query metadata
                new_creation_metadata = new_creation if new_creation else "any"
                
            elif mutation_type == "topic":
                # Add or change topic
                # Find current topic
                topic_part = None
                other_parts = []
                
                for part in parent["query_text"].split():
                    if part.startswith("topic:"):
                        topic_part = part
                    else:
                        other_parts.append(part)
                
                # Choose a new topic
                new_topic = random.choice([t for t in self.topics if t and t != topic_part])
                
                if topic_part:
                    # Replace existing topic part
                    new_query_text = " ".join([p for p in parent["query_text"].split() if not p.startswith("topic:")])
                    if new_topic:
                        new_query_text = f"{new_query_text} {new_topic}"
                else:
                    # Add new topic part
                    if new_topic:
                        new_query_text = f"{parent['query_text']} {new_topic}"
                    else:
                        new_query_text = parent["query_text"]
                
                # Store topic in new query metadata
                new_topic_metadata = new_topic.replace("topic:", "") if new_topic else ""
                
            elif mutation_type == "extreme_stars":
                # Make dramatic jumps in star ranges - opposite of current range
                # Find current star range
                idx = None
                for i, (min_stars, max_stars, range_name) in enumerate(self.star_ranges):
                    if range_name == parent["star_range"]:
                        idx = i
                        break
                
                if idx is not None:
                    # Choose opposite range (high->low, low->high)
                    if idx < len(self.star_ranges) // 2:  # Currently low stars
                        new_idx = random.randint(len(self.star_ranges) * 3 // 4, len(self.star_ranges) - 1)
                    else:  # Currently high stars
                        new_idx = random.randint(0, len(self.star_ranges) // 4)
                    
                    # Get new star range
                    min_stars, max_stars, range_name = self.star_ranges[new_idx]
                    
                    # Extract non-star parts from the query
                    non_star_parts = [part for part in parent["query_text"].split() if not part.startswith("stars:")]
                    
                    # Build new star query parts
                    new_star_parts = []
                    if min_stars is not None:
                        new_star_parts.append(f"stars:>={min_stars}")
                    if max_stars is not None:
                        new_star_parts.append(f"stars:<{max_stars}")
                    
                    # Combine to form the new query
                    new_query_text = " ".join(new_star_parts + non_star_parts)
                    
                    # Store star range in new query metadata
                    new_star_range_metadata = range_name
                else:
                    # Fallback to default behavior
                    new_query_text = parent["query_text"]
                    new_star_range_metadata = parent["star_range"]
            
            elif mutation_type == "compound":
                # Change 2-3 parameters at once (stars + language + topic/sort)
                # Select 2-3 parameters to change
                parameters = ["stars", "language", "sort", "creation", "topic"]
                num_to_change = random.randint(2, 3)
                params_to_change = random.sample(parameters, k=num_to_change)
                
                # Copy the original query parts
                query_parts = parent["query_text"].split()
                
                # Handle each parameter
                new_star_range_metadata = parent["star_range"]
                new_language_metadata = parent["language"]
                new_sort_metadata = parent["sort"]
                new_creation_metadata = parent["creation"]
                new_topic_metadata = parent.get("topic", "")
                
                # Stars change
                if "stars" in params_to_change:
                    # Remove existing star parts
                    query_parts = [part for part in query_parts if not part.startswith("stars:")]
                    
                    # Add new star range
                    idx = None
                    for i, (min_stars, max_stars, range_name) in enumerate(self.star_ranges):
                        if range_name == parent["star_range"]:
                            idx = i
                            break
                    
                    # Choose a different range
                    if idx is not None:
                        new_idx = (idx + random.randint(1, len(self.star_ranges) - 1)) % len(self.star_ranges)
                        min_stars, max_stars, range_name = self.star_ranges[new_idx]
                        
                        if min_stars is not None:
                            query_parts.append(f"stars:>={min_stars}")
                        if max_stars is not None:
                            query_parts.append(f"stars:<{max_stars}")
                        
                        new_star_range_metadata = range_name
                
                # Language change
                if "language" in params_to_change:
                    # Remove existing language part
                    query_parts = [part for part in query_parts if not part.startswith("language:")]
                    
                    # Add new language
                    new_language = random.choice([l for l in self.languages if l != parent["language"]])
                    if new_language:
                        query_parts.append(f"language:{new_language}")
                        new_language_metadata = new_language
                    else:
                        new_language_metadata = "any"
                
                # Sort change
                if "sort" in params_to_change:
                    # Remove existing sort part
                    query_parts = [part for part in query_parts if not part.startswith("sort:")]
                    
                    # Add new sort
                    new_sort = random.choice([s for s in self.sort_options if s != parent["sort"]])
                    query_parts.append(f"sort:{new_sort}")
                    new_sort_metadata = new_sort
                
                # Creation date change
                if "creation" in params_to_change:
                    # Remove existing creation part
                    query_parts = [part for part in query_parts if not part.startswith("created:")]
                    
                    # Add new creation part
                    new_creation = random.choice([c for c in self.creation_periods if c and c != parent["creation"]])
                    if new_creation:
                        query_parts.append(new_creation)
                        new_creation_metadata = new_creation
                    else:
                        new_creation_metadata = "any"
                
                # Topic change
                if "topic" in params_to_change:
                    # Remove existing topic part
                    query_parts = [part for part in query_parts if not part.startswith("topic:")]
                    
                    # Add new topic
                    if self.topics:
                        new_topic = random.choice([t for t in self.topics if t and t != parent.get("topic", "")])
                        if new_topic:
                            query_parts.append(f"topic:{new_topic}")
                            new_topic_metadata = new_topic
                        else:
                            new_topic_metadata = ""
                    else:
                        new_topic_metadata = ""
                
                # Construct the new query
                new_query_text = " ".join(query_parts)
            
            elif mutation_type == "temporal_partition":
                # Split date ranges into much finer increments
                # Extract current creation part and other parts
                creation_part = None
                other_parts = []
                
                for part in parent["query_text"].split():
                    if part.startswith("created:"):
                        creation_part = part
                    else:
                        other_parts.append(part)
                
                # Default to last year if no creation part
                if not creation_part:
                    last_year = datetime.now().year - 1
                    start_date = f"{last_year}-01-01"
                    end_date = f"{last_year}-12-31"
                else:
                    # Parse the existing date range
                    date_range = creation_part.replace("created:", "")
                    if ".." in date_range:
                        start_date, end_date = date_range.split("..")
                    else:
                        # Default to a reasonable range if no range specified
                        current_year = datetime.now().year
                        start_date = f"{current_year-2}-01-01"
                        end_date = f"{current_year}-12-31"
                
                # Convert to datetime objects
                try:
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                except ValueError:
                    # If date parsing fails, use default
                    current_year = datetime.now().year
                    start_dt = datetime(current_year-1, 1, 1)
                    end_dt = datetime(current_year, 12, 31)
                
                # Calculate total days and divide into chunks
                total_days = (end_dt - start_dt).days
                if total_days <= 14:
                    # Already a small range, no need to partition further
                    new_query_text = parent["query_text"]
                    new_creation_metadata = parent["creation"]
                else:
                    # Divide into smaller chunks
                    chunk_size = random.randint(7, 60)  # Between 1 week and 2 months
                    
                    # Choose a random chunk
                    max_offset = total_days - chunk_size
                    if max_offset <= 0:
                        offset = 0
                    else:
                        offset = random.randint(0, max_offset)
                    
                    # Calculate new start and end dates
                    new_start_dt = start_dt + timedelta(days=offset)
                    new_end_dt = new_start_dt + timedelta(days=chunk_size)
                    
                    # Format dates
                    new_start_date = new_start_dt.strftime("%Y-%m-%d")
                    new_end_date = new_end_dt.strftime("%Y-%m-%d")
                    
                    # Create new creation part
                    new_creation_part = f"created:{new_start_date}..{new_end_date}"
                    
                    # Combine with other parts
                    new_query_text = " ".join(other_parts + [new_creation_part])
                    new_creation_metadata = new_creation_part
            
            elif mutation_type == "parameter_inversion":
                # Invert sort order and flip other parameters
                query_parts = []
                
                # Invert sort order
                sort_inverted = False
                for part in parent["query_text"].split():
                    if part.startswith("sort:"):
                        current_sort = part.replace("sort:", "")
                        # Invert order if possible
                        if current_sort == "stars":
                            query_parts.append("sort:stars-asc")
                            new_sort = "stars-asc"
                        elif current_sort == "stars-asc":
                            query_parts.append("sort:stars")
                            new_sort = "stars"
                        elif current_sort == "updated":
                            query_parts.append("sort:updated-asc")
                            new_sort = "updated-asc"
                        elif current_sort == "updated-asc":
                            query_parts.append("sort:updated")
                            new_sort = "updated"
                        else:
                            # Use random different sort
                            new_sort = random.choice([s for s in self.sort_options if s != current_sort])
                            query_parts.append(f"sort:{new_sort}")
                        sort_inverted = True
                    else:
                        query_parts.append(part)
                
                # If no sort parameter found, add one
                if not sort_inverted:
                    new_sort = random.choice(self.sort_options)
                    query_parts.append(f"sort:{new_sort}")
                
                # Construct the new query
                new_query_text = " ".join(query_parts)
                
                # Set metadata
                new_star_range_metadata = parent["star_range"]
                new_language_metadata = parent["language"]
                new_sort_metadata = new_sort
                new_creation_metadata = parent["creation"]
                new_topic_metadata = parent.get("topic", "")
            
            elif mutation_type == "micro_date_range":
                # Split date ranges into small windows (days or weeks)
                # Remove existing creation part
                other_parts = [part for part in parent["query_text"].split() if not part.startswith("created:")]
                
                # Generate a micro date range
                # Find a busy time period (like early month, hackathon season, etc.)
                busy_periods = [
                    # Hacktoberfest season
                    {"month": 10, "day_start": 1, "day_end": 31},
                    # GitHub Universe typical dates
                    {"month": 11, "day_start": 7, "day_end": 14},
                    # Google Summer of Code announcement period
                    {"month": 2, "day_start": 15, "day_end": 28},
                    # New year activity
                    {"month": 1, "day_start": 5, "day_end": 15},
                    # Back to school/work after summer
                    {"month": 9, "day_start": 5, "day_end": 15}
                ]
                
                # Choose a random year from the last 5 years
                current_year = datetime.now().year
                year = random.randint(current_year - 5, current_year - 1)
                
                # Select a random busy period
                period = random.choice(busy_periods)
                
                # Create a date range within the busy period
                range_start = datetime(year, period["month"], period["day_start"])
                max_end_day = min(period["day_end"], 28)  # Avoid month end issues
                range_end = datetime(year, period["month"], max_end_day)
                
                # Choose a window of 1-7 days within this period
                window_size = random.randint(1, min(7, (range_end - range_start).days))
                if (range_end - range_start).days > window_size:
                    start_offset = random.randint(0, (range_end - range_start).days - window_size)
                else:
                    start_offset = 0
                
                window_start = range_start + timedelta(days=start_offset)
                window_end = window_start + timedelta(days=window_size)
                
                # Format the date range
                date_range = f"created:{window_start.strftime('%Y-%m-%d')}..{window_end.strftime('%Y-%m-%d')}"
                
                # Combine with other parts
                new_query_text = " ".join(other_parts + [date_range])
                new_creation_metadata = date_range
                
                # Pass through other metadata
                new_star_range_metadata = parent["star_range"]
                new_language_metadata = parent["language"]
                new_sort_metadata = parent["sort"]
                new_topic_metadata = parent.get("topic", "")
            
            elif mutation_type == "language_combination":
                # Combine multiple languages (for future OR support in GitHub)
                # Remove existing language part
                other_parts = [part for part in parent["query_text"].split() if not part.startswith("language:")]
                
                # Select 2-3 languages
                if len(self.languages) >= 2:
                    num_languages = random.randint(2, min(3, len(self.languages)))
                    languages = random.sample(self.languages, k=num_languages)
                    
                    # GitHub API doesn't support OR for language, so we currently can only use one
                    # This prepares for future OR support or special handling
                    language_part = f"language:{languages[0]}"
                    new_language_metadata = languages[0]
                    
                    # Combine with other parts
                    new_query_text = " ".join(other_parts + [language_part])
                else:
                    # Fallback if not enough languages
                    new_query_text = parent["query_text"]
                    new_language_metadata = parent["language"]
                
                # Pass through other metadata
                new_star_range_metadata = parent["star_range"]
                new_sort_metadata = parent["sort"]
                new_creation_metadata = parent["creation"]
                new_topic_metadata = parent.get("topic", "")
            
            elif mutation_type == "star_bracket_shift":
                # Strategically shift star ranges up or down based on performance data
                # This creates targeted queries exploring neighboring star ranges
                
                # Find current star components in the query
                star_components = [part for part in parent["query_text"].split() if part.startswith("stars:")]
                non_star_components = [part for part in parent["query_text"].split() if not part.startswith("stars:")]
                
                # Get collection-wide star distribution
                star_distribution = self.context_features.get("star_distribution", {})
                
                # Determine if we should shift up or down based on collection needs
                shift_direction = "up"  # Default to shifting up for higher stars
                
                # If the parent query has been successful with high stars, try shifting up
                # Otherwise, consider the distribution in our collection
                if "avg_star_count" in parent and parent["avg_star_count"] > 10000:
                    # This query finds high-star repos - shift up to find even higher ones
                    shift_direction = "up"
                elif star_distribution:
                    # Analyze collection needs
                    high_star_percent = 0
                    for bucket in ["100K+", "50K-100K", "10K-50K"]:
                        high_star_percent += star_distribution.get(bucket, 0)
                    
                    # If less than 10% of our collection has high stars, shift up to find more
                    total_repos = sum(star_distribution.values()) or 1
                    if high_star_percent / total_repos < 0.1:
                        shift_direction = "up"
                    else:
                        shift_direction = random.choice(["up", "down"])  # Random if we already have many high stars
                
                # Parse current star range and create new one
                min_stars = None
                max_stars = None
                for part in star_components:
                    if ">" in part:
                        # Extract min stars
                        try:
                            min_stars = int(part.replace("stars:>", "").replace("stars:>=", ""))
                        except ValueError:
                            min_stars = 1000  # Default fallback
                    if "<" in part:
                        # Extract max stars
                        try:
                            max_stars = int(part.replace("stars:<", "").replace("stars:<=", ""))
                        except ValueError:
                            max_stars = 50000  # Default fallback
                
                # If no star range found, create a default one
                if min_stars is None and max_stars is None:
                    min_stars = 1000
                    max_stars = 50000
                
                # Apply shift
                new_star_parts = []
                if shift_direction == "up":
                    # Shift minimum and maximum up by 50-100%
                    if min_stars is not None:
                        shift_factor = random.uniform(1.5, 2.0)
                        new_min = int(min_stars * shift_factor)
                        new_star_parts.append(f"stars:>={new_min}")
                    
                    if max_stars is not None:
                        shift_factor = random.uniform(1.5, 2.0)
                        new_max = int(max_stars * shift_factor)
                        new_star_parts.append(f"stars:<{new_max}")
                else:  # shift down
                    # Shift minimum and maximum down by 30-50%
                    if min_stars is not None:
                        shift_factor = random.uniform(0.5, 0.7)
                        new_min = max(10, int(min_stars * shift_factor))
                        new_star_parts.append(f"stars:>={new_min}")
                    
                    if max_stars is not None:
                        shift_factor = random.uniform(0.5, 0.7)
                        new_max = int(max_stars * shift_factor)
                        new_star_parts.append(f"stars:<{new_max}")
                
                # Combine to form new query
                new_query_text = " ".join(new_star_parts + non_star_components)
                
                # Determine star range for metadata
                # Find which of our preset ranges this falls into
                new_star_range_metadata = "custom"
                for min_s, max_s, range_name in self.star_ranges:
                    min_match = (min_s is None) or (min_stars is not None and min_stars >= min_s)
                    max_match = (max_s is None) or (max_stars is not None and max_stars < max_s)
                    if min_match and max_match:
                        new_star_range_metadata = range_name
                        break
                
                # Keep other metadata the same
                new_language_metadata = parent["language"]
                new_sort_metadata = parent["sort"]
                new_creation_metadata = parent["creation"]
                new_topic_metadata = parent.get("topic", "")
                
            elif mutation_type == "viral_repo_finder":
                # Target repositories with rapid star growth rates (trending repos)
                # Focus on repos that are gaining popularity quickly
                
                # Start with parts of the parent query that aren't related to stars/sorting
                query_parts = [part for part in parent["query_text"].split() 
                             if not part.startswith("stars:") and not part.startswith("sort:")]
                
                # Create a sort by recently starred
                query_parts.append("sort:stars")
                
                # Add a recent timeframe to focus on trending repos
                # Either by creation date or push date
                timeframe_type = random.choice(["created", "pushed"])
                
                # Generate timeframe with recent bias
                current_year = datetime.now().year
                current_month = datetime.now().month
                
                # Favor very recent timeframes (last few months)
                if random.random() < 0.7:  # 70% recent focus
                    # Last 1-6 months
                    months_ago = random.randint(1, 6)
                    if current_month > months_ago:
                        start_date = f"{current_year}-{current_month-months_ago:02d}-01"
                    else:
                        start_date = f"{current_year-1}-{12-(months_ago-current_month):02d}-01"
                    
                    # Add timeframe constraint
                    query_parts.append(f"{timeframe_type}:>{start_date}")
                else:
                    # Last 1-2 years but not too recent
                    years_ago = random.randint(1, 2)
                    months_span = random.randint(3, 6)
                    
                    year = current_year - years_ago
                    month = random.randint(1, 12 - months_span)
                    end_month = month + months_span
                    
                    start_date = f"{year}-{month:02d}-01"
                    end_date = f"{year}-{end_month:02d}-30"
                    
                    # Add timeframe constraint
                    query_parts.append(f"{timeframe_type}:{start_date}..{end_date}")
                
                # Add star range with modestly high thresholds
                # The idea is to find repos that have gotten popular quickly
                min_stars = random.randint(500, 5000)
                max_stars = min_stars * (2 + random.randint(1, 3))
                query_parts.append(f"stars:>={min_stars}")
                query_parts.append(f"stars:<{max_stars}")
                
                # Create full query
                new_query_text = " ".join(query_parts)
                
                # Set metadata
                new_star_range_metadata = "custom"  # This won't match standard ranges
                new_language_metadata = parent["language"]
                new_sort_metadata = "stars"
                new_creation_metadata = "custom"
                new_topic_metadata = parent.get("topic", "")
                
            elif mutation_type == "star_topic_correlation":
                # Combine high-performing star ranges with successful topics
                # Uses data about which topics tend to have higher star counts
                
                # Start with a clean slate
                query_parts = []
                
                # Select a promising star range
                # Either from the parent or from global high-performing queries
                if random.random() < 0.5 and "avg_star_count" in parent and parent["avg_star_count"] > 1000:
                    # Use parent's star range since it's good
                    star_parts = [part for part in parent["query_text"].split() if part.startswith("stars:")]
                    if star_parts:
                        query_parts.extend(star_parts)
                    else:
                        # Fallback if no star parts found
                        min_stars = max(1000, int(parent["avg_star_count"] * 0.5))
                        max_stars = int(parent["avg_star_count"] * 2.0)
                        query_parts.append(f"stars:>={min_stars}")
                        query_parts.append(f"stars:<{max_stars}")
                else:
                    # Use a high-star range more likely to yield valuable repositories
                    high_star_ranges = [
                        ("stars:>=10000 stars:<50000", "10K-50K"),
                        ("stars:>=5000 stars:<10000", "5K-10K"),
                        ("stars:>=20000 stars:<100000", "20K-100K"),
                        ("stars:>=50000", "50K+")
                    ]
                    star_query, new_star_range_metadata = random.choice(high_star_ranges)
                    query_parts.append(star_query)
                
                # Select a promising topic
                # Either from the parent or from trending topics
                if random.random() < 0.3 and parent.get("topic") and parent.get("topic") != "":
                    # Use parent's topic since it might correlate well with stars
                    query_parts.append(f"topic:{parent['topic']}")
                    new_topic_metadata = parent["topic"]
                else:
                    # Use a topic more likely to have high-star repos
                    star_rich_topics = [
                        "machine-learning", "deep-learning", "data-science", "ai",
                        "javascript", "react", "vue", "nextjs", "typescript",
                        "rust", "golang", "python", "kubernetes", "devops",
                        "blockchain", "graphql", "chatgpt", "llm", "transformers"
                    ]
                    new_topic = random.choice(star_rich_topics)
                    query_parts.append(f"topic:{new_topic}")
                    new_topic_metadata = new_topic
                
                # Select a language if parent has one
                if parent["language"] and parent["language"] != "any":
                    query_parts.append(f"language:{parent['language']}")
                    new_language_metadata = parent["language"]
                else:
                    # Optionally add a language known for high stars
                    if random.random() < 0.7:
                        star_rich_languages = ["javascript", "typescript", "python", "rust", "go"]
                        new_language = random.choice(star_rich_languages)
                        query_parts.append(f"language:{new_language}")
                        new_language_metadata = new_language
                    else:
                        new_language_metadata = "any"
                
                # Use sort by stars
                query_parts.append("sort:stars")
                new_sort_metadata = "stars"
                
                # Random recent timeframe
                if random.random() < 0.3:
                    years = random.randint(1, 3)
                    query_parts.append(f"created:>{datetime.now().year - years}")
                    new_creation_metadata = f"created:>{datetime.now().year - years}"
                else:
                    new_creation_metadata = parent["creation"]
                
                # Create the final query
                new_query_text = " ".join(query_parts)
                
            elif mutation_type == "star_threshold_exploration":
                # Create very specific star thresholds based on successful queries
                # Fine-tunes the star ranges to target specific thresholds where valuable repos exist
                
                # Get all parts except star parts
                non_star_parts = [part for part in parent["query_text"].split() if not part.startswith("stars:")]
                
                # Choose a specific star threshold strategy
                threshold_strategy = random.choice(["precise_window", "odd_thresholds", "power_threshold"])
                
                if threshold_strategy == "precise_window":
                    # Create a narrow window around a specific threshold
                    base = random.choice([1000, 2500, 5000, 10000, 15000, 25000, 30000, 50000])
                    window = base * random.uniform(0.1, 0.3)  # 10-30% window
                    min_stars = int(base - window/2)
                    max_stars = int(base + window/2)
                    star_parts = [f"stars:>={min_stars}", f"stars:<{max_stars}"]
                    
                elif threshold_strategy == "odd_thresholds":
                    # Use non-standard thresholds that might reveal unique repositories
                    # People often use round numbers, so search between those thresholds
                    options = [
                        (1001, 1999),   # Between 1K and 2K
                        (2001, 2999),   # Between 2K and 3K
                        (5001, 9999),   # Between 5K and 10K
                        (10001, 19999), # Between 10K and 20K
                        (20001, 29999), # Between 20K and 30K
                        (50001, 99999)  # Between 50K and 100K
                    ]
                    min_stars, max_stars = random.choice(options)
                    star_parts = [f"stars:>={min_stars}", f"stars:<{max_stars}"]
                    
                else:  # power_threshold
                    # Use powers of a number as thresholds
                    base = random.choice([2, 3, 5])
                    power_min = random.randint(8, 12)  # 2^10 = 1024, 3^8 = 6561, 5^8 = 390625
                    power_max = power_min + random.randint(1, 2)
                    min_stars = base ** power_min
                    max_stars = base ** power_max
                    star_parts = [f"stars:>={min_stars}", f"stars:<{max_stars}"]
                
                # Combine to create new query
                new_query_text = " ".join(star_parts + non_star_parts)
                
                # Set metadata
                new_star_range_metadata = "custom"  # Custom ranges won't match standard ones
                new_language_metadata = parent["language"]
                new_sort_metadata = parent["sort"]
                new_creation_metadata = parent["creation"]
                new_topic_metadata = parent.get("topic", "")
                
            elif mutation_type == "star_band_narrowing":
                # Create very narrow star range queries (e.g., stars:5000..5100)
                # Targets specific star bands to find niche high-quality repositories
                
                # Get non-star parts from parent
                non_star_parts = [part for part in parent["query_text"].split() if not part.startswith("stars:")]
                
                # Choose a promising star threshold as center point
                base_thresholds = [1000, 2500, 5000, 7500, 10000, 15000, 20000, 25000, 30000, 50000, 75000, 100000]
                
                # If the parent has been successful with high stars, use that as a guide
                if "avg_star_count" in parent and parent["avg_star_count"] > 1000:
                    # Find the closest base threshold to parent's average
                    closest_base = min(base_thresholds, key=lambda x: abs(x - parent["avg_star_count"]))
                    base_threshold = closest_base
                else:
                    # Otherwise select randomly with bias toward higher values
                    if random.random() < 0.7:
                        # 70% chance to pick from the higher half
                        base_threshold = random.choice(base_thresholds[len(base_thresholds)//2:])
                    else:
                        base_threshold = random.choice(base_thresholds)
                
                # Create a very narrow band (1-10% of base value)
                band_width = base_threshold * random.uniform(0.01, 0.1)
                
                # Create exact boundaries
                min_stars = int(base_threshold - band_width/2)
                max_stars = int(base_threshold + band_width/2)
                
                # Ensure minimum of 10 stars difference
                if max_stars - min_stars < 10:
                    max_stars = min_stars + 10
                    
                # Create star parts
                star_parts = [f"stars:>={min_stars}", f"stars:<{max_stars}"]
                
                # Combine to create new query
                new_query_text = " ".join(star_parts + non_star_parts)
                
                # Set metadata
                new_star_range_metadata = "custom"
                new_language_metadata = parent["language"]
                new_sort_metadata = parent["sort"]
                new_creation_metadata = parent["creation"]
                new_topic_metadata = parent.get("topic", "")
                
            elif mutation_type == "language_star_correlation":
                # Target languages with unusually high star counts
                # Finds correlations between programming languages and star counts
                
                # Start with a clean slate
                query_parts = []
                
                # Choose a language likely to have high stars
                high_star_languages = {
                    "javascript": (5000, 50000),  # min and max stars
                    "typescript": (3000, 40000),
                    "python": (3000, 40000),
                    "rust": (2000, 30000),
                    "go": (2000, 30000),
                    "java": (3000, 30000),
                    "kotlin": (1000, 20000),
                    "swift": (1000, 20000),
                    "csharp": (1000, 25000),
                    "cpp": (2000, 30000),
                    "ruby": (2000, 25000),
                    "php": (1000, 20000)
                }
                
                # Determine language - either use parent's or select a promising one
                if parent["language"] != "any" and parent["language"] in high_star_languages:
                    selected_language = parent["language"]
                else:
                    # Choose a language based on probability weighted by expected star counts
                    languages = list(high_star_languages.keys())
                    weights = [max_stars - min_stars for min_stars, max_stars in high_star_languages.values()]
                    selected_language = random.choices(languages, weights=weights, k=1)[0]
                
                # Add language to query
                query_parts.append(f"language:{selected_language}")
                new_language_metadata = selected_language
                
                # Get star range for this language
                min_stars, max_stars = high_star_languages.get(selected_language, (1000, 10000))
                
                # Adjust star range slightly for randomness
                adjustment = random.uniform(0.8, 1.2)
                min_stars = int(min_stars * adjustment)
                max_stars = int(max_stars * adjustment)
                
                # Add star range to query
                query_parts.append(f"stars:>={min_stars}")
                query_parts.append(f"stars:<{max_stars}")
                
                # Add sorting
                query_parts.append("sort:stars")
                new_sort_metadata = "stars"
                
                # Set metadata
                new_star_range_metadata = "custom"
                new_creation_metadata = parent["creation"]
                new_topic_metadata = parent.get("topic", "")
                
                # Create the final query
                new_query_text = " ".join(query_parts)
                
            elif mutation_type == "creation_star_correlation":
                # Find relationships between creation dates and star counts
                # Identifies time periods when high-star repositories were created
                
                # Start with a clean slate
                query_parts = []
                
                # Define time periods when high-star repos were more likely to be created
                promising_periods = [
                    # Early GitHub period - established projects with long histories
                    ("created:2008-01-01..2010-12-31", "stars:>=5000"),
                    
                    # Mid-period - established projects that aren't as old
                    ("created:2011-01-01..2015-12-31", "stars:>=7500"),
                    
                    # Modern frameworks and libraries
                    ("created:2016-01-01..2019-12-31", "stars:>=10000"),
                    
                    # Recent high-fliers (pandemic period)
                    ("created:2020-01-01..2021-12-31", "stars:>=5000"),
                    
                    # Very recent trending projects
                    ("created:2022-01-01..", "stars:>=2000")
                ]
                
                # Select a promising time period
                creation_query, star_query = random.choice(promising_periods)
                query_parts.append(creation_query)
                query_parts.append(star_query)
                
                # Add language if parent has one
                if parent["language"] != "any":
                    query_parts.append(f"language:{parent['language']}")
                    new_language_metadata = parent["language"]
                else:
                    # 50% chance to add a random language
                    if random.random() < 0.5:
                        languages = ["javascript", "typescript", "python", "rust", "go", "java"]
                        selected_language = random.choice(languages)
                        query_parts.append(f"language:{selected_language}")
                        new_language_metadata = selected_language
                    else:
                        new_language_metadata = "any"
                
                # Add sorting
                query_parts.append("sort:stars")
                new_sort_metadata = "stars"
                
                # Potentially add a topic
                if parent.get("topic") and random.random() < 0.5:
                    query_parts.append(f"topic:{parent['topic']}")
                    new_topic_metadata = parent["topic"]
                else:
                    new_topic_metadata = ""
                
                # Set metadata
                new_star_range_metadata = "custom"
                new_creation_metadata = creation_query
                
                # Create the final query
                new_query_text = " ".join(query_parts)
                
            elif mutation_type == "niche_finder":
                # Target repositories with specific file types or structures
                # and repositories with recent updates but few stars
                # Keep most of the existing query
                query_parts = parent["query_text"].split()
                
                # Choose a niche filter
                niche_filters = [
                    "filename:Dockerfile",
                    "filename:docker-compose.yml",
                    "filename:requirements.txt",
                    "filename:package.json",
                    "filename:Cargo.toml",
                    "filename:go.mod",
                    "filename:build.gradle",
                    "filename:pom.xml",
                    "path:/.github/workflows",
                    "path:/kubernetes",
                    "pushed:>2023-01-01",
                    "fork:false",
                    "good-first-issue:>0"
                ]
                
                # Add a niche filter
                niche_filter = random.choice(niche_filters)
                
                # Adjust star range to focus on lower stars for niche repos
                # Replace any existing star filters with more specific ones
                query_parts = [part for part in query_parts if not part.startswith("stars:")]
                
                # Add small stars range - better for finding niche repos
                min_stars = random.randint(3, 50)
                max_stars = random.randint(min_stars+50, min_stars+500)
                star_parts = [f"stars:>={min_stars}", f"stars:<{max_stars}"]
                
                # Find the matching star range
                star_range_name = "custom"
                for min_s, max_s, range_name in self.star_ranges:
                    if (min_s is None or min_s <= min_stars) and (max_s is None or max_s >= max_stars):
                        star_range_name = range_name
                        break
                
                # Combine everything
                new_query_text = " ".join(query_parts + star_parts + [niche_filter])
                
                # Update metadata
                new_star_range_metadata = star_range_name
                new_language_metadata = parent["language"]
                new_sort_metadata = parent["sort"]
                new_creation_metadata = parent["creation"]
                new_topic_metadata = parent.get("topic", "")
            
            elif mutation_type == "topic_explorer":
                # Generate queries with multiple topic combinations
                # Include emerging topics from GitHub trends
                
                # Remove existing topic parts
                other_parts = [part for part in parent["query_text"].split() if not part.startswith("topic:")]
                
                # Add trending topics - these should be periodically updated in the actual implementation
                trending_topics = [
                    "machine-learning", "data-science", "deep-learning", "ai",
                    "web-development", "javascript-framework", "react", "vue", "svelte", "nextjs",
                    "blockchain", "web3", "ethereum", "solidity", "nft",
                    "kubernetes", "devops", "cloud-native", "microservices", "serverless",
                    "rust", "typescript", "go", "flutter", "kotlin", "swift"
                ]
                
                # Combine available topics with trending topics
                all_topics = list(set(self.topics + trending_topics))
                
                # Select 2-3 topics
                if len(all_topics) >= 2:
                    num_topics = random.randint(2, 3)
                    selected_topics = random.sample(all_topics, k=min(num_topics, len(all_topics)))
                    
                    # Create topic parts
                    topic_parts = [f"topic:{topic}" for topic in selected_topics if topic]
                    
                    # Combine with other parts
                    new_query_text = " ".join(other_parts + topic_parts)
                    
                    # Use first topic for metadata (limitation of current structure)
                    new_topic_metadata = selected_topics[0] if selected_topics else ""
                else:
                    # Fallback if not enough topics
                    new_query_text = parent["query_text"]
                    new_topic_metadata = parent.get("topic", "")
                
                # Pass through other metadata
                new_star_range_metadata = parent["star_range"]
                new_language_metadata = parent["language"]
                new_sort_metadata = parent["sort"]
                new_creation_metadata = parent["creation"]
            
            elif mutation_type == "crossover":
                # Multi-parent recombination: select 3-5 high-performing parents
                # Use weighted selection based on parent performance
                
                # Select multiple parents for recombination (excluding the current parent)
                available_parents = [p for p in weighted_parents if p != parent]
                if len(available_parents) < 2:
                    # Not enough parents for recombination, fall back to basic crossover
                    other_parent = random.choice(weighted_parents)
                    multi_parents = [parent, other_parent]
                else:
                    # Select 2-4 additional parents (for a total of 3-5 including the current one)
                    num_additional = min(random.randint(2, 4), len(available_parents))
                    additional_parents = random.sample(available_parents, k=num_additional)
                    multi_parents = [parent] + additional_parents
                
                # For multi-parent recombination, select components from each parent 
                # based on their success rate
                components = ["stars", "language", "sort", "creation", "topic"]
                
                # Extract components from all parents
                parent_components = []
                for p in multi_parents:
                    # Initialize component dict for this parent
                    p_components = {
                        "stars": [],
                        "language": None,
                        "sort": None,
                        "creation": None, 
                        "topic": None,
                        "parent": p,
                        "reward_ratio": p["reward"] / max(1, p["usage_count"])
                    }
                    
                    # Extract components from this parent's query
                    for part in p["query_text"].split():
                        if part.startswith("stars:"):
                            p_components["stars"].append(part)
                        elif part.startswith("language:"):
                            p_components["language"] = part
                        elif part.startswith("sort:"):
                            p_components["sort"] = part
                        elif part.startswith("created:"):
                            p_components["creation"] = part
                        elif part.startswith("topic:"):
                            p_components["topic"] = part
                    
                    parent_components.append(p_components)
                
                # Select components for the child based on weighted selection
                # with preference for components from higher-performing parents
                star_parts = []
                lang_part = None
                sort_part = None
                creation_part = None
                topic_part = None
                
                # Weight parents by their performance
                parent_weights = [pc["reward_ratio"] for pc in parent_components]
                
                # For each component type, select from a parent based on weights
                for component in components:
                    # Select a parent for this component
                    selected_parent_idx = random.choices(
                        range(len(parent_components)),
                        weights=parent_weights,
                        k=1
                    )[0]
                    
                    selected_parent = parent_components[selected_parent_idx]
                    
                    # Use this parent's component
                    if component == "stars":
                        star_parts = selected_parent["stars"]
                    elif component == "language":
                        lang_part = selected_parent["language"]
                    elif component == "sort":
                        sort_part = selected_parent["sort"]
                    elif component == "creation":
                        creation_part = selected_parent["creation"]
                    elif component == "topic":
                        topic_part = selected_parent["topic"]
                
                # For metadata tracking, default to the first parent's values
                # and override with selected components
                new_star_range_metadata = parent["star_range"]
                new_language_metadata = parent["language"]
                new_sort_metadata = parent["sort"]
                new_creation_metadata = parent["creation"]
                new_topic_metadata = parent.get("topic", "")
                
                # Combine parts for the new query
                new_parts = []
                new_parts.extend(star_parts)
                if lang_part:
                    new_parts.append(lang_part)
                if sort_part:
                    new_parts.append(sort_part)
                if creation_part:
                    new_parts.append(creation_part)
                if topic_part:
                    new_parts.append(topic_part)
                
                new_query_text = " ".join(new_parts)
                
                # Extract metadata from parts
                if "stars" in selected_components:
                    new_star_range_metadata = other_parent["star_range"]
                else:
                    new_star_range_metadata = parent["star_range"]
                    
                if "language" in selected_components:
                    new_language_metadata = other_parent["language"]
                else:
                    new_language_metadata = parent["language"]
                    
                if "sort" in selected_components:
                    new_sort_metadata = other_parent["sort"]
                else:
                    new_sort_metadata = parent["sort"]
                    
                if "creation" in selected_components:
                    new_creation_metadata = other_parent["creation"]
                else:
                    new_creation_metadata = parent["creation"]
                    
                if "topic" in selected_components:
                    new_topic_metadata = other_parent.get("topic", "")
                else:
                    new_topic_metadata = parent.get("topic", "")
            
            # Create the new query if it doesn't already exist
            if new_query_text and new_query_text not in existing_queries:
                # Create new query object
                new_query = {
                    "query_text": new_query_text,
                    "star_range": new_star_range_metadata if mutation_type in ["stars", "crossover"] else parent["star_range"],
                    "language": new_language_metadata if mutation_type in ["language", "crossover"] else parent["language"],
                    "sort": new_sort_metadata if mutation_type in ["sort", "crossover"] else parent["sort"],
                    "creation": new_creation_metadata if mutation_type in ["creation", "crossover"] else parent["creation"],
                    "topic": new_topic_metadata if mutation_type in ["topic", "crossover"] else parent.get("topic", ""),
                    
                    # Query execution stats (start fresh)
                    "usage_count": 0,
                    "success_count": 0,
                    "error_count": 0,
                    
                    # Performance metrics (inherit some from parent)
                    "total_results": 0,
                    "unique_results": 0,
                    "unique_rate": 0.0,
                    "api_efficiency": parent["api_efficiency"] * 0.5,  # Inherit half the parent's efficiency
                    "quality_score": parent["quality_score"] * 0.5,    # Inherit half the parent's quality
                    "reward": 0.0,  # Start with zero reward
                    "ucb_score": 100.0,  # High initial UCB to encourage exploration
                    
                    # Evolutionary parameters
                    "generation": parent["generation"] + 1 if "generation" in parent else 1,
                    "parent": parent["query_text"],
                    "mutation_type": mutation_type,
                    "duplication_rate": parent["duplication_rate"] * 0.5,  # Inherit half the parent's duplication rate
                    
                    # Thompson sampling parameters (slightly optimistic)
                    "alpha": 1.5,  # Slightly optimistic prior
                    "beta": 1.0,
                }
                
                # Add to queries and tracking
                self.queries.append(new_query)
                existing_queries.add(new_query_text)
                new_queries.append(new_query)
        
        if new_queries:
            # Log details about the new evolved queries
            logger.info(f"Evolved {len(new_queries)} new queries, total pool size: {len(self.queries)}")
            
            # Log detailed information about the mutations
            for query in new_queries[:5]:  # Limit to first 5 for readability
                mutation_type = query.get("mutation_type", "unknown")
                parent_text = query.get("parent", "")[:30] + "..." if query.get("parent") else "none"
                gen = query.get("generation", 0)
                logger.info(f"Query Evolution: {mutation_type.upper()} mutation (gen {gen}) - "
                          f"{query['query_text'][:50]}... (parent: {parent_text})")
            
            # Save immediately when we evolve new queries
            self._save_performance_data()
            
            # Return the new queries (used by local aggressive mutations)
            return new_queries
            
        # If we get here with no mutations created, return empty list
        return []
    
    def _generate_random_query(self):
        """Generate a completely random query to explore new areas of the search space.
        
        Returns:
            A random query string
        """
        query_parts = []
        
        # 70% chance to include star range
        if random.random() < 0.7:
            # 20% chance for extreme stars
            if random.random() < 0.2:
                # Generate extreme star ranges
                if random.random() < 0.5:
                    # Very high stars
                    query_parts.append("stars:>50000")
                else:
                    # Very low stars but with constraints to make them interesting
                    query_parts.append("stars:<100 stars:>10")
            else:
                # Random star range from our predefined ranges
                min_stars, max_stars, _ = random.choice(self.star_ranges)
                if min_stars is not None:
                    query_parts.append(f"stars:>={min_stars}")
                if max_stars is not None:
                    query_parts.append(f"stars:<{max_stars}")
        
        # 60% chance to include language
        if random.random() < 0.6:
            # 20% chance for language combination (if we support it with OR)
            if random.random() < 0.2:
                # Pick 2-3 languages
                num_languages = random.randint(2, 3)
                languages = random.sample(self.languages, k=min(num_languages, len(self.languages)))
                # Combine with OR if the GitHub API supports it, otherwise use first one
                # GitHub API does not directly support OR for languages, but we can handle this
                # in the post-processing logic or by making multiple queries
                query_parts.append(f"language:{languages[0]}")
            else:
                # Pick one language
                if self.languages:
                    query_parts.append(f"language:{random.choice(self.languages)}")
        
        # 50% chance to include sort
        if random.random() < 0.5:
            query_parts.append(f"sort:{random.choice(self.sort_options)}")
        
        # 60% chance to include creation date
        if random.random() < 0.6:
            # 30% chance for micro date range
            if random.random() < 0.3:
                # Generate a very narrow date range (1-7 days)
                days_ago = random.randint(7, 1000)
                range_days = random.randint(1, 7)
                end_date = datetime.now() - timedelta(days=days_ago)
                start_date = end_date - timedelta(days=range_days)
                # Format: created:2019-01-01..2019-01-07
                query_parts.append(f"created:{start_date.strftime('%Y-%m-%d')}..{end_date.strftime('%Y-%m-%d')}")
            else:
                # Use standard creation periods if available
                if self.creation_periods:
                    query_parts.append(random.choice([p for p in self.creation_periods if p]))
        
        # 40% chance to include topic
        if random.random() < 0.4:
            # 25% chance for topic combination
            if random.random() < 0.25 and len(self.topics) >= 2:
                # Pick 2-3 topics
                num_topics = random.randint(2, 3)
                topics = random.sample(self.topics, k=min(num_topics, len(self.topics)))
                # Add them separately (GitHub allows multiple topic: filters)
                for topic in topics:
                    if topic:
                        query_parts.append(f"topic:{topic}")
            else:
                # Just one topic
                if self.topics:
                    topic = random.choice([t for t in self.topics if t])
                    if topic:
                        query_parts.append(f"topic:{topic}")
        
        # 10% chance to include niche filters
        if random.random() < 0.1:
            # Add file-based or structure-based filters
            niche_filters = [
                "filename:Dockerfile",
                "filename:docker-compose.yml",
                "filename:requirements.txt",
                "filename:package.json",
                "filename:Cargo.toml",
                "filename:go.mod",
                "filename:build.gradle",
                "filename:pom.xml",
                "path:/.github/workflows",
                "path:/kubernetes",
                "pushed:>2023-01-01",
                "fork:false",
                "good-first-issue:>0"
            ]
            query_parts.append(random.choice(niche_filters))
        
        # Make sure we have at least one part
        if not query_parts:
            # Default to a safe query if nothing was selected
            query_parts = ["stars:>100", "sort:stars"]
        
        # Join all parts
        return " ".join(query_parts)
    
    def get_top_performing_queries(self, count=10):
        """Get the top performing queries based on reward/usage ratio.
        
        Args:
            count: Number of top queries to return
            
        Returns:
            List of (query, score) tuples
        """
        # Only consider queries used multiple times
        used_queries = [q for q in self.queries if q["usage_count"] >= 3]
        
        if not used_queries:
            return []
            
        # Score by reward per usage
        scored_queries = [(q, q["reward"] / max(1, q["usage_count"])) for q in used_queries]
        
        # Sort by score (descending)
        scored_queries.sort(key=lambda x: x[1], reverse=True)
        
        # Return top N
        return scored_queries[:count]

