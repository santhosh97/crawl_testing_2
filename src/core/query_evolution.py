"""
Query evolution module for GitHub Stars Crawler.

This module provides a unified approach to query evolution and optimization,
incorporating mutation strategies, novelty detection, and adaptive search patterns.
"""
import logging
import random
import math
import zlib
import threading
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from datetime import datetime, timedelta
from collections import defaultdict

# Configure logging
logger = logging.getLogger(__name__)

class QueryEvolver:
    """
    Manages evolution of query configurations with adaptive mutation strategies.
    
    This class encapsulates all the mutation operators and evolution logic for
    generating new query configurations based on successful parent queries.
    It implements progress-aware mutation intensities and novelty guidance.
    """
    
    def __init__(self, similarity_engine=None):
        """
        Initialize the query evolver.
        
        Args:
            similarity_engine: Optional QuerySimilarityEngine for novelty guidance
        """
        self.similarity_engine = similarity_engine
        
        # Mutation operators with weighted probabilities
        self.mutation_operators = [
            # (name, method_name, weight in early phase, weight in late phase)
            ("parameter_adjustment", "mutate_parameter_adjustment", 0.6, 0.2),
            ("parameter_shifting", "mutate_parameter_shifting", 0.2, 0.1),
            ("parameter_addition", "mutate_parameter_addition", 0.1, 0.2),
            ("value_substitution", "mutate_value_substitution", 0.1, 0.25),
            ("strategy_shift", "mutate_strategy_shift", 0.0, 0.15),
            ("time_window_mutation", "mutate_time_window", 0.0, 0.1),
        ]
        
        # Parameter value options for mutations
        self.value_options = {
            'language': [
                'python', 'javascript', 'typescript', 'java', 'cpp', 'go', 
                'rust', 'php', 'ruby', 'swift', 'kotlin', 'scala', 'c', 
                'csharp', 'html', 'css', 'dart', 'shell', 'r', 'julia',
                'vue', 'elixir', 'clojure', 'haskell', 'perl'
            ],
            'topic': [
                'web', 'api', 'machine-learning', 'cli', 'database', 
                'framework', 'library', 'tool', 'ui', 'game', 'backend', 
                'frontend', 'mobile', 'desktop', 'ai', 'blockchain',
                'security', 'test', 'automation', 'infrastructure', 'cloud',
                'devops', 'serverless', 'microservices', 'visualization',
                'education', 'web3', 'productivity', 'utilities'
            ],
            'sort': ['stars', 'forks', 'updated', 'help-wanted-issues'],
            'date_ranges': ['7d', '14d', '30d', '90d', '180d', '365d'],
        }
        
        logger.info("Initialized QueryEvolver with %d mutation operators", 
                    len(self.mutation_operators))
    
    def query_to_string(self, query: Dict[str, Any]) -> str:
        """
        Convert a query configuration to a string for similarity comparison.
        
        Args:
            query: Query configuration dictionary
            
        Returns:
            String representation of the query
        """
        components = []
        for key in sorted(query.keys()):
            if query[key] is not None:
                components.append(f"{key}:{query[key]}")
        return " ".join(components)
    
    def evolve_queries(self, 
                      collection_progress: float, 
                      parent_queries: List[Dict[str, Any]], 
                      num_mutations: int = 5,
                      max_mutations: int = 10) -> List[Dict[str, Any]]:
        """
        Evolve query configurations with adaptive mutation strategy.
        
        Args:
            collection_progress: Collection progress (0.0 to 1.0)
            parent_queries: List of parent query configurations
            num_mutations: Base number of mutations to generate
            max_mutations: Maximum number of mutations to generate
            
        Returns:
            List of mutated query configurations
        """
        if not parent_queries:
            return []
        
        # Calculate mutation intensity based on collection progress
        mutation_intensity = compute_mutation_intensity(collection_progress)
        
        # Scale the number of mutations based on progress
        scaled_mutations = int(num_mutations + (max_mutations - num_mutations) * collection_progress)
        
        # Store mutations
        mutations = []
        
        # Novelty-guided mutation approach when similarity engine is available
        if self.similarity_engine is not None:
            self._evolve_with_novelty_guidance(
                parent_queries, 
                scaled_mutations, 
                mutation_intensity, 
                mutations
            )
        else:
            # Fall back to the original approach without novelty guidance
            self._evolve_without_novelty(
                parent_queries, 
                scaled_mutations, 
                collection_progress, 
                mutation_intensity, 
                mutations
            )
        
        return mutations
    
    def _evolve_with_novelty_guidance(self, 
                                     parent_queries: List[Dict[str, Any]],
                                     mutations_count: int,
                                     mutation_intensity: float,
                                     mutations: List[Dict[str, Any]]) -> None:
        """
        Evolve queries with novelty guidance using similarity engine.
        
        Args:
            parent_queries: List of parent query configurations
            mutations_count: Number of mutations to generate
            mutation_intensity: Intensity of mutations (0.0-1.0)
            mutations: List to store the generated mutations
        """
        for _ in range(mutations_count):
            # Select a parent - prioritize parents that are more novel
            if len(parent_queries) > 1 and random.random() < 0.6:
                # Get novelty scores for parents
                parent_novelties = []
                for parent in parent_queries:
                    query_str = self.query_to_string(parent)
                    novelty = self.similarity_engine.compute_novelty_score(query_str)
                    parent_novelties.append((parent, novelty))
                
                # Sort by novelty (highest first) and select with weighted random
                parent_novelties.sort(key=lambda x: x[1], reverse=True)
                
                # Use softmax-like weighting to favor novel parents
                weights = [math.exp(nov * 2) for _, nov in parent_novelties]
                total = sum(weights)
                if total > 0:
                    weights = [w / total for w in weights]
                    parent_idx = random.choices(range(len(parent_novelties)), weights=weights, k=1)[0]
                    parent = parent_novelties[parent_idx][0]
                else:
                    parent = random.choice(parent_queries)
            else:
                parent = random.choice(parent_queries)
            
            # Try multiple mutation operators and select the most novel result
            candidates = []
            
            # Get parent novelty
            parent_str = self.query_to_string(parent)
            parent_novelty = self.similarity_engine.compute_novelty_score(parent_str)
            
            # Adjust operator weights based on parent novelty
            weights = []
            for i, (op_name, method_name, early_weight, late_weight) in enumerate(self.mutation_operators):
                if parent_novelty < 0.25:
                    # For low-novelty parents, prefer more radical mutations
                    if op_name in ["strategy_shift", "value_substitution"]:
                        weight = late_weight * 1.3
                    else:
                        weight = early_weight * 0.8
                else:
                    # Normal weight calculation for average novelty parents
                    weight = early_weight + 0.5 * (late_weight - early_weight)
                weights.append(weight)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            
            # Try multiple operators
            num_operators_to_try = min(5, len(self.mutation_operators))
            operator_indices = random.choices(range(len(self.mutation_operators)), weights=weights, k=num_operators_to_try)
            
            for operator_idx in operator_indices:
                _, method_name, _, _ = self.mutation_operators[operator_idx]
                
                # Generate a candidate mutation using the method
                intensity = mutation_intensity
                
                # Increase intensity for low-novelty parents
                if parent_novelty < 0.25:
                    intensity = min(0.85, intensity * 1.3)
                
                # Call the mutation method using getattr
                mutation_method = getattr(self, method_name)
                candidate = mutation_method(parent, intensity)
                
                if candidate:
                    # Compute novelty score
                    cand_str = self.query_to_string(candidate)
                    novelty = self.similarity_engine.compute_novelty_score(cand_str)
                    candidates.append((candidate, novelty))
            
            if candidates:
                # Select the most novel candidate
                candidates.sort(key=lambda x: x[1], reverse=True)
                best_candidate = candidates[0][0]
                mutations.append(best_candidate)
                
                # Add this mutation to the similarity engine
                self.similarity_engine.add_query(self.query_to_string(best_candidate))
    
    def _evolve_without_novelty(self, 
                               parent_queries: List[Dict[str, Any]],
                               mutations_count: int,
                               collection_progress: float,
                               mutation_intensity: float,
                               mutations: List[Dict[str, Any]]) -> None:
        """
        Evolve queries without novelty guidance.
        
        Args:
            parent_queries: List of parent query configurations
            mutations_count: Number of mutations to generate
            collection_progress: Collection progress (0.0 to 1.0)
            mutation_intensity: Intensity of mutations (0.0-1.0)
            mutations: List to store the generated mutations
        """
        # Compute weights based on collection progress
        weights = []
        for _, _, early_weight, late_weight in self.mutation_operators:
            # Linear interpolation between early and late weights
            weight = early_weight + collection_progress * (late_weight - early_weight)
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        
        # Generate mutations
        for _ in range(mutations_count):
            # Select a random parent
            parent = random.choice(parent_queries)
            
            # Select a mutation operator based on weights
            operator_idx = random.choices(range(len(self.mutation_operators)), weights=weights, k=1)[0]
            _, method_name, _, _ = self.mutation_operators[operator_idx]
            
            # Apply mutation using the method name
            mutation_method = getattr(self, method_name)
            mutation = mutation_method(parent, mutation_intensity)
            
            if mutation:
                mutations.append(mutation)
    
    def mutate_parameter_adjustment(self, query_config: Dict[str, Any], intensity: float) -> Dict[str, Any]:
        """
        Adjust numeric parameters like stars range.
        
        Args:
            query_config: Original query configuration
            intensity: Mutation intensity (0.0-1.0)
            
        Returns:
            Mutated query configuration
        """
        # Create a copy to avoid modifying the original
        new_config = dict(query_config)
        
        # Parse q parameter to find components
        q = new_config.get("q", "")
        if not q:
            return new_config

        # Look for stars pattern in the query
        parts = q.split()
        new_parts = []
        modified = False

        for part in parts:
            if part.startswith("stars:") and random.random() < intensity:
                # This is a star range, try to modify it
                stars_value = part[6:]
                
                if ".." in stars_value:
                    # It's a range like stars:100..1000
                    try:
                        min_stars, max_stars = stars_value.split("..")
                        min_stars = int(min_stars)
                        max_stars = int(max_stars) if max_stars else float('inf')
                        
                        # Apply random adjustment
                        adjustment_factor = 1.0 + (random.random() * 2 - 1) * intensity
                        adjustment_factor = max(0.9, min(1.1, adjustment_factor))
                        
                        new_min = max(1, int(min_stars * adjustment_factor))
                        
                        if max_stars < float('inf'):
                            new_max = max(new_min + 1, int(max_stars * adjustment_factor))
                            new_parts.append(f"stars:{new_min}..{new_max}")
                        else:
                            new_parts.append(f"stars:{new_min}..")
                        
                        modified = True
                    except (ValueError, TypeError):
                        new_parts.append(part)  # Keep original if parsing fails
                elif stars_value.startswith(">"):
                    # It's a range like stars:>100
                    try:
                        min_stars = int(stars_value[1:])
                        
                        # Apply random adjustment
                        adjustment_factor = 1.0 + (random.random() * 2 - 1) * intensity
                        adjustment_factor = max(0.9, min(1.1, adjustment_factor))
                        
                        new_min = max(1, int(min_stars * adjustment_factor))
                        new_parts.append(f"stars:>{new_min}")
                        
                        modified = True
                    except (ValueError, TypeError):
                        new_parts.append(part)  # Keep original if parsing fails
                else:
                    new_parts.append(part)  # Keep other forms unchanged
            else:
                new_parts.append(part)  # Keep other parts unchanged
        
        if modified:
            new_config["q"] = " ".join(new_parts)
        
        return new_config
    
    def mutate_parameter_shifting(self, query_config: Dict[str, Any], intensity: float) -> Dict[str, Any]:
        """
        Shift parameters up or down significantly.
        
        Args:
            query_config: Original query configuration
            intensity: Mutation intensity (0.0-1.0)
            
        Returns:
            Mutated query configuration
        """
        # Create a copy to avoid modifying the original
        new_config = dict(query_config)
        
        # Parse q parameter to find components
        q = new_config.get("q", "")
        if not q:
            return new_config

        # Look for stars pattern in the query
        parts = q.split()
        new_parts = []
        modified = False

        for part in parts:
            if part.startswith("stars:") and random.random() < intensity:
                # This is a star range, try to modify it
                stars_value = part[6:]
                
                # Apply significant shift (double or halve)
                shift_factor = 2.0 if random.random() < 0.5 else 0.5
                
                if ".." in stars_value:
                    # It's a range like stars:100..1000
                    try:
                        min_stars, max_stars = stars_value.split("..")
                        min_stars = int(min_stars)
                        max_stars = int(max_stars) if max_stars else float('inf')
                        
                        new_min = max(1, int(min_stars * shift_factor))
                        
                        if max_stars < float('inf'):
                            new_max = max(new_min + 1, int(max_stars * shift_factor))
                            new_parts.append(f"stars:{new_min}..{new_max}")
                        else:
                            new_parts.append(f"stars:{new_min}..")
                        
                        modified = True
                    except (ValueError, TypeError):
                        new_parts.append(part)  # Keep original if parsing fails
                elif stars_value.startswith(">"):
                    # It's a range like stars:>100
                    try:
                        min_stars = int(stars_value[1:])
                        new_min = max(1, int(min_stars * shift_factor))
                        new_parts.append(f"stars:>{new_min}")
                        
                        modified = True
                    except (ValueError, TypeError):
                        new_parts.append(part)  # Keep original if parsing fails
                else:
                    new_parts.append(part)  # Keep other forms unchanged
            else:
                new_parts.append(part)  # Keep other parts unchanged
        
        if modified:
            new_config["q"] = " ".join(new_parts)
        
        return new_config
    
    def mutate_parameter_addition(self, query_config: Dict[str, Any], intensity: float) -> Dict[str, Any]:
        """
        Add or remove parameters.
        
        Args:
            query_config: Original query configuration
            intensity: Mutation intensity (0.0-1.0)
            
        Returns:
            Mutated query configuration
        """
        # Create a copy to avoid modifying the original
        new_config = dict(query_config)
        
        # Get original q parameter
        q = new_config.get("q", "")
        
        # Add a parameter with probability based on intensity
        if random.random() < intensity:
            # Possible parameters to add
            language_params = [f"language:{lang}" for lang in self.value_options['language']]
            topic_params = [f"topic:{topic}" for topic in self.value_options['topic']]
            
            # Check if we already have language or topic in the query
            has_language = "language:" in q
            has_topic = "topic:" in q
            
            # Select what to add based on what's missing
            if has_language and not has_topic and random.random() < 0.7:
                param_to_add = random.choice(topic_params)
            elif has_topic and not has_language and random.random() < 0.7:
                param_to_add = random.choice(language_params)
            else:
                # Choose randomly
                all_params = language_params + topic_params
                param_to_add = random.choice(all_params)
            
            # Add to query
            parts = q.split()
            if param_to_add not in parts:  # Avoid duplicates
                parts.append(param_to_add)
                new_config["q"] = " ".join(parts)
        
        # Add sort parameter if missing with lower probability
        if "sort" not in new_config and random.random() < intensity * 0.5:
            sort_value = random.choice(self.value_options['sort'])
            new_config["sort"] = sort_value
            
            # Add order if adding sort
            if "order" not in new_config:
                new_config["order"] = "desc" if random.random() < 0.8 else "asc"
        
        # Remove a parameter with even lower probability
        if random.random() < intensity * 0.2:
            if q:
                parts = q.split()
                if len(parts) > 1:  # Don't remove if only one part
                    part_to_remove = random.choice(parts)
                    parts.remove(part_to_remove)
                    new_config["q"] = " ".join(parts)
        
        return new_config
    
    def mutate_value_substitution(self, query_config: Dict[str, Any], intensity: float) -> Dict[str, Any]:
        """
        Substitute parameter values.
        
        Args:
            query_config: Original query configuration
            intensity: Mutation intensity (0.0-1.0)
            
        Returns:
            Mutated query configuration
        """
        # Create a copy to avoid modifying the original
        new_config = dict(query_config)
        
        # Get original q parameter
        q = new_config.get("q", "")
        
        # Substitute values in q parameter with probability based on intensity
        if q and random.random() < intensity:
            parts = q.split()
            new_parts = []
            modified = False
            
            for part in parts:
                # Check if part is a qualifier (contains ":")
                if ":" in part:
                    qualifier, value = part.split(":", 1)
                    
                    # Substitute language if present
                    if qualifier == "language" and random.random() < intensity:
                        new_language = random.choice(self.value_options["language"])
                        if new_language != value:
                            new_parts.append(f"language:{new_language}")
                            modified = True
                        else:
                            new_parts.append(part)
                    
                    # Substitute topic if present
                    elif qualifier == "topic" and random.random() < intensity:
                        new_topic = random.choice(self.value_options["topic"])
                        if new_topic != value:
                            new_parts.append(f"topic:{new_topic}")
                            modified = True
                        else:
                            new_parts.append(part)
                    
                    else:
                        new_parts.append(part)
                else:
                    new_parts.append(part)
            
            if modified:
                new_config["q"] = " ".join(new_parts)
        
        # Substitute sort parameter if present
        if "sort" in new_config and random.random() < intensity:
            current_sort = new_config["sort"]
            options = [s for s in self.value_options["sort"] if s != current_sort]
            if options:
                new_config["sort"] = random.choice(options)
        
        # Substitute order parameter if present
        if "order" in new_config and random.random() < intensity:
            current_order = new_config["order"]
            new_order = "asc" if current_order == "desc" else "desc"
            new_config["order"] = new_order
        
        return new_config
    
    def mutate_strategy_shift(self, query_config: Dict[str, Any], intensity: float) -> Dict[str, Any]:
        """
        Radical mutation that completely changes the query strategy.
        
        Args:
            query_config: Original query configuration
            intensity: Mutation intensity (0.0-1.0)
            
        Returns:
            Mutated query configuration
        """
        # Only apply this with probability based on intensity
        if random.random() > intensity:
            return dict(query_config)  # Return unchanged
        
        # Define completely new query strategies
        strategies = [
            # Popular repositories strategy
            {'q': f"stars:>1000", 'sort': 'stars', 'order': 'desc'},
            # Recently updated strategy
            {'q': f"stars:>100", 'sort': 'updated', 'order': 'desc'},
            # Language-focused strategy
            {'q': f"stars:>50 language:{random.choice(self.value_options['language'])}", 'sort': 'stars', 'order': 'desc'},
            # Topic-focused strategy
            {'q': f"stars:>10 topic:{random.choice(self.value_options['topic'])}", 'sort': 'stars', 'order': 'desc'},
            # Recent repositories
            {'q': f"created:>{self._get_recent_date_range(30)} stars:>10", 'sort': 'stars', 'order': 'desc'},
            # Help wanted strategy
            {'q': f"stars:>50", 'sort': 'help-wanted-issues', 'order': 'desc'},
        ]
        
        # Select a completely new strategy
        return random.choice(strategies)
    
    def mutate_time_window(self, query_config: Dict[str, Any], intensity: float) -> Dict[str, Any]:
        """
        Add or modify time window constraints in the query.
        
        Args:
            query_config: Original query configuration
            intensity: Mutation intensity (0.0-1.0)
            
        Returns:
            Mutated query configuration
        """
        # Create a copy to avoid modifying the original
        new_config = dict(query_config)
        
        # Get original q parameter
        q = new_config.get("q", "")
        
        # Only apply with probability based on intensity
        if random.random() > intensity:
            return new_config
            
        # Check if the query already has a time window
        has_time_window = any(term in q for term in ["pushed:>", "created:>", "updated:>"])
        
        if has_time_window and random.random() < 0.7:
            # Modify existing time window
            parts = q.split()
            new_parts = []
            modified = False
            
            for part in parts:
                if any(time_term in part for time_term in ["pushed:", "created:", "updated:"]):
                    # Choose a new time term with 30% probability, otherwise keep the same
                    time_term = part.split(":")[0]
                    if random.random() < 0.3:
                        time_types = ["pushed", "created", "updated"]
                        time_types.remove(time_term)
                        time_term = random.choice(time_types)
                    
                    # Choose a new date range
                    date_range = random.choice(self.value_options["date_ranges"])
                    new_parts.append(f"{time_term}:>{date_range}")
                    modified = True
                else:
                    new_parts.append(part)
            
            if modified:
                new_config["q"] = " ".join(new_parts)
        else:
            # Add new time window
            time_term = random.choice(["pushed", "created", "updated"])
            date_range = random.choice(self.value_options["date_ranges"])
            
            # Add to query
            parts = q.split()
            parts.append(f"{time_term}:>{date_range}")
            new_config["q"] = " ".join(parts)
        
        return new_config
    
    def _get_recent_date_range(self, days_ago):
        """
        Generate a date string for a query (format: 'YYYY-MM-DD').
        
        Args:
            days_ago: Number of days to look back
            
        Returns:
            Date string in format 'YYYY-MM-DD'
        """
        date = datetime.now() - timedelta(days=days_ago)
        return date.strftime('%Y-%m-%d')

def compute_collection_progress(found_repos, target_repos, min_progress=0.0, max_progress=1.0):
    """
    Compute collection progress as a value between 0.0 and 1.0
    
    Args:
        found_repos: Number of unique repositories found
        target_repos: Target number of repositories to find
        min_progress: Minimum progress value (default: 0.0)
        max_progress: Maximum progress value (default: 1.0)
    
    Returns:
        Progress value between min_progress and max_progress
    """
    if target_repos <= 0:
        return min_progress
    
    progress = min(found_repos / target_repos, 1.0)
    # Scale to the specified range
    scaled_progress = min_progress + progress * (max_progress - min_progress)
    return scaled_progress

def compute_novelty_alpha(collection_progress):
    """
    Compute alpha value for blending novelty with standard reward
    
    Args:
        collection_progress: Collection progress (0.0 to 1.0)
    
    Returns:
        Alpha value between 0.8 (early) and 0.2 (late)
    """
    # Early phase (0-0.3): α = 0.8 (favor reward)
    # Mid phase (0.3-0.7): Linear transition from 0.8 to 0.2
    # Late phase (0.7-1.0): α = 0.2 (favor novelty)
    
    if collection_progress < 0.3:
        return 0.8
    elif collection_progress > 0.7:
        return 0.2
    else:
        # Linear interpolation between 0.8 and 0.2
        progress_in_mid_phase = (collection_progress - 0.3) / 0.4
        return 0.8 - progress_in_mid_phase * 0.6

def compute_novelty_threshold(collection_progress):
    """
    Compute novelty threshold for filtering queries
    
    Args:
        collection_progress: Collection progress (0.0 to 1.0)
    
    Returns:
        Novelty threshold (0.0 = no filtering, higher values = stricter filtering)
    """
    if collection_progress < 0.7:
        # First 70%: No filtering
        return 0.0
    else:
        # Last 30%: Gradually increase threshold from 0.3 to 0.7
        phase_progress = (collection_progress - 0.7) / 0.3
        return 0.3 + phase_progress * 0.4

def compute_mutation_intensity(collection_progress):
    """
    Compute mutation intensity based on collection progress
    
    Args:
        collection_progress: Collection progress (0.0 to 1.0)
    
    Returns:
        Mutation intensity between 0.2 (early) and 0.8 (late)
    """
    # Start conservative (0.2) and become more radical (0.8) as collection progresses
    base_intensity = 0.2
    max_intensity = 0.8
    
    # Sigmoid-like curve that increases more rapidly in the later stages
    if collection_progress < 0.5:
        # First half: slow increase
        return base_intensity + collection_progress * 0.2
    else:
        # Second half: faster increase
        return base_intensity + 0.1 + (collection_progress - 0.5) * 1.0

class QuerySimilarityEngine:
    """
    MinHash + LSH based query similarity engine for efficiently detecting query novelty.
    Uses n-grams of tokens to represent queries and efficiently computes similarity.
    """
    
    def __init__(self, num_permutations=128, num_bands=10, n_gram_size=3):
        """
        Initialize the query similarity engine with MinHash parameters.
        
        Args:
            num_permutations: Number of hash permutations for MinHash
            num_bands: Number of bands for LSH bucketing
            n_gram_size: Size of n-grams to create from query tokens
        """
        self.num_permutations = num_permutations
        self.num_bands = num_bands
        self.n_gram_size = n_gram_size
        self.rows_per_band = self.num_permutations // self.num_bands
        
        # LSH index: {band_id: {bucket_id: set(query_ids)}}
        self.lsh_index = defaultdict(lambda: defaultdict(set))
        
        # Store query signatures
        self.query_signatures = {}
        
        # Store query string representations
        self.query_strings = {}
        
        # Random hash functions for MinHash
        self.seed_masks = [(random.randint(1, 2**32 - 1), random.randint(1, 2**32 - 1)) 
                            for _ in range(self.num_permutations)]
        
        # Query counter for ID generation
        self.query_counter = 0
        
        logger.info(f"Initialized QuerySimilarityEngine with {num_permutations} permutations, "
                   f"{num_bands} bands, and {n_gram_size}-grams")
    
    def _tokenize_query(self, query_str):
        """Convert query string to tokens"""
        # Split by spaces and remove empty tokens
        tokens = [t.strip() for t in query_str.split() if t.strip()]
        return tokens
    
    def _create_ngrams(self, tokens, n=3):
        """Create n-grams from a list of tokens"""
        return [" ".join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)] if len(tokens) >= n else tokens
    
    def _compute_hash(self, token, seed, mask):
        """Compute a single hash for a token"""
        # Use MurmurHash-like approach for speed
        h = zlib.crc32(token.encode()) ^ seed
        h = ((h * seed) & 0xFFFFFFFF) ^ mask
        return h
    
    def _compute_minhash_signature(self, tokens):
        """Compute MinHash signature for a set of tokens"""
        if not tokens:
            return [2**32 - 1] * self.num_permutations
        
        # Initialize signature to max value
        signature = [2**32 - 1] * self.num_permutations
        
        # For each token and each hash function
        for token in tokens:
            for i, (seed, mask) in enumerate(self.seed_masks):
                h = self._compute_hash(token, seed, mask)
                signature[i] = min(signature[i], h)
                
        return signature
    
    def add_query(self, query_str):
        """
        Add a query to the LSH index
        
        Args:
            query_str: The query string to add
            
        Returns:
            query_id: Unique identifier for the added query
        """
        # Generate unique ID for this query
        query_id = self.query_counter
        self.query_counter += 1
        
        # Store original query string
        self.query_strings[query_id] = query_str
        
        # Tokenize the query
        tokens = self._tokenize_query(query_str)
        
        # Create n-grams
        ngrams = self._create_ngrams(tokens, self.n_gram_size)
        
        # Compute MinHash signature
        signature = self._compute_minhash_signature(ngrams)
        
        # Store signature
        self.query_signatures[query_id] = signature
        
        # Add to LSH index
        for band_idx in range(self.num_bands):
            # Compute band value by combining hash values in this band
            start_idx = band_idx * self.rows_per_band
            end_idx = start_idx + self.rows_per_band
            band_signature = tuple(signature[start_idx:end_idx])
            
            # Use band signature as bucket key
            bucket_id = hash(band_signature)
            
            # Add query to bucket
            self.lsh_index[band_idx][bucket_id].add(query_id)
        
        return query_id
    
    def _jaccard_similarity(self, sig1, sig2):
        """Estimate Jaccard similarity from MinHash signatures"""
        if not sig1 or not sig2:
            return 0.0
            
        matches = sum(1 for i in range(len(sig1)) if sig1[i] == sig2[i])
        return matches / len(sig1)
    
    def find_similar_queries(self, query_str, threshold=0.5):
        """
        Find queries similar to the given query
        
        Args:
            query_str: The query string to compare against
            threshold: Minimum similarity threshold
            
        Returns:
            List of (query_id, similarity) tuples for similar queries
        """
        # Tokenize the query
        tokens = self._tokenize_query(query_str)
        
        # Create n-grams
        ngrams = self._create_ngrams(tokens, self.n_gram_size)
        
        # Compute MinHash signature
        signature = self._compute_minhash_signature(ngrams)
        
        # Find candidate matches using LSH
        candidate_ids = set()
        for band_idx in range(self.num_bands):
            start_idx = band_idx * self.rows_per_band
            end_idx = start_idx + self.rows_per_band
            band_signature = tuple(signature[start_idx:end_idx])
            bucket_id = hash(band_signature)
            
            # Get all queries in the same bucket
            matches = self.lsh_index[band_idx].get(bucket_id, set())
            candidate_ids.update(matches)
        
        # Compute actual similarities for candidates
        similarities = []
        for cand_id in candidate_ids:
            cand_sig = self.query_signatures[cand_id]
            sim = self._jaccard_similarity(signature, cand_sig)
            if sim >= threshold:
                similarities.append((cand_id, sim))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities
    
    def compute_novelty_score(self, query_str, default_score=1.0):
        """
        Compute novelty score for a query
        
        Novelty score is 1.0 - (similarity to closest match),
        or default_score if no queries have been indexed yet
        
        Args:
            query_str: Query string to evaluate
            default_score: Default novelty score for the first query
            
        Returns:
            Novelty score between 0.0 (identical to existing) and 1.0 (completely novel)
        """
        if not self.query_signatures:
            return default_score
            
        # Find similar queries
        similar_queries = self.find_similar_queries(query_str, threshold=0.0)
        
        if not similar_queries:
            return default_score
            
        # Get highest similarity
        max_similarity = max([sim for _, sim in similar_queries])
        
        # Novelty is inverse of similarity
        novelty_score = 1.0 - max_similarity
        return novelty_score

def generate_diverse_queries(count=10):
    """
    Generate a diverse set of GitHub search queries to seed the query pool.
    
    Args:
        count: Number of queries to generate
        
    Returns:
        List of query configuration dictionaries
    """
    # Languages to sample from
    languages = [
        'python', 'javascript', 'typescript', 'java', 'go', 'rust', 
        'cpp', 'csharp', 'php', 'ruby', 'swift', 'kotlin', 'scala'
    ]
    
    # Star ranges to sample from
    star_ranges = [
        "stars:>10000", 
        "stars:5000..9999",
        "stars:1000..4999", 
        "stars:500..999",
        "stars:100..499",
        "stars:10..99"
    ]
    
    # Topics to sample from
    topics = [
        "machine-learning", "data-science", "web-development", 
        "game-development", "blockchain", "artificial-intelligence",
        "deep-learning", "database", "serverless", "devops",
        "frontend", "backend", "mobile", "cloud", "security"
    ]
    
    # Sort options
    sort_options = [
        {"sort": "stars", "order": "desc"},
        {"sort": "updated", "order": "desc"},
        {"sort": "forks", "order": "desc"},
        {}  # Default sort
    ]
    
    # Generate diverse queries
    queries = []
    
    # Add some language-based queries
    for _ in range(count // 4):
        lang = random.choice(languages)
        stars = random.choice(star_ranges)
        sort_opt = random.choice(sort_options)
        
        query = {"q": f"language:{lang} {stars}"}
        query.update(sort_opt)
        queries.append(query)
    
    # Add some topic-based queries
    for _ in range(count // 4):
        topic = random.choice(topics)
        stars = random.choice(star_ranges)
        sort_opt = random.choice(sort_options)
        
        query = {"q": f"topic:{topic} {stars}"}
        query.update(sort_opt)
        queries.append(query)
    
    # Add some combined queries
    for _ in range(count // 4):
        lang = random.choice(languages)
        topic = random.choice(topics)
        stars = random.choice(star_ranges)
        sort_opt = random.choice(sort_options)
        
        query = {"q": f"language:{lang} topic:{topic} {stars}"}
        query.update(sort_opt)
        queries.append(query)
    
    # Add some specialized queries
    for _ in range(count - len(queries)):
        query_type = random.choice([
            "recent_activity", "high_stars", "high_forks", 
            "high_contributors", "documentation"
        ])
        
        if query_type == "recent_activity":
            days = random.choice([7, 14, 30])
            query = {"q": f"pushed:>{days}d stars:>50", "sort": "updated", "order": "desc"}
        elif query_type == "high_stars":
            query = {"q": "stars:>5000", "sort": "stars", "order": "desc"}
        elif query_type == "high_forks":
            query = {"q": "forks:>1000", "sort": "forks", "order": "desc"}
        elif query_type == "high_contributors":
            query = {"q": "stars:>100 forks:>50", "sort": "stars", "order": "desc"}
        else:  # documentation
            query = {"q": "stars:>100 topic:documentation", "sort": "stars", "order": "desc"}
            
        queries.append(query)
    
    # Ensure we have exactly the requested count
    return queries[:count]

def evolve_queries(collection_progress, parent_queries, num_mutations=5, max_mutations=10, similarity_engine=None):
    """
    Evolve query configurations with novelty-guided adaptive mutation strategy
    
    Args:
        collection_progress: Collection progress (0.0 to 1.0)
        parent_queries: List of parent query configurations to evolve from
        num_mutations: Base number of mutations to generate
        max_mutations: Maximum number of mutations to generate
        similarity_engine: Optional QuerySimilarityEngine for novelty-guided mutations
    
    Returns:
        List of mutated query configurations
    """
    # Create QueryEvolver instance
    evolver = QueryEvolver(similarity_engine=similarity_engine)
    
    # Delegate to the query evolver
    return evolver.evolve_queries(
        collection_progress=collection_progress,
        parent_queries=parent_queries,
        num_mutations=num_mutations,
        max_mutations=max_mutations
    )