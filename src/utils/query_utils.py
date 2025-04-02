"""
Shared query utilities for GitHub API interactions.

This module centralizes common functionality for GitHub GraphQL query execution,
pagination handling, and related helper functions to eliminate code duplication.
"""
import logging
from typing import Dict, Tuple, List, Any, Optional

# Configure logging
logger = logging.getLogger(__name__)


def get_pagination_thresholds(query_text: str) -> Tuple[float, int, int]:
    """Get pagination thresholds based on star range in query.
    
    Args:
        query_text: The query text to analyze
        
    Returns:
        Tuple of (duplication_threshold, max_pages_threshold, max_skip_factor)
    """
    query_lower = query_text.lower()
    
    # Check for high-star ranges in query
    if any(pattern in query_lower for pattern in [
        "stars:>100000", "stars:>90000", "stars:>80000", "stars:>70000"
    ]):
        # Very high star repositories - be extremely persistent
        return 0.99, 20, 1  # 99% duplication threshold, 20 pages, max skip 1
    elif any(pattern in query_lower for pattern in [
        "stars:>50000", "stars:>40000", "stars:>30000"
    ]):
        # High star repositories - be very persistent
        return 0.98, 16, 1  # 98% duplication threshold, 16 pages, max skip 1
    elif any(pattern in query_lower for pattern in [
        "stars:>20000", "stars:>15000", "stars:>10000"
    ]):
        # Moderately high star repositories - be persistent
        return 0.95, 12, 1  # 95% duplication threshold, 12 pages, max skip 1
    elif any(pattern in query_lower for pattern in [
        "stars:>5000", "stars:>3000", "stars:>1000"
    ]):
        # Medium star repositories - be somewhat persistent
        return 0.8, 5, 2  # 80% duplication threshold, 5 pages, max skip 2
    else:
        # Default for low-star queries - standard behavior
        return 0.7, 3, 4  # 70% duplication threshold, 3 pages, max skip 4


def get_query_exhaustion_parameters(query_text: str, collection_stage: float = 0.0) -> Dict[str, Any]:
    """Get query exhaustion parameters based on star range in query.
    
    Args:
        query_text: The query text to analyze
        collection_stage: Progress of collection from 0.0 to 1.0
        
    Returns:
        Dictionary with exhaustion parameters:
            - base_threshold: Base duplication threshold
            - min_results: Minimum results needed before applying threshold
            - exhaustion_threshold: Adjusted threshold based on collection stage
            - cooling_multiplier: Cooling period multiplier
    """
    query_lower = query_text.lower()
    
    # Base thresholds
    if any(pattern in query_lower for pattern in ["stars:>50000", "stars:>100000"]):
        # Very high star queries - extremely high threshold before exhaustion
        base_threshold = 0.98
        min_results = 30  # Need more samples for statistical confidence
        cooling_multiplier = 0.3  # 30% of standard cooling period for very high-star queries
    elif any(pattern in query_lower for pattern in ["stars:>20000", "stars:>30000", "stars:>40000"]):
        # High star queries - very high threshold
        base_threshold = 0.95
        min_results = 25
        cooling_multiplier = 0.4  # 40% of standard cooling for high-star queries
    elif any(pattern in query_lower for pattern in ["stars:>10000", "stars:>15000"]):
        # Moderately high star queries - high threshold
        base_threshold = 0.9
        min_results = 20
        cooling_multiplier = 0.6  # 60% of standard cooling for moderately high-star queries
    else:
        # Standard queries - lower threshold to trigger mutations more easily
        base_threshold = 0.05
        min_results = 5
        cooling_multiplier = 1.0  # Standard cooling period
    
    # Adjust thresholds based on collection stage - make them stricter as we collect more
    # Maximum adjustment: reduce threshold by 5-10% when collection is complete
    threshold_adjustment = collection_stage * 0.08  # Up to 8% reduction
    exhaustion_threshold = max(0.75, base_threshold - threshold_adjustment)
    
    return {
        "base_threshold": base_threshold,
        "min_results": min_results,
        "exhaustion_threshold": exhaustion_threshold,
        "cooling_multiplier": cooling_multiplier
    }


def calculate_adaptive_skip_factor(
    duplication_rates: List[float],
    pages_since_useful_content: int,
    skip_factor: int,
    max_skip_factor: int
) -> int:
    """Calculate adaptive skip factor for smart pagination.

    Args:
        duplication_rates: Recent duplication rates history
        pages_since_useful_content: Count of consecutive pages with high duplication
        skip_factor: Current skip factor
        max_skip_factor: Maximum allowed skip factor

    Returns:
        Updated skip factor
    """
    if len(duplication_rates) < 3:
        return skip_factor
        
    # Calculate duplication gradient (rate of increase)
    recent_rates = duplication_rates[-3:]
    duplication_gradient = (recent_rates[-1] - recent_rates[0]) / 2
    
    if duplication_gradient > 0.1:  # Rapidly increasing duplication
        # Skip more pages as duplication increases faster
        return min(max_skip_factor, skip_factor + 1)
    elif duplication_gradient > 0.05:  # Moderately increasing duplication
        return min(max_skip_factor, skip_factor + 1)
    elif pages_since_useful_content >= 2:
        # Haven't found much in recent pages, try jumping ahead
        return min(max_skip_factor, skip_factor + 1)
    elif duplication_gradient < -0.1:  # Decreasing duplication - finding new content
        return 1  # Slow down to explore this area
        
    # No changes needed
    return skip_factor


def adjust_skip_factor_for_query_type(query_text: str, skip_factor: int) -> int:
    """Apply special skip factor adjustments based on query characteristics.
    
    Args:
        query_text: The query text
        skip_factor: Current skip factor
        
    Returns:
        Adjusted skip factor
    """
    # Apply a skip factor slowdown for high-star queries
    if "stars:>10000" in query_text.lower() or "stars:>20000" in query_text.lower():
        # Apply a 0.5 multiplier for high-star queries
        if skip_factor > 1:
            return max(1, int(skip_factor * 0.5))
    
    return skip_factor


def get_smart_pagination_config(query_text: str) -> Dict[str, Any]:
    """Get comprehensive pagination configuration for a query.
    
    Args:
        query_text: The query text to analyze
        
    Returns:
        Dictionary with pagination configuration parameters
    """
    # Default thresholds - conservative
    config = {
        "max_pages": 10,             # Maximum pages to fetch
        "early_stop_threshold": 7,   # Stop after this many consecutive high-duplication pages
        "skip_factor": 1,            # Initial skip factor (1 = don't skip)
        "duplication_threshold": 0.7, # Duplication rate threshold
        "min_api_calls": 2,          # Minimum API calls before early termination
        "high_star_min_calls": 0      # Minimum calls for high-star queries
    }
    
    # Star range affects result quality and duplication likelihood
    if "stars:>50000" in query_text.lower() or "stars:>30000" in query_text.lower():
        # Very high star repos - fetch more pages with less early stopping
        config["max_pages"] = 20
        config["early_stop_threshold"] = 10
        config["duplication_threshold"] = 0.99
        config["skip_factor"] = 1
        config["high_star_min_calls"] = 5  # Ensure at least 5 API calls for high-star queries
    elif "stars:>10000" in query_text.lower():
        # High star repos - fetch more pages
        config["max_pages"] = 15
        config["early_stop_threshold"] = 8
        config["duplication_threshold"] = 0.95
        config["skip_factor"] = 1
        config["high_star_min_calls"] = 3  # Ensure at least 3 API calls for moderately high-star queries
    elif "stars:<100" in query_text.lower() or "stars:<50" in query_text.lower():
        # Very low star repos - more likely to have high duplication
        config["max_pages"] = 5
        config["early_stop_threshold"] = 3
        config["duplication_threshold"] = 0.7
        config["skip_factor"] = 2
        
    # Creation date affects uniqueness
    if "created:>2022" in query_text.lower():
        # Recent repos - less likely to be duplicates
        config["max_pages"] += 2
        config["early_stop_threshold"] += 1
    elif "created:<2015" in query_text.lower():
        # Older repos - more likely to be well-known
        config["max_pages"] -= 1
        config["early_stop_threshold"] -= 1
        
    return config


def should_terminate_pagination(
    query_text: str,
    query_results: int, 
    query_duplicates: int,
    api_calls_for_query: int,
    pages_since_useful_content: int,
    has_next_page: bool,
    pagination_config: Optional[Dict[str, Any]] = None
) -> Tuple[bool, str]:
    """Determine if pagination should be terminated based on various criteria.
    
    Args:
        query_text: The query text
        query_results: Number of results fetched so far
        query_duplicates: Number of duplicates found
        api_calls_for_query: Number of API calls made for this query
        pages_since_useful_content: Consecutive pages with high duplication
        has_next_page: Whether API indicates more pages exist
        pagination_config: Optional pagination configuration, will be fetched if not provided
        
    Returns:
        Tuple of (should_terminate, reason)
    """
    # Get pagination configuration if not provided
    if pagination_config is None:
        pagination_config = get_smart_pagination_config(query_text)
    
    # Extract configuration values
    duplication_threshold = pagination_config.get("duplication_threshold", 0.7)
    max_pages_threshold = pagination_config.get("early_stop_threshold", 7)
    min_api_calls = pagination_config.get("min_api_calls", 2)
    high_star_min_calls = pagination_config.get("high_star_min_calls", 0)
    
    # Check if there are no more pages
    if not has_next_page:
        return True, "No more pages available"
    
    # Calculate current duplication rate
    if query_results > 0:
        duplication_rate = query_duplicates / query_results
    else:
        duplication_rate = 0
    
    # Early termination based on high duplication rate
    if (query_results > 0 and 
        duplication_rate > duplication_threshold and 
        api_calls_for_query >= max(min_api_calls, max_pages_threshold // 2)):
        return True, f"High duplication rate: {duplication_rate:.1%} (threshold: {duplication_threshold:.1%})"
    
    # Early termination based on consecutive high-duplication pages
    if pages_since_useful_content >= max_pages_threshold:
        return True, f"Too many consecutive high-duplication pages: {pages_since_useful_content} (threshold: {max_pages_threshold})"
    
    # Special logic for high-star queries to ensure minimum API calls
    if high_star_min_calls > 0 and api_calls_for_query < high_star_min_calls:
        # For high-star queries, ensure we make at least the minimum API calls
        return False, f"Persisting with high-star query ({api_calls_for_query}/{high_star_min_calls} minimum calls)"
    
    # Default - continue pagination
    return False, ""


def calculate_quality_score(star_distribution: Dict[str, int], star_weights: Optional[Dict[str, float]] = None) -> float:
    """Calculate quality score based on star distribution.
    
    Args:
        star_distribution: Dictionary mapping star ranges to counts
        star_weights: Optional weights for each star range, defaults to pre-defined weights if not provided
        
    Returns:
        Quality score from 0.0 to 1.0
    """
    # Default star weights if not provided
    if star_weights is None:
        star_weights = {
            "100K+": 1.0,     # Highest quality
            "75K-100K": 0.98, # Increased
            "50K-75K": 0.95,  # Increased
            "40K-50K": 0.92,  # Increased
            "30K-40K": 0.88,  # Increased
            "25K-30K": 0.85,  # Increased
            "20K-25K": 0.82,  # Increased
            "15K-20K": 0.78,  # Increased
            "10K-15K": 0.75,  # Increased
            "7.5K-10K": 0.7,  # Increased
            "5K-7.5K": 0.65,  # Increased
            "2.5K-5K": 0.55,  # Increased
            "1K-2.5K": 0.45,  # Increased
            "500-1K": 0.35,
            "100-500": 0.3,
            "50-100": 0.25,
            "10-50": 0.2,
            "<10": 0.15
        }
    
    # If no distribution data, return default score
    if not star_distribution:
        return 0.0
    
    # Calculate weighted sum and total
    total_weight = 0.0
    weighted_sum = 0.0
    
    for range_name, count in star_distribution.items():
        weight = star_weights.get(range_name, 0.5)  # Default to 0.5 if range not found
        weighted_sum += count * weight
        total_weight += count
    
    # Calculate and return quality score
    if total_weight > 0:
        return weighted_sum / total_weight
    else:
        return 0.0