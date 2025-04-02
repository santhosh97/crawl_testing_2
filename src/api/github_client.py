"""
GitHub API client for the repository crawler.

This module provides classes and functions for interacting with the GitHub API,
including query building, rate limit handling, and result processing.
It supports both REST and GraphQL API endpoints with proper caching and error handling.
"""

import os
import logging
import time
import threading
import random
import hashlib
import json
from datetime import datetime
from pathlib import Path
import math
from typing import Dict, List, Optional, Any, Set, Union, Iterator, Tuple
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Import required modules
from typing import TYPE_CHECKING

# Import interfaces instead of implementation classes
from src.interfaces import IConfig, IQueryPool, IMetricsCollector

# Use TYPE_CHECKING for imports only needed for type checking
if TYPE_CHECKING:
    from src.core.config import Config
    from src.metrics.collector.collector import MetricsCollector
    from src.core.query_pool import QueryPool

# Use centralized logging configuration
logger = logging.getLogger(__name__)

# Use existing bandit logger from centralized logging configuration
bandit_logger = logging.getLogger('bandit_algorithm')

# We don't need to import and use global directories here - they should be
# passed through the metrics_collector or provided via proper DI

# Default metrics logging interval (in seconds)
CACHE_METRICS_LOG_INTERVAL = 300  # Default: log every 5 minutes

# Import connection management
from src.utils.connection_manager import ConnectionManager

# GitHub GraphQL API endpoint
GITHUB_API_URL = "https://api.github.com/graphql"

# Use CacheManager directly
from src.utils.cache_utils import CacheManager

# Default TTL value - this should be passed in from components that have config
CACHE_DEFAULT_TTL = 1800  # Default TTL for cached queries (30 minutes)



# Import the unified TokenManager from token_management
from src.api.token_management import TokenManager

# TokenManager is passed as a parameter to functions in this module
# Token authentication format and validation is handled by the TokenManager class

# Function to validate token auth formats - delegates to TokenManager

# Import exceptions from central location
from src.api.github_exceptions import (
    GitHubRateLimitError,
    GitHubAPIError,
)


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((requests.exceptions.RequestException, GitHubRateLimitError)),
    reraise=True,
)
def execute_graphql_query(query: str, variables: Optional[Dict[str, Any]] = None, assigned_token: Optional[str] = None, token_manager: Optional[Any] = None, connection_manager=None) -> Dict[str, Any]:
    """Execute a GraphQL query against GitHub's API with retries and token rotation.
    
    Args:
        query: The GraphQL query string.
        variables: Optional variables for the query.
        assigned_token: Optional specific token to use.
        token_manager: TokenManager instance for managing GitHub API tokens.
        connection_manager: ConnectionManager instance for HTTP connections.
        
    Returns:
        The JSON response from the API.
        
    Raises:
        GitHubAPIError: If the API returns an error.
        GitHubRateLimitError: If rate limit is exceeded for all tokens.
    """
    # Get the token to use
    # The enhanced token manager supports high_priority parameter
    if assigned_token:
        token = assigned_token
    elif token_manager:
        # Get token from token manager
        token = token_manager.get_token(high_priority=False)
    else:
        raise ValueError("Either assigned_token or token_manager must be provided")
        
    # Check if we've exhausted all tokens
    if token_manager and not token_manager.tokens:
        logger.error("All tokens have been marked as invalid! Unable to continue.")
        raise GitHubAPIError("All tokens are invalid")
    
    # Use GitHub token authentication format (Bearer)
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/vnd.github.v4+json"  # Explicitly request v4 API (GraphQL)
    }
    
    logger.debug(f"Using token {token[:4]}...{token[-4:]}")
    
    json_data = {"query": query}
    if variables:
        json_data["variables"] = variables
    
    try:
        # Get or create session for this token to reuse connections
        if connection_manager:
            # Use provided ConnectionManager directly
            session = connection_manager.get_session(token)
        else:
            # Use TokenManager's connection_manager
            session = token_manager.connection_manager.get_session(token)
        
        # Execute request with authentication header
        response = session.post(GITHUB_API_URL, json=json_data, headers=headers, timeout=30)
        
        # Get rate limit info from headers
        remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
        reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
        
        # Update token's rate limit info in the token manager
        token_manager.update_token_usage(token=token, response=response, remaining=remaining, reset_time=reset_time)
        
        # Check for rate limiting
        if remaining == 0:
            # Mark token as depleted in token manager
            # The token manager will move it to the rate_limited_tokens queue
            token_manager.mark_token_depleted(token=token, reset_time=reset_time)
            # Log at debug level since this is a normal operational event
            logger.debug(f"Token {token[:4]}...{token[-4:]} depleted. Handled by token queue system.")
            # Let retry mechanism try with another token from the available queue
            raise GitHubRateLimitError("Token depleted")
        
        # Check for HTTP errors
        if response.status_code != 200:
            logger.error(f"GitHub API error: {response.status_code} - {response.text}")
            
            # If still unauthorized after trying both auth formats, mark token as invalid
            if response.status_code == 401:
                logger.error(f"Token {token[:4]}...{token[-4:]} is invalid or expired with both auth formats. Removing from token pool.")
                
                # Remove the token from all active token lists
                token_manager.remove_invalid_token(token)
                
                # Check if we have any valid tokens left
                if not token_manager.tokens:
                    logger.critical("All tokens have been marked as invalid! Unable to continue.")
                    raise GitHubAPIError("All tokens are invalid")
                    
                # Log how many valid tokens remain
                logger.warning(f"Invalid token removed. {len(token_manager.tokens)} valid tokens remaining")
                
                # Let retry mechanism try with another token
                raise GitHubRateLimitError("Invalid token")
                
            raise GitHubAPIError(f"GitHub API error: {response.status_code}")
        
        result = response.json()
        
        # Check for GraphQL errors
        if "errors" in result:
            errors = result["errors"]
            # Check for rate limit related errors
            rate_limit_messages = ["rate limit", "secondary rate limit", "abuse detection"]
            if any(any(msg in str(err).lower() for msg in rate_limit_messages) for err in errors):
                # Mark token as depleted in token manager
                # The token manager will handle moving it to rate_limited_tokens queue
                token_manager.mark_token_depleted(token=token, reset_time=reset_time)
                # Log the error at debug level since this is handled by queue system
                logger.debug(f"GraphQL rate limit error for token {token[:4]}...{token[-4:]}. Handled by token queue system. Error: {errors}")
                raise GitHubRateLimitError(f"GraphQL rate limit exceeded: {errors}")
            else:
                logger.error(f"GraphQL error: {errors}")
                raise GitHubAPIError(f"GraphQL error: {errors}")
        
        return result
    
    except requests.exceptions.Timeout:
        logger.warning(f"Request timeout using token {token[:4]}...{token[-4:]}.")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Request exception: {e}")
        raise


def process_repository_data(data: Dict[str, Any], check_duplicates: bool = False, cache_mgr=None) -> List[Dict[str, Any]]:
    """
    Process repository data from GitHub API response with efficient duplicate detection.
    
    Args:
        data: GraphQL response data.
        check_duplicates: Whether to check and filter out duplicates during processing.
        cache_mgr: CacheManager instance
        
    Returns:
        List of processed repository data.
    """
    repositories = []
    duplicates = 0
    timestamp = datetime.utcnow()  # Use same timestamp for batch for consistency
    
    for edge in data["data"]["search"]["edges"]:
        node = edge["node"]
        repo_id = node["id"]
        
        # Check if this is a duplicate repository (if requested)
        if check_duplicates and cache_mgr and cache_mgr.is_duplicate_repository(repo_id):
            duplicates += 1
            continue
        
        # Preprocess data for efficiency - parse dates once
        created_at = datetime.fromisoformat(node["createdAt"].replace("Z", "+00:00"))
        updated_at = datetime.fromisoformat(node["updatedAt"].replace("Z", "+00:00"))
        
        repository = {
            "github_id": repo_id,
            "name": node["name"],
            "owner": node["owner"]["login"],
            "full_name": node["nameWithOwner"],
            "url": node["url"],
            "description": node["description"],
            "created_at": created_at,
            "updated_at": updated_at,
            "star_count": node["stargazerCount"],
            "fetched_at": timestamp,
        }
        
        # Mark as seen if we're checking duplicates
        if check_duplicates and cache_mgr:
            cache_mgr.mark_repository_seen(repo_id)
        
        repositories.append(repository)
    
    if check_duplicates and duplicates > 0:
        # Calculate and log duplication rate
        total = len(repositories) + duplicates
        duplication_rate = duplicates / total if total > 0 else 0
        if duplication_rate > 0.2:  # Only log if significant duplication
            logger.debug(f"Filtered {duplicates} duplicates ({duplication_rate:.1%}) during processing")
    
    return repositories




def fetch_repositories_with_query(query_text: str, limit: int = 100, cursor: Optional[str] = None, 
                        assigned_token: Optional[str] = None, token_manager: Optional[Any] = None, 
                        cache_mgr=None, query_pool=None, connection_manager=None,
                        ttl: Optional[int] = None, force_refresh: bool = False) -> Dict[str, Any]:
    """Fetch repositories using a specific query with enhanced caching.
    
    Args:
        query_text: The query text to use for repository search
        limit: Maximum number of repositories to fetch per request
        cursor: Pagination cursor
        assigned_token: Optional specific token to use for this request
        token_manager: TokenManager instance for managing GitHub API tokens
        cache_mgr: CacheManager instance (optional)
        query_pool: QueryPool instance (optional)
        connection_manager: ConnectionManager instance (optional)
        ttl: Optional time-to-live for cache entries
        force_refresh: If True, bypass cache and force a fresh API call
        
    Returns:
        GraphQL API response
    """
    # GraphQL query
    query = """
    query FetchRepositories($query: String!, $cursor: String, $limit: Int!) {
      search(query: $query, type: REPOSITORY, first: $limit, after: $cursor) {
        repositoryCount
        pageInfo {
          endCursor
          hasNextPage
        }
        edges {
          node {
            ... on Repository {
              id
              name
              owner {
                login
              }
              nameWithOwner
              url
              description
              createdAt
              updatedAt
              stargazerCount
            }
          }
        }
      }
    }
    """
    
    variables = {
        "query": query_text,
        "cursor": cursor,
        "limit": limit
    }
    
    # Generate cache key for this query+cursor+limit combination
    key_components = f"{query_text}|{cursor or 'None'}|{limit}"
    cache_key = hashlib.md5(key_components.encode()).hexdigest()
    
    # Try to get cached result first directly from cache manager if not forcing refresh
    cached_result = None
    if cache_mgr and not force_refresh:
        cached_result = cache_mgr.get_cached_query_result(cache_key)
        
    if cached_result:
        logger.debug(f"Cache hit for query: {query_text[:30]}... cursor: {cursor}")
        
        # Update query stats for this successful cache hit
        if query_pool:
            query_pool.update_query_stats(query_text, success=True, unique_rate=0.8)  # Assume good unique rate for popular queries
        
        return cached_result
    
    # No cache hit, execute query
    result = execute_graphql_query(
        query=query, 
        variables=variables, 
        assigned_token=assigned_token, 
        token_manager=token_manager,
        connection_manager=connection_manager
    )
    
    # Determine cache TTL based on query characteristics
    if ttl is None:
        # Base TTL
        base_ttl = CACHE_DEFAULT_TTL
        
        # Adjust TTL for different query types and cursor positions
        if cursor is None:
            # First page - cache longer
            ttl = base_ttl * 2
        elif "stars:>10000" in query_text.lower():
            # High-star queries change less often - cache longer
            ttl = base_ttl * 1.5
        elif "stars:>1000" in query_text.lower():
            # Medium-star queries - cache slightly longer
            ttl = base_ttl * 1.2
        else:
            # Standard cache time
            ttl = base_ttl
    
    # Don't cache errors
    if "errors" not in result and cache_mgr:
        # Cache the successful result directly with the cache manager
        cache_mgr.cache_query_result(cache_key, result, ttl)
        
        # Update query statistics with estimation of uniqueness
        edges = result.get("data", {}).get("search", {}).get("edges", [])
        if edges and query_pool and cache_mgr:
            repo_count = len(edges)
            # Check a random sample for uniqueness (for efficiency)
            sample_size = min(10, repo_count)
            if repo_count > 0:
                sample = random.sample(edges, sample_size)
                
                dupes = 0
                for edge in sample:
                    repo_id = edge["node"]["id"]
                    if cache_mgr.is_duplicate_repository(repo_id):
                        dupes += 1
                
                # Estimate unique rate from sample
                unique_rate = 1.0 - (dupes / sample_size)
                query_pool.update_query_stats(query_text, success=True, unique_rate=unique_rate)
    
    return result


