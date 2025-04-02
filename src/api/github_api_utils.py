"""
GitHub API utilities for the Stars Crawler.

This module provides helper functions for interacting with the GitHub GraphQL API,
processing API responses, and managing repository data.
"""
import logging
import hashlib
import time
import json
from typing import Dict, List, Optional, Any, Tuple, Set, Union

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Import from our modules
from src.utils.cache_utils import CacheManager  # Use CacheManager directly
from src.utils.connection_manager import ConnectionManager
from src.api.token_management import TokenManager

# Configure logging
logger = logging.getLogger(__name__)

# GitHub GraphQL API endpoint
GITHUB_API_URL = "https://api.github.com/graphql"

# Standard GitHub Search Query - this is the GraphQL query that will be used
# It accepts search query, limit, and cursor as variables
GITHUB_SEARCH_QUERY = """
query SearchRepositories($query: String!, $limit: Int!, $cursor: String) {
  search(query: $query, type: REPOSITORY, first: $limit, after: $cursor) {
    repositoryCount
    pageInfo {
      endCursor
      hasNextPage
    }
    edges {
      node {
        ... on Repository {
          databaseId
          nameWithOwner
          owner { login }
          stargazerCount
          forkCount
          primaryLanguage { name }
          isFork
        }
      }
    }
  }
}

"""

# Import exception classes for retry handling
from src.api.github_exceptions import GitHubRateLimitError, GitHubAPIError

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((
        requests.exceptions.RequestException, 
        json.JSONDecodeError,
        GitHubRateLimitError,  # Retry on rate limiting
        GitHubAPIError  # Retry on other GitHub API errors
    ))
)
def execute_graphql_query(
    query_text: str, 
    token_manager: TokenManager, 
    variables: Optional[Dict[str, Any]] = None,
    cursor: Optional[str] = None, 
    limit: int = 100,
    connection_manager: Optional[ConnectionManager] = None
) -> Dict[str, Any]:
    """Execute a GraphQL query against GitHub's API with retries and token rotation.
    
    Args:
        query_text: The GraphQL query string
        token_manager: TokenManager instance for token handling
        variables: Optional variables dictionary to pass to the GraphQL query
        cursor: Optional pagination cursor (for search queries only)
        limit: Results limit (for search queries only)
        connection_manager: Optional connection manager for HTTP sessions
        
    Returns:
        Dictionary with query results
        
    Raises:
        GitHubAPIError: If the API returns an error.
        GitHubRateLimitError: If rate limit is exceeded.
    """
    # Get a valid token from the TokenManager
    token = token_manager.get_token()
    
    if not token:
        logger.error("No valid token available")
        raise Exception("No valid token available to execute query")
    
    # Prepare the query variables
    query_variables = variables if variables is not None else {}
    
    # If this is a standard search query (using our GITHUB_SEARCH_QUERY template)
    if not variables:
        query_variables = {
            "query": query_text,  # This is the actual search query string (e.g., "stars:>10000")
            "limit": limit
        }
        
        if cursor:
            query_variables["cursor"] = cursor
            
        # Prepare the full GraphQL request using our standard query
        payload = {
            "query": GITHUB_SEARCH_QUERY,  # Use the standard GraphQL query template
            "variables": query_variables
        }
    else:
        # This is a custom GraphQL query with its own variables
        payload = {
            "query": query_text,  # Use the provided query directly
            "variables": query_variables
        }
    
    # Get a session from the connection_manager
    if connection_manager:
        session = connection_manager.get_session(token)
    else:
        # Use the token_manager's connection_manager or create a new one
        if token_manager and token_manager.connection_manager:
            session = token_manager.connection_manager.get_session(token)
        else:
            # Create a new connection manager if none provided
            conn_mgr = ConnectionManager()
            session = conn_mgr.get_session(token)
        
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    try:
        # Execute the query
        response = session.post(GITHUB_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        
        # Update token usage with response headers
        token_manager.update_token_usage(token, response)
        
        # Check for GraphQL errors
        if "errors" in result:
            error_messages = [error.get("message", "Unknown error") for error in result["errors"]]
            
            # Check for rate limiting or token issues
            for error in error_messages:
                if "rate limit" in error.lower():
                    # Mark token as depleted on rate limit
                    token_manager.mark_token_depleted(token)
                    logger.warning(f"Rate limit reached for token. Marked as depleted.")
                elif "authorization" in error.lower() or "authenticate" in error.lower():
                    # Remove invalid token
                    token_manager.remove_invalid_token(token)
                    logger.warning(f"Token authentication failed. Removed invalid token.")
            
            # Log the errors
            logger.error(f"GraphQL errors: {', '.join(error_messages)}")
            
            # Update token usage to reflect the error
            token_manager.update_token_usage(token, success=False)
            
            # Check if this is a rate limit error
            if any("rate limit" in err.lower() for err in error_messages):
                # Import and raise the standardized error class
                from src.api.github_client_parallel import GitHubRateLimitError
                raise GitHubRateLimitError(f"GraphQL rate limit exceeded: {', '.join(error_messages)}")
            else:
                # Import and raise the standardized error class for other errors
                from src.api.github_client_parallel import GitHubAPIError
                raise GitHubAPIError(f"GraphQL errors: {', '.join(error_messages)}")
        
        # Mark successful request
        token_manager.update_token_usage(token, success=True)
        return result
        
    except requests.exceptions.RequestException as e:
        # Handle request-related errors
        logger.error(f"Request error: {str(e)}")
        
        # Update token usage to reflect the error
        token_manager.update_token_usage(token, success=False)
        
        # Re-raise to trigger retry
        raise
        
    except json.JSONDecodeError as e:
        # Handle JSON parsing errors
        logger.error(f"JSON decode error: {str(e)}")
        
        # Update token usage to reflect the error
        token_manager.update_token_usage(token, success=False)
        
        # Re-raise to trigger retry
        raise
        
    except Exception as e:
        # Handle other errors
        logger.error(f"Unexpected error: {str(e)}")
        
        # Update token usage to reflect the error
        token_manager.update_token_usage(token, success=False)
        
        # Re-raise without retry
        raise

def process_repository_data(data: Dict[str, Any], cache_manager: CacheManager) -> Tuple[List[Dict[str, Any]], int]:
    """Process repository data from GitHub API response with efficient duplicate detection.
    
    Args:
        data: Query result data from GitHub API
        cache_manager: Cache manager for deduplication
        
    Returns:
        Tuple containing:
            - List of unique repository data
            - Count of duplicates found
    """
    if not data or "data" not in data:
        logger.warning("Invalid data format received")
        return [], 0
        
    # Extract the search results
    try:
        search_data = data["data"]["search"]
        edges = search_data.get("edges", [])
    except (KeyError, AttributeError) as e:
        logger.error(f"Error extracting search data: {e}")
        return [], 0
        
    unique_repos = []
    duplicate_count = 0
    
    # Process each repository
    for edge in edges:
        try:
            repo = edge["node"]
            repo_id = str(repo["databaseId"])
            
            # Check if this is a duplicate (safely handle possible None cache_manager)
            if cache_manager and cache_manager.is_duplicate_repository(repo_id):
                duplicate_count += 1
                continue
                
            # Mark as seen (safely handle possible None cache_manager)
            if cache_manager:
                cache_manager.mark_repository_seen(repo_id)
            
            # Add to results
            unique_repos.append(repo)
            
        except KeyError as e:
            logger.warning(f"Missing expected field in repository data: {e}")
            continue
            
    logger.debug(f"Processed {len(edges)} repos, found {len(unique_repos)} unique and {duplicate_count} duplicates")
    return unique_repos, duplicate_count

def generate_cache_key(query_text: str, cursor: Optional[str] = None, limit: int = 100) -> str:
    """Generate a unique cache key for a query.
    
    Args:
        query_text: The query text
        cursor: Pagination cursor
        limit: Batch size limit
        
    Returns:
        A unique hash to use as cache key
    """
    # Create a string with all parameters that affect the query result
    key_components = f"{query_text}|{cursor or 'None'}|{limit}"
    # Generate a hash using MD5 (not for security, just for uniqueness)
    return hashlib.md5(key_components.encode()).hexdigest()