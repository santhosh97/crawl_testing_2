"""
GitHub API client for interacting with GitHub's GraphQL and REST APIs.

This module provides a unified client for making requests to GitHub's APIs,
handling authentication, rate limiting, retries, and error handling.
"""
import logging
import time
import json
from typing import Dict, List, Optional, Any, Union

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.utils.connection_manager import ConnectionManager
from src.api.token_management import TokenManager
from src.api.github_exceptions import GitHubRateLimitError, GitHubAPIError

# Configure logging
logger = logging.getLogger(__name__)

# GitHub API endpoints
GITHUB_GRAPHQL_API_URL = "https://api.github.com/graphql"
GITHUB_REST_API_URL = "https://api.github.com"

# Standard GraphQL query templates
REPOSITORY_SEARCH_QUERY = """
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

class GitHubApiClient:
    """Client for interacting with GitHub's APIs with rate limit handling and retries."""
    
    def __init__(self, token_manager: TokenManager, connection_manager: Optional[ConnectionManager] = None):
        """Initialize the API client.
        
        Args:
            token_manager: Manager for GitHub API tokens
            connection_manager: Optional manager for HTTP connections
        """
        self.token_manager = token_manager
        self.connection_manager = connection_manager or ConnectionManager()
        
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((requests.exceptions.RequestException, GitHubRateLimitError)),
        reraise=True,
    )
    def execute_graphql_query(self, query: str, variables: Optional[Dict[str, Any]] = None, 
                             assigned_token: Optional[str] = None, high_priority: bool = False) -> Dict[str, Any]:
        """Execute a GraphQL query against GitHub's API with retries and token rotation.
        
        Args:
            query: The GraphQL query string.
            variables: Optional variables for the query.
            assigned_token: Optional specific token to use.
            high_priority: Whether to use a high-priority token.
            
        Returns:
            The JSON response from the API.
            
        Raises:
            GitHubAPIError: If the API returns an error.
            GitHubRateLimitError: If rate limit is exceeded for all tokens.
        """
        # Get the token to use
        if assigned_token:
            token = assigned_token
        else:
            # Get token from token manager
            token = self.token_manager.get_token(high_priority=high_priority)
            
        if not token:
            raise GitHubRateLimitError("No available tokens")
            
        # Set up headers with token authentication
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        
        # Prepare request payload
        payload = {
            "query": query,
            "variables": variables or {},
        }
        
        # Execute the request
        start_time = time.time()
        
        try:
            session = self.connection_manager.get_session()
            response = session.post(
                GITHUB_GRAPHQL_API_URL,
                headers=headers,
                json=payload,
                timeout=30,
            )
            
            # Calculate request duration
            request_duration = time.time() - start_time
            
            # Update token usage statistics
            self.token_manager.record_usage(token, request_duration)
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Parse the response
            result = response.json()
            
            # Check for GraphQL errors
            if "errors" in result:
                # Check if it's a rate limit error
                for error in result.get("errors", []):
                    message = error.get("message", "").lower()
                    if "rate limit" in message or "rate_limit" in message:
                        # Mark this token as rate limited
                        self.token_manager.mark_rate_limited(token)
                        raise GitHubRateLimitError(f"Rate limit exceeded: {message}")
                        
                # Handle other GraphQL errors
                error_message = "; ".join([error.get("message", "Unknown error") 
                                         for error in result.get("errors", [])])
                raise GitHubAPIError(f"GraphQL error: {error_message}")
                
            return result
            
        except requests.exceptions.RequestException as e:
            # Log connection errors
            logger.error(f"Request error: {str(e)}")
            raise
            
    def search_repositories(self, query_text: str, limit: int = 100, cursor: Optional[str] = None,
                           assigned_token: Optional[str] = None) -> Dict[str, Any]:
        """Search for repositories using GitHub's GraphQL API.
        
        Args:
            query_text: The search query text
            limit: Maximum number of results to return
            cursor: Optional pagination cursor
            assigned_token: Optional specific token to use
            
        Returns:
            The search results from the API
        """
        variables = {
            "query": query_text,
            "limit": limit,
            "cursor": cursor,
        }
        
        return self.execute_graphql_query(
            query=REPOSITORY_SEARCH_QUERY,
            variables=variables,
            assigned_token=assigned_token
        )
        
    def execute_rest_api_call(self, endpoint: str, method: str = "GET", data: Optional[Dict[str, Any]] = None,
                             params: Optional[Dict[str, Any]] = None, assigned_token: Optional[str] = None) -> Dict[str, Any]:
        """Execute a REST API call to GitHub.
        
        Args:
            endpoint: API endpoint path (e.g., "/repos/owner/repo")
            method: HTTP method to use
            data: Optional request body for POST/PUT
            params: Optional URL parameters
            assigned_token: Optional specific token to use
            
        Returns:
            The JSON response from the API
        """
        # Get the token to use
        if assigned_token:
            token = assigned_token
        else:
            # Get token from token manager
            token = self.token_manager.get_token()
            
        if not token:
            raise GitHubRateLimitError("No available tokens")
            
        # Set up headers with token authentication
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json",
        }
        
        # Prepare the URL
        url = f"{GITHUB_REST_API_URL}{endpoint}"
        
        # Execute the request
        start_time = time.time()
        
        try:
            session = self.connection_manager.get_session()
            response = session.request(
                method,
                url,
                headers=headers,
                json=data,
                params=params,
                timeout=30,
            )
            
            # Calculate request duration
            request_duration = time.time() - start_time
            
            # Update token usage statistics
            self.token_manager.record_usage(token, request_duration)
            
            # Check for HTTP errors
            response.raise_for_status()
            
            # Return the JSON response if available
            if response.content:
                return response.json()
            return {}
            
        except requests.exceptions.RequestException as e:
            # Check for rate limit errors
            if hasattr(e, "response") and e.response is not None:
                if e.response.status_code == 403:
                    # Check if it's a rate limit error
                    response_body = e.response.json() if e.response.content else {}
                    if "rate limit" in response_body.get("message", "").lower():
                        # Mark this token as rate limited
                        self.token_manager.mark_rate_limited(token)
                        raise GitHubRateLimitError("Rate limit exceeded")
            
            # Log and re-raise other errors
            logger.error(f"REST API error: {str(e)}")
            raise GitHubAPIError(f"REST API error: {str(e)}")