"""
Repository fetcher for the GitHub Stars Crawler.

This module provides functionality for fetching GitHub repositories
using different strategies, with proper caching, pagination, and metrics tracking.
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Set, Tuple, Union, Iterator
from pathlib import Path

from src.api.api_client import GitHubApiClient
from src.api.token_management import TokenManager
from src.utils.cache_utils import CacheManager
from src.metrics.collector.collector import MetricsCollector
from src.api.github_exceptions import GitHubRateLimitError, GitHubAPIError

# Configure logging
logger = logging.getLogger(__name__)

class RepositoryFetcher:
    """Fetches GitHub repositories using optimized strategies."""
    
    def __init__(self, token_manager: TokenManager, api_client: Optional[GitHubApiClient] = None,
                cache_manager: Optional[CacheManager] = None, metrics_collector: Optional[MetricsCollector] = None):
        """Initialize the repository fetcher.
        
        Args:
            token_manager: Manager for GitHub API tokens
            api_client: Optional GitHubApiClient instance
            cache_manager: Optional cache manager for caching results
            metrics_collector: Optional metrics collector for tracking metrics
        """
        self.token_manager = token_manager
        self.api_client = api_client or GitHubApiClient(token_manager)
        self.cache_manager = cache_manager
        self.metrics_collector = metrics_collector
        
        # Store repository ID cache for deduplication
        self._seen_repo_ids = set()
        self._seen_repos_lock = threading.RLock()
        
    def fetch_repositories(self, query: Dict[str, Any], limit: int = 100, 
                         assigned_token: Optional[str] = None) -> List[Dict[str, Any]]:
        """Fetch repositories based on a search query.
        
        Args:
            query: Search query parameters
            limit: Maximum number of repositories to fetch
            assigned_token: Optional specific token to use
            
        Returns:
            List of repository data dictionaries
        """
        query_text = self._build_query_text(query)
        repositories = []
        cursor = None
        
        # Check cache first if available
        cached_results = None
        if self.cache_manager:
            cache_key = self.cache_manager.generate_cache_key(query_text, cursor, limit)
            cached_results = self.cache_manager.get_cached_query_result(cache_key)
            
        if cached_results:
            # Use cached results
            logger.debug(f"Cache hit for query: {query_text[:50]}...")
            if self.metrics_collector:
                self.metrics_collector.record_api_request(success=True, time_taken=0.0)
            
            # Process and return cached repositories
            return self._process_repository_batch(cached_results)
            
        # Cache miss, fetch from API
        try:
            # Execute the search query
            start_time = time.time()
            result = self.api_client.search_repositories(
                query_text=query_text,
                limit=limit,
                cursor=cursor,
                assigned_token=assigned_token
            )
            request_time = time.time() - start_time
            
            # Track API request in metrics
            if self.metrics_collector:
                self.metrics_collector.record_api_request(success=True, time_taken=request_time)
            
            # Cache the result if cache manager is available
            if self.cache_manager:
                cache_key = self.cache_manager.generate_cache_key(query_text, cursor, limit)
                self.cache_manager.cache_query_result(cache_key, result)
            
            # Process repository data
            repositories = self._process_repository_batch(result)
            
            return repositories
            
        except (GitHubRateLimitError, GitHubAPIError) as e:
            # Record API error in metrics
            if self.metrics_collector:
                self.metrics_collector.record_api_request(success=False)
                
            logger.error(f"Error fetching repositories: {str(e)}")
            return []
            
    def fetch_repositories_paginated(self, query: Dict[str, Any], max_results: int = 1000, 
                                   stop_on_duplicates: bool = True, 
                                   assigned_token: Optional[str] = None) -> Iterator[Dict[str, Any]]:
        """Fetch repositories with pagination, yielding results as they're fetched.
        
        Args:
            query: Search query parameters
            max_results: Maximum number of repositories to fetch in total
            stop_on_duplicates: Whether to stop pagination if too many duplicates are found
            assigned_token: Optional specific token to use
            
        Yields:
            Repository data dictionaries as they're fetched
        """
        query_text = self._build_query_text(query)
        cursor = None
        batch_size = min(100, max_results)  # GitHub's max per page is 100
        fetched_count = 0
        duplicate_count = 0
        has_next_page = True
        
        # Continue fetching as long as there are more pages and we haven't hit our limit
        while has_next_page and fetched_count < max_results:
            # Check cache first
            cached_results = None
            if self.cache_manager:
                cache_key = self.cache_manager.generate_cache_key(query_text, cursor, batch_size)
                cached_results = self.cache_manager.get_cached_query_result(cache_key)
                
            if cached_results:
                # Process cached results
                result = cached_results
                logger.debug(f"Cache hit for query: {query_text[:50]}... (cursor: {cursor})")
                if self.metrics_collector:
                    self.metrics_collector.record_api_request(success=True, time_taken=0.0)
            else:
                # Cache miss, fetch from API
                try:
                    start_time = time.time()
                    result = self.api_client.search_repositories(
                        query_text=query_text,
                        limit=batch_size,
                        cursor=cursor,
                        assigned_token=assigned_token
                    )
                    request_time = time.time() - start_time
                    
                    # Record API request in metrics
                    if self.metrics_collector:
                        self.metrics_collector.record_api_request(success=True, time_taken=request_time)
                        
                    # Cache the result
                    if self.cache_manager:
                        cache_key = self.cache_manager.generate_cache_key(query_text, cursor, batch_size)
                        self.cache_manager.cache_query_result(cache_key, result)
                        
                except (GitHubRateLimitError, GitHubAPIError) as e:
                    # Record API error in metrics
                    if self.metrics_collector:
                        self.metrics_collector.record_api_request(success=False)
                    
                    logger.error(f"Error in paginated fetch: {str(e)}")
                    break
            
            # Process the repositories
            page_repositories = self._process_repository_batch(result)
            page_duplicate_count = sum(1 for repo in page_repositories if self._is_duplicate(repo))
            duplicate_count += page_duplicate_count
            
            # Update pagination info
            search_data = result.get("data", {}).get("search", {})
            page_info = search_data.get("pageInfo", {})
            cursor = page_info.get("endCursor")
            has_next_page = page_info.get("hasNextPage", False)
            
            # Check for too many duplicates
            duplicate_rate = page_duplicate_count / len(page_repositories) if page_repositories else 0
            if stop_on_duplicates and duplicate_rate > 0.8 and fetched_count > 100:
                logger.info(f"Stopping pagination due to high duplicate rate: {duplicate_rate:.2f}")
                break
                
            # Yield non-duplicate repositories
            for repo in page_repositories:
                if not self._is_duplicate(repo):
                    fetched_count += 1
                    yield repo
                    
                    # Stop if we've hit our limit
                    if fetched_count >= max_results:
                        break
                        
    def _process_repository_batch(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a batch of repository results from the API.
        
        Args:
            result: The API response data
            
        Returns:
            List of processed repository data dictionaries
        """
        repositories = []
        try:
            # Extract repository data from GraphQL result
            search_data = result.get("data", {}).get("search", {})
            edges = search_data.get("edges", [])
            
            for edge in edges:
                node = edge.get("node", {})
                if not node:
                    continue
                    
                # Extract repository data
                repo_data = {
                    "id": node.get("databaseId"),
                    "name_with_owner": node.get("nameWithOwner"),
                    "owner": node.get("owner", {}).get("login"),
                    "stars": node.get("stargazerCount", 0),
                    "forks": node.get("forkCount", 0),
                    "is_fork": node.get("isFork", False),
                }
                
                # Add language if available
                if node.get("primaryLanguage"):
                    repo_data["language"] = node.get("primaryLanguage", {}).get("name")
                
                repositories.append(repo_data)
                
            return repositories
            
        except Exception as e:
            logger.error(f"Error processing repository batch: {str(e)}")
            return []
            
    def _is_duplicate(self, repository: Dict[str, Any]) -> bool:
        """Check if a repository has been seen before.
        
        Args:
            repository: Repository data dictionary
            
        Returns:
            True if the repository is a duplicate
        """
        repo_id = repository.get("id")
        if not repo_id:
            return False
            
        with self._seen_repos_lock:
            if repo_id in self._seen_repo_ids:
                return True
                
            # Mark as seen
            self._seen_repo_ids.add(repo_id)
            
            # Also mark in cache manager if available
            if self.cache_manager:
                self.cache_manager.mark_repository_seen(str(repo_id))
                
            return False
            
    def _build_query_text(self, query: Dict[str, Any]) -> str:
        """Build a GitHub search query string from a query dictionary.
        
        Args:
            query: Query parameters dictionary
            
        Returns:
            GitHub search query string
        """
        # Extract the 'q' parameter if it exists
        if "q" in query:
            return query["q"]
            
        # Otherwise, build from components
        query_parts = []
        
        # Add language filter
        if "language" in query:
            query_parts.append(f"language:{query['language']}")
            
        # Add stars filter
        if "stars" in query:
            query_parts.append(f"stars:{query['stars']}")
            
        # Add creation date filter
        if "created" in query:
            query_parts.append(f"created:{query['created']}")
            
        # Add topic filter
        if "topic" in query:
            query_parts.append(f"topic:{query['topic']}")
            
        # Return joined query string or default to all repositories
        return " ".join(query_parts) if query_parts else "stars:>=10"