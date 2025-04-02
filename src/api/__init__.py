"""
GitHub API interface package for the GitHub Stars Crawler.

This package provides a modular interface for interacting with GitHub's APIs,
including API clients, repository fetching, worker coordination, and token management.
"""

# Import main components for convenient access
from src.api.api_client import GitHubApiClient
from src.api.repository_fetcher import RepositoryFetcher
from src.api.worker_coordinator import WorkerCoordinator
from src.api.token_management import TokenManager
from src.api.github_exceptions import GitHubAPIError, GitHubRateLimitError