"""
Token management for GitHub API access.

This module handles GitHub API token management including:
- Token validation and rate limit tracking
- Token rotation and prioritization
- Token usage optimization
"""

import re
import os
import time
import logging
import heapq
import random
import threading
import concurrent.futures
import inspect  # For caller inspection
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Union, Any, Callable

import requests

import logging
from src.utils.error_handling import retry_with_backoff, log_error, format_error_context
from src.utils.connection_manager import ConnectionManager
from src.api.github_exceptions import (
    GitHubRateLimitError, GitHubAuthenticationError, 
    GitHubAPIError, TokenManagementError, TransientError
)

# Configure logging
logger = logging.getLogger(__name__)


def parse_github_tokens(token_string: str) -> List[str]:
    """Parse a string of GitHub tokens into a list.
    
    This function supports multiple formats:
    - Single token: "abc123"
    - Comma-separated list: "abc123,def456"
    - Newline-separated list: "abc123\ndef456"
    - JSON array: '["abc123", "def456"]'
    
    Args:
        token_string: String containing one or more GitHub tokens
        
    Returns:
        List of tokens as strings
    """
    if not token_string or not isinstance(token_string, str):
        return []
        
    # Check if it's a JSON array
    if token_string.strip().startswith('[') and token_string.strip().endswith(']'):
        try:
            import json
            tokens = json.loads(token_string)
            if isinstance(tokens, list):
                return [t.strip() for t in tokens if t and isinstance(t, str)]
        except (json.JSONDecodeError, TypeError):
            pass
    
    # Split by commas and/or newlines
    tokens = re.split(r'[,\n]+', token_string)
    return [t.strip() for t in tokens if t.strip()]


class TokenManager:
    """GitHub API token manager for handling rate limits and token rotation."""
    
    def __init__(
        self, 
        tokens: List[str],
        connection_manager: Optional[ConnectionManager] = None,
        min_token_usage_interval: float = 0.5,
        token_request_threshold: int = 50
    ):
        """Initialize the token manager.
        
        Args:
            tokens: List of GitHub API tokens
            connection_manager: ConnectionManager for HTTP requests
            min_token_usage_interval: Minimum time between uses of the same token in seconds
            token_request_threshold: Minimum requests remaining to consider a token healthy
        
        Raises:
            TokenManagementError: If no valid tokens are provided
        """
        if not tokens:
            error_msg = "No GitHub API tokens provided"
            log_error(logger, error_msg, level="critical", component="TokenManager")
            raise TokenManagementError(error_msg)

        self.tokens = tokens
        self.min_token_usage_interval = min_token_usage_interval
        self.token_request_threshold = token_request_threshold
        
        # Connection manager for HTTP requests
        self.connection_manager = connection_manager or ConnectionManager()
        
        # Implement fine-grained lock strategy to reduce contention
        # Use separate locks for different parts of the token manager state
        self.metadata_lock = threading.RLock()  # For token metadata
        self.queue_lock = threading.RLock()  # For token queues
        self.lock = threading.RLock()  # For operations that need all locks
        
        # Token metadata tracking
        self.token_metadata: Dict[str, Dict[str, Any]] = {}
        for token in tokens:
            self.token_metadata[token] = {
                "remaining": 5000,  # Default rate limit
                "limit": 5000,
                "reset_time": datetime.now() + timedelta(hours=1),
                "last_used": datetime.now() - timedelta(hours=1),  # Start with tokens unused
                "success_rate": 1.0,
                "consecutive_errors": 0,
                "request_velocity": 0.0,
                "total_requests": 0,
                "priority_score": 5000,
                "status": "UNKNOWN"
            }
            
        # Track invalid tokens
        self.invalid_tokens: Set[str] = set()
        
        # Token queues for available and rate-limited tokens
        self.available_tokens: List[Tuple[float, str]] = []  # Priority queue for available tokens
        self.rate_limited_tokens: Dict[str, float] = {}  # Map of rate-limited tokens to their reset times
        
        # Start background worker for token monitoring
        self.token_monitor_active = True
        self.token_monitor_thread = threading.Thread(
            target=self._token_monitor_worker,
            daemon=True,
            name="TokenMonitorWorker"
        )
        self.token_monitor_thread.start()
        
        # Instance-level tracking of validated tokens
        self.validated_tokens = set()
        
        # Log initialization
        logger.info(f"TokenManager initialized with {len(self.tokens)} tokens")
        
        # Validate all tokens during initialization
        self.token_stats = self.validate_token_auth_formats()
        logger.info(f"Token validation complete: {self.token_stats['token']} valid tokens, " + 
                  f"{self.token_stats['invalid']} invalid tokens")
    
    def _check_sample_rate_limits(self, sample_size: int = 3) -> None:
        """Check rate limits for a sample of tokens to establish baseline.
        
        Args:
            sample_size: Number of tokens to check initially
        """
        sample_tokens = random.sample(self.tokens, min(sample_size, len(self.tokens)))
        for token in sample_tokens:
            try:
                self._check_rate_limit(token)
            except Exception as e:
                logger.warning(f"Failed to check initial rate limit for token {token[:4]}...{token[-4:]}: {e}")
    
    def _token_monitor_worker(self) -> None:
        """Background worker that monitors rate-limited tokens and moves them back when ready.
        This avoids blocking the main thread with sleeps when tokens are rate-limited.
        
        Returns:
            None
        """
        logger.info("Token monitor worker started")
        
        try:
            while self.token_monitor_active:
                moved_tokens = False
                ready_tokens: List[str] = []
                
                # Check if any rate-limited tokens are ready to be moved back
                with self.queue_lock:
                    current_time = time.time()
                    
                    # Find tokens that have passed their reset time
                    for token, reset_time in list(self.rate_limited_tokens.items()):
                        if current_time >= reset_time:
                            ready_tokens.append(token)
                            self._move_to_available_queue(token)
                            del self.rate_limited_tokens[token]
                            moved_tokens = True
                            logger.debug(f"Token {token[:4]}...{token[-4:]} is now available")
                
                # If we moved tokens, update rate limits for better priority calculation
                # Only if we have at least one token ready
                if moved_tokens and ready_tokens:
                    try:
                        # Use a random sample of tokens if we have many to avoid overloading API
                        tokens_to_check = ready_tokens
                        if len(ready_tokens) > 3:
                            tokens_to_check = random.sample(ready_tokens, 3)
                            
                        # Check rate limits in parallel with a thread pool
                        with concurrent.futures.ThreadPoolExecutor(max_workers=min(3, len(tokens_to_check))) as executor:
                            futures = {
                                executor.submit(self._check_rate_limit, token): token 
                                for token in tokens_to_check
                            }
                            
                            for future in concurrent.futures.as_completed(futures):
                                token = futures[future]
                                try:
                                    future.result()
                                except Exception as e:
                                    logger.warning(f"Error checking rate limit for reactivated token {token[:4]}...{token[-4:]}: {e}")
                                    
                    except Exception as e:
                        logger.error(f"Error updating rate limits for reactivated tokens: {e}")
                
                # Adaptive sleep - longer when we have fewer tokens or no activity
                sleep_time = 0.5
                if len(self.rate_limited_tokens) < 3:
                    sleep_time = 2.0  # Sleep longer when we have few rate-limited tokens
                elif not moved_tokens:
                    sleep_time = 1.0  # Sleep longer when nothing happened
                    
                time.sleep(sleep_time)
                
        except Exception as e:
            logger.error(f"Token monitor worker encountered an error: {e}", exc_info=True)
        finally:
            logger.info("Token monitor worker stopped")
    
    def _rebuild_priority_queue(self) -> None:
        """Rebuild the available tokens queue based on current metadata."""
        with self.queue_lock:
            # Clear the existing queue
            self.available_tokens = []
            
            for token, metadata in self.token_metadata.items():
                if token in self.invalid_tokens:
                    continue
                    
                # Check if token is rate limited
                if metadata["remaining"] <= 0:
                    # Calculate when the token will be available again
                    reset_time = metadata["reset_time"].timestamp()
                    # Add to rate limited tokens if not already there
                    if token not in self.rate_limited_tokens:
                        logger.debug(f"Token {token[:4]}...{token[-4:]} added to rate-limited queue")
                        self.rate_limited_tokens[token] = reset_time
                    continue
                
                # Only add tokens with remaining capacity to the available queue
                self._move_to_available_queue(token)
    
    def _move_to_available_queue(self, token: str) -> None:
        """Move a token to the available queue with proper priority.
        
        Args:
            token: Token to move to the available queue
        """
        if token in self.invalid_tokens or token not in self.token_metadata:
            return
            
        metadata = self.token_metadata[token]
        
        # Calculate priority score
        time_to_reset = (metadata["reset_time"] - datetime.now()).total_seconds()
        time_factor = max(1, min(3600, time_to_reset)) / 3600  # Normalize to 0-1 range
        success_factor = metadata["success_rate"]
        base_score = metadata["remaining"]
        
        # Boost tokens with longer time to reset
        priority_score = base_score * time_factor * success_factor
        
        # Higher score = higher priority
        metadata["priority_score"] = priority_score
        
        # Use negative score for max-heap behavior (we want highest score first)
        heapq.heappush(self.available_tokens, (-priority_score, token))
    
    def update_rate_limits(self, check_all: bool = False, force: bool = False) -> Dict[str, Any]:
        """Update rate limits for all tokens or a sample of tokens.
        
        Args:
            check_all: Whether to check all tokens or just a sample
            force: Whether to force check even recently checked tokens
            
        Returns:
            Dict: Summary of token statuses after update
        
        Raises:
            TokenManagementError: If all tokens are invalid or unreachable
        """
        check_tokens: List[str] = []
        
        with self.lock:
            valid_tokens = [t for t in self.tokens if t not in self.invalid_tokens]
            
            if not valid_tokens:
                error_msg = "No valid tokens available to check rate limits"
                log_error(logger, error_msg, level="critical", component="TokenManager", operation="update_rate_limits")
                raise TokenManagementError(error_msg)
            
            if check_all:
                check_tokens = valid_tokens
            else:
                # Check about 25% of tokens, at least 3, or all if we have few
                sample_size = max(3, len(valid_tokens) // 4)
                sample_size = min(sample_size, len(valid_tokens))
                check_tokens = random.sample(valid_tokens, sample_size)
        
        # Track tokens with errors
        error_tokens: List[str] = []
        
        # Check rate limits for selected tokens
        for token in check_tokens:
            try:
                self._check_rate_limit(token, force=force)
            except Exception as e:
                error_context = format_error_context(e, operation="check_rate_limit", token_prefix=token[:4])
                log_error(
                    logger, 
                    f"Failed to check rate limit for token {token[:4]}...{token[-4:]}", 
                    exception=e, 
                    level="error", 
                    **error_context
                )
                error_tokens.append(token)
        
        # Rebuild the priority queue
        self._rebuild_priority_queue()
        
        # Get summary after update
        with self.lock:
            total_remaining = sum(m["remaining"] for t, m in self.token_metadata.items() 
                                if t not in self.invalid_tokens)
            total_tokens = len(self.tokens) - len(self.invalid_tokens)
            avg_remaining = total_remaining / max(1, total_tokens)
            
            return {
                "total_tokens": total_tokens,
                "valid_tokens": total_tokens - len(error_tokens),
                "total_remaining": total_remaining,
                "avg_remaining": avg_remaining,
                "checked_tokens": len(check_tokens),
                "error_tokens": len(error_tokens)
            }
    
    def get_token(self, high_priority: bool = False) -> str:
        """Get the best available token based on rate limits.
        
        Args:
            high_priority: Whether this is a high priority request
            
        Returns:
            Best available token
            
        Raises:
            GitHubRateLimitError: If no tokens with remaining requests are available
        """
        with self.queue_lock:
            # If we have no available tokens, check if we need to rebuild
            if not self.available_tokens:
                self._rebuild_priority_queue()
                
                # If still no available tokens after rebuild, check if we're just waiting
                if not self.available_tokens:
                    if self.rate_limited_tokens:
                        # We have tokens, but they're all rate limited
                        # Calculate the soonest one will be available
                        next_available = min(self.rate_limited_tokens.values())
                        wait_time = max(0, next_available - time.time())
                        
                        if high_priority:
                            # For high priority, take a rate-limited token if it's close to resetting
                            if wait_time < 10:  # If we're within 10 seconds of a reset
                                # Find the token that will reset soonest
                                next_token = min(self.rate_limited_tokens.items(), key=lambda x: x[1])[0]
                                logger.info(f"Taking soon-to-reset token {next_token[:4]}...{next_token[-4:]} for high priority request")
                                return next_token
                                
                        error_msg = f"All tokens rate limited, next available in {wait_time:.1f} seconds"
                    else:
                        error_msg = "No valid tokens available"
                        
                    log_error(logger, error_msg, level="warning" if self.rate_limited_tokens else "critical", 
                             component="TokenManager", operation="get_token")
                    raise GitHubRateLimitError(error_msg)
            
            # Get the best available token
            best_score, token = heapq.heappop(self.available_tokens)
            best_score = -best_score  # Convert back to positive score
            metadata = self.token_metadata[token]
            
            # Update last_used time
            metadata["last_used"] = datetime.now()
            
            # Double-check that token still has requests remaining
            # (might have changed since we last checked)
            if metadata["remaining"] <= 0:
                # This token is actually depleted, move it to rate-limited and try again
                reset_time = metadata["reset_time"].timestamp()
                self.rate_limited_tokens[token] = reset_time
                
                # Try to get another token
                if self.available_tokens:
                    return self.get_token(high_priority)
                else:
                    error_msg = "All tokens depleted"
                    log_error(logger, error_msg, level="critical", component="TokenManager", operation="get_token")
                    raise GitHubRateLimitError(error_msg)
            
            # Check if the token is being used too frequently
            time_since_last_use = (datetime.now() - metadata["last_used"]).total_seconds()
            if time_since_last_use < self.min_token_usage_interval and not high_priority:
                # Token used too recently, put back with temporarily reduced score
                temp_score = best_score * 0.8
                heapq.heappush(self.available_tokens, (-temp_score, token))
                
                # Try the next best token
                if self.available_tokens:
                    return self.get_token(high_priority)
                else:
                    # If no other tokens available, use this one anyway
                    heapq.heappush(self.available_tokens, (-best_score, token))
            else:
                # Put token back with its normal priority
                self._move_to_available_queue(token)
            
            return token
            
    def update_token_usage(self, token: str, response: Optional[requests.Response] = None,
                           remaining: Optional[int] = None, reset_time: Optional[Union[int, datetime]] = None, 
                           success: bool = True) -> bool:
        """Update token usage statistics and parse rate limit headers.
        
        Args:
            token: GitHub API token
            response: API response with rate limit headers (if available)
            remaining: Remaining rate limit (used if response not provided)
            reset_time: Reset time for rate limit (used if response not provided)
            success: Whether the operation was successful
            
        Returns:
            bool: True if update was successful, False if token wasn't found
            
        Raises:
            TokenManagementError: If there's an error processing the headers
        """
        # Use metadata_lock instead of global lock to reduce contention
        with self.metadata_lock:
            if token not in self.token_metadata:
                error_msg = f"Tried to update unknown token: {token[:4]}...{token[-4:]}"
                log_error(logger, error_msg, level="warning", component="TokenManager", operation="update_token_usage")
                return False
                
            metadata = self.token_metadata[token]
            current_time = datetime.now()
            
            # Track old remaining value for velocity calculations
            old_remaining = metadata["remaining"]
            old_reset_time = metadata["reset_time"]
            
            # Extract rate limit headers from response if provided
            if response is not None:
                try:
                    # Parse headers outside of lock to minimize lock time
                    parsed_headers = {}
                    
                    # GitHub rate limit headers
                    remaining_header = response.headers.get("X-RateLimit-Remaining")
                    reset_header = response.headers.get("X-RateLimit-Reset")
                    limit_header = response.headers.get("X-RateLimit-Limit")
                    
                    # Pre-parse headers
                    if remaining_header is not None:
                        parsed_headers["remaining"] = int(remaining_header)
                        
                    if limit_header is not None:
                        parsed_headers["limit"] = int(limit_header)
                        
                    if reset_header is not None:
                        parsed_headers["reset_time"] = datetime.fromtimestamp(int(reset_header))
                    
                    # Apply all updates at once to minimize lock time
                    if "remaining" in parsed_headers:
                        metadata["remaining"] = parsed_headers["remaining"]
                    if "limit" in parsed_headers:
                        metadata["limit"] = parsed_headers["limit"]
                    if "reset_time" in parsed_headers:
                        metadata["reset_time"] = parsed_headers["reset_time"]
                        
                except Exception as e:
                    error_context = format_error_context(e, operation="parse_headers", token_prefix=token[:4])
                    log_error(
                        logger, 
                        f"Error parsing rate limit headers for {token[:4]}...{token[-4:]}", 
                        exception=e, 
                        level="error", 
                        **error_context
                    )
                    raise TokenManagementError(f"Failed to parse rate limit headers: {str(e)}") from e
            else:
                # Use provided values if no response
                if remaining is not None:
                    metadata["remaining"] = remaining
                    
                if reset_time is not None:
                    # Handle both timestamp and datetime
                    if isinstance(reset_time, int):
                        metadata["reset_time"] = datetime.fromtimestamp(reset_time)
                    else:
                        metadata["reset_time"] = reset_time
            
            # Update success/error metrics
            if success:
                metadata["consecutive_errors"] = 0
                metadata["success_rate"] = min(1.0, metadata["success_rate"] * 0.95 + 0.05)
            else:
                metadata["consecutive_errors"] += 1
                metadata["success_rate"] = max(0.1, metadata["success_rate"] * 0.8)
            
            # Track total requests
            metadata["total_requests"] += 1
            
            # Calculate token velocity (requests per minute)
            time_diff = (current_time - old_reset_time).total_seconds()
            if time_diff > 0 and old_remaining > metadata["remaining"]:
                # We've used (old - current) requests in time_diff seconds
                requests_used = old_remaining - metadata["remaining"]
                velocity = (requests_used / time_diff) * 60  # Convert to per minute
                
                # Weighted moving average for velocity
                if metadata["request_velocity"] > 0:
                    metadata["request_velocity"] = (
                        metadata["request_velocity"] * 0.7 + velocity * 0.3
                    )
                else:
                    metadata["request_velocity"] = velocity
            
            # Update token status (must happen under lock)
            self._update_token_status(token)
        
        # Only rebuild queue if there are significant changes
        # Use queue_lock specifically for this operation
        if not success or metadata["remaining"] <= self.token_request_threshold:
            with self.queue_lock:
                self._rebuild_priority_queue()
                
        return True
    
    def _update_token_status(self, token: str) -> None:
        """Update token status based on remaining requests and reset time.
        
        Args:
            token: GitHub API token
        """
        with self.lock:
            if token not in self.token_metadata:
                return
                
            metadata = self.token_metadata[token]
            
            # Calculate minutes until reset
            now = datetime.now()
            time_to_reset = (metadata["reset_time"] - now).total_seconds() / 60
            limit = metadata["limit"]
            remaining = metadata["remaining"]
            
            # Edge case for invalid reset time
            if time_to_reset < 0:
                metadata["status"] = "UNKNOWN"
                return
            
            # Determine status based on remaining percentage
            remaining_pct = remaining / limit if limit > 0 else 0
            
            if remaining <= 0:
                metadata["status"] = "DEPLETED"
            elif remaining_pct < 0.05 or remaining < 10:
                metadata["status"] = "CRITICAL"
            elif remaining_pct < 0.15 or remaining < 50:
                metadata["status"] = "LOW"
            elif remaining_pct < 0.3:
                metadata["status"] = "AT_RISK"
            else:
                metadata["status"] = "HEALTHY"
    
    def _check_rate_limit(self, token: str, force: bool = False) -> Dict[str, Any]:
        """Check the rate limit for a token by making a rate_limit API call.
        
        Args:
            token: GitHub API token
            force: Whether to force check even if recently checked
            
        Returns:
            Dict with token metadata
            
        Raises:
            GitHubAuthenticationError: If authentication fails
            GitHubAPIError: If API returns an error
            TokenManagementError: If there's an error processing the rate limit
        """
        with self.lock:
            metadata = self.token_metadata[token]
            
            # Skip if recently checked and not forced
            last_checked = metadata.get("last_checked")
            if not force and last_checked and (datetime.now() - last_checked).total_seconds() < 60:
                return metadata
            
            # Use Bearer token format as per GitHub API v4 requirements
            if token in self.invalid_tokens:
                error_msg = f"Token {token[:4]}...{token[-4:]} is already marked as invalid"
                log_error(logger, error_msg, level="error", component="TokenManager", operation="check_rate_limit")
                raise GitHubAuthenticationError(error_msg)
                
            headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            # Track when we checked the rate limit
            metadata["last_checked"] = datetime.now()
            
            try:
                # Get session from connection manager
                session = self.connection_manager.get_session(token)
                response = session.get(
                    "https://api.github.com/rate_limit", 
                    headers=headers, 
                    timeout=10
                )
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        rate_data = data.get("resources", {}).get("core", {})
                        
                        remaining = rate_data.get("remaining")
                        limit = rate_data.get("limit")
                        reset_time = rate_data.get("reset")
                        
                        if None in (remaining, limit, reset_time):
                            error_msg = f"Incomplete rate limit data for token {token[:4]}...{token[-4:]}"
                            log_error(logger, error_msg, level="warning", component="TokenManager", operation="check_rate_limit")
                            raise TokenManagementError(error_msg)
                        
                        # Update token metadata
                        metadata["remaining"] = remaining
                        metadata["limit"] = limit
                        metadata["reset_time"] = datetime.fromtimestamp(reset_time)
                        
                        # Reset error counter on successful check
                        metadata["consecutive_errors"] = 0
                        metadata["success_rate"] = min(1.0, metadata["success_rate"] * 0.95 + 0.05)
                        
                        # Update token status
                        self._update_token_status(token)
                        
                        # Calculate minutes until reset
                        now = datetime.now()
                        time_to_reset = (metadata["reset_time"] - now).total_seconds() / 60
                        
                        # Log token status based on remaining requests
                        if remaining < limit * 0.3 or metadata["consecutive_errors"] > 0:
                            logger.info(
                                f"Token {token[:4]}...{token[-4:]} has {remaining}/{limit} "
                                f"requests remaining, resets in {time_to_reset:.1f}m, "
                                f"velocity: {metadata['request_velocity']:.1f} req/min"
                            )
                        else:
                            logger.debug(
                                f"Token {token[:4]}...{token[-4:]} has {remaining}/{limit} "
                                f"requests remaining, resets in {time_to_reset:.1f}m, "
                                f"velocity: {metadata['request_velocity']:.1f} req/min"
                            )
                        
                        return metadata
                        
                    except Exception as e:
                        error_context = format_error_context(e, operation="parse_rate_limit", token_prefix=token[:4])
                        log_error(
                            logger, 
                            f"Error parsing rate limit response for {token[:4]}...{token[-4:]}", 
                            exception=e, 
                            level="error", 
                            **error_context
                        )
                        raise TokenManagementError(f"Failed to parse rate limit response: {str(e)}") from e
                
                elif response.status_code == 401:
                    error_msg = f"Authentication failed for token {token[:4]}...{token[-4:]}"
                    log_error(logger, error_msg, level="error", component="TokenManager", operation="check_rate_limit")
                    self.remove_invalid_token(token)
                    raise GitHubAuthenticationError(error_msg)
                    
                else:
                    error_msg = (
                        f"Failed to check rate limit for token {token[:4]}...{token[-4:]}, "
                        f"status code: {response.status_code}"
                    )
                    log_error(logger, error_msg, level="error", component="TokenManager", operation="check_rate_limit")
                    raise GitHubAPIError(error_msg)
                    
            except (requests.exceptions.RequestException, ConnectionError) as e:
                error_context = format_error_context(e, operation="request", token_prefix=token[:4])
                log_error(
                    logger, 
                    f"Network error checking rate limit for token {token[:4]}...{token[-4:]}", 
                    exception=e, 
                    level="error", 
                    **error_context
                )
                
                # Update error metrics
                metadata["consecutive_errors"] += 1
                metadata["success_rate"] = max(0.1, metadata["success_rate"] * 0.9)
                
                raise TokenManagementError(f"Network error checking rate limit: {str(e)}") from e
                
    # Note: All external code now calls get_token directly
    
    def mark_token_depleted(self, token: str, reset_time: Optional[Union[int, datetime]] = None) -> bool:
        """Mark a token as depleted (0 remaining requests).
        
        Args:
            token: GitHub API token
            reset_time: Optional reset time to set
            
        Returns:
            bool: True if token was marked depleted, False if token not found
        """
        with self.lock:
            if token not in self.token_metadata:
                return False
                
            metadata = self.token_metadata[token]
            metadata["remaining"] = 0
            metadata["status"] = "DEPLETED"
            
            # Update reset time if provided
            if reset_time is not None:
                try:
                    if isinstance(reset_time, int):
                        metadata["reset_time"] = datetime.fromtimestamp(reset_time)
                    else:
                        metadata["reset_time"] = reset_time
                except Exception as e:
                    # Log but don't fail the operation
                    error_context = format_error_context(e, operation="parse_reset_time", token_prefix=token[:4])
                    log_error(
                        logger, 
                        f"Error parsing reset time for {token[:4]}...{token[-4:]}", 
                        exception=e, 
                        level="warning", 
                        **error_context
                    )
            
            # Rebuild priority queue
            self._rebuild_priority_queue()
            
            return True
    
    def remove_invalid_token(self, token: str) -> bool:
        """Remove an invalid token from the pool.
        
        Args:
            token: GitHub API token to remove
            
        Returns:
            bool: True if token was removed, False if token not found
        """
        with self.lock:
            if token not in self.token_metadata:
                return False
                
            # Mark as invalid
            self.invalid_tokens.add(token)
            
            # Rebuild priority queue
            self._rebuild_priority_queue()
            
            logger.warning(f"Removed invalid token {token[:4]}...{token[-4:]} from pool")
            return True
    
    # The get_auth_format and get_auth_header methods were removed as they were never used
    # and were leftover from a backward compatibility layer for different token authentication formats.
    # Always using Bearer token format now as per GitHub API v4 requirements.
    
    def get_token_stats(self) -> Dict[str, Any]:
        """Get statistics about all tokens.
        
        Returns:
            Dict with token statistics
        """
        with self.lock:
            stats: Dict[str, Any] = {}
            
            # Calculate statistics for each token
            for token, metadata in self.token_metadata.items():
                if token in self.invalid_tokens:
                    continue
                    
                time_to_reset = (metadata["reset_time"] - datetime.now()).total_seconds() / 60
                time_to_reset = max(0, time_to_reset)
                
                stats[token] = {
                    "remaining": metadata["remaining"],
                    "limit": metadata["limit"],
                    "reset_time": metadata["reset_time"].isoformat(),
                    "minutes_to_reset": time_to_reset,
                    "success_rate": metadata["success_rate"],
                    "velocity": metadata["request_velocity"],
                    "priority_score": metadata["priority_score"],
                    "status": metadata["status"],
                    "consecutive_errors": metadata["consecutive_errors"]
                }
            
            # Add summary statistics
            total_tokens = len(self.tokens) - len(self.invalid_tokens)
            total_remaining = sum(m["remaining"] for t, m in self.token_metadata.items() 
                               if t not in self.invalid_tokens)
            
            stats["_summary"] = {
                "total_tokens": total_tokens,
                "valid_tokens": total_tokens,
                "invalid_tokens": len(self.invalid_tokens),
                "total_remaining": total_remaining,
                "avg_remaining": total_remaining / max(1, total_tokens),
                
                # Token status counts
                "healthy_tokens": sum(1 for t, m in stats.items() 
                                     if t != "_summary" and m.get("status") == "HEALTHY"),
                "at_risk_tokens": sum(1 for t, m in stats.items() 
                                     if t != "_summary" and m.get("status") == "AT_RISK"),
                "low_tokens": sum(1 for t, m in stats.items() 
                                 if t != "_summary" and m.get("status") == "LOW"),
                "critical_tokens": sum(1 for t, m in stats.items() 
                                      if t != "_summary" and m.get("status") == "CRITICAL"),
                "depleted_tokens": sum(1 for t, m in stats.items() 
                                      if t != "_summary" and m.get("status") == "DEPLETED"),
            }
            
            return stats
            
    def get_total_remaining_requests(self) -> int:
        """Get the total number of remaining requests across all tokens.
        
        Returns:
            int: Total remaining requests
        """
        with self.lock:
            stats = self.get_token_stats()
            return stats["_summary"]["total_remaining"]
            
    def validate_token_auth_formats(self, tokens_to_validate: List[str] = None, log_results: bool = True) -> Dict[str, int]:
        """Validate tokens to identify invalid ones.
        
        Args:
            tokens_to_validate: Optional list of specific tokens to validate (defaults to all self.tokens)
            log_results: Whether to log validation results after completion
            
        Returns:
            Dict with counts of tokens by validity
        
        Raises:
            TokenManagementError: If all tokens are invalid
        """
        # Use provided tokens or fall back to all tokens
        tokens_to_check = tokens_to_validate or self.tokens
        results = {"token": 0, "invalid": 0, "total": len(tokens_to_check)}
        
        for token in tokens_to_check:
            # Skip if already marked as invalid
            if token in self.invalid_tokens:
                results["invalid"] += 1
                continue
            
            headers = {
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            try:
                # Get session from connection manager
                session = self.connection_manager.get_session(token)
                
                # Use rate_limit endpoint as a simple validation
                response = session.get("https://api.github.com/rate_limit", 
                                    headers=headers, timeout=10)
                
                # If successful, token is valid
                if response.status_code == 200:
                    token_valid = True
                    logger.debug(f"Token {token[:4]}...{token[-4:]} validated successfully")
                    results["token"] += 1
                else:
                    token_valid = False
                    logger.warning(f"Token {token[:4]}...{token[-4:]} failed validation: {response.status_code}")
            
            except Exception as e:
                error_context = format_error_context(e, operation="validate_token", token_prefix=token[:4])
                log_error(
                    logger, 
                    f"Error testing token {token[:4]}...{token[-4:]}", 
                    exception=e, 
                    level="warning", 
                    **error_context
                )
                token_valid = False
            
            # Handle token results
            if not token_valid:
                # Mark token as invalid and remove completely
                logger.error(f"Token {token[:4]}...{token[-4:]} is invalid. Removing from token pool.")
                self.remove_invalid_token(token)
                results["invalid"] += 1
        
        # After validation, check if we have any valid tokens left
        if results["token"] == 0:
            error_msg = f"All {len(self.tokens)} tokens are invalid."
            log_error(logger, error_msg, level="critical", component="TokenManager", operation="validate_token_auth_formats")
            raise TokenManagementError(error_msg)
            
        # Rebuild priority queue after validation
        self._rebuild_priority_queue()
            
        return results
        
    def cleanup(self) -> None:
        """Clean up resources used by the token manager."""
        self.token_monitor_active = False
        
        # Wait for background thread to finish
        if self.token_monitor_thread and self.token_monitor_thread.is_alive():
            logger.debug(f"Waiting for token monitor thread to exit...")
            self.token_monitor_thread.join(timeout=2.0)
        
        logger.info(f"TokenManager cleaned up")