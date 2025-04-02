"""
Error handling utilities for GitHub Stars Crawler.

This module provides utility functions and classes for consistent error handling
across the codebase.
"""

import time
import logging
import traceback
from datetime import datetime
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Union, Generic

from src.api.github_exceptions import TransientError

# Configure logging
logger = logging.getLogger(__name__)

# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')

def retry_with_backoff(
    max_retries: int = 3, 
    initial_delay: float = 1.0, 
    backoff_factor: float = 2.0,
    exceptions: tuple = (TransientError,)
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator for retrying a function with exponential backoff on certain exceptions.
    
    Args:
        max_retries: Maximum number of retries
        initial_delay: Initial delay between retries in seconds
        backoff_factor: Factor to increase delay by after each retry
        exceptions: Tuple of exceptions to catch for retry
        
    Returns:
        Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            nonlocal initial_delay
            retry_count = 0
            delay = initial_delay
            
            while True:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    retry_count += 1
                    
                    if retry_count > max_retries:
                        # Log the final failure and re-raise
                        logger.error(
                            f"Failed after {max_retries} retries: {func.__name__}. Error: {str(e)}"
                        )
                        raise
                    
                    # Log retry attempt
                    logger.warning(
                        f"Retrying {func.__name__} after error: {str(e)}. "
                        f"Retry {retry_count}/{max_retries} in {delay:.2f}s"
                    )
                    
                    # Wait with backoff
                    time.sleep(delay)
                    delay *= backoff_factor
        
        return wrapper
    
    return decorator


class CircuitBreaker:
    """Circuit breaker pattern implementation to prevent repeated calls to failing services."""
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        half_open_timeout: float = 30.0
    ):
        """Initialize the circuit breaker.
        
        Args:
            name: Name of the circuit breaker for logging
            failure_threshold: Number of failures before opening circuit
            reset_timeout: Time in seconds before attempting to close circuit
            half_open_timeout: Time to wait in half-open state
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.half_open_timeout = half_open_timeout
        
        # Internal state
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.successes_in_half_open = 0
        self.success_threshold = 3  # Successful calls to close circuit
    
    def execute(self, operation: Callable[..., T], *args, **kwargs) -> T:
        """Execute an operation with circuit breaker protection.
        
        Args:
            operation: Function to execute
            *args: Arguments to pass to operation
            **kwargs: Keyword arguments to pass to operation
            
        Returns:
            Result of the operation
            
        Raises:
            CircuitOpenError: When circuit is open
            Exception: Any exception raised by the operation
        """
        self._check_state()
        
        try:
            result = operation(*args, **kwargs)
            self._handle_success()
            return result
        except Exception as e:
            self._handle_failure(e)
            raise
    
    def _check_state(self):
        """Check and potentially update the circuit state."""
        current_time = time.time()
        
        if self.state == "OPEN":
            # Check if we should try half-open state
            if current_time - self.last_failure_time > self.reset_timeout:
                logger.info(f"Circuit {self.name} transitioning from OPEN to HALF_OPEN")
                self.state = "HALF_OPEN"
                self.successes_in_half_open = 0
            else:
                raise CircuitOpenError(f"Circuit {self.name} is OPEN")
                
        elif self.state == "HALF_OPEN":
            # Only allow one request at a time in half-open state
            # We could use a semaphore here for more sophisticated control
            pass
    
    def _handle_success(self):
        """Handle successful operation."""
        if self.state == "HALF_OPEN":
            self.successes_in_half_open += 1
            
            if self.successes_in_half_open >= self.success_threshold:
                logger.info(f"Circuit {self.name} closed after {self.successes_in_half_open} successful operations")
                self.state = "CLOSED"
                self.failure_count = 0
        
        elif self.state == "CLOSED":
            # Reset failure count after successful operations
            self.failure_count = 0
    
    def _handle_failure(self, exception: Exception):
        """Handle operation failure."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == "CLOSED" and self.failure_count >= self.failure_threshold:
            logger.warning(
                f"Circuit {self.name} opening after {self.failure_count} failures. "
                f"Latest error: {str(exception)}"
            )
            self.state = "OPEN"
            
        elif self.state == "HALF_OPEN":
            logger.warning(
                f"Circuit {self.name} re-opening after failure in HALF_OPEN state. "
                f"Error: {str(exception)}"
            )
            self.state = "OPEN"


class CircuitOpenError(Exception):
    """Exception raised when a circuit breaker is open."""
    pass


def format_error_context(error: Exception, operation: str = "", **additional_context) -> dict:
    """Format error information with context for consistent error reporting.
    
    Args:
        error: The exception that was raised
        operation: Name of the operation that failed
        **additional_context: Additional context to include
        
    Returns:
        Dictionary with formatted error information
    """
    error_type = error.__class__.__name__
    error_info = {
        "error_type": error_type,
        "message": str(error),
        "timestamp": datetime.now().isoformat(),
        "operation": operation,
    }
    
    # Add additional context
    error_info.update(additional_context)
    
    return error_info


def log_error(
    logger: logging.Logger,
    message: str,
    exception: Optional[Exception] = None,
    level: str = "error",
    **context
) -> None:
    """Log errors with consistent format and context.
    
    Centralized error logging function that ensures consistent formatting and
    contextual information across the codebase. Use this function for all error
    logging to maintain uniformity and capture sufficient troubleshooting information.
    
    Args:
        logger: Logger to use
        message: Error message
        exception: Optional exception that caused the error
        level: Log level (critical, error, warning, info)
        **context: Additional context to include (component, operation, etc.)
    
    Example:
        log_error(logger, "Failed to fetch data", exception=e, 
                 component="TokenManager", operation="refresh_tokens")
    """
    # Build full message with context
    context_str = ""
    if context:
        context_str = " Context: " + ", ".join(f"{k}={v}" for k, v in context.items())
    
    full_message = f"{message}{context_str}"
    
    # Determine if exception details should be included
    exc_info = exception is not None
    
    # Log with appropriate level
    if level == "critical":
        logger.critical(full_message, exc_info=exc_info)
    elif level == "error":
        logger.error(full_message, exc_info=exc_info)
    elif level == "warning":
        logger.warning(full_message, exc_info=exc_info)
    else:
        logger.info(full_message)