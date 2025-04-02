"""
Connection management utilities for HTTP requests.

This module provides centralized management of HTTP connections with
efficient connection pooling, retry logic, and thread safety.
"""

import logging
import threading
from typing import Dict, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# Configure connection pooling for better performance
MAX_POOL_CONNECTIONS = 50  # Increased from default 10
MAX_POOL_MAXSIZE = 50      # Increased from default 10
MAX_RETRIES = 3            # Number of retries for HTTP requests
RETRY_BACKOFF_FACTOR = 0.5  # Backoff factor for retries

class ConnectionManager:
    """Manages HTTP connections with efficient connection pooling.
    
    This class centralizes the creation and caching of HTTP sessions,
    ensuring optimal connection reuse and configuration.
    
    Features:
    - Per-token connection pooling
    - Automatic retry configuration
    - TCP keepalive settings
    - Thread-safe session management
    """
    
    def __init__(self):
        """Initialize the connection manager."""
        self.session_pool = {}  # Maps tokens to sessions
        self.default_session = None  # Default session (no token)
        self.lock = threading.RLock()  # Thread safety
        
    def get_session(self, token: Optional[str] = None) -> requests.Session:
        """Get or create an optimized session.
        
        Args:
            token: Optional token to associate with the session
            
        Returns:
            Requests session configured for optimal connection reuse
        """
        # Use token as cache key if provided
        cache_key = token if token else "__default__"
        
        with self.lock:
            # Check if session exists in pool
            if cache_key in self.session_pool:
                return self.session_pool[cache_key]
            
            # Create new session
            session = self._create_optimized_session()
            
            # Add to pool
            self.session_pool[cache_key] = session
            
            # Log session creation (only show token prefix/suffix for security)
            if token:
                token_display = f"{token[:4]}...{token[-4:]}"
                logger.debug(f"Created new connection pool for token {token_display}")
            else:
                logger.debug("Created new default connection pool")
                
            return session
    
    def _create_optimized_session(self) -> requests.Session:
        """Create a requests session with optimized connection pooling.
        
        Returns:
            Configured requests session
        """
        session = requests.Session()
        
        # Configure retry strategy with backoff
        retry_strategy = Retry(
            total=MAX_RETRIES,
            backoff_factor=RETRY_BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        
        # Create connection adapters with increased pool size
        adapter = HTTPAdapter(
            pool_connections=MAX_POOL_CONNECTIONS,
            pool_maxsize=MAX_POOL_MAXSIZE,
            max_retries=retry_strategy
        )
        
        # Mount adapters for both http and https
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set TCP keepalive options
        session.keep_alive = True
        
        return session
    
    def clear_session(self, token: Optional[str] = None) -> bool:
        """Clear a specific session from the pool.
        
        Args:
            token: Token associated with the session to clear
            
        Returns:
            True if session was cleared, False if not found
        """
        cache_key = token if token else "__default__"
        
        with self.lock:
            if cache_key in self.session_pool:
                # Close the session to release connections
                try:
                    self.session_pool[cache_key].close()
                except Exception as e:
                    logger.warning(f"Error closing session: {e}")
                
                # Remove from pool
                del self.session_pool[cache_key]
                return True
            
            return False
    
    def clear_all_sessions(self):
        """Clear all sessions from the pool."""
        with self.lock:
            # Close all sessions to release connections
            for token, session in self.session_pool.items():
                try:
                    session.close()
                except Exception as e:
                    logger.warning(f"Error closing session for {token}: {e}")
            
            # Clear the pool
            self.session_pool.clear()
            logger.info("Cleared all connection pools")