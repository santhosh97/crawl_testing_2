"""
Database management module for GitHub Stars Crawler.

This module provides a DatabaseManager class and utilities for handling 
database connections, sessions, and transactions. It's designed to support
high concurrency with connection pooling and proper error handling.

Key features:
- Connection pooling with optimized settings for PostgreSQL
- Transaction management with context managers
- Consistent error handling with custom exceptions
- Automatic reconnection for transient database errors
"""

import os
from contextlib import contextmanager
from typing import Generator, Optional, Callable

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from .models import Base
from src.api.github_exceptions import (
    DatabaseException, DatabaseConnectionError, 
    DatabaseInitError, DatabaseQueryError
)
from src.utils.error_handling import log_error, retry_with_backoff

class DatabaseManager:
    """Manages database connections and operations.
    
    This class handles database connection pooling, session management,
    and provides methods for database initialization and cleanup.
    It is optimized for high-concurrency operations with PostgreSQL.
    
    Attributes:
        engine: SQLAlchemy engine instance for database connections
        session_factory: Factory function for creating new database sessions
        logger: Optional logger for database operations
    """
    
    def __init__(self, database_url: str, pool_size: int = None,
                 max_overflow: int = None, pool_timeout: int = 30,
                 pool_recycle: int = 1800, logger = None):
        """Initialize the database manager with optimized pool settings for high concurrency.
        
        Args:
            database_url: Database connection URL (PostgreSQL only)
            pool_size: Connection pool size - if None, calculated based on system resources
            max_overflow: Maximum number of connections to overflow - if None, calculated based on system resources
            pool_timeout: Seconds to wait for a connection from the pool
            pool_recycle: Seconds after which a connection is recycled
            logger: Optional logger instance for recording database operations
            
        Raises:
            DatabaseConnectionError: When database URL is invalid or connection fails
        """
        self.logger = logger
        
        # Validate PostgreSQL URL
        if not database_url.startswith("postgresql://"):
            error_msg = "Only PostgreSQL is supported. DATABASE_URL must start with 'postgresql://'"
            if self.logger:
                log_error(self.logger, error_msg, level="critical", component="DatabaseManager")
            raise DatabaseConnectionError(error_msg)
            
        # Mask password in logs
        masked_url = self._mask_password(database_url)
        if self.logger:
            self.logger.info(f"Creating database manager with URL: {masked_url}")
        
        # Determine optimal pool size based on available system resources
        if pool_size is None or max_overflow is None:
            try:
                # Import locally to avoid unnecessary dependencies
                import psutil
                import os
                
                # Get available memory
                mem = psutil.virtual_memory()
                available_mem_gb = mem.available / (1024 * 1024 * 1024)
                
                # Get CPU count
                cpu_count = os.cpu_count() or 4
                
                # Calculate pool size based on available resources
                # Each PostgreSQL connection typically uses ~10-50MB
                # We'll assume 50MB per connection to be conservative
                max_connections_by_memory = int(available_mem_gb * 1024 / 50)
                
                # Balance between CPU and memory constraints
                # For I/O bound operations, we can use more threads than CPUs
                optimal_pool_size = min(max_connections_by_memory, cpu_count * 4)
                
                # Ensure reasonable bounds
                if pool_size is None:
                    pool_size = max(4, min(optimal_pool_size, 32))
                if max_overflow is None:
                    max_overflow = min(pool_size * 2, 50)
                
                if self.logger:
                    self.logger.debug(f"Database connection pool: size={pool_size}, max_overflow={max_overflow}")
            except Exception as e:
                # Fall back to reasonable defaults if resource detection fails
                if self.logger:
                    log_error(self.logger, f"Could not determine optimal connection pool size: {e}. Using defaults.",
                             level="warning", component="DatabaseManager")
                if pool_size is None:
                    pool_size = 20
                if max_overflow is None:
                    max_overflow = 40
        
        try:
            # Create SQLAlchemy engine with optimized PostgreSQL configurations
            self.engine = create_engine(
                database_url, 
                pool_size=pool_size,
                max_overflow=max_overflow,
                pool_timeout=pool_timeout,
                pool_recycle=pool_recycle,
                pool_pre_ping=True,
                isolation_level="READ COMMITTED",
                # Optimize connection parameters for PostgreSQL
                connect_args={
                    "connect_timeout": 10,
                    "application_name": "github_crawler",
                    "keepalives": 1,
                    "keepalives_idle": 60,
                    "keepalives_interval": 10,
                    "keepalives_count": 5,
                    # Add server-side statement timeout to avoid long-running queries
                    "options": "-c statement_timeout=30000"  # 30 second timeout
                },
                # Add JSON serialization using the fastest available serializer
                json_serializer=lambda obj: __import__('json').dumps(obj, default=str)
            )
            
            # Create session factory
            self.session_factory = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        except Exception as e:
            error_msg = f"Failed to create database engine: {str(e)}"
            if self.logger:
                log_error(self.logger, error_msg, exception=e, level="critical", component="DatabaseManager")
            raise DatabaseConnectionError(error_msg) from e
        
    def _mask_password(self, database_url: str) -> str:
        """Mask the password in a database URL for logging.
        
        Args:
            database_url: Database URL with password
            
        Returns:
            Database URL with password masked
        """
        # Try to extract password from URL
        if "@" in database_url and ":" in database_url:
            parts = database_url.split("@")
            if len(parts) >= 2:
                credentials = parts[0].split(":")
                if len(credentials) >= 3:  # protocol:username:password
                    password = credentials[2]
                    return database_url.replace(password, "*" * len(password))
        
        # Return as is if we can't extract the password
        return database_url
        
    def init_db(self, clean: bool = False) -> bool:
        """Initialize the database by creating all tables.
        
        Args:
            clean: If True, drops all tables before creating them
            
        Returns:
            True if successful, False otherwise
            
        Raises:
            DatabaseInitError: When database initialization fails
        """
        try:
            if clean and self.logger:
                log_error(self.logger, "Cleaning database - dropping all tables", level="warning", 
                         component="DatabaseManager", operation="init_db", clean_mode=True)
                Base.metadata.drop_all(bind=self.engine)
                if self.logger:
                    self.logger.info("Database cleaned, recreating tables")
            
            Base.metadata.create_all(bind=self.engine)
            if self.logger:
                self.logger.info("Database tables created successfully")
            return True
            
        except SQLAlchemyError as e:
            error_msg = f"Database initialization error: {str(e)}"
            if self.logger:
                log_error(self.logger, error_msg, exception=e, level="error", 
                         component="DatabaseManager", operation="init_db")
                self.logger.error("Ensure PostgreSQL is running and the connection details are correct")
            raise DatabaseInitError(error_msg) from e
            
    def get_session(self) -> Session:
        """Get a new database session.
        
        Returns:
            Database session
        """
        return self.session_factory()
        
    def test_connection(self) -> bool:
        """Test the database connection.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
                return True
        except SQLAlchemyError as e:
            if self.logger:
                log_error(self.logger, "Database connection test failed", exception=e, 
                         level="error", component="DatabaseManager", operation="test_connection")
            return False
            
    def cleanup(self) -> None:
        """Clean up database resources.
        
        Disposes of the engine and all its database connections.
        Should be called when the application is shutting down.
        
        Returns:
            None
        """
        if self.engine:
            self.engine.dispose()
            if self.logger:
                self.logger.debug("Database engine disposed")


@contextmanager
def session_scope(db_manager: DatabaseManager) -> Generator[Session, None, None]:
    """Provide a transactional scope around a series of operations.
    
    Args:
        db_manager: Database manager instance
        
    Yields:
        Database session
        
    Raises:
        DatabaseQueryError: On database query errors
        Exception: On other unexpected errors
    """
    session = db_manager.get_session()
    logger = db_manager.logger
    
    try:
        yield session
        session.commit()
    except SQLAlchemyError as e:
        error_msg = f"Database operation failed: {str(e)}"
        if logger:
            log_error(logger, error_msg, exception=e, level="error", 
                    component="session_scope", operation="commit")
        session.rollback()
        raise DatabaseQueryError(error_msg) from e
    except (KeyboardInterrupt, SystemExit):
        # Always allow keyboard interrupts and system exits to propagate
        session.rollback()
        raise
    except Exception as e:
        error_msg = f"Unexpected error during database session: {str(e)}"
        if logger:
            log_error(logger, error_msg, exception=e, level="error", 
                    component="session_scope", operation="execute")
        session.rollback()
        
        # Convert to DatabaseQueryError for consistency, but preserve original exception as cause
        # This makes error handling more consistent for callers
        raise DatabaseQueryError(f"Unexpected database error: {str(e)}") from e
    finally:
        session.close()
        if logger:
            logger.debug("Database session closed")


# Rather than using a singleton, DatabaseManager instances should be created by the application
# and injected where needed. This ensures proper lifecycle management and testability.