"""Application class for GitHub Stars Crawler.

This module provides the main Application class that manages component lifecycle,
dependencies, and configuration for the GitHub Stars Crawler. It serves as the 
central orchestration point that initializes and coordinates all system components.

The Application class follows dependency injection principles to maintain clean
separation of concerns and testability. Components are initialized in a specific
order to ensure proper dependencies are available when needed, and all resources
are cleaned up properly when the application shuts down.

Key responsibilities:
- Component initialization and dependency management
- Runtime coordination and execution flow
- Resource cleanup and error handling
- Configuration loading and validation
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union

from src.cli.args import parse_args
from src.cli.environment import Environment
from src.utils.logging_config import LogManager
from src.db.database import DatabaseManager
from src.metrics.collector.collector import MetricsCollector, MetricsExporter
from src.utils.cache_utils import CacheManager
from src.utils.path_manager import PathManager
from src.api.token_management import TokenManager
from src.core.query_pool import QueryPool
from src.api.github_client_parallel import ParallelFetcher
from src.api.token_stats import print_token_stats, start_token_stats_reporter, print_final_token_stats
from src.core.config import Config
from src.db.repository_manager_parallel import process_batch_with_new_session
from src.utils.connection_manager import ConnectionManager
from src.metrics.updater import start_metrics_updater
from src.api.github_exceptions import (
    ApplicationException, InitializationError, DatabaseInitError, 
    MissingConfigError, GitHubAuthenticationError, ResourceCleanupError
)
from src.utils.error_handling import log_error

# Application instance is now properly passed via dependency injection
# No global state or singleton pattern

class Application:
    """Main application for GitHub Stars Crawler.
    
    This class manages component lifecycle, dependency injection,
    and execution flow for the GitHub Stars Crawler. It orchestrates
    the initialization, execution, and cleanup of all system components.
    
    The Application follows a structured initialization process to ensure
    proper dependency management and resource allocation. It also provides
    centralized error handling and cleanup to maintain system consistency.
    
    Attributes:
        args: Command-line arguments for configuration
        components: Dictionary of initialized components
        logger: Logger instance for application logs
    """
    
    def __init__(self, args=None, log_manager=None, path_manager=None, environment=None):
        """Initialize the application with optional injected dependencies.
        
        Creates a new Application instance with the specified dependencies.
        Any dependencies not provided will be created during initialization.
        
        Args:
            args: Command-line arguments (optional, will parse if not provided)
            log_manager: LogManager instance for logging configuration and access
            path_manager: PathManager instance for consistent file path handling
            environment: Environment instance for configuration and env variables
        """
        # Parse arguments if not provided
        self.args = args or parse_args()
        
        # Store components
        self.components = {}
        
        # Store injected dependencies
        if log_manager:
            self.components['log_manager'] = log_manager
            self._init_logger = log_manager.get_logger(__name__)
        else:
            self._init_logger = logging.getLogger(__name__)
            
        if path_manager:
            self.components['path_manager'] = path_manager
            
        if environment:
            self.components['environment'] = environment
        
    def initialize(self):
        """Initialize all application components.
        
        Returns:
            Self for method chaining
            
        Raises:
            InitializationError: When component initialization fails
        """
        try:
            # Initialize only components that haven't been injected
            if 'log_manager' not in self.components:
                self._init_logging()
                
            if 'environment' not in self.components:
                self._init_environment()
                
            if 'config' not in self.components:
                self._init_config()
                
            if 'db_manager' not in self.components:
                self._init_database()
                
            if 'connection_manager' not in self.components:
                self._init_connection_manager()
                
            if 'token_manager' not in self.components:
                self._init_token_manager()
                
            if 'path_manager' not in self.components:
                self._init_path_manager()
                
            if 'cache_manager' not in self.components:
                self._init_cache_manager()
                
            if 'metrics_collector' not in self.components:
                self._init_metrics_collector()
                
            if 'query_pool' not in self.components:
                self._init_query_pool()
            
            # Get proper logger after logging is configured
            self.logger = self.get_component('log_manager').get_logger(__name__)
            self.logger.info("Application initialized successfully")
            
            return self
        except Exception as e:
            # Use standardized error logging
            log_error(self._init_logger, "Application initialization failed", exception=e, 
                      level="critical", component="Application", operation="initialize",
                      init_phase=True)
            
            # Re-raise as InitializationError for consistent handling
            raise InitializationError(f"Failed to initialize application: {str(e)}") from e
    
    def _init_logging(self):
        """Initialize logging system."""
        log_level = getattr(logging, self.args.log_level)
        log_manager = LogManager(log_level=log_level)
        self.components['log_manager'] = log_manager
        
        # Update logger reference
        self._init_logger = log_manager.get_logger(__name__)
        self._init_logger.debug("Logging initialized")
        
    def _init_environment(self):
        """Initialize environment configuration."""
        env = Environment()
        self.components['environment'] = env
        self._init_logger.debug("Environment initialized")
        
    def _init_config(self):
        """Initialize configuration."""
        # Get environment instance to pass to config
        env = self.get_component('environment')
        
        # Create config with proper dependency injection
        app_config = Config(environment=env)
        self.components['config'] = app_config
        self._init_logger.debug("Configuration initialized")
        
    def _init_database(self):
        """Initialize database connection.
        
        Raises:
            InitializationError: When database initialization fails
            ConfigurationError: When database configuration is invalid
        """
        
        try:
            env = self.get_component('environment')
            database_url = env.get_database_url()
            
            if not database_url:
                error_msg = "Invalid database URL configuration"
                log_error(self._init_logger, error_msg, level="critical", 
                         component="Application", operation="_init_database")
                raise MissingConfigError(error_msg)
                
            # Create database manager with proper error handling
            try:
                db_manager = DatabaseManager(database_url)
                self.components['db_manager'] = db_manager
            except Exception as db_error:
                error_msg = f"Failed to create database manager: {str(db_error)}"
                log_error(self._init_logger, error_msg, exception=db_error, 
                         level="critical", component="Application", operation="_init_database")
                raise DatabaseInitError(error_msg) from db_error
            
            # Initialize database (clean if requested)
            try:
                if self.args.clean_db:
                    self._init_logger.info("Initializing database with clean option (dropping all existing data)...")
                    db_manager.init_db(clean=True)
                else:
                    self._init_logger.info("Initializing database...")
                    db_manager.init_db()
                
                self._init_logger.info("Database initialized successfully")
            except Exception as init_error:
                error_msg = f"Database initialization error: {str(init_error)}"
                log_error(self._init_logger, error_msg, exception=init_error, 
                         level="critical", component="Application", operation="_init_database")
                raise DatabaseInitError(error_msg) from init_error
                
        except MissingConfigError:
            # Re-raise configuration errors directly
            raise
        except Exception as e:
            # Wrap other exceptions in InitializationError
            error_msg = f"Failed to initialize database: {str(e)}"
            log_error(self._init_logger, error_msg, exception=e, 
                     level="critical", component="Application", operation="_init_database")
            raise InitializationError(error_msg) from e
            
    def _init_token_manager(self):
        """Initialize token manager.
        
        Raises:
            InitializationError: When token manager initialization fails
            MissingConfigError: When no GitHub tokens are available
        """
        
        try:
            env = self.get_component('environment')
            # db_manager is not used in this method
            
            tokens = env.get_github_tokens()
            if not tokens:
                error_msg = "No valid GitHub tokens available"
                log_error(self._init_logger, error_msg, level="critical", 
                         component="Application", operation="_init_token_manager")
                raise MissingConfigError(error_msg)
                
            # Create token manager with proper error handling
            try:
                # Initialize connection manager first if needed
                if 'connection_manager' not in self.components:
                    self._init_connection_manager()

                connection_manager = self.components['connection_manager']
                
                # Create TokenManager with dependencies injected
                token_manager = TokenManager(
                    tokens=tokens,
                    connection_manager=connection_manager
                )
                
                # Store in components registry
                self.components['token_manager'] = token_manager
                self._init_logger.debug("Stored token_manager in application components")
                
                # Validate at least one token works
                valid_tokens = len(tokens) - len(token_manager.invalid_tokens)
                if valid_tokens == 0:
                    error_msg = "No valid GitHub tokens available (all tokens failed validation)"
                    log_error(self._init_logger, error_msg, level="critical", 
                             component="Application", operation="_init_token_manager")
                    raise GitHubAuthenticationError(error_msg)
                    
            except Exception as token_error:
                error_msg = f"Failed to initialize token manager: {str(token_error)}"
                log_error(self._init_logger, error_msg, exception=token_error, 
                         level="critical", component="Application", operation="_init_token_manager")
                raise InitializationError(error_msg) from token_error
                
        except (MissingConfigError, GitHubAuthenticationError):
            # Re-raise configuration errors directly
            raise
        except Exception as e:
            # Wrap other exceptions in InitializationError
            error_msg = f"Failed to initialize token manager: {str(e)}"
            log_error(self._init_logger, error_msg, exception=e, 
                     level="critical", component="Application", operation="_init_token_manager")
            raise InitializationError(error_msg) from e
        
    def _init_connection_manager(self):
        """Initialize connection manager."""
        connection_manager = ConnectionManager()
        self.components['connection_manager'] = connection_manager
        self._init_logger.debug("Connection manager initialized")
        
    def _init_path_manager(self):
        """Initialize path manager."""
        if not self.get_component('path_manager'):
            path_manager = PathManager()
            self.components['path_manager'] = path_manager
            if self._init_logger:
                self._init_logger.debug("Path manager initialized")
        
    def _init_cache_manager(self):
        """Initialize cache manager."""
        config = self.get_component('config')
        metrics_collector = self.components.get('metrics_collector')  # May be None
        path_manager = self.get_component('path_manager')
        logger = self.get_component('log_manager').get_logger(__name__) if 'log_manager' in self.components else None
        
        # Create cache manager with all dependencies injected
        cache_manager = CacheManager(
            max_size=config.get("cache.max_size", 10000),
            default_ttl=config.get("cache.default_ttl", 1800),
            strategy=config.get("cache.strategy", "hybrid"),
            metrics_collector=metrics_collector,
            path_manager=path_manager,
            logger=logger
        )
        
        self.components['cache_manager'] = cache_manager
        self._init_logger.debug("Cache manager initialized")
        
    def _init_metrics_collector(self):
        """Initialize metrics collector."""
        log_manager = self.get_component('log_manager')
        path_manager = self.get_component('path_manager')
        
        # Get metrics directory from path manager instead of log manager
        metrics_dir = path_manager.get_metrics_dir()
        
        # Create exporter with dependency injection
        metrics_exporter = MetricsExporter(metrics_dir=metrics_dir)
        
        # Create collector with proper dependency injection
        metrics_collector = MetricsCollector(
            metrics_dir=metrics_dir,
            exporter=metrics_exporter,
            path_manager=path_manager  # Pass path manager as dependency
        )
        
        # Reset for fresh start
        metrics_collector.reset_stats()
        
        # Set global target count
        metrics_collector.set_global_target_count(self.args.repos)
        
        self.components['metrics_collector'] = metrics_collector
        self.components['metrics_exporter'] = metrics_exporter
        
        self._init_logger.debug("Metrics collector initialized")
        
    def _init_query_pool(self):
        """Initialize query pool."""
        config = self.get_component('config')
        metrics_collector = self.get_component('metrics_collector')
        
        # Override the target repos in config with args value first
        config.set("crawler.total_count", self.args.repos)
        
        query_pool = QueryPool(
            metrics_collector=metrics_collector,
            config=config
        )
        
        self.components['query_pool'] = query_pool
        self._init_logger.debug("Query pool initialized")
        
    def get_component(self, name):
        """Get a component by name.
        
        Args:
            name: Component name
            
        Returns:
            Component instance or None if not found
        """
        return self.components.get(name)
        
    def run(self):
        """Run the crawler.
        
        Returns:
            Exit code (0 for success, non-zero for errors)
            
        Raises:
            ApplicationException: When a critical application error occurs
        """
        # Start timing
        start_time = time.time()
        logger = None
        metrics_thread = None
        stats_thread = None
        
        try:
            # Get required components
            logger = self.get_component('log_manager').get_logger(__name__)
            token_manager = self.get_component('token_manager')
            connection_manager = self.get_component('connection_manager')
            cache_manager = self.get_component('cache_manager')
            metrics_collector = self.get_component('metrics_collector')
            query_pool = self.get_component('query_pool')
            
            # Show initial token stats if requested
            if self.args.token_stats:
                print_token_stats(token_manager)
                
            # Start metrics updater thread
            metrics_thread = start_metrics_updater(metrics_collector)
            
            # Start token stats reporting if requested
            if self.args.token_stats:
                stats_thread = start_token_stats_reporter(token_manager)
                
            # Start crawling
            max_workers = self.args.workers if self.args.workers > 0 else min(10, len(token_manager.tokens))
            logger.info(f"Starting to crawl {self.args.repos:,} repositories using query pool strategy...")
            
            # Create a ParallelFetcher instance
            fetcher = ParallelFetcher(
                token_manager=token_manager,
                query_pool=query_pool,
                cache_mgr=cache_manager,
                metrics_collector=metrics_collector,
                connection_manager=connection_manager
            )
            
            # Track processed repositories for reporting
            processed_count = 0
            seen_repo_ids = set()
            
            # Use a smaller batch size for incremental database updates
            db_batch_size = min(50, self.args.batch_size)
            current_batch = []
            
            # Fetch repositories in chunks and process them incrementally
            for chunk in fetcher.fetch_repositories_with_query_pool(
                self.args.repos, max_workers, yield_chunks=True
            ):
                fetch_time = time.time() - start_time
                # Only log when significant chunks are received
                if len(chunk) > 10:
                    logger.debug(f"Fetched chunk of {len(chunk)} repositories in {fetch_time:.2f}s")
                
                # Filter out duplicates we've already seen in this session
                unique_repos = [repo for repo in chunk if repo["github_id"] not in seen_repo_ids]
                
                # Only take as many as we need to reach the target
                repos_needed = self.args.repos - processed_count
                if len(unique_repos) > repos_needed:
                    logger.info(f"Trimming chunk from {len(unique_repos)} to {repos_needed} to meet target of {self.args.repos}")
                    unique_repos = unique_repos[:repos_needed]
                
                for repo in unique_repos:
                    seen_repo_ids.add(repo["github_id"])
                    current_batch.append(repo)
                    
                    # Process batch when it reaches the target size
                    if len(current_batch) >= db_batch_size:
                        try:
                            batch_processed = process_batch_with_new_session(
                                current_batch, 
                                db_manager=self.get_component('db_manager'), 
                                metrics_collector=metrics_collector
                            )
                            processed_count += batch_processed
                            
                            # Only log progress at 10% intervals or every 1000 repos
                            log_progress = processed_count % 1000 == 0
                            if self.args.repos >= 10:  # Only use percentage intervals for larger targets
                                log_progress = log_progress or processed_count % max(1, int(self.args.repos * 0.1)) == 0
                            
                            if log_progress:
                                elapsed = time.time() - start_time
                                rate = processed_count / elapsed if elapsed > 0 else 0
                                percent = (processed_count / self.args.repos) * 100
                                logger.info(f"Progress: {processed_count}/{self.args.repos} repositories ({percent:.1f}%) at {rate:.1f} repos/sec")
                            
                            current_batch = []
                            
                            # Stop if we've reached the target
                            if processed_count >= self.args.repos:
                                logger.info(f"Reached target of {self.args.repos} repositories, stopping fetch.")
                                break
                        except Exception as batch_error:
                            # Log batch error but continue with next batch
                            log_error(logger, "Error processing batch", exception=batch_error, 
                                      level="error", component="Application", operation="process_batch",
                                      batch_size=len(current_batch))
                            # Clear the batch and continue
                            current_batch = []
                
                # Stop if we've reached the target
                if processed_count >= self.args.repos:
                    break
            
            # Process any remaining repositories in the final batch
            if current_batch:
                # Trim final batch if needed to exactly meet the target
                if processed_count + len(current_batch) > self.args.repos:
                    extra = processed_count + len(current_batch) - self.args.repos
                    logger.info(f"Trimming final batch from {len(current_batch)} to {len(current_batch) - extra} repositories to meet target of {self.args.repos}")
                    current_batch = current_batch[:len(current_batch) - extra]
                
                if current_batch:  # Check if we still have repos after trimming
                    try:
                        batch_processed = process_batch_with_new_session(
                            current_batch, 
                            db_manager=self.get_component('db_manager'),
                            metrics_collector=metrics_collector
                        )
                        processed_count += batch_processed
                    except Exception as final_batch_error:
                        # Log final batch error but continue to summary
                        log_error(logger, "Error processing final batch", exception=final_batch_error, 
                                 level="error", component="Application", operation="process_final_batch",
                                 batch_size=len(current_batch))
            
            # Log processing results
            if processed_count > 0:
                logger.info(f"Successfully processed {processed_count} repositories")
            else:
                logger.warning("No repositories were fetched, nothing to process")
            
            # Calculate elapsed time and log completion
            elapsed_time = time.time() - start_time
            
            # Format the elapsed time
            time_str = f"{int(elapsed_time)}s"
            
            logger.info(f"Query pool crawling completed in {time_str} - processed {processed_count:,} repositories")
            
            # Log metrics summary from metrics collector
            if not self.args.no_summary:
                metrics_collector.log_summary()
                
                # Export metrics to JSON file
                metrics_file = metrics_collector.export_metrics_json()
                logger.info(f"Detailed metrics saved to: {metrics_file}")
            
            # Show final token statistics if requested
            if self.args.token_stats:
                print_final_token_stats(token_manager)
            
            # Note if we used a clean database
            if self.args.clean_db:
                logger.info("Note: Started with a clean database (existing data was backed up)")
            else:
                logger.info("Note: Used existing database (use --clean-db to start fresh next time)")
            
            return 0
            
        except KeyboardInterrupt:
            if logger:
                log_error(logger, "Crawler interrupted by user. Partial data has been saved.", 
                         level="warning", component="Application", operation="run")
            return 130  # Standard exit code for SIGINT
            
        except Exception as e:
            if not logger:
                logger = logging.getLogger(__name__)
            
            # Use standardized error logging
            log_error(logger, "Error running query pool crawler", exception=e, 
                     level="critical", component="Application", operation="run")
            
            # Re-raise as ApplicationException if it's not already a specific application exception
            if not isinstance(e, ApplicationException):
                raise ApplicationException(f"Application run failed: {str(e)}") from e
            raise
        finally:
            # Clean up background threads if they were started
            if metrics_thread:
                try:
                    # All thread objects created by our application implement stop method
                    metrics_thread.stop()
                except (AttributeError, Exception):
                    pass
                    
            if stats_thread:
                try:
                    # All thread objects created by our application implement stop method
                    stats_thread.stop()
                except (AttributeError, Exception):
                    pass
        
    def cleanup(self):
        """Clean up all resources used by the application.
        
        Raises:
            ResourceCleanupError: When there's a critical error during resource cleanup
        """
        try:
            logger = self.get_component('log_manager').get_logger(__name__)
        except Exception:
            # Fallback logger if logging system is unavailable
            logger = logging.getLogger(__name__)
            
        logger.info("Cleaning up application resources...")
        
        # Clean up components in reverse initialization order
        components_to_cleanup = [
            'query_pool',
            'metrics_collector',
            'cache_manager',
            'connection_manager',
            'token_manager',
            'db_manager',
            'config'
        ]
        
        cleanup_errors = []
        
        for name in components_to_cleanup:
            if name in self.components:
                component = self.components[name]
                try:
                    # Check if component has cleanup method
                    if hasattr(component, 'cleanup') and callable(component.cleanup):
                        logger.debug(f"Cleaning up {name}...")
                        component.cleanup()
                    else:
                        logger.debug(f"Component {name} has no cleanup method, skipping")
                except Exception as e:
                    # Use the imported log_error function
                    log_error(logger, f"Error cleaning up {name}", exception=e, 
                             level="error", component="Application", operation="cleanup",
                             component_name=name)
                    cleanup_errors.append((name, str(e)))
        
        if cleanup_errors:
            logger.warning(f"Encountered {len(cleanup_errors)} errors during cleanup")
            
            # If critical components failed to clean up, raise an exception
            critical_components = ['db_manager', 'token_manager']
            critical_failures = [name for name, _ in cleanup_errors if name in critical_components]
            
            if critical_failures:
                # Use the imported ResourceCleanupError
                raise ResourceCleanupError(f"Failed to clean up critical components: {', '.join(critical_failures)}")
        else:
            logger.info("Application cleanup complete")
            
        # No need to clear global references as we're using dependency injection