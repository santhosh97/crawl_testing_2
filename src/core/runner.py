"""GitHub Stars Crawler main execution logic."""

import logging
import time
import os
from typing import List, Dict, Any

from src.db.database import init_db
from src.db.repository_manager_parallel import process_batch_with_new_session
from src.api.github_client_parallel import ParallelFetcher
from src.utils.cache_utils import CacheManager
from src.utils.connection_manager import ConnectionManager
from src.core.config import Config
from src.metrics.collector.collector import MetricsCollector
from src.core.query_pool import QueryPool
from src.api.token_management import TokenManager
from src.api.token_stats import print_token_stats, start_token_stats_reporter, print_final_token_stats
from src.metrics.updater import start_metrics_updater

logger = logging.getLogger(__name__)

def run_crawler(args):
    """Run the GitHub stars crawler with query pool strategy and advanced token management.
    
    Args:
        args: Command-line arguments.
        
    Returns:
        int: Exit code (0 for success, non-zero for errors).
    """
    try:
        # Initialize database (clean if requested)
        try:
            if args.clean_db:
                logger.info("Initializing database with clean option (dropping all existing data)...")
                init_db(clean=True)
            else:
                logger.info("Initializing database...")
                init_db()
        except Exception as e:
            logger.error(f"Database initialization error: {e}")
            # We'll continue - some functionality will work without the database
        
        # Initialize all core components in a centralized way
        # Create config instance
        app_config = Config()
        
        # Get token_manager from the caller (must be provided as a required parameter)
        if not args.token_manager:
            logger.error("No TokenManager provided - required for API access")
            return 1
        token_manager = args.token_manager
            
        # Check if we have enough valid tokens
        if len(token_manager.tokens) == 0:
            logger.error("No valid GitHub tokens available. Please check your tokens and try again.")
            return 1
        
        # Show initial token stats if requested
        if args.token_stats:
            print_token_stats(token_manager)
        
        # 1. ConnectionManager - centralized HTTP connection management
        connection_manager = ConnectionManager()
        
        # 2. CacheManager - direct instantiation instead of singleton
        cache_manager = CacheManager(
            max_size=app_config.get("cache.max_size", 10000),
            default_ttl=app_config.get("cache.default_ttl", 1800),
            strategy=app_config.get("cache.strategy", "hybrid")
        )
        
        # 3. MetricsCollector - direct instantiation instead of singleton
        metrics_collector = MetricsCollector()
        metrics_collector.reset_stats()  # Fresh start for this run
        
        # Set the connection manager in token_manager
        token_manager.connection_manager = connection_manager
        
        # Start metrics updater thread
        metrics_thread = start_metrics_updater(metrics_collector)
        
        # 3. QueryPool - initialize with exploration weight from config and metrics_collector
        query_pool = QueryPool(
            metrics_collector=metrics_collector,  # Pass metrics_collector for dependency injection
            config=app_config  # Pass the config instance containing exploration_weight and target_repos parameters
        )
        
        # Start token stats reporting if requested
        if args.token_stats:
            stats_thread = start_token_stats_reporter(token_manager)
        
        # Start crawling with query pool strategy
        max_workers = args.workers if args.workers > 0 else min(10, len(token_manager.tokens))  # Default to 10 workers or token count
        logger.info(f"Starting to crawl {args.repos:,} repositories using query pool strategy...")
        start_time = time.time()
        
        # Create a ParallelFetcher instance to manage parallel fetching
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
        db_batch_size = min(50, args.batch_size)
        current_batch = []
        
        # Fetch repositories in chunks and process them incrementally
        for chunk in fetcher.fetch_repositories_with_query_pool(args.repos, max_workers, yield_chunks=True):
            fetch_time = time.time() - start_time
            # Only log when significant chunks are received
            if len(chunk) > 10:
                logger.debug(f"Fetched chunk of {len(chunk)} repositories in {fetch_time:.2f}s")
            
            # Filter out duplicates we've already seen in this session
            unique_repos = [repo for repo in chunk if repo["github_id"] not in seen_repo_ids]
            
            # Only take as many as we need to reach the target
            repos_needed = args.repos - processed_count
            if len(unique_repos) > repos_needed:
                logger.info(f"Trimming chunk from {len(unique_repos)} to {repos_needed} to meet target of {args.repos}")
                unique_repos = unique_repos[:repos_needed]
            
            for repo in unique_repos:
                seen_repo_ids.add(repo["github_id"])
                current_batch.append(repo)
                
                # Process batch when it reaches the target size
                if len(current_batch) >= db_batch_size:
                    batch_processed = process_batch_with_new_session(current_batch, metrics_collector=metrics_collector)
                    processed_count += batch_processed
                    
                    # Only log progress at 10% intervals or every 1000 repos
                    log_progress = processed_count % 1000 == 0
                    if args.repos >= 10:  # Only use percentage intervals for larger targets
                        log_progress = log_progress or processed_count % max(1, int(args.repos * 0.1)) == 0
                    
                    if log_progress:
                        elapsed = time.time() - start_time
                        rate = processed_count / elapsed if elapsed > 0 else 0
                        percent = (processed_count / args.repos) * 100
                        logger.info(f"Progress: {processed_count}/{args.repos} repositories ({percent:.1f}%) at {rate:.1f} repos/sec")
                    
                    current_batch = []
                    
                    # Stop if we've reached the target
                    if processed_count >= args.repos:
                        logger.info(f"Reached target of {args.repos} repositories, stopping fetch.")
                        break
            
            # Stop if we've reached the target
            if processed_count >= args.repos:
                break
        
        # Process any remaining repositories in the final batch
        if current_batch:
            # Trim final batch if needed to exactly meet the target
            if processed_count + len(current_batch) > args.repos:
                extra = processed_count + len(current_batch) - args.repos
                logger.info(f"Trimming final batch from {len(current_batch)} to {len(current_batch) - extra} repositories to meet target of {args.repos}")
                current_batch = current_batch[:len(current_batch) - extra]
            
            if current_batch:  # Check if we still have repos after trimming
                batch_processed = process_batch_with_new_session(current_batch, metrics_collector=metrics_collector)
                processed_count += batch_processed
        
        # Log processing results
        if processed_count > 0:
            logger.info(f"Successfully processed {processed_count} repositories")
        else:
            logger.warning("No repositories were fetched, nothing to process")
        
        # Calculate elapsed time and log completion
        elapsed_time = time.time() - start_time
        
        # Format elapsed time in a human-readable format
        minutes, seconds = divmod(int(elapsed_time), 60)
        hours, minutes = divmod(minutes, 60)
        
        time_str = ""
        if hours > 0:
            time_str += f"{hours}h "
        if minutes > 0 or hours > 0:
            time_str += f"{minutes}m "
        time_str += f"{seconds}s"
        
        logger.info(f"Query pool crawling completed in {time_str} - processed {processed_count:,} repositories")
        
        # Log metrics summary from metrics collector
        if not args.no_summary:
            metrics_collector.log_summary()
            
            # Export metrics to JSON file
            metrics_file = metrics_collector.export_metrics_json()
            logger.info(f"Detailed metrics saved to: {metrics_file}")
        
        # Show final token statistics if requested
        if args.token_stats:
            print_final_token_stats(token_manager)
        
        # Note if we used a clean database
        if args.clean_db:
            logger.info("Note: Started with a clean database (existing data was backed up)")
        else:
            logger.info("Note: Used existing database (use --clean-db to start fresh next time)")
        
        return 0
    
    except KeyboardInterrupt:
        logger.warning("Crawler interrupted by user. Partial data has been saved.")
        return 130  # Standard exit code for SIGINT
        
    except Exception as e:
        logger.error(f"Error running query pool crawler: {e}", exc_info=True)
        return 1