"""
Stats monitoring module for GitHub API operations.

This module provides a dedicated StatsMonitor class for tracking performance metrics,
monitoring collection progress, and providing insights during repository fetching.
"""

import time
import logging
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List, Set

from src.metrics.collector.collector import MetricsCollector
from src.core.query_pool import QueryPool
from src.api.token_management import TokenManager

logger = logging.getLogger(__name__)

class StatsMonitor:
    """
    Monitors and reports statistics during repository fetching operations.
    
    This class is responsible for tracking metrics, reporting progress, and
    providing insights during repository fetching operations. It runs in a
    background thread and periodically updates statistics.
    """
    
    def __init__(
        self,
        metrics_collector: MetricsCollector,
        query_pool: QueryPool,
        token_manager: TokenManager
    ):
        """
        Initialize the stats monitor.
        
        Args:
            metrics_collector: MetricsCollector for metrics tracking
            query_pool: QueryPool for query performance tracking
            token_manager: TokenManager for token health monitoring
        """
        self.metrics_collector = metrics_collector
        self.query_pool = query_pool
        self.token_manager = token_manager
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        self.monitor_interval = 5.0  # seconds
        self.last_stats_time = time.time()
        
        # Callbacks
        self.worker_scale_callback = None
        
    def start_monitoring(self, target_count: int, callback = None):
        """
        Start the monitoring thread.
        
        Args:
            target_count: Total number of repositories to fetch
            callback: Optional callback for worker scaling
        """
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Monitoring already active, not starting another thread")
            return
            
        self.monitoring_active = True
        self.worker_scale_callback = callback
        self.metrics_collector.set_global_target_count(target_count)
        
        self.monitoring_thread = threading.Thread(
            target=self._monitor_worker,
            daemon=True,
            name="StatsMonitorWorker"
        )
        self.monitoring_thread.start()
        logger.debug("Stats monitoring started")
        
    def stop_monitoring(self):
        """Stop the monitoring thread."""
        self.monitoring_active = False
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.debug("Waiting for monitoring thread to exit...")
            self.monitoring_thread.join(timeout=2.0)
            
        logger.debug("Stats monitoring stopped")
        
    def _monitor_worker(self):
        """Background worker that periodically collects and reports statistics."""
        try:
            while self.monitoring_active:
                current_time = time.time()
                
                # Only update stats periodically to avoid overwhelming logs
                if current_time - self.last_stats_time >= self.monitor_interval:
                    self._collect_and_report_stats()
                    self.last_stats_time = current_time
                    
                # Sleep for a short time
                time.sleep(1.0)
                
        except Exception as e:
            logger.error(f"Error in stats monitor worker: {e}", exc_info=True)
        finally:
            logger.debug("Stats monitor worker stopped")
    
    def _collect_and_report_stats(self):
        """Collect and report statistics about the current fetch operation."""
        try:
            # Get metrics from metrics collector
            metrics = self.metrics_collector.get_current_metrics()
            
            # Calculate throughput metrics
            duration = metrics.get('duration', 0)
            unique_results = metrics.get('unique_repositories', 0)
            total_results = metrics.get('total_processed', 0)
            duplicate_rate = metrics.get('duplicate_rate', 0) * 100
            global_target = metrics.get('global_target', 0)
            
            # Calculate rates
            repos_per_sec = unique_results / max(1, duration)
            total_per_sec = total_results / max(1, duration)
            
            # Determine completion percentage
            if global_target > 0:
                completion_pct = min(100, (unique_results / global_target) * 100)
            else:
                completion_pct = 0
                
            # Log general status update
            logger.info(
                f"Progress: {unique_results}/{global_target} repositories "
                f"({completion_pct:.1f}%) in {duration:.1f}s "
                f"[{repos_per_sec:.2f} repos/sec, {duplicate_rate:.1f}% duplicates]"
            )
            
            # Every third time (approximately every 15s), log more detailed stats
            if int(time.time()) % 15 < 5:
                # Get token stats
                token_stats = self.token_manager.get_token_stats()
                summary = token_stats.get('_summary', {})
                
                healthy_tokens = summary.get('healthy_tokens', 0)
                total_tokens = summary.get('total_tokens', 0) 
                total_remaining = summary.get('total_remaining', 0)
                
                # Log token health
                logger.info(
                    f"Token health: {healthy_tokens}/{total_tokens} healthy, "
                    f"{total_remaining} requests remaining"
                )
                
                # Get query performance
                top_queries = self.query_pool.get_top_performing_queries(5)
                if top_queries:
                    # Log top performing queries
                    top_query_info = []
                    for query_info in top_queries[:3]:
                        name = query_info.get('name', 'Unknown')
                        reward = query_info.get('reward', 0)
                        top_query_info.append(f"{name} ({reward:.2f})")
                        
                    logger.info(f"Top queries: {', '.join(top_query_info)}")
                    
                # Update query pool context
                self.query_pool.update_context({
                    'duration': duration,
                    'unique_results': unique_results,
                    'duplicate_rate': duplicate_rate
                })
            
            # Calculate optimal worker count based on token health
            if self.worker_scale_callback:
                healthy_count = token_stats['_summary'].get('healthy_tokens', 0)
                at_risk_count = token_stats['_summary'].get('at_risk_tokens', 0)
                low_count = token_stats['_summary'].get('low_tokens', 0)
                
                # Get total token count and usable tokens
                total_tokens = token_stats['_summary'].get('total_tokens', 0)
                usable_tokens = healthy_count + at_risk_count
                
                # Determine optimal worker count based on token health
                if usable_tokens == 0:
                    optimal_workers = 1  # Minimum worker count
                else:
                    # Scale workers based on token health
                    workers_per_token = 2 if healthy_count > total_tokens / 2 else 1
                    optimal_workers = max(1, usable_tokens * workers_per_token)
                    
                # Call the worker scale callback with the optimal count
                self.worker_scale_callback(optimal_workers)
                
            # Check if target is reached
            if unique_results >= global_target:
                logger.info(f"Reached target of {global_target} repositories. Auto-terminating to save API usage.")
                self.metrics_collector.mark_auto_completed(True)
                
        except Exception as e:
            logger.error(f"Error collecting stats: {e}", exc_info=True)