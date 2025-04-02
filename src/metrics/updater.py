"""Metrics updater for repository crawler."""

import logging
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

def metrics_updater(collector: Any, interval: int = 5):
    """Thread function to update performance metrics periodically.
    
    Args:
        collector: Instance of MetricsCollector to use for updates
        interval: Number of seconds between updates (default: 5)
    """
    if not collector:
        logger.error("No metrics collector provided to updater thread")
        return
    
    # Flag to indicate if the thread should stop
    should_stop = False
        
    while not should_stop:
        try:
            # Sleep first to allow for initial setup
            time.sleep(interval)
            
            # Check if we should stop
            if hasattr(collector, "_is_shutting_down") and collector._is_shutting_down:
                should_stop = True
                continue
                
            # Only update if collector has the method
            if hasattr(collector, "update_performance_metrics"):
                collector.update_performance_metrics()
        except Exception as e:
            logger.warning(f"Error updating performance metrics: {e}")

def start_metrics_updater(metrics_collector: Any, interval: int = 5) -> Optional[threading.Thread]:
    """Start a metrics updater thread.
    
    Args:
        metrics_collector: MetricsCollector instance
        interval: Number of seconds between updates (default: 5)
        
    Returns:
        Thread: The started thread, or None if no collector provided
    """
    if not metrics_collector:
        logger.warning("No metrics collector provided, not starting updater thread")
        return None
    
    # Add shutdown flag to collector
    if not hasattr(metrics_collector, "_is_shutting_down"):
        metrics_collector._is_shutting_down = False
    
    # Add stop method to thread
    class MetricsThread(threading.Thread):
        def __init__(self, target, args, **kwargs):
            super().__init__(target=target, args=args, **kwargs)
            self.collector = args[0]
            
        def stop(self):
            if hasattr(self.collector, "_is_shutting_down"):
                self.collector._is_shutting_down = True
                logger.debug("Signaling metrics thread to stop")
    
    # Create and start thread
    metrics_thread = MetricsThread(
        target=metrics_updater, 
        args=(metrics_collector, interval), 
        daemon=True
    )
    metrics_thread.start()
    logger.info(f"Started metrics updater thread (update interval: {interval}s)")
    return metrics_thread