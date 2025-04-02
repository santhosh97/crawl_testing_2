"""GitHub token statistics reporting."""

import logging
import threading
import time

logger = logging.getLogger(__name__)

def print_token_stats(token_manager):
    """Print token statistics.
    
    Args:
        token_manager: TokenManager instance.
    """
    stats = token_manager.get_token_stats()
    summary = stats["_summary"]
    logger.info("Token Statistics:")
    logger.info(f"Total Tokens: {summary['total_tokens']}, Reserved: {summary['reserved_tokens']}")
    logger.info(f"Total Available Requests: {summary['total_remaining']:,}/{summary['total_limit']:,} "
               f"({summary['usage_percentage']:.1f}% used)")
    logger.info(f"Average Requests per Token: {summary['avg_remaining']:.1f}")
    
    # Show token health summary
    health_summary = (
        f"Status: "
        f"{summary['healthy_tokens']} healthy, "
        f"{summary['at_risk_tokens']} at risk, "
        f"{summary['low_tokens']} low, "
        f"{summary['critical_tokens']} critical, "
        f"{summary['depleted_tokens']} depleted"
    )
    logger.info(health_summary)


def start_token_stats_reporter(token_manager):
    """Start a thread to periodically log token statistics.
    
    Args:
        token_manager: TokenManager instance.
        
    Returns:
        Thread: The started thread.
    """
    def report_token_stats():
        while True:
            try:
                time.sleep(60)  # Report every minute
                stats = token_manager.get_token_stats()
                summary = stats["_summary"]
                
                # Determine log level based on status
                strategy = summary.get("recommended_strategy", "NORMAL")
                if strategy == "CRITICAL":
                    log_method = logger.warning
                else:
                    log_method = logger.info
                    
                log_method("Token Statistics Update:")
                log_method(f"Total Available: {summary['total_remaining']:,}/{summary['total_limit']:,} "
                          f"({summary['usage_percentage']:.1f}% used)")
                log_method(f"Global Velocity: {summary['global_velocity']:.1f} req/min "
                          f"({summary['requests_per_second']:.2f} req/sec)")
                
                # Show predicted time until exhaustion if available
                if summary.get("minutes_until_all_depleted"):
                    mins = summary["minutes_until_all_depleted"]
                    log_method(f"Predicted time until all tokens exhausted: {mins:.1f} minutes")
                    
                # Show recommended strategy
                log_method(f"Recommended strategy: {strategy}")
                
                # Show token health summary
                health_summary = (
                    f"Status: "
                    f"{summary['healthy_tokens']} healthy, "
                    f"{summary['at_risk_tokens']} at risk, "
                    f"{summary['low_tokens']} low, "
                    f"{summary['critical_tokens']} critical, "
                    f"{summary['depleted_tokens']} depleted"
                )
                log_method(health_summary)
                
                # List tokens with concerning status
                concerning_tokens = []
                for token, metadata in stats.items():
                    if token != "_summary" and isinstance(metadata, dict):
                        status = metadata.get("status", "")
                        if status in ["AT_RISK", "LOW", "CRITICAL", "DEPLETED"]:
                            # Mask token for privacy in logs
                            if len(token) > 8:
                                masked_token = f"{token[:4]}...{token[-4:]}"
                            else:
                                masked_token = "xxxx...xxxx"  # For invalid tokens
                                
                            # Format token status info
                            remaining = metadata.get("remaining", 0)
                            velocity = metadata.get("request_velocity", 0)
                            if metadata.get("predicted_exhaustion_in_minutes") is not None:
                                exhaust_mins = metadata["predicted_exhaustion_in_minutes"]
                                exhaust_info = f", exhausts in {exhaust_mins:.1f}m"
                            else:
                                exhaust_info = ""
                                
                            concerning_tokens.append(
                                f"{masked_token}: {remaining} remaining, {velocity:.1f} req/min{exhaust_info} [{status}]"
                            )
                
                if concerning_tokens:
                    logger.warning(f"Concerning tokens: {', '.join(concerning_tokens)}")
                    
            except Exception as e:
                logger.error(f"Error in token stats reporting: {e}")
    
    # Start stats reporting thread
    stats_thread = threading.Thread(target=report_token_stats, daemon=True)
    stats_thread.start()
    return stats_thread


def print_final_token_stats(token_manager):
    """Print final token statistics at the end of a run.
    
    Args:
        token_manager: TokenManager instance.
    """
    stats = token_manager.get_token_stats()
    summary = stats["_summary"]
    logger.info("\n------------------------------------------------------------")
    logger.info("FINAL TOKEN STATISTICS")
    logger.info("------------------------------------------------------------")
    
    # Summary section
    logger.info("Summary:")
    logger.info(f"Total Tokens: {summary['total_tokens']}, Reserved: {summary['reserved_tokens']}")
    logger.info(f"Total Available Requests: {summary['total_remaining']:,}/{summary['total_limit']:,} "
               f"({summary['usage_percentage']:.1f}% used)")
    logger.info(f"Final Global Velocity: {summary['global_velocity']:.1f} req/min "
              f"({summary['requests_per_second']:.2f} req/sec)")
    
    # Show token health summary
    health_summary = (
        f"Token Health: "
        f"{summary['healthy_tokens']} healthy, "
        f"{summary['at_risk_tokens']} at risk, "
        f"{summary['low_tokens']} low, "
        f"{summary['critical_tokens']} critical, "
        f"{summary['depleted_tokens']} depleted"
    )
    logger.info(health_summary)
    
    # Print individual token usage
    logger.info("\nIndividual Token Usage:")
    try:
        # Filter out non-token entries and sort by usage
        token_items = []
        for t, m in stats.items():
            if t != "_summary" and isinstance(m, dict) and "total_used" in m:
                token_items.append((t, m))
        
        # Sort by total_used (descending)        
        sorted_tokens = sorted(token_items, key=lambda x: x[1].get("total_used", 0), reverse=True)
        
        # Table header
        header = f"{'Token':<13} {'Status':<9} {'Used':<6} {'Remaining':<10} {'Velocity':<10} {'Exhausts In':<12} {'Resets In':<10} {'Success':<8}"
        logger.info(header)
        logger.info("-" * 80)
        
        for token, metadata in sorted_tokens:
            # Mask token for privacy in logs
            if len(token) > 8:
                masked_token = f"{token[:4]}...{token[-4:]}"
            else:
                masked_token = "xxxx...xxxx"  # For invalid tokens
                
            used = metadata.get("total_used", 0)
            remaining = metadata.get("remaining", 0)
            velocity = metadata.get("request_velocity", 0)
            status = metadata.get("status", "UNKNOWN")
            
            # Format reset time
            reset_mins = round(metadata.get("reset_in_minutes", 0))
            
            # Format exhaustion time if available
            if metadata.get("predicted_exhaustion_in_minutes") is not None:
                exhaust_mins = metadata["predicted_exhaustion_in_minutes"]
                exhaust_str = f"{exhaust_mins:.1f}m"
            else:
                exhaust_str = "N/A"
                
            success_rate = metadata.get("success_rate", 0) * 100
            
            # Reserved indicator
            token_type = "[R]" if metadata.get("reserved", False) else ""
            
            # Assemble the row
            row = (
                f"{masked_token+token_type:<13} "
                f"{status:<9} "
                f"{used:<6} "
                f"{remaining:<10} "
                f"{velocity:.1f}/min  "
                f"{exhaust_str:<12} "
                f"{reset_mins}m      "
                f"{success_rate:.1f}%"
            )
            logger.info(row)
        
        logger.info("------------------------------------------------------------")
        
    except Exception as e:
        logger.error(f"Error printing token statistics: {e}")