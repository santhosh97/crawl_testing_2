"""Command-line argument parsing for the GitHub stars crawler."""

import argparse
import logging

def parse_args():
    """Parse command-line arguments for the GitHub stars crawler.
    
    Returns:
        Namespace containing the parsed arguments.
    """
    parser = argparse.ArgumentParser(description="GitHub Stars Query Pool Crawler")
    parser.add_argument(
        "--repos", 
        type=int, 
        default=1000,
        help="Number of repositories to crawl (default: 1000)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of parallel workers (default: auto based on tokens)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of repositories to process in each database batch (default: 100)"
    )
    parser.add_argument(
        "--clean-db",
        action="store_true",
        help="Clean the database before starting (removes all existing repositories and star records)"
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip printing the final summary (useful for automated runs)"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set logging level (default: INFO)"
    )
    parser.add_argument(
        "--reserve-tokens",
        type=float,
        default=0.2,
        help="Percentage of tokens to reserve for priority operations (0.0-0.5, default: 0.2)"
    )
    parser.add_argument(
        "--token-stats",
        action="store_true",
        help="Show detailed token statistics during execution"
    )
    
    return parser.parse_args()