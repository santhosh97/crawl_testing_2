#!/usr/bin/env python3
"""GitHub Stars Crawler main entry point."""

import sys
from pathlib import Path

from src.core.application import Application
from src.api.github_exceptions import ApplicationException, InitializationError
from src.utils.error_handling import log_error
from src.utils.logging_config import LogManager
from src.utils.path_manager import PathManager
from src.cli.environment import Environment

def main():
    """Main entry point for the GitHub Stars Crawler."""
    # Set up path manager
    path_manager = PathManager()
    
    # Set up log manager using path manager
    log_manager = LogManager(
        logs_dir=path_manager.get_logs_dir(),
        metrics_dir=path_manager.get_metrics_dir()
    )
    logger = log_manager.get_logger(__name__)
    
    try:
        # Create all dependencies explicitly
        environment = Environment()
        
        # Create and initialize application with explicit dependencies
        app = Application(
            log_manager=log_manager,
            path_manager=path_manager,
            environment=environment
        ).initialize()
        
        # Run the application and get exit code
        exit_code = app.run()
        
        # Clean up resources
        app.cleanup()
        
        return exit_code
    except InitializationError as e:
        # Handle specific initialization errors
        log_error(logger, "Application initialization failed", exception=e, level="critical", 
                  component="main", operation="initialize")
        return 2
    except ApplicationException as e:
        # Handle known application errors
        log_error(logger, "Application error", exception=e, level="critical",
                  component="main", operation="run")
        return 1
    except Exception as e:
        # Handle unexpected errors
        log_error(logger, "Fatal error", exception=e, level="critical",
                  component="main", operation="unknown")
        return 3

if __name__ == "__main__":
    sys.exit(main())