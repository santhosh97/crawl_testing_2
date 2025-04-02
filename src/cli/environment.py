"""Environment configuration management for the GitHub stars crawler."""

import os
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from dotenv import load_dotenv, dotenv_values

logger = logging.getLogger(__name__)

class Environment:
    """Manages environment configuration without global state."""
    
    def __init__(self, env_file: Optional[str] = None):
        """Initialize the environment configuration.
        
        Args:
            env_file: Path to .env file (optional)
        """
        self._values = {}
        self.env_file_path = None
        
        # Load from specified .env file or search for one
        if env_file:
            self.load_env_file(env_file)
        else:
            # Try to find default .env file
            project_root = Path(__file__).resolve().parents[2]  # Go up 2 dirs from cli/environment.py
            default_env_path = project_root / '.env'
            if default_env_path.exists():
                self.load_env_file(str(default_env_path))
                
        # Always load from os.environ to allow overrides
        self._values.update(os.environ)
        
        logger.debug("Environment initialized")
        
    def load_env_file(self, dotenv_path: str) -> bool:
        """Load environment variables from .env file.
        
        Args:
            dotenv_path: Path to .env file
            
        Returns:
            True if file was loaded successfully, False otherwise
        """
        env_path = Path(dotenv_path)
        if not env_path.exists():
            logger.warning(f".env file not found at {dotenv_path}")
            return False
            
        logger.info(f"Loading environment variables from: {dotenv_path}")
        
        # Load values from .env file without modifying os.environ
        env_values = dotenv_values(dotenv_path=dotenv_path)
        self._values.update(env_values)
        
        # Store the loaded file path
        self.env_file_path = dotenv_path
        
        # Log loaded values (masked sensitive data)
        self._log_loaded_values()
        
        return True
        
    def _log_loaded_values(self):
        """Log loaded environment variables with sensitive data masked."""
        for key in ['DB_USER', 'DB_HOST', 'DB_NAME']:
            if key in self._values:
                logger.info(f"Loaded {key}={self._values[key]}")
                
        # Mask passwords and tokens
        for key in ['DB_PASSWORD', 'TOKEN']:
            if key in self._values and self._values[key]:
                masked = '*' * len(self._values[key])
                logger.info(f"Loaded {key}={masked}")
                
    def get(self, key: str, default: Any = None) -> Any:
        """Get an environment variable.
        
        Args:
            key: Environment variable name
            default: Default value if not found
            
        Returns:
            Value of environment variable or default
        """
        return self._values.get(key, default)
        
    def set(self, key: str, value: Any):
        """Set an environment variable (does not modify os.environ).
        
        Args:
            key: Environment variable name
            value: Value to set
        """
        self._values[key] = value
        
    def get_database_url(self) -> Optional[str]:
        """Get validated DATABASE_URL.
        
        Returns:
            Valid DATABASE_URL string or None if invalid
        """
        # Try to get explicit DATABASE_URL
        database_url = self.get("DATABASE_URL")
        
        if not database_url:
            # Build from components
            db_user = self.get("DB_USER", "postgres")
            db_password = self.get("DB_PASSWORD", "postgres")
            db_host = self.get("DB_HOST", "localhost")
            db_port = self.get("DB_PORT", "5432")
            db_name = self.get("DB_NAME", "github_stars")
            
            # Construct URL
            database_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
            logger.warning(f"DATABASE_URL not set, constructed from components: {self._mask_password(database_url)}")
            
            # Store for future use
            self.set("DATABASE_URL", database_url)
            
        # Validate PostgreSQL URL
        if not database_url.startswith("postgresql://"):
            logger.error("Only PostgreSQL is supported. DATABASE_URL must start with 'postgresql://'")
            return None
            
        return database_url
        
    def _mask_password(self, database_url: str) -> str:
        """Mask password in database URL for logging.
        
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
                    if password:
                        return database_url.replace(password, "*" * len(password))
        
        # Return as is if we can't extract the password
        return database_url
        
    def get_github_tokens(self) -> Optional[List[str]]:
        """Get and validate GitHub tokens.
        
        Returns:
            List of valid GitHub tokens or None if none found
        """
        github_token = self.get("TOKEN")
        if not github_token:
            logger.error("TOKEN environment variable is not set")
            return None
            
        # Import the token parsing function
        from src.api.token_management import parse_github_tokens
        tokens = parse_github_tokens(github_token)
        
        if not tokens:
            logger.error("No valid GitHub tokens found")
            return None
            
        logger.info(f"Using {len(tokens)} GitHub tokens")
        return tokens
        
    def validate_postgres_config(self) -> bool:
        """Validate PostgreSQL database configuration.
        
        Returns:
            True if all required fields are valid, False otherwise
        """
        db_host = self.get("DB_HOST", "")
        db_port = self.get("DB_PORT", "")
        db_user = self.get("DB_USER", "")
        db_password = self.get("DB_PASSWORD", "")
        db_name = self.get("DB_NAME", "")
        
        # Check for all required PostgreSQL fields
        missing_fields = []
        required_fields = {
            "DB_HOST": db_host,
            "DB_PORT": db_port,
            "DB_USER": db_user,
            "DB_PASSWORD": db_password,
            "DB_NAME": db_name
        }
        
        for field, value in required_fields.items():
            if not value:
                missing_fields.append(field)
                
        # Log missing fields
        if missing_fields:
            logger.error("CRITICAL ERROR: Missing required PostgreSQL configuration")
            for field in missing_fields:
                logger.error(f"Missing or invalid: {field}")
            logger.error("Please update .env file to match .env.template format")
            return False
            
        # Check if explicit DATABASE_URL is set and valid
        database_url = self.get("DATABASE_URL", "")
        if database_url and not database_url.startswith("postgresql://"):
            logger.error("DATABASE_URL must start with 'postgresql://'")
            return False
            
        return True
        
    def as_dict(self) -> Dict[str, str]:
        """Get all environment variables as a dictionary.
        
        Returns:
            Dictionary of all environment variables
        """
        return dict(self._values)


