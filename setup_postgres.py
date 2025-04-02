#!/usr/bin/env python3
"""
Script to setup the PostgreSQL database for GitHub Stars Crawler.
Creates the database if it doesn't exist and initializes the schema.
"""

import os
import sys
import logging
import argparse
import psycopg2
from sqlalchemy import create_engine, inspect, text
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
load_dotenv(dotenv_path=dotenv_path, verbose=True)

def create_database(host, port, user, password, dbname):
    """Create the PostgreSQL database if it doesn't exist."""
    # Connect to PostgreSQL server without specifying a database
    conn = psycopg2.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname="postgres"  # Connect to default database first
    )
    conn.autocommit = True  # Required for CREATE DATABASE
    cursor = conn.cursor()

    # Check if database exists
    cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
    exists = cursor.fetchone()
    
    if not exists:
        logger.info(f"Creating database {dbname}...")
        cursor.execute(f"CREATE DATABASE {dbname}")
        logger.info(f"Database {dbname} created successfully")
    else:
        logger.info(f"Database {dbname} already exists")
    
    cursor.close()
    conn.close()

def main():
    parser = argparse.ArgumentParser(description="Setup PostgreSQL database for GitHub Stars Crawler")
    parser.add_argument(
        "--host", 
        default=os.getenv("DB_HOST", "localhost"),
        help="PostgreSQL host (default: from DB_HOST env var or localhost)"
    )
    parser.add_argument(
        "--port", 
        type=int,
        default=int(os.getenv("DB_PORT", "5432")),
        help="PostgreSQL port (default: from DB_PORT env var or 5432)"
    )
    parser.add_argument(
        "--user",
        default=os.getenv("DB_USER", "postgres"),
        help="PostgreSQL username (default: from DB_USER env var or postgres)"
    )
    parser.add_argument(
        "--password",
        default=os.getenv("DB_PASSWORD", "postgres"),
        help="PostgreSQL password (default: from DB_PASSWORD env var or postgres)"
    )
    parser.add_argument(
        "--dbname",
        default=os.getenv("DB_NAME", "github_stars"),
        help="PostgreSQL database name (default: from DB_NAME env var or github_stars)"
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Drop and recreate all tables (WARNING: destroys all data)"
    )
    parser.add_argument(
        "--force-migrate-from-sqlite",
        action="store_true",
        help="Migrate data from an existing SQLite database to PostgreSQL"
    )
    args = parser.parse_args()

    # Build DATABASE_URL for SQLAlchemy
    db_url = f"postgresql://{args.user}:{args.password}@{args.host}:{args.port}/{args.dbname}"
    
    # Update environment variable for other modules to use
    os.environ["DATABASE_URL"] = db_url
    
    # Remove any existing SQLite-based DATABASE_URL from .env file
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r') as file:
            lines = file.readlines()
        
        with open(env_path, 'w') as file:
            for line in lines:
                if line.startswith('DATABASE_URL=sqlite:'):
                    file.write(f'DATABASE_URL={db_url}\n')
                    logger.info(f"Updated .env file, changed SQLite URL to PostgreSQL: {db_url}")
                else:
                    file.write(line)
                    
    # Check if we need to migrate from SQLite
    if args.force_migrate_from_sqlite:
        logger.warning("SQLite to PostgreSQL migration not implemented yet. Please manually export/import data.")
        logger.info("Continuing with fresh PostgreSQL setup...")
    
    try:
        # Create database if it doesn't exist
        create_database(args.host, args.port, args.user, args.password, args.dbname)
        
        # Import after environment variables are set
        from src.db.database import init_db
        
        # Initialize the schema
        if args.clean:
            logger.warning("--clean flag set, dropping and recreating all tables!")
            init_db(clean=True)
        else:
            init_db()
            
        logger.info("Database setup completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error setting up database: {e}", exc_info=True)
        return 1
        
if __name__ == "__main__":
    sys.exit(main())