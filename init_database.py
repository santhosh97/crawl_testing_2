#!/usr/bin/env python3
"""
Script to initialize the database schema for GitHub Stars Crawler.
Creates the repositories and star_records tables if they don't exist.
"""

from sqlalchemy import create_engine, text
import os

def init_database():
    # Get database URL from environment
    db_url = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/github_stars')
    print(f'Connecting to database: {db_url}')

    # Connect to database
    engine = create_engine(db_url)

    # Create tables
    with engine.connect() as conn:
        # Check if tables exist
        result = conn.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema='public' 
            AND table_name IN ('repositories', 'star_records')
        """))
        existing_tables = [row[0] for row in result]
        
        print(f'Existing tables: {existing_tables}')
        
        # Create repositories table if it doesn't exist
        if 'repositories' not in existing_tables:
            print('Creating repositories table...')
            conn.execute(text("""
                CREATE TABLE repositories (
                    id SERIAL PRIMARY KEY,
                    github_id VARCHAR NOT NULL UNIQUE,
                    name VARCHAR NOT NULL,
                    owner VARCHAR NOT NULL,
                    full_name VARCHAR NOT NULL,
                    url VARCHAR NOT NULL,
                    description TEXT,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    fetched_at TIMESTAMP NOT NULL DEFAULT NOW()
                )
            """))
            conn.execute(text('CREATE INDEX idx_repo_github_id ON repositories (github_id)'))
            conn.execute(text('CREATE INDEX idx_repo_full_name ON repositories (full_name)'))
            conn.commit()
            print('Created repositories table')
        
        # Create star_records table if it doesn't exist
        if 'star_records' not in existing_tables:
            print('Creating star_records table...')
            conn.execute(text("""
                CREATE TABLE star_records (
                    id SERIAL PRIMARY KEY,
                    repository_id INTEGER NOT NULL REFERENCES repositories(id),
                    star_count INTEGER NOT NULL,
                    recorded_at TIMESTAMP NOT NULL DEFAULT NOW()
                )
            """))
            conn.execute(text('CREATE INDEX idx_star_repository_id ON star_records (repository_id)'))
            conn.commit()
            print('Created star_records table')
        
        print('Database initialization completed')

if __name__ == "__main__":
    init_database()