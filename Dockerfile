FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set environment variables for the database connection
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Command to run when container starts
# Default is to set up the database
ENTRYPOINT ["python", "setup_postgres.py"]

# Default command (can be overridden)
CMD ["--host", "${DB_HOST:-postgres}", "--port", "${DB_PORT:-5432}", "--user", "${DB_USER:-postgres}", "--password", "${DB_PASSWORD:-postgres}", "--dbname", "${DB_NAME:-github_stars}"]