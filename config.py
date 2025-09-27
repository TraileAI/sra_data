"""
Configuration management for SRA Data Processing.
Handles environment variables and database configuration for Render deployment.
"""

import os
from dotenv import load_dotenv

# Load .env file for local development
load_dotenv()

class Config:
    """Configuration class that handles both local and Render environments."""

    def __init__(self):
        self._db_config = None
        self._load_config()

    def _load_config(self):
        """Load configuration from environment variables."""

        # Check if we're running on Render (has DATABASE_URL)
        database_url = os.getenv('DATABASE_URL')

        if database_url:
            # Parse Render's DATABASE_URL format
            self._parse_database_url(database_url)
        else:
            # Use individual environment variables (local development)
            self._load_individual_vars()

    def _parse_database_url(self, database_url):
        """Parse DATABASE_URL into individual components."""
        try:
            # Format: postgresql://user:password@host:port/database
            from urllib.parse import urlparse

            parsed = urlparse(database_url)

            self._db_config = {
                'host': parsed.hostname,
                'port': parsed.port or 5432,
                'database': parsed.path.lstrip('/'),
                'user': parsed.username,
                'password': parsed.password
            }

            print(f"📊 Using DATABASE_URL configuration")
            print(f"   Host: {self._db_config['host']}")
            print(f"   Database: {self._db_config['database']}")
            print(f"   User: {self._db_config['user']}")

        except Exception as e:
            print(f"❌ Error parsing DATABASE_URL: {e}")
            self._load_individual_vars()

    def _load_individual_vars(self):
        """Load individual DB environment variables."""
        self._db_config = {
            'host': os.getenv('DB_HOST'),
            'port': int(os.getenv('DB_PORT', '5432')),
            'database': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD', '')
        }

        print(f"🔧 Using individual environment variables")
        print(f"   Host: {self._db_config['host']}")
        print(f"   Database: {self._db_config['database']}")
        print(f"   User: {self._db_config['user']}")

        # Validate required variables
        required_vars = ['host', 'database', 'user']
        missing_vars = [var for var in required_vars if not self._db_config[var]]

        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")

    @property
    def db_config(self):
        """Get database configuration dictionary."""
        return self._db_config.copy()

    def get_connection_string(self):
        """Get PostgreSQL connection string."""
        return (f"postgresql://{self._db_config['user']}:{self._db_config['password']}"
                f"@{self._db_config['host']}:{self._db_config['port']}/{self._db_config['database']}")

# Global configuration instance
config = Config()