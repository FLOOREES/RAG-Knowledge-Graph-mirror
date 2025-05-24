# PROJECT_ROOT/config.py
import os
from dotenv import load_dotenv
from pathlib import Path

# Define the project root dynamically
# This assumes config.py is at PROJECT_ROOT/config.py
PROJECT_ROOT = Path(__file__).parent.resolve()
# Load environment variables from .env file located at the project root
ENV_PATH = PROJECT_ROOT / '.env'

load_dotenv(dotenv_path=ENV_PATH)

class AppConfig:
    """
    Application configuration class.
    Loads necessary configurations from environment variables.
    """
    NUCLIA_KB_URL: str = os.getenv("NUCLIA_KB_URL", "")
    NUCLIA_API_KEY: str = os.getenv("NUCLIA_API_KEY", "")

    @staticmethod
    def validate_config() -> None:
        """
        Validates that essential configurations are set.
        Raises ValueError if a required config is missing.
        """
        if not AppConfig.NUCLIA_KB_URL:
            raise ValueError("NUCLIA_KB_URL is not set in the .env file or environment variables.")
        if not AppConfig.NUCLIA_API_KEY:
            raise ValueError("NUCLIA_API_KEY is not set in the .env file or environment variables.")

# Instantiate and validate configuration globally for easy access
# This way, validation happens once at import time if this module is imported.
try:
    AppConfig.validate_config()
except ValueError as e:
    # Handle this critical error, perhaps by logging and exiting if in a script context
    # For now, we'll print and allow to proceed to show where it would be caught
    print(f"Configuration Error: {e}") # In a real app, use logger here