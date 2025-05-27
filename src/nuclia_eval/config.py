# PROJECT_ROOT/config.py
import os
from dotenv import load_dotenv
from pathlib import Path

# Define the project root dynamically
PROJECT_ROOT = Path(__file__).parent.resolve()
ENV_PATH = PROJECT_ROOT / '.env'

load_dotenv(override=True)  # Load .env file, overriding existing env vars

class AppConfig:
    """
    Application configuration class.
    Loads necessary configurations from environment variables.
    """
    NUCLIA_KB_URL: str = os.getenv("NUCLIA_KB_URL", "")
    NUCLIA_API_KEY: str = os.getenv("NUCLIA_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "") # New
    

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
        if not AppConfig.OPENAI_API_KEY: # New validation
            raise ValueError("OPENAI_API_KEY is not set in the .env file or environment variables. This is needed for LLM-based evaluation.")

# Instantiate and validate configuration globally
try:
    AppConfig.validate_config()
except ValueError as e:
    # In a real app, use logger here and potentially exit
    print(f"CRITICAL Configuration Error: {e}. Please check your .env file.")
    # For this script, if OPENAI_API_KEY is missing, evaluation will fail later.
    # The script logic should ideally handle this more gracefully if evaluation is optional.
    # For now, we make it a critical config.