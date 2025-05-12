# legal_rag_mvp/config.py
"""
Handles application configuration, including loading environment variables
and defining static paths and parameters.
"""

import os
import warnings
from dotenv import load_dotenv
from pathlib import Path

# --- Project Root and Data Directories ---
PROJECT_ROOT_DIR: Path = Path(__file__).resolve().parent.parent
ENV_PATH: Path = PROJECT_ROOT_DIR / ".env"

# Load environment variables from .env file
if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
    # print(f"INFO: Loaded configurations from: {ENV_PATH}") # Verbose, for debugging
else:
    warnings.warn(
        f".env file not found at {ENV_PATH}. "
        "Ensure it exists for loading sensitive credentials.",
        UserWarning
    )

# --- Nuclia API Credentials and Knowledge Box (KB) Identifiers ---
# These should be set in your .env file.
NUCLIA_USER_TOKEN: str | None = os.getenv("NUCLIA_USER_TOKEN")
NUCLIA_API_KEY: str | None = os.getenv("NUCLIA_API_KEY") # NUA key for a specific KB
NUCLIA_KB_ID: str | None = os.getenv("NUCLIA_KB_ID")     # Default KB ID if using one main KB
NUCLIA_KB_URL: str | None = os.getenv("NUCLIA_KB_URL")   # Base URL for the KB API

# Example: If you plan to use different KB IDs per corpus, you might manage them like this:
# NUCLIA_KB_ID_PRIVACY_QA: str | None = os.getenv("NUCLIA_KB_ID_PRIVACY_QA")
# NUCLIA_KB_ID_CONTRACTNLI: str | None = os.getenv("NUCLIA_KB_ID_CONTRACTNLI")
# Or, more generically, a base slug for KBs if you create them programmatically or follow a pattern.

# Perform a basic check for essential Nuclia configurations
_essential_nuclia_configs = {
    "NUCLIA_API_KEY": NUCLIA_API_KEY,
    "NUCLIA_KB_ID": NUCLIA_KB_ID, # Or ensure a mechanism to select the correct KB ID is in place
    "NUCLIA_KB_URL": NUCLIA_KB_URL
}
for config_name, config_value in _essential_nuclia_configs.items():
    if not config_value:
        warnings.warn(
            f"Essential Nuclia configuration '{config_name}' is missing. "
            "This may impact the application's ability to interact with Nuclia. "
            "Please check your .env file or environment variables.",
            UserWarning
        )

# --- Data Paths Configuration ---
_BASE_DATA_ROOT_NAME = "data"
_BENCHMARK_SET_NAME = "LegalBench-RAG" # Main folder for the benchmark

BASE_DATA_DIR: Path = PROJECT_ROOT_DIR / _BASE_DATA_ROOT_NAME / _BENCHMARK_SET_NAME
CORPUS_BASE_DIR: Path = BASE_DATA_DIR / "corpus"
BENCHMARK_BASE_DIR: Path = BASE_DATA_DIR / "benchmarks"

# --- Benchmark File Configuration ---
# One JSON file per sub-corpus in the 'benchmarks' folder.
# The DataLoader will expect filenames like 'privacy_qa.json', 'contractnli.json', etc.

# --- Available Corpora & Default Settings ---
AVAILABLE_CORPUS_NAMES: list[str] = ["privacy_qa", "contractnli", "cuad", "maud"]
DEFAULT_CORPUS_NAME: str = os.getenv("DEFAULT_CORPUS_NAME", "privacy_qa")
if DEFAULT_CORPUS_NAME not in AVAILABLE_CORPUS_NAMES:
    warnings.warn(
        f"DEFAULT_CORPUS_NAME ('{DEFAULT_CORPUS_NAME}') from .env is not in "
        f"AVAILABLE_CORPUS_NAMES. Falling back to 'privacy_qa'.",
        UserWarning
    )
    DEFAULT_CORPUS_NAME = "privacy_qa"

MVP_DEFAULT_NUM_QUESTIONS: int = 15 # Default number of questions for MVP evaluation

# --- Logging Configuration ---
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
# Ensure LOG_LEVEL is one of the standard Python logging levels for robustness
_valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
if LOG_LEVEL not in _valid_log_levels:
    warnings.warn(
        f"Invalid LOG_LEVEL '{LOG_LEVEL}' from .env. Must be one of {_valid_log_levels}. "
        "Defaulting to 'INFO'.",
        UserWarning
    )
    LOG_LEVEL = "INFO"

# --- Type Hinting for Configuration Variables (Optional but good practice) ---
# This doesn't change runtime behavior but helps with static analysis and IDEs.
# (Already done by adding type hints to the variable declarations above)


# --- Optional: Function to print loaded config for debugging ---
def print_loaded_configuration() -> None:
    """Prints a summary of the loaded configurations for verification."""
    print("--- Application Configuration Summary ---")
    print(f"Project Root Directory: {PROJECT_ROOT_DIR}")
    print(f"Data Directory (LegalBench-RAG): {BASE_DATA_DIR}")
    print(f"Corpus Base Directory: {CORPUS_BASE_DIR}")
    print(f"Benchmarks Base Directory: {BENCHMARK_BASE_DIR}")
    print(f"Default Corpus for MVP: {DEFAULT_CORPUS_NAME}")
    print(f"Default # Questions for MVP: {MVP_DEFAULT_NUM_QUESTIONS}")
    print(f"Log Level: {LOG_LEVEL}")
    print(f"Nuclia KB URL: {NUCLIA_KB_URL if NUCLIA_KB_URL else 'NOT SET'}")
    print(f"Nuclia KB ID (Default/Active): {NUCLIA_KB_ID if NUCLIA_KB_ID else 'NOT SET'}")
    print(f"Nuclia API Key Loaded: {'YES' if NUCLIA_API_KEY else 'NO - ESSENTIAL!'}")
    print(f"Nuclia User Token Loaded: {'YES' if NUCLIA_USER_TOKEN else 'NO (Potentially needed for some ops)'}")
    print("---------------------------------------")

# Example of how you might call this for debugging when the app starts:
# if __name__ == "__main__":
#     print_loaded_configuration()