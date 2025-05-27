# PROJECT_ROOT/scripts/pre_fetch_all_paragraphs.py
import argparse
import json
import logging
from pathlib import Path
from typing import List, Dict, Set, Optional

# Assuming AppConfig is in a top-level config.py
# and retrieve_paragraphs is in a utility module.
# Adjust these imports based on your project structure.
try:
    from src.nuclia_eval.config import AppConfig
    # Assuming retrieve_paragraphs is in a utility file, e.g., src/utils/retrieval_utils.py
    # If it's defined elsewhere (like in your gnn_pipeline.py), adjust the import path.
    # For this script, it's cleaner if retrieve_paragraphs is a general utility.
    # If not easily importable, you might need to copy its definition here or to a shared util.
    # Let's assume it's in:
    from src.graph_processing_utils import retrieve_paragraphs
except ImportError as e:
    print(f"Error importing necessary modules. Ensure config.py and data_retrieval_utils.py (with retrieve_paragraphs) are accessible: {e}")
    print("You might need to set your PYTHONPATH or adjust import paths.")
    exit(1)

# Setup logger for this script
# You can use your existing setup_logger or a simple basicConfig for standalone scripts
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)


def extract_and_fetch_paragraphs(
    relations_filepath: Path, 
    output_paragraphs_filepath: Path,
    kb_url: str,
    api_key: str
) -> None:
    """
    Extracts all unique paragraph_ids from a relations JSON file,
    retrieves their text content using the Nuclia API, and saves
    them to an output JSON file.

    Args:
        relations_filepath: Path to the input JSON file containing relations data.
                            Expected format: List of dicts, each with metadata.paragraph_id.
        output_paragraphs_filepath: Path to save the retrieved paragraphs (JSON mapping ID to text).
        kb_url: Nuclia Knowledge Box URL.
        api_key: Nuclia API Key.
    """
    if not relations_filepath.exists():
        logger.error(f"Input relations file not found: {relations_filepath}")
        return

    logger.info(f"Loading relations from: {relations_filepath}")
    try:
        with open(relations_filepath, 'r', encoding='utf-8') as f:
            relations_data = json.load(f)
        if not isinstance(relations_data, list):
            logger.error(f"Relations file {relations_filepath} does not contain a JSON list.")
            return
    except json.JSONDecodeError:
        logger.error(f"Error decoding JSON from {relations_filepath}.", exc_info=True)
        return
    except Exception as e:
        logger.error(f"Error reading relations file {relations_filepath}: {e}", exc_info=True)
        return

    unique_paragraph_ids: Set[str] = set()
    for i, relation_obj in enumerate(relations_data):
        if not isinstance(relation_obj, dict):
            logger.debug(f"Item {i} in relations data is not a dictionary. Skipping.")
            continue
        
        metadata = relation_obj.get("metadata")
        if isinstance(metadata, dict):
            paragraph_id = metadata.get("paragraph_id")
            if isinstance(paragraph_id, str) and paragraph_id.strip():
                unique_paragraph_ids.add(paragraph_id.strip())
            else:
                logger.debug(f"Relation object {i} has metadata but no valid 'paragraph_id' string: {metadata}")
        else:
            logger.debug(f"Relation object {i} has no 'metadata' dictionary or it's invalid.")
            
    if not unique_paragraph_ids:
        logger.warning(f"No unique paragraph_ids found in {relations_filepath}. Output file will not be created.")
        return

    logger.info(f"Found {len(unique_paragraph_ids)} unique paragraph IDs to fetch.")

    # Fetch paragraphs using the provided utility function
    # This function already includes internal caching for resources per run.
    retrieved_paragraph_texts: Dict[str, str] = retrieve_paragraphs(
        list(unique_paragraph_ids), kb_url, api_key
    )

    if not retrieved_paragraph_texts:
        logger.warning("No paragraph texts were retrieved. Output file will be empty or not created.")
        # Depending on desired behavior, you might still want to write an empty JSON object.
        # For now, let's only write if we have content.
        return

    # Ensure output directory exists
    output_paragraphs_filepath.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving {len(retrieved_paragraph_texts)} retrieved paragraphs to: {output_paragraphs_filepath}")
    try:
        with open(output_paragraphs_filepath, 'w', encoding='utf-8') as f:
            json.dump(retrieved_paragraph_texts, f, indent=4)
        logger.info("Successfully saved paragraphs.")
    except IOError:
        logger.error(f"Error writing paragraphs to {output_paragraphs_filepath}.", exc_info=True)
    except TypeError: # For non-serializable content, though retrieve_paragraphs should return Dict[str,str]
        logger.error(f"TypeError, content might not be JSON serializable when writing to {output_paragraphs_filepath}.", exc_info=True)


def main():
    parser = argparse.ArgumentParser(
        description="Pre-fetches all paragraph texts from a Nuclia KB based on paragraph IDs "
                    "found in a relations JSON file and saves them to a local cache file."
    )
    parser.add_argument(
        "input_relations_file", 
        type=Path,
        help="Path to the input JSON file containing relations (e.g., legal_graph.json)."
    )
    parser.add_argument(
        "output_paragraphs_file", 
        type=Path,
        help="Path to the output JSON file where fetched paragraphs will be saved (e.g., data/kb_paragraphs_cache.json)."
    )
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)."
    )
    
    args = parser.parse_args()

    # Reconfigure logger level based on CLI argument
    logger.setLevel(args.log_level.upper())
    # If retrieve_paragraphs uses its own logger, you might want to set its level too.
    logging.getLogger("src.utils.data_retrieval_utils").setLevel(args.log_level.upper()) # Example

    logger.info("Starting paragraph pre-fetch process...")
    
    try:
        app_config = AppConfig()
        # Validate essential configs from AppConfig if not done globally
        if not app_config.NUCLIA_KB_URL or not app_config.NUCLIA_API_KEY:
            logger.critical("Nuclia KB URL or API Key is not configured in .env or AppConfig. Exiting.")
            return
        # AppConfig.validate_config() # If you have a method that raises on missing keys
            
        extract_and_fetch_paragraphs(
            args.input_relations_file,
            args.output_paragraphs_file,
            app_config.NUCLIA_KB_URL,
            app_config.NUCLIA_API_KEY
        )
        logger.info("Paragraph pre-fetch process completed.")
        
    except SystemExit: # Catch SystemExit if AppConfig validation itself fails critically
        logger.info("Exiting due to configuration issues.")
    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    # Example usage:
    # python scripts/pre_fetch_all_paragraphs.py data/gnn_data/legal_graph_for_gnn.json data/kb_paragraphs_cache.json --log_level DEBUG
    main()