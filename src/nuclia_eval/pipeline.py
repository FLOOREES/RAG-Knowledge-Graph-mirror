# PROJECT_ROOT/nuclia_eval/pipeline.py
import logging
from typing import Dict, Any

from config import AppConfig # Assuming config.py is in PROJECT_ROOT, adjust if placed elsewhere
from nuclia_client import NucliaClientWrapper # Relative import
from utils.logger_setup import setup_logger # Assuming utils is in PYTHONPATH or relative import works

# Configure a module-specific logger
logger = setup_logger(__name__)

class EvaluationPipeline:
    """
    Orchestrates the process of querying Nuclia for evaluation purposes.
    """

    def __init__(self):
        """
        Initializes the EvaluationPipeline.
        Loads configuration and sets up the Nuclia client.
        """
        logger.info("Initializing Evaluation Pipeline...")
        # Configuration is loaded when config.py is imported.
        # AppConfig.validate_config() is called in config.py to ensure vars are present.
        self.config = AppConfig()
        
        try:
            self.nuclia_client = NucliaClientWrapper(
                kb_url=self.config.NUCLIA_KB_URL,
                api_key=self.config.NUCLIA_API_KEY
            )
            logger.info("NucliaClientWrapper initialized successfully.")
        except Exception as e:
            logger.critical(f"Failed to initialize NucliaClientWrapper: {e}", exc_info=True)
            # This is a critical failure for the pipeline.
            raise SystemExit(f"Critical: Could not initialize Nuclia Client. Details: {e}")


    def run_single_query_evaluation(self, question: str) -> Dict[str, Any]:
        """
        Runs a single query against Nuclia and returns the processed results.

        Args:
            question (str): The question to send to Nuclia.

        Returns:
            Dict[str, Any]: A dictionary containing the question, answer,
                            relations, and citations.
        """
        logger.info(f"Running single query evaluation for: '{question}'")
        try:
            results = self.nuclia_client.query_knowledge_graph(question)
            
            # Log the key parts of the result for now
            logger.info(f"Question: {results.get('question')}")
            logger.info(f"Generated Answer: {results.get('answer')}")
            logger.info(f"Number of Retrieved Relations: {len(results.get('relations', []))}")
            logger.info(f"Number of Citations: {len(results.get('citations', []))}")
            logger.info(f"Number of Retrieved Context Paragraphs: {len(results.get('retrieved_context_paragraphs', []))}")

            if results.get('relations'):
                logger.debug(f"First retrieved relation (sample): {results['relations'][0] if results['relations'] else 'N/A'}")
            
            return results
        except ConnectionError as e: # Example of catching a specific error from client
            logger.error(f"Could not connect or query Nuclia: {e}", exc_info=True)
            # For now, re-raise or return error state. In a batch job, might continue.
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during query evaluation for '{question}': {e}", exc_info=True)
            raise