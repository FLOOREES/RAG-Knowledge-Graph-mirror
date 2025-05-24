# PROJECT_ROOT/nuclia_eval/pipeline.py
import logging
from typing import Dict, Any, Optional # Added Optional

from src.nuclia_eval.config import AppConfig
from src.nuclia_eval.nuclia_client import NucliaClientWrapper
from src.utils.logger_setup import setup_logger

logger = setup_logger(__name__)

class EvaluationPipeline:
    def __init__(self):
        logger.info("Initializing Evaluation Pipeline...")
        self.config = AppConfig()
        try:
            self.nuclia_client = NucliaClientWrapper(
                kb_url=self.config.NUCLIA_KB_URL,
                api_key=self.config.NUCLIA_API_KEY
            )
            logger.info("NucliaClientWrapper initialized successfully.")
        except Exception as e:
            logger.critical(f"Failed to initialize NucliaClientWrapper: {e}", exc_info=True)
            raise SystemExit(f"Critical: Could not initialize Nuclia Client. Details: {e}")

    def run_single_query_evaluation(
            self, 
            question: str, 
            generative_model_override: Optional[str] = None # New parameter
        ) -> Dict[str, Any]:
        """
        Runs a single query against Nuclia and returns the processed results.
        Optionally allows overriding the generative model used by Nuclia.
        """
        logger.info(f"Running single query evaluation for: '{question}'")
        if generative_model_override:
            logger.info(f"  Using Nuclia generation model override: {generative_model_override}")
        
        try:
            # Pass the override to the nuclia_client
            results = self.nuclia_client.query_knowledge_graph(
                question,
                generative_model_override=generative_model_override # New
            )
            
            logger.info(f"Question: {results.get('question')}")
            logger.info(f"Generated Answer (snippet): {str(results.get('answer'))[:100]}...")
            # ... (other specific logging of results can remain or be adjusted)
            return results
        except ConnectionError as e:
            logger.error(f"Could not connect or query Nuclia: {e}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred for '{question}': {e}", exc_info=True)
            raise