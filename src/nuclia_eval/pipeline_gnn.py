# PROJECT_ROOT/nuclia_eval/pipeline.py
import logging
from typing import Dict, Any, Optional # Added Optional

from src.nuclia_eval.config import AppConfig
from src.nuclia_eval.similarity_querying import PathRGCNRetriever
from src.nuclia_eval.gnn_querying import PathRGCNRetrieverTrained
from src.nuclia_eval.msps_querying import MSPN_Search
from src.utils.logger_setup import setup_logger
from typing import Literal

logger = setup_logger(__name__)

class GNNEvaluationPipeline:
    def __init__(self, gnn_method: Literal["GNN", "MSPN", "Similarity"] = "GNN"):
        logger.info("Initializing Evaluation Pipeline...")
        self.config = AppConfig()
        try:
            if gnn_method == "MSPN":
                logger.info("Using MPNN method for GNN evaluation.")
                self.gnn_model = MSPN_Search()
            elif gnn_method == "Similarity":
                logger.info("Using Similarty method for evaluation.")
                self.gnn_model = PathRGCNRetriever()
            else:
                logger.info("Using default GNN method for evaluation.")
                self.gnn_model = PathRGCNRetrieverTrained()
        except Exception as e:
            logger.critical(f"Failed to initialize NucliaClientWrapper: {e}", exc_info=True)
            raise SystemExit(f"Critical: Could not initialize Nuclia Client. Details: {e}")

    def run_single_query_evaluation(
            self, 
            question: str, 
            gnn_method: Optional[str] = None, # Can be GNN, MPNN or similarity
            generative_model_override: Optional[str] = None # New parameter for model override
        ) -> Dict[str, Any]:
        """
        Runs a single query against Nuclia and returns the processed results.
        Optionally allows overriding the generative model used by Nuclia.
        """
        logger.info(f"Running single query evaluation for: '{question}'")
        if gnn_method:
            logger.info(f"  Using Nuclia generation model override: {gnn_method}")
        
        try:
            # Pass the override to the nuclia_client
            results = self.gnn_model.query_knowledge_graph(
                question,
                model_override=generative_model_override
            )
            print(f"Results: {results}, type {type(results)}")  # Debugging line to see the raw results
            
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