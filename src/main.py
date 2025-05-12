# src/main.py
"""
Main executable script for the Knowledge Graph-based Legal RAG Pipeline.

Supports distinct phases:
1. 'ingest': Uploads corpus documents to Nuclia for KG generation.
2. 'evaluate': Runs retrieval and evaluation against an existing Nuclia KG.
"""

import argparse
import logging
import sys
import time # For a potential wait, though not fully implemented for MVP

from . import config
from .pipeline import LegalRAGPipeline # Will be updated to handle phases


def setup_logging(level_str: str) -> None:
	"""Configures basic stream logging for the application."""
	numeric_level = getattr(logging, level_str.upper(), logging.INFO)
	if not isinstance(numeric_level, int):
		logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
		logging.warning(f"Invalid log level: {level_str}. Defaulting to INFO.")
	else:
		logging.basicConfig(level=numeric_level, format="%(asctime)s - %(levelname)s - %(message)s")
	logging.info(f"Logging configured at level: {logging.getLevelName(numeric_level)}")


def run():
	"""Parses command-line arguments and orchestrates the pipeline execution based on phases."""
	parser = argparse.ArgumentParser(
		description="Run the Knowledge Graph-based Legal RAG Pipeline in specified phases.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter
	)

	parser.add_argument(
		"phase", # Positional argument for the phase
		type=str,
		choices=["ingest", "evaluate"],
		help="The pipeline phase to execute: 'ingest' for uploading data to Nuclia, "
			 "'evaluate' for running retrieval and evaluation."
	)
	parser.add_argument(
		"--corpus",
		type=str,
		default=config.DEFAULT_CORPUS_NAME,
		choices=config.AVAILABLE_CORPUS_NAMES,
		help="Name of the sub-corpus from LegalBench-RAG-mini to process."
	)
	parser.add_argument(
		"--num_questions",
		type=int,
		default=config.MVP_DEFAULT_NUM_QUESTIONS,
		help="Number of questions to select and evaluate (only used in 'evaluate' phase)."
	)
	parser.add_argument(
		"--log_level",
		type=str,
		default=config.LOG_LEVEL,
		choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
		help="Set the console logging output level."
	)
	# Add --max_files_to_upload for ingest phase
	parser.add_argument(
		"--max_files_to_ingest",
		type=int,
		default=None, # Process all files in the corpus folder by default
		help="Maximum number of files to upload during the 'ingest' phase (for quick testing)."
	)

	args = parser.parse_args()

	setup_logging(args.log_level)
	logger = logging.getLogger(__name__)

	logger.info(f"Application started. Executing Phase: '{args.phase}'")
	logger.info(f"Target Corpus: {args.corpus}")

	try:
		pipeline_instance = LegalRAGPipeline(
			corpus_name=args.corpus
			# Nuclia credentials will be picked up from config by the handler
		)

		if args.phase == "ingest":
			logger.info("Starting INGESTION phase...")
			# The pipeline's ingest method will need the path to the corpus files
			corpus_folder_to_ingest = config.CORPUS_BASE_DIR / args.corpus
			pipeline_instance.execute_ingestion_phase(
				corpus_folder_path=corpus_folder_to_ingest,
				max_files_to_upload=args.max_files_to_ingest
			)
			logger.info(
				f"INGESTION phase for corpus '{args.corpus}' submitted to Nuclia. "
				f"Knowledge Graph generation may take time. "
				f"You can run the 'evaluate' phase later."
			)

		elif args.phase == "evaluate":
			logger.info("Starting EVALUATION phase...")
			logger.info(f"Number of Questions for this run: {args.num_questions}")
			pipeline_instance.execute_evaluation_phase(
				num_questions_to_process=args.num_questions
			)
			logger.info("EVALUATION phase completed.")

	except FileNotFoundError as fnf_error:
		logger.error(f"Data file/folder not found: {fnf_error}. Please check paths in config.py and data directory structure for corpus '{args.corpus}'.")
		sys.exit(1)
	except ConnectionError as conn_error: # More specific for network/auth issues with Nuclia
		logger.error(f"Nuclia connection or authentication error: {conn_error}. Check KB_URL, KB_ID, API_KEY, and network.", exc_info=True)
		sys.exit(1)
	except ValueError as val_error: # For issues like missing config or bad data structures
		logger.error(f"Configuration or data validation error: {val_error}", exc_info=True)
		sys.exit(1)
	except Exception as e:
		logger.error(f"An unexpected critical error occurred: {e}", exc_info=True)
		sys.exit(1)

if __name__ == "__main__":
	run()