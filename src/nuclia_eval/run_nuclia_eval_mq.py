# PROJECT_ROOT/run_benchmark_evaluation.py
import argparse
import logging
import time
from pathlib import Path
from src.nuclia_eval.benchmark_runner import BenchmarkRunner # Import the BenchmarkRunner class
from src.utils.logger_setup import setup_logger

# Setup a logger for this script
logger = setup_logger(__name__)

# Define a directory for benchmark data if it's consistent
DEFAULT_BENCHMARK_DIR = Path("data/LegalBench-RAG/benchmarks")
# Define a directory for outputs
DEFAULT_OUTPUT_DIR = Path("eval_results_nuclia_fullrag_4o_mini_eval_4o_mini")

DEFAULT_EVALUATION_MODEL = "gpt-4.1-nano"

DEFAULT_SCORE_THRESHOLD = 5.0

DEFAULT_SOLUTION_METHOD="nuclia"

def main():
    parser = argparse.ArgumentParser(description="Run RAG benchmark evaluations.")
    parser.add_argument(
        "--benchmarks", nargs="+", required=True,
        help="List of benchmark names (e.g., privacy_qa)."
    )
    parser.add_argument(
        "--benchmark_dir", type=Path, default=DEFAULT_BENCHMARK_DIR,
        help=f"Directory containing benchmark JSON files (default: {DEFAULT_BENCHMARK_DIR})"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save evaluation results (default: {DEFAULT_OUTPUT_DIR})"
    )
    parser.add_argument(
        "--log_level", type=str, default="ERROR",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)"
    )
    parser.add_argument(
        "--generation_model", type=str, default=None,
        help="Optional: Nuclia generation model override. If not set, Nuclia KB default is used."
    )
    parser.add_argument(
        "--evaluation_model", type=str, default=DEFAULT_EVALUATION_MODEL, # Default will be handled by BenchmarkRunner/llm_utils
        help=f"OpenAI model for LLM-as-judge evaluation (default: {DEFAULT_EVALUATION_MODEL} or llm_utils default)."
    )
    parser.add_argument( # New argument for score threshold
        "--score_threshold", type=float, default=DEFAULT_SCORE_THRESHOLD,
        help=f"Score threshold (inclusive) for accuracy calculation (default: {DEFAULT_SCORE_THRESHOLD})"
    )
    parser.add_argument( # New argument
        "--force_reevaluation",
        action="store_true", # Makes it a boolean flag, True if present
        help="Force re-evaluation of benchmarks even if results files exist."
    )

    parser.add_argument(
        "--solution", type=str, default=DEFAULT_SOLUTION_METHOD,
        choices=["nuclia", "GNN", "Similarity", "MSPN", "llm"], # llm not implemented yet
        help=f"Solution method to use for evaluation (default: {DEFAULT_SOLUTION_METHOD})."
    )


    args = parser.parse_args()

    # Configure logging level for all loggers based on CLI
    # This simple basicConfig might be overridden if other modules also call it.
    # A more robust approach involves getting the root logger and setting its level and handlers.
    # For now, this might suffice if setup_logger in utils doesn't conflict heavily.
    logging.basicConfig(level=args.log_level.upper(), 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(module)s.%(funcName)s:%(lineno)d - %(message)s',
                        force=True) # force=True can help if other basicConfigs were called.
    
    # Optionally set levels for specific loggers if basicConfig isn't granular enough or causes issues
    logging.getLogger('src.benchmark_runner').setLevel(args.log_level.upper())
    logging.getLogger('nuclia_eval.pipeline').setLevel(args.log_level.upper())
    logging.getLogger('nuclia_client').setLevel(args.log_level.upper())
    logging.getLogger('src.llm_utils').setLevel(args.log_level.upper())
    logging.getLogger('src.benchmark_utils').setLevel(args.log_level.upper())
    logger.setLevel(args.log_level.upper()) # For this script's own logger

    logger.info(f"Starting benchmark evaluation run with arguments: {args}")

    try:
        if not args.benchmark_dir.exists() or not args.benchmark_dir.is_dir():
            logger.critical(f"Benchmark data directory not found or not a directory: {args.benchmark_dir}")
            return

        runner = BenchmarkRunner(
            benchmark_data_dir=args.benchmark_dir,
            output_dir=args.output_dir,
            force_reevaluation=args.force_reevaluation, # Pass force reevaluation flag
            score_threshold=args.score_threshold, # Pass threshold
            generation_model_override=args.generation_model,
            evaluation_model=args.evaluation_model,
            solution_method=args.solution # Pass solution method
        )
        
        start_time = time.time()
        runner.run_evaluations_on_benchmarks(args.benchmarks) # Call the method
        end_time = time.time()
        
        logger.warning("Benchmark evaluation run completed.")
        
        # Log total time only if new evaluations were performed
        if runner.was_any_benchmark_fully_run:
            total_time = end_time - start_time
            logger.info(f"Total time for performing new evaluations: {total_time:.2f} seconds.")
        else:
            logger.info("All benchmark results were loaded from existing files. No new evaluations performed.")

    except SystemExit: # Catch SystemExit from BenchmarkRunner init
        logger.critical("Exiting due to critical error during BenchmarkRunner initialization.")
    except Exception as e:
        logger.critical(f"An unhandled error occurred during the benchmark run: {e}", exc_info=True)

if __name__ == "__main__":
    main()