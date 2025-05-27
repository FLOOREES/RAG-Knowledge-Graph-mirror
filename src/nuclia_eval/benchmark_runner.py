# Assumed to be in PROJECT_ROOT/src/benchmark_runner.py
# Make sure all necessary imports are at the top of your file:
import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.benchmark_utils import load_benchmark_qa_pairs, BenchmarkQAItem
from src.llm_utils import evaluate_answer_with_openai
from src.nuclia_eval.pipeline import NucliaEvaluationPipeline 
from src.nuclia_eval.config import AppConfig
from src.nuclia_eval.pipeline_gnn import GNNEvaluationPipeline

logger = logging.getLogger(__name__)

class BenchmarkRunner:

    def __init__(self, 
                 benchmark_data_dir: Path, 
                 output_dir: Path,
                 force_reevaluation: bool = False,
                 score_threshold: float = 7.0,
                 generation_model_override: Optional[str] = None,
                 evaluation_model: Optional[str] = None,
                 solution_method: str = "nuclia"):
        
        self.benchmark_data_dir = benchmark_data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.force_reevaluation = force_reevaluation
        self.score_threshold = score_threshold
        self.generation_model_override = generation_model_override
        self.evaluation_model = evaluation_model # Will use llm_utils default if None
        
        # Assuming AppConfig is imported and available
        self.config = AppConfig()
        try:
            AppConfig.validate_config() # Ensure all keys are present
        except ValueError as e:
            logger.critical(f"Configuration error: {e}. Exiting.")
            raise SystemExit(f"Configuration error: {e}")

        logger.info("Initializing RAG Evaluation Pipeline for Nuclia...")
        try:
            if solution_method.lower() == "nuclia":
                logger.info("Using Nuclia Evaluation Pipeline.")
                self.rag_pipeline = NucliaEvaluationPipeline()
            elif solution_method.lower() == "gnn":
                logger.info(f"Using GNN Evaluation Pipeline with method.")
                self.rag_pipeline = GNNEvaluationPipeline(gnn_method="GNN")
            elif solution_method.lower() == "mspn":
                logger.info("Using MSPN Evaluation Pipeline.")
                self.rag_pipeline = GNNEvaluationPipeline(gnn_method="MSPN")
            elif solution_method.lower() == "similarity":
                logger.info("Using Similarity Evaluation Pipeline.")
                self.rag_pipeline = GNNEvaluationPipeline(gnn_method="Similarity")
            else:
                logger.error(f"Unsupported solution method: {solution_method}. Supported methods are 'nuclia' and 'gnn' right now.")
                raise ValueError(f"Unsupported solution method: {solution_method}. Supported methods are 'nuclia' and 'gnn' right now.")

            logger.info("RAG Evaluation Pipeline initialized successfully.")
        except SystemExit as e: # Catch SystemExit if pipeline init fails critically
            logger.critical(f"Failed to initialize RAG Evaluation Pipeline: {e}")
            raise 
        except Exception as e:
            logger.critical(f"An unexpected error occurred during RAG pipeline initialization: {e}", exc_info=True)
            raise
        
        self.was_any_benchmark_fully_run: bool = False


    def _load_results_from_jsonl(self, filepath: Path) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    results.append(json.loads(line))
            logger.info(f"Successfully loaded {len(results)} existing results from {filepath}")
        except FileNotFoundError: # Should be checked before calling, but good to have
            logger.error(f"Result file {filepath} not found for loading.")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {filepath}: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Error reading results file {filepath}: {e}", exc_info=True)
            return []
        return results

    def _save_results_to_jsonl(self, results: List[Dict[str, Any]], filepath: Path) -> None:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for result_item in results:
                    f.write(json.dumps(result_item) + "\n")
            logger.info(f"Successfully saved {len(results)} results to {filepath}")
        except IOError as e:
            logger.error(f"Failed to write results to {filepath}: {e}", exc_info=True)
        except TypeError as e:
            logger.error(f"TypeError during JSON serialization for {filepath}. Error: {e}", exc_info=True)

    def _calculate_and_log_summary_metrics(self, benchmark_name: str, results: List[Dict[str, Any]]) -> None:
        if not results:
            logger.warning(f"No results to summarize for benchmark: {benchmark_name}")
            return

        scores: List[float] = []
        valid_scores_for_accuracy_count = 0
        hits_count = 0

        for res in results:
            score = res.get("llm_eval_score")
            eval_error = res.get("llm_eval_error")

            if eval_error is None and isinstance(score, (int, float)):
                scores.append(float(score))
                valid_scores_for_accuracy_count += 1
                if score >= self.score_threshold:
                    hits_count += 1
            elif eval_error:
                logger.debug(f"Query ID {res.get('query_id')} had an LLM evaluation error: {eval_error}, excluded from summary.")
            elif score is None and eval_error is None :
                 logger.debug(f"Query ID {res.get('query_id')} has no LLM score and no eval error, excluded from summary stats.")
        
        logger.info(f"--- Summary Metrics for Benchmark: {benchmark_name} ---")
        if scores:
            average_score = sum(scores) / len(scores)
            logger.info(f"  Average LLM Evaluation Score: {average_score:.2f} (based on {len(scores)} scored items)")
        else:
            logger.info("  Average LLM Evaluation Score: N/A (no valid scores found)")

        if valid_scores_for_accuracy_count > 0:
            accuracy = (hits_count / valid_scores_for_accuracy_count) * 100
            logger.info(f"  Accuracy (Scores >= {self.score_threshold}): {accuracy:.2f}% ({hits_count} hits out of {valid_scores_for_accuracy_count} validly scored items)")
        else:
            logger.info(f"  Accuracy (Scores >= {self.score_threshold}): N/A (no validly scored items for accuracy calculation)")
        logger.info("----------------------------------------------------")

    def run_evaluations_on_benchmarks(self, benchmark_names: List[str]) -> None:
        """
        Runs the evaluation process for all specified benchmarks.
        If a results file exists and force_reevaluation is False, loads it.
        Otherwise, runs the full evaluation pipeline.
        Calculates and logs summary metrics for each benchmark.
        """
        logger.info(f"Starting evaluations for benchmarks: {benchmark_names}")
        if self.force_reevaluation:
            logger.warning("Force re-evaluation is ENABLED. Existing results files will be ignored and overwritten if new evaluations are run.")
        
        # Log model choices
        if self.generation_model_override:
            logger.info(f"Using Nuclia generation model override: {self.generation_model_override}")
        else:
            logger.info("Using Nuclia Knowledge Box default generation model.")
        
        # Use self.evaluation_model directly; if it's None, llm_utils function will use its own default.
        logger.info(f"Using LLM evaluation model: {self.evaluation_model if self.evaluation_model else 'llm_utils.py default (e.g., gpt-4.1)'}")


        for benchmark_name in benchmark_names:
            logger.info(f"--- Processing benchmark: {benchmark_name} ---")
            output_filepath = self.output_dir / f"{benchmark_name}_eval_results.jsonl"
            
            current_benchmark_results: List[Dict[str, Any]] = []
            run_full_evaluation_for_this_benchmark = True

            if not self.force_reevaluation and output_filepath.exists():
                logger.info(f"Attempting to load existing results for '{benchmark_name}' from {output_filepath}.")
                current_benchmark_results = self._load_results_from_jsonl(output_filepath)
                if current_benchmark_results: # Successfully loaded non-empty results
                    logger.info(f"Successfully loaded {len(current_benchmark_results)} existing results for '{benchmark_name}'. Skipping re-evaluation.")
                    run_full_evaluation_for_this_benchmark = False
                else:
                    logger.warning(f"Failed to load results or file was empty for '{benchmark_name}'. Will proceed with full evaluation.")
                    # Optionally delete the problematic file if it's considered corrupt/empty and should be regenerated
                    # output_filepath.unlink(missing_ok=True) 
            
            if run_full_evaluation_for_this_benchmark:
                if self.force_reevaluation and output_filepath.exists():
                    logger.info(f"Force re-evaluation: Will overwrite existing results at {output_filepath}")

                self.was_any_benchmark_fully_run = True 
                logger.info(f"Running full evaluation for '{benchmark_name}'...")
                
                benchmark_filepath_to_load = self.benchmark_data_dir / f"{benchmark_name}.json"
            
                if not benchmark_filepath_to_load.exists():
                    logger.error(f"Benchmark file for '{benchmark_name}' not found at {benchmark_filepath_to_load}. Skipping this benchmark.")
                    continue # Skip to the next benchmark

                qa_items: List[BenchmarkQAItem] = load_benchmark_qa_pairs(str(benchmark_filepath_to_load))
                if not qa_items:
                    logger.warning(f"No QA items loaded for benchmark '{benchmark_name}'. Skipping this benchmark.")
                    continue
                
                logger.info(f"Loaded {len(qa_items)} QA items from {benchmark_name}.")
                
                current_benchmark_results = [] # Reset for new results
                for i, qa_item in enumerate(qa_items):
                    question = qa_item.get("question")
                    ground_truth_answer = qa_item.get("ground_truth_answer")

                    if not question:
                        logger.warning(f"QA item {i+1} in {benchmark_name} is missing 'question'. Skipping.")
                        continue
                    
                    logger.info(f"Processing Q{i+1}/{len(qa_items)} for {benchmark_name}: '{question[:100]}...'")
                    
                    query_result_package = {
                        "benchmark_name": benchmark_name,
                        "query_id": f"{benchmark_name}_{i+1}",
                        "question": question,
                        "ground_truth_answer": ground_truth_answer,
                        "generation_model_used": self.generation_model_override or "Nuclia KB Default",
                        "evaluation_model_used": self.evaluation_model or "llm_utils.py Default",
                        "generated_answer": None, "retrieved_relations": [], "retrieved_paragraphs": [],
                        "retrieved_citations": [], "llm_eval_score": None, "llm_eval_explanation": None,
                        "llm_eval_raw_output": None, "llm_eval_error": None, "generation_error": None
                    }

                    try:
                        nuclia_response = self.rag_pipeline.run_single_query_evaluation(
                            question,
                            generative_model_override=self.generation_model_override
                        )
                        query_result_package.update({
                            "generated_answer": nuclia_response.get("answer"),
                            "retrieved_relations": nuclia_response.get("relations"),
                            "retrieved_paragraphs": nuclia_response.get("retrieved_context_paragraphs"),
                            "retrieved_citations": nuclia_response.get("citations")
                        })

                        # Perform LLM-based evaluation if conditions are met
                        if query_result_package["generated_answer"] and ground_truth_answer:
                            logger.info(f"Performing LLM-based evaluation for Q{i+1}...")
                            evaluation_result = evaluate_answer_with_openai(
                                api_key=self.config.OPENAI_API_KEY,
                                question=question,
                                generated_answer=query_result_package["generated_answer"],
                                ground_truth_answer=ground_truth_answer,
                                model_name=self.evaluation_model # Pass configured model
                            )
                            if evaluation_result:
                                query_result_package.update({
                                    "llm_eval_score": evaluation_result.get("score"),
                                    "llm_eval_explanation": evaluation_result.get("explanation"),
                                    "llm_eval_raw_output": evaluation_result.get("raw_output")
                                })
                                # Check if the explanation field itself contains an error message from the LLM eval function
                                explanation_str = str(evaluation_result.get("explanation", ""))
                                if "Error:" in explanation_str or "error:" in explanation_str.lower() :
                                    query_result_package["llm_eval_error"] = explanation_str
                                logger.info(f"LLM Eval Score for Q{i+1}: {evaluation_result.get('score')}")
                            else: # evaluation_result itself is None (e.g., critical pre-API error in evaluate_answer_with_openai)
                                query_result_package["llm_eval_error"] = "Evaluation function (evaluate_answer_with_openai) returned None."
                                logger.warning(f"LLM-based evaluation function returned None for Q{i+1} of {benchmark_name}.")
                        elif not ground_truth_answer:
                            logger.info(f"Skipping LLM-based evaluation for Q{i+1} of {benchmark_name} as no ground truth answer is available.")
                        else: # No generated answer
                            logger.info(f"Skipping LLM-based evaluation for Q{i+1} of {benchmark_name} as no answer was generated by RAG pipeline.")
                    
                    except Exception as e:
                        logger.error(f"Error processing question Q{i+1} ('{question[:50]}...') from {benchmark_name}: {e}", exc_info=True)
                        query_result_package["generation_error"] = str(e)
                    
                    current_benchmark_results.append(query_result_package)
                
                # Save the newly generated results
                self._save_results_to_jsonl(current_benchmark_results, output_filepath)
            
            # Calculate and log summary metrics using the results (either loaded or newly generated)
            if current_benchmark_results:
                self._calculate_and_log_summary_metrics(benchmark_name, current_benchmark_results)
            else:
                # This case should be rare if loading fails and then generation also fails to produce any items.
                logger.warning(f"No results (neither loaded nor newly generated) available to summarize for benchmark: {benchmark_name}")

            logger.info(f"--- Finished processing benchmark: {benchmark_name} ---")