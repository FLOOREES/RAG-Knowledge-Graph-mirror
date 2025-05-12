# src/pipeline.py
"""
Defines the main pipeline for the Knowledge Graph-based Legal RAG system.

This class orchestrates distinct operational phases:
1. Ingestion Phase: Uploads corpus documents to Nuclia for Knowledge Graph generation.
2. Evaluation Phase: Executes context retrieval using the generated Knowledge Graph
   and evaluates the performance against benchmark questions.
"""

import logging
from pathlib import Path # Ensure Path is imported
from typing import Any, Dict, List, Optional

# Project-specific imports
from . import config
from .data_loader import DataLoader
from .nuclia_interface import NucliaKGHandler
from .retrieval_model import BaselineRetrievalModel
from .evaluation import Evaluation


class LegalRAGPipeline:
	"""
	Orchestrates ingestion and/or evaluation flows for the Legal RAG system.
	It initializes and uses specialized components depending on the phase of operation.
	"""

	def __init__(self,
				 corpus_name: str,
				 # Nuclia credentials can be passed directly or loaded from config by NucliaKGHandler
				 nuclia_api_key: Optional[str] = None,
				 nuclia_kb_id: Optional[str] = None,
				 nuclia_kb_url: Optional[str] = None):
		"""
		Initializes the pipeline with necessary configurations and components.

		Args:
			corpus_name (str): The name of the sub-corpus to process (e.g., "privacy_qa").
			nuclia_api_key (Optional[str]): Nuclia API key. Defaults to config.
			nuclia_kb_id (Optional[str]): Nuclia Knowledge Box ID. Defaults to config.
			nuclia_kb_url (Optional[str]): Nuclia Knowledge Box URL. Defaults to config.
		"""
		self.logger = logging.getLogger(__name__)
		self.logger.info(f"Initializing LegalRAGPipeline for corpus: '{corpus_name}'")

		self.corpus_name: str = corpus_name

		# Components are initialized here. They will be used by specific phase methods.
		self.data_loader: DataLoader = DataLoader(
			corpus_base_path=str(config.CORPUS_BASE_DIR),
			benchmark_base_path=str(config.BENCHMARK_BASE_DIR)
		)
		
		# NucliaKGHandler will attempt to load credentials from config if not provided directly
		self.nuclia_handler: NucliaKGHandler = NucliaKGHandler(
			api_key=nuclia_api_key, # Will fall back to config.NUCLIA_API_KEY if None
			kb_id=nuclia_kb_id,       # Will fall back to config.NUCLIA_KB_ID if None
			kb_url=nuclia_kb_url      # Will fall back to config.NUCLIA_KB_URL if None
		)
		
		# These components are primarily for the 'evaluate' phase
		self.retrieval_model: BaselineRetrievalModel = BaselineRetrievalModel(self.nuclia_handler)
		self.evaluator: Evaluation = Evaluation()

		# Data attributes that will be populated by specific phases
		self.corpus_documents: Optional[Dict[str, str]] = None
		self.benchmark_questions: Optional[List[Dict[str, Any]]] = None

		self.logger.info("LegalRAGPipeline initialized with all components.")

	def execute_ingestion_phase(self, 
								corpus_folder_path: Path, 
								max_files_to_upload: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
		"""
		Executes the ingestion phase: uploads documents from the specified corpus folder to Nuclia.

		Args:
			corpus_folder_path (Path): The local directory path containing .txt files to upload.
									   (e.g., PROJECT_ROOT/data/LegalBench-RAG/corpus/privacy_qa).
			max_files_to_upload (Optional[int]): Maximum number of files to attempt to upload.
												 If None or 0, attempts all found .txt files.

		Returns:
			Dict[str, Dict[str, Any]]: A dictionary summarizing the upload results for each file.
		
		Raises:
			FileNotFoundError: If the corpus_folder_path does not exist.
			ConnectionError: If there are issues connecting to or authenticating with Nuclia.
			# Other exceptions from NucliaKGHandler.upload_documents_from_folder might also propagate.
		"""
		self.logger.info(f"--- Starting INGESTION Phase for Corpus: '{self.corpus_name}' ---")
		self.logger.info(f"Target folder for document upload: {corpus_folder_path}")

		if not corpus_folder_path.is_dir():
			msg = f"Corpus folder for ingestion not found: {corpus_folder_path}"
			self.logger.error(msg)
			raise FileNotFoundError(msg)

		# No need to load benchmark questions for ingestion.
		# Corpus documents are read directly by NucliaKGHandler during upload.
		
		upload_summary = self.nuclia_handler.upload_documents_from_folder(
			folder_path=corpus_folder_path,
			max_files_to_upload=max_files_to_upload
		)
		
		self.logger.info("--- INGESTION Phase Completed ---")
		successful_uploads = sum(1 for result in upload_summary.values() if result["status"] == "success" or result["status"] == "accepted_no_rid")
		failed_uploads = len(upload_summary) - successful_uploads
		self.logger.info(f"Upload summary: {successful_uploads} successful/accepted, {failed_uploads} failed.")
		
		return upload_summary

	def _prepare_evaluation_data(self, num_questions_to_process: int) -> None:
		"""
		Loads corpus documents (for reference) and a subset of benchmark questions
		specifically for the evaluation phase.

		Args:
			num_questions_to_process (int): The number of questions to load.
		
		Raises:
			FileNotFoundError, ValueError: If data loading fails.
		"""
		self.logger.info(f"Preparing data for EVALUATION phase - Corpus: '{self.corpus_name}'")
		
		# Load all documents from the corpus; they might be needed for context or
		# to map retrieval results back to original text if Nuclia only returns pointers.
		self.logger.info(f"Loading all corpus documents from: {config.CORPUS_BASE_DIR / self.corpus_name}")
		self.corpus_documents = self.data_loader.load_corpus_documents(self.corpus_name)
		if not self.corpus_documents:
			msg = f"No documents loaded for corpus '{self.corpus_name}'. Evaluation cannot proceed."
			self.logger.error(msg)
			raise FileNotFoundError(msg)

		benchmark_filename = f"{self.corpus_name}.json"
		self.logger.info(f"Loading benchmark questions from: {config.BENCHMARK_BASE_DIR / benchmark_filename}")
		
		self.benchmark_questions = self.data_loader.load_benchmark_questions(
			corpus_name=self.corpus_name,
			benchmark_file_name=benchmark_filename, # Corrected argument name
			num_questions_to_select=num_questions_to_process # Corrected argument name
		)
		if not self.benchmark_questions:
			msg = f"No benchmark questions loaded for corpus '{self.corpus_name}'. Evaluation cannot proceed."
			self.logger.error(msg)
			raise ValueError(msg)
		
		self.logger.info(f"Evaluation data preparation complete. Loaded {len(self.corpus_documents)} documents and {len(self.benchmark_questions)} benchmark questions.")


	def execute_evaluation_phase(self, num_questions_to_process: int = config.MVP_DEFAULT_NUM_QUESTIONS) -> List[Dict[str, Any]]:
		"""
		Executes the evaluation phase: retrieves context from Nuclia KG and evaluates it.
		This phase assumes the Knowledge Graph has already been generated by Nuclia
		from a prior ingestion phase.

		Args:
			num_questions_to_process (int): Number of benchmark questions to process.

		Returns:
			List[Dict[str, Any]]: Aggregated results for each processed question.
		"""
		self.logger.info(f"--- Starting EVALUATION Phase for Corpus: '{self.corpus_name}' ---")
		self.logger.info(f"Processing up to {num_questions_to_process} questions.")

		try:
			self._prepare_evaluation_data(num_questions_to_process)
		except (FileNotFoundError, ValueError) as e:
			self.logger.error(f"Failed to prepare data for evaluation phase: {e}", exc_info=True)
			raise # Re-raise for main.py to catch

		aggregated_results: List[Dict[str, Any]] = []

		for question_data in self.benchmark_questions: # This is the selected subset
			query_id = question_data.get("query_id", "unknown_query_id")
			query_text = question_data.get("query", "N/A")
			
			self.logger.info(f"Processing query ID: {query_id} - '{query_text[:80]}...'")

			doc_filename = question_data.get("file_path")
			if not doc_filename or doc_filename not in self.corpus_documents:
				self.logger.warning(f"Document '{doc_filename}' for query '{query_id}' not found in loaded corpus documents. Skipping evaluation for this question.")
				continue
			
			original_doc_text: str = self.corpus_documents[doc_filename]
			ground_truth_snippets: List[Dict[str, Any]] = question_data.get("snippets", [])

			if not ground_truth_snippets:
				self.logger.warning(f"No ground truth snippets found for query ID: {query_id}. Evaluation will be limited.")

			# 1. Retrieve Context using the baseline model (which interacts with Nuclia KG)
			self.logger.debug(f"Invoking retrieval model for query ID: {query_id}")
			retrieved_contexts_with_info: List[Dict[str, Any]] = self.retrieval_model.retrieve_context(
				question_data,
				self.corpus_documents
			)
			self.logger.debug(f"Retrieved {len(retrieved_contexts_with_info)} context snippets for query ID: {query_id}")

			# 2. Evaluate Retrieval
			self.logger.debug(f"Evaluating retrieved context for query ID: {query_id}")
			evaluation_result: Dict[str, Any] = self.evaluator.evaluate_retrieved_contexts(
				retrieved_snippets_with_info=retrieved_contexts_with_info,
				ground_truth_benchmark_snippets=ground_truth_snippets,
				original_doc_text=original_doc_text
			)
			
			aggregated_results.append({
				"query_id": query_id,
				"query_text": query_text,
				"document_filename": doc_filename,
				"retrieval_evaluation": evaluation_result,
				"retrieved_snippets_for_review": retrieved_contexts_with_info
			})

		self.logger.info(f"--- EVALUATION Phase Completed. Processed {len(aggregated_results)} questions. ---")
		# Further summary can be done in main.py or here
		return aggregated_results