# src/data_loader.py
"""
Handles loading and pre-processing of corpus documents and benchmark data
for the Legal RAG pipeline.
"""

import json
import logging
import os # Retained for os.listdir, though Path is used for construction
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple # Added Tuple for return type

# Import from our own config module within the 'src' package
from . import config

class DataLoader:
	"""
	Responsible for loading corpus documents and benchmark questions
	from the specified file structure.
	"""

	def __init__(self, corpus_base_path: str, benchmark_base_path: str):
		"""
		Initializes the DataLoader with base paths for data.

		Args:
			corpus_base_path (str): The absolute base path to the corpus directory
									(e.g., ".../data/LegalBench-RAG/corpus").
			benchmark_base_path (str): The absolute base path to the benchmarks directory
									   (e.g., ".../data/LegalBench-RAG/benchmarks").
		"""
		self.logger = logging.getLogger(__name__)
		self.corpus_base_path = Path(corpus_base_path)
		self.benchmark_base_path = Path(benchmark_base_path)

		if not self.corpus_base_path.is_dir():
			self.logger.error(f"Corpus base path does not exist or is not a directory: {self.corpus_base_path}")
			raise FileNotFoundError(f"Corpus base path not found: {self.corpus_base_path}")
		if not self.benchmark_base_path.is_dir():
			self.logger.error(f"Benchmark base path does not exist or is not a directory: {self.benchmark_base_path}")
			raise FileNotFoundError(f"Benchmark base path not found: {self.benchmark_base_path}")

		self.logger.info(f"DataLoader initialized. Corpus: '{self.corpus_base_path}', Benchmarks: '{self.benchmark_base_path}'")

	def load_corpus_documents(self, corpus_name: str) -> Dict[str, str]:
		"""
		Loads all .txt documents from the specified sub-corpus directory.

		The sub-corpus directory is expected to be: <corpus_base_path>/<corpus_name>/

		Args:
			corpus_name (str): The name of the sub-corpus (e.g., "privacy_qa").

		Returns:
			Dict[str, str]: A dictionary mapping document filenames to their text content.
							Example: {"23andMe.txt": "Full text of 23andMe privacy policy..."}
		
		Raises:
			FileNotFoundError: If the specific sub-corpus directory is not found.
		"""
		specific_corpus_path: Path = self.corpus_base_path / corpus_name
		self.logger.info(f"Attempting to load documents from: {specific_corpus_path}")

		if not specific_corpus_path.is_dir():
			self.logger.error(f"Sub-corpus directory not found: {specific_corpus_path}")
			raise FileNotFoundError(f"Sub-corpus directory not found: {specific_corpus_path}")

		documents: Dict[str, str] = {}
		for filename in os.listdir(specific_corpus_path): # os.listdir is fine for simple listing
			if filename.lower().endswith(".txt"):
				filepath: Path = specific_corpus_path / filename
				try:
					with open(filepath, 'r', encoding='utf-8') as f:
						documents[filename] = f.read()
				except Exception as e:
					self.logger.error(f"Error reading document {filepath}: {e}", exc_info=True)
					# For MVP, we might skip a problematic file, or raise the error
					# depending on how critical individual files are.
					# For now, let's log and continue if one file fails.
					continue 
		
		if not documents:
			self.logger.warning(f"No .txt documents found or loaded from {specific_corpus_path}.")
		else:
			self.logger.info(f"Successfully loaded {len(documents)} documents from corpus '{corpus_name}'.")
		return documents

	def load_benchmark_questions(self,
								 corpus_name: str, # Used to find the correct benchmark file.
								 num_questions_to_select: int = -1
								 ) -> List[Dict[str, Any]]:
		"""
		Loads benchmark questions and ground truth for a specific sub-corpus.
		It expects a JSON file named <corpus_name>.json in the benchmark_base_path.
		(e.g., "privacy_qa.json").

		Each item in the returned list should represent a question and its associated
		ground truth data, minimally including:
		- "query_id": A unique identifier for the question.
		- "query": The text of the question.
		- "file_path": The filename of the source document in the corpus.
		- "snippets": A list of ground truth snippets, where each snippet is a dict
					  e.g., {"text": "relevant text...", "char_index_range": [start, end]}

		Args:
			corpus_name (str): The name of the sub-corpus (e.g., "privacy_qa").
							   This determines which benchmark JSON file to load.
			num_questions_to_select (int): Number of questions to select for the run.
										   If -1 or 0, all questions are loaded.
										   If > 0, a subset is selected (currently first N).

		Returns:
			List[Dict[str, Any]]: A list of benchmark question entries.
		
		Raises:
			FileNotFoundError: If the benchmark JSON file is not found.
			ValueError: If the benchmark JSON is empty or improperly structured.
		"""
		benchmark_filename = f"{corpus_name}.json" # e.g., privacy_qa.json
		benchmark_filepath: Path = self.benchmark_base_path / benchmark_filename
		self.logger.info(f"Attempting to load benchmark questions from: {benchmark_filepath}")

		if not benchmark_filepath.is_file():
			self.logger.error(f"Benchmark file not found: {benchmark_filepath}")
			raise FileNotFoundError(f"Benchmark file not found: {benchmark_filepath}")

		try:
			with open(benchmark_filepath, 'r', encoding='utf-8') as f:
				# LegalBench-RAG-mini's individual JSONs (like privacy_qa.json)
				# typically contain a list of question objects directly.
				all_questions_for_corpus: List[Dict[str, Any]] = json.load(f)
		except json.JSONDecodeError as e:
			self.logger.error(f"Error decoding JSON from {benchmark_filepath}: {e}", exc_info=True)
			raise ValueError(f"Invalid JSON in benchmark file: {benchmark_filepath}")
		except Exception as e:
			self.logger.error(f"Unexpected error loading benchmark file {benchmark_filepath}: {e}", exc_info=True)
			raise # Re-raise other unexpected errors

		if not all_questions_for_corpus:
			self.logger.warning(f"No questions found in benchmark file: {benchmark_filepath} for corpus '{corpus_name}'.")
			return []
		
		self.logger.info(f"Loaded {len(all_questions_for_corpus)} total questions for corpus '{corpus_name}'.")

		# Validate structure of the first question (as a sample)
		if all_questions_for_corpus:
			sample_q = all_questions_for_corpus[0]
			required_keys = ["query_id", "query", "file_path", "snippets"]
			if not all(key in sample_q for key in required_keys):
				self.logger.error(f"Benchmark data in {benchmark_filepath} has unexpected structure. Missing one of {required_keys}. Sample: {sample_q}")
				raise ValueError(f"Benchmark data in {benchmark_filepath} has unexpected structure.")
			if sample_q["snippets"] and not all("char_index_range" in snip and "text" in snip for snip in sample_q["snippets"]):
				self.logger.error(f"Benchmark snippet data in {benchmark_filepath} has unexpected structure. Snippets should contain 'text' and 'char_index_range'. Sample: {sample_q['snippets'][0]}")
				raise ValueError(f"Benchmark snippet data in {benchmark_filepath} has unexpected structure.")


		if num_questions_to_select > 0 and len(all_questions_for_corpus) > num_questions_to_select:
			# For consistent MVP runs, select the first N questions.
			# random.sample could be used for random selection if desired later.
			selected_questions = all_questions_for_corpus[:num_questions_to_select]
			self.logger.info(f"Selected the first {len(selected_questions)} questions for this run, as requested.")
		else:
			selected_questions = all_questions_for_corpus
			self.logger.info(f"Using all {len(selected_questions)} loaded questions for this run.")
			
		return selected_questions