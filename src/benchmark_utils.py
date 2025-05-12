# benchmark_utils.py
import json
import random
import os
from typing import List, Dict, Any

def load_benchmark_questions(filepath: str) -> List[str]:
	"""
	Loads questions from a specified LegalBench-RAG benchmark JSON file.

	Args:
		filepath: The path to the benchmark JSON file (e.g., "data/.../privacy_qa.json").

	Returns:
		A list of question strings.

	Raises:
		FileNotFoundError: If the specified file does not exist.
		ValueError: If the file is not valid JSON or has unexpected structure.
		KeyError: If the expected keys ("tests", "question") are missing.
	"""
	if not os.path.exists(filepath):
		raise FileNotFoundError(f"Benchmark file not found at: {filepath}")

	try:
		with open(filepath, 'r', encoding='utf-8') as f:
			data = json.load(f)
	except json.JSONDecodeError as e:
		raise ValueError(f"Error decoding JSON from {filepath}: {e}")
	except Exception as e:
		raise ValueError(f"Error reading file {filepath}: {e}")

	if not isinstance(data, Dict):
		raise ValueError(f"Expected JSON root to be a dictionary, found {type(data)} in {filepath}")

	if "tests" not in data:
		raise KeyError(f"Missing 'tests' key in benchmark file: {filepath}")

	tests = data["tests"]
	if not isinstance(tests, List):
		raise ValueError(f"Expected 'tests' key to contain a list, found {type(tests)} in {filepath}")

	questions = []
	for i, test_item in enumerate(tests):
		if not isinstance(test_item, Dict):
			print(f"Warning: Item at index {i} in 'tests' is not a dictionary, skipping.")
			continue
		if "query" not in test_item:
			print(f"Warning: Item at index {i} in 'tests' is missing the 'question' key, skipping.")
			continue
		question_text = test_item["query"]
		if isinstance(question_text, str) and question_text.strip():
			questions.append(question_text.strip())
		else:
			print(f"Warning: Item at index {i} has an invalid or empty 'question', skipping.")


	if not questions:
		print(f"Warning: No valid questions extracted from {filepath}")

	return questions

def get_random_question(questions: List[str]) -> str:
	"""
	Selects a random question from a list of questions.

	Args:
		questions: A list of question strings.

	Returns:
		A randomly selected question string, or an empty string if the list is empty.
	"""
	if not questions:
		return "" # Handle empty list gracefully
	return random.choice(questions)