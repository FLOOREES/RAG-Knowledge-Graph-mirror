# src/benchmark_utils.py
import json
import random
import os
from typing import List, Dict, Any, Optional # Keep Optional
import logging

logger = logging.getLogger(__name__)

# Type alias for clarity - a dictionary containing a question and its ground truth answer
BenchmarkQAItem = Dict[str, str]

def load_benchmark_qa_pairs(filepath: str) -> List[BenchmarkQAItem]:
    """
    Loads question and combined ground truth answer pairs from a specified
    LegalBench-RAG benchmark JSON file. If multiple answer snippets exist
    for a question, they are concatenated into a single ground_truth_answer string.

    Args:
        filepath: The path to the benchmark JSON file.
                  Expected structure: {"tests": [{"query": "...", "snippets": [{"answer": "..."}, ...]}, ...]}

    Returns:
        A list of dictionaries, each containing a "question" and a "ground_truth_answer".
        Returns an empty list if issues occur or no valid pairs are found.
    """
    if not os.path.exists(filepath):
        logger.error(f"Benchmark file not found at: {filepath}")
        return [] # Or raise FileNotFoundError

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {filepath}: {e}", exc_info=True)
        return [] # Or raise ValueError
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {e}", exc_info=True)
        return [] # Or raise ValueError

    if not isinstance(data, Dict) or "tests" not in data:
        logger.error(f"Invalid benchmark file format in {filepath}: Missing 'tests' key or not a root dictionary.")
        return [] # Or raise ValueError

    tests = data.get("tests") # Use .get for safer access
    if not isinstance(tests, List):
        logger.error(f"Invalid benchmark file format in {filepath}: 'tests' key does not contain a list.")
        return [] # Or raise ValueError

    qa_pairs: List[BenchmarkQAItem] = []
    for i, test_item in enumerate(tests):
        if not isinstance(test_item, Dict):
            logger.warning(f"Item at index {i} in 'tests' is not a dictionary, skipping.")
            continue

        question_text = test_item.get("query") # Question from "query" key

        # --- > MODIFIED LOGIC to handle 'snippets' as a LIST < ---
        all_answer_snippets_texts: List[str] = []
        snippets_list = test_item.get("snippets") # Get the list of snippets

        if isinstance(snippets_list, list): # Check if snippets is indeed a list
            for j, snippet_dict in enumerate(snippets_list): # Iterate through each item in the list
                if isinstance(snippet_dict, dict):
                    answer_text_from_snippet = snippet_dict.get("answer") # Get 'answer' from the dict
                    if isinstance(answer_text_from_snippet, str) and answer_text_from_snippet.strip():
                        all_answer_snippets_texts.append(answer_text_from_snippet.strip())
                    else:
                        logger.debug(f"Item {i}, snippet {j}: 'answer' key missing in snippet, not a string, or empty. Skipping this snippet.")
                else:
                    logger.debug(f"Item {i}, snippet {j}: item within 'snippets' list is not a dictionary. Skipping this snippet.")
        elif snippets_list is not None: # If 'snippets' key exists but isn't a list
             logger.warning(f"Item at index {i}: 'snippets' key was found but is not a list (type: {type(snippets_list)}). Cannot extract answer snippets.")
        # If snippets_list is None, all_answer_snippets_texts will remain empty.

        combined_ground_truth_answer: Optional[str] = None
        if all_answer_snippets_texts:
            # Join all valid answer snippets with a clear separator.
            # This creates one comprehensive ground truth string for evaluation.
            combined_ground_truth_answer = "\n\n---\n\n".join(all_answer_snippets_texts)
            logger.debug(f"Item {i}: Combined {len(all_answer_snippets_texts)} answer snippets into one ground truth.")
        # --- > END OF MODIFIED LOGIC < ---

        # Ensure both question and the (now potentially combined) answer are valid non-empty strings
        if isinstance(question_text, str) and question_text.strip() and \
           isinstance(combined_ground_truth_answer, str) and combined_ground_truth_answer.strip():
            qa_pairs.append({
                "question": question_text.strip(),
                "ground_truth_answer": combined_ground_truth_answer # Use the combined answer
            })
            logger.debug(f"Successfully added QA pair for item {i}. Question: '{question_text[:50]}...'")
        else:
            # Updated logging to be more informative about why an item might be skipped
            valid_question = isinstance(question_text, str) and question_text.strip()
            valid_answer = isinstance(combined_ground_truth_answer, str) and combined_ground_truth_answer.strip()
            logger.warning(
                f"Item at index {i} in 'tests': Skipped due to missing/invalid 'query' or unable to form a valid combined 'answer' from snippets. "
                f"Valid Question: {valid_question} (text: '{str(question_text)[:50]}...'), "
                f"Valid Combined Answer: {valid_answer} (from {len(all_answer_snippets_texts)} snippet(s) found)."
            )

    if not qa_pairs:
        logger.warning(f"No valid QA pairs extracted from {filepath}. "
                       "Please check file structure, ensure 'query' key exists with text, "
                       "and 'snippets' key contains a list of dictionaries, each with a non-empty 'answer' string.")
    else:
        logger.info(f"Successfully loaded {len(qa_pairs)} QA pairs from {filepath}.")
    return qa_pairs

def get_random_qa_pair(qa_pairs: List[BenchmarkQAItem]) -> Optional[BenchmarkQAItem]:
    """
    Selects a random QA pair dictionary from a list.

    Args:
        qa_pairs: A list of QA pair dictionaries.

    Returns:
        A randomly selected QA pair dictionary, or None if the list is empty.
    """
    if not qa_pairs:
        logger.warning("Cannot get random QA pair: the list of QA pairs is empty.")
        return None
    return random.choice(qa_pairs)