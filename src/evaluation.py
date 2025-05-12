class Evaluation:
	"""
	Evaluation class to assess the performance of the retrieval model against benchmark questions.

	Attributes:
		corpus_name (str): Name of the corpus being evaluated.
		retrieval_model (RetrievalModel): Instance of the retrieval model to be evaluated.
		evaluator (Evaluator): Instance of the evaluator to assess the retrieval results.
		corpus_documents (Dict[str, str]): Dictionary mapping document filenames to their text content.
		benchmark_questions (List[Dict[str, Any]]): List of benchmark questions for evaluation.
		logger (Logger): Logger instance for logging messages.

	Methods:
		evaluate(num_questions_to_process: int) -> List[Dict[str, Any]]:
			Evaluate the retrieval model using benchmark questions and return aggregated results.
	"""