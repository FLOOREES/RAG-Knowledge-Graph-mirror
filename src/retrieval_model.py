class BaselineRetrievalModel:
	"""
	A class to handle the retrieval of documents from a Nuclia Knowledge Box (KB) using a baseline model.
	This model is designed to retrieve documents based on a given query.
	"""

	def __init__(self):
		"""
		Initializes the BaselineRetrievalModel with a NucliaKGHandler instance.

		Args:
			nuclia_handler (NucliaKGHandler): An instance of NucliaKGHandler to interact with the Nuclia API.
		"""