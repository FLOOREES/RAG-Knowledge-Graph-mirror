from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

class RAGPipelineBase(ABC):
    """
    Abstract Base Class for RAG pipelines to ensure a consistent interface
    for the BenchmarkRunner.
    """

    @abstractmethod
    def get_rag_response(self, question: str, generative_model_override: Optional[str] = None) -> Dict[str, Any]:
        """
        Processes a question through the RAG pipeline and returns a structured response.

        Args:
            question: The input question string.
            generative_model_override: Optional model identifier for the generation step.
                                       The pipeline should handle or ignore this as appropriate.

        Returns:
            A dictionary containing at least the following keys:
            - "generated_answer": str | None
            - "retrieved_paragraphs": List[str]
            - "retrieved_relations": List[Dict[str, Any]]
            - "retrieved_citations": List[Dict[str, Any]] (can be empty)
            - "error": str | None (if an error occurred during processing this specific query)
        """
        pass

    @abstractmethod
    def get_pipeline_name(self) -> str:
        """
        Returns a unique name for the pipeline (e.g., "NucliaDefault", "GNN_SimpleMPNN_v1").
        This will be used for logging and in the output results file.
        """
        pass

    def initialize(self) -> bool:
        """
        Optional method for complex initializations that might fail.
        Returns True if successful, False otherwise.
        Default implementation assumes success.
        Pipelines can override this if they have fallible setup steps
        beyond basic constructor errors (which should raise exceptions).
        """
        return True