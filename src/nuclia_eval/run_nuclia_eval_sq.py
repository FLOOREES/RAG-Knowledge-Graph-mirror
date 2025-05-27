# PROJECT_ROOT/run.py
import logging # Import standard logging
from .pipeline import NucliaEvaluationPipeline # Relative import
from .pipeline_gnn import GNNEvaluationPipeline
from src.utils.logger_setup import app_logger # Import the pre-configured app_logger

# You might want to set the root logger level if you want to see logs from libraries (e.g. Nuclia SDK)
# logging.basicConfig(level=logging.WARNING) # Example: Show WARNING and above from all loggers

def main(pipeline: str = 'GNN'):
    """
    Main function to execute the Nuclia query evaluation.
    """
    app_logger.info("Starting Nuclia Evaluation Script...")

    try:
        pipeline = GNNEvaluationPipeline()
        if pipeline == 'Nuclia':
            pipeline = NucliaEvaluationPipeline()
        
        # Example query - replace with your actual test query
        # This question is inspired by the report's example [cite: 165]
        test_question = "How is entity 'Privacy Shield' related to 'GDPR' within the legal documents?"
        
        app_logger.info(f"Sending test question to Nuclia: '{test_question}'")
        results = pipeline.run_single_query_evaluation(test_question)

        app_logger.info("Query evaluation finished.")
        app_logger.info("Results:")
        app_logger.info(f"  Question: {results.get('question')}")
        app_logger.info(f"  Answer: {results.get('answer')}")
        
        if results.get('relations'):
            app_logger.info(f"  Retrieved Relations ({len(results['relations'])}):")
            for i, relation in enumerate(results['relations'][:3]): # Log first 3 relations
                app_logger.info(f"    Relation {i+1}: {relation}")
        else:
            app_logger.info("  No relations retrieved or an issue occurred.")
            
        if results.get('citations'):
            app_logger.info(f"  Citations ({len(results['citations'])}):")
            for i, citation in enumerate(results['citations'][:3]): # Log first 3 citations
                app_logger.info(f"    Citation {i+1}: {citation}")
        else:
            app_logger.info("  No citations retrieved.")

        if results.get('retrieved_context_paragraphs'):
            app_logger.info(f"  Retrieved Context Paragraphs ({len(results['retrieved_context_paragraphs'])}):")
            # for i, para_text in enumerate(results['retrieved_context_paragraphs'][:2]):
            #     app_logger.info(f"    Paragraph {i+1} (sample): {para_text[:100]}...") # Log snippet
        else:
            app_logger.info("  No context paragraphs retrieved.")


    except SystemExit as e:
        app_logger.critical(f"Pipeline initialization failed: {e}")
    except Exception as e:
        app_logger.critical(f"An unhandled exception occurred in the main script: {e}", exc_info=True)

if __name__ == "__main__":
    # To make imports work smoothly if you run this script directly from PROJECT_ROOT,
    # ensure PROJECT_ROOT is in PYTHONPATH or structure your project as a package.
    # For simplicity, if PROJECT_ROOT/utils and PROJECT_ROOT/nuclia_eval are directories,
    # Python's import system should handle it if `run.py` is in PROJECT_ROOT.
    
    # If you have PROJECT_ROOT/src/nuclia_eval and PROJECT_ROOT/src/utils,
    # you might need to add 'src' to sys.path or run as `python -m src.run`
    # from the directory containing 'src'.
    # For current structure (nuclia_eval and utils directly under PROJECT_ROOT):
    # Ensure your PYTHONPATH includes PROJECT_ROOT or run from PROJECT_ROOT.
    main()