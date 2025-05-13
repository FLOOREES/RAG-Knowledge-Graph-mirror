# app.py
# --- Core Imports ---
import streamlit as st
import pandas as pd
import os
import json
from dotenv import load_dotenv # For loading environment variables from .env file
import random
import logging # For application-level logging
import sys # For modifying Python path
from typing import List, Dict, Any, Optional # For type hinting clarity

# --- Configuration for Logging ---
# Provides informative messages during execution, helpful for debugging and monitoring.
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)

# --- Ensure Application Modules are Discoverable ---
# Adds the project's root directory to the Python path.
# This allows importing local modules from 'src' and 'benchmarking' directories.
# Assumes app.py is in the root directory of the project.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
logging.info(f"Project root added to sys.path: {os.path.abspath(os.path.join(os.path.dirname(__file__), './'))}")

# --- Import Custom Application Modules ---
# Loads utility functions for benchmark handling, graph processing,
# Nuclia API interaction (fallback), the baseline GNN model wrapper, and LLM interaction.
try:
    from src.benchmark_utils import load_benchmark_qa_pairs, get_random_qa_pair
    from benchmarking.wrappers import CustomWrapper
    from benchmarking.general_utils import extract_triplets # Used for Nuclia API fallback
    from src.graph_processing_utils import find_relations_for_entities, retrieve_paragraphs
    from src.llm_utils import generate_answer_with_openai, evaluate_answer_with_openai
    logging.info("Successfully imported all custom application modules.")
except ImportError as e:
    # Critical error if modules can't be imported. Display in app and stop.
    st.error(f"Fatal Error: Failed to import required application modules: {e}. "
             "Please ensure file structure is correct (e.g., 'src', 'benchmarking' folders are present) "
             "and that necessary '__init__.py' files exist in these directories.")
    logging.error(f"Fatal import error on startup: {e}", exc_info=True)
    st.stop()
except Exception as e:
    # Catch any other unexpected errors during these initial imports.
    st.error(f"Fatal Error: An unexpected error occurred during initial module imports: {e}")
    logging.error(f"Unexpected fatal import error: {e}", exc_info=True)
    st.stop()


# --- Configuration & Constants ---
# Load environment variables from a .env file in the project root.
# This file should contain API keys and other sensitive configurations.
if load_dotenv(override=True):
    logging.info(".env file found and loaded successfully.")
else:
    logging.warning(".env file not found. Relying on pre-set environment variables if available.")

# Nuclia configurations (used for API fallback if local graph fails or for paragraph retrieval)
NUCLIA_KB_URL = os.getenv('NUCLIA_KB_URL')
NUCLIA_API_KEY = os.getenv('NUCLIA_API_KEY') # Used by retrieve_paragraphs and extract_triplets
logging.info(f"NUCLIA_KB_URL loaded from environment: '{NUCLIA_KB_URL}'")
if not NUCLIA_KB_URL: logging.warning("NUCLIA_KB_URL not found. API fallback/paragraph retrieval might fail.")
if not NUCLIA_API_KEY: logging.warning("NUCLIA_API_KEY not found. API fallback/paragraph retrieval might fail.")

# OpenAI API Key for LLM-based answer generation
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    logging.warning("OPENAI_API_KEY not found in environment. LLM answer generation will be disabled.")
    # User will be notified in the UI if they try to generate an answer.

# Define path for the benchmark questions JSON file.
# Assumes 'data' folder is sibling to 'src', 'benchmarking', and app.py.
BENCHMARK_FILE_PATH = os.path.join(os.path.dirname(__file__), "data", "LegalBench-RAG", "benchmarks", "privacy_qa.json")
logging.info(f"Benchmark questions JSON path set to: {BENCHMARK_FILE_PATH}")

# Define path for the pre-extracted local Knowledge Graph JSON file.
LOCAL_GRAPH_FILE = os.path.join(os.path.dirname(__file__), "data", "legal_graph.json")
logging.info(f"Local Knowledge Graph JSON path set to: {LOCAL_GRAPH_FILE}")

# Configuration: How many top entities (from GNN) to use for finding relevant relations.
TOP_N_ENTITIES_FOR_RELATIONS: int = 5
logging.info(f"Will use top {TOP_N_ENTITIES_FOR_RELATIONS} entities for relation searching.")


# --- Caching Functions ---
# These functions use Streamlit's caching to avoid expensive re-computation
# or data re-loading on every UI interaction or script rerun, improving performance.

@st.cache_data(show_spinner="Loading Knowledge Graph...")
def load_or_fetch_graph(local_graph_path: str, kb_url_fallback: Optional[str]) -> List[Dict[str, Any]]:
    """
    Loads graph data preferentially from a local JSON file.
    If the file is not found or fails to load, falls back to fetching
    from the Nuclia API via extract_triplets (if kb_url_fallback and API key are provided).
    """
    # ... (Implementation from your provided code - assumed correct and tested) ...
    graph_data = []
    loaded_from_local = False
    logging.info(f"Attempting to load graph from local path: '{local_graph_path}'")
    st.info(f"Checking for local graph file at: '{os.path.basename(local_graph_path)}'")

    if os.path.exists(local_graph_path):
        st.success("Found local graph file!")
        logging.info("Local graph file found.")
        try:
            with open(local_graph_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            st.success(f"Successfully loaded graph from {os.path.basename(local_graph_path)}!")
            logging.info(f"Successfully loaded graph from {local_graph_path}.")
            if not isinstance(graph_data, list):
                 st.warning("Warning: Loaded graph data is not a list type. Processing might fail.")
                 logging.warning(f"Loaded graph data from {local_graph_path} is not a list.")
            elif not graph_data:
                 st.warning("Warning: Loaded graph data is empty.")
                 logging.warning(f"Loaded graph data from {local_graph_path} is empty.")
            loaded_from_local = True
            return graph_data
        except json.JSONDecodeError as json_err:
            st.error(f"Error: Failed to decode JSON from {local_graph_path}. File may be corrupted.")
            st.warning("Will attempt to fetch from Nuclia API instead (if configured).")
            logging.error(f"JSONDecodeError for {local_graph_path}: {json_err}", exc_info=True)
        except Exception as e:
             st.error(f"Error reading local graph file {local_graph_path}: {e}")
             st.warning("Will attempt to fetch from Nuclia API instead (if configured).")
             logging.error(f"Error reading {local_graph_path}: {e}", exc_info=True)
    else:
         st.info("Local graph file not found.")
         logging.info(f"Local graph file not found at {local_graph_path}.")

    if not loaded_from_local:
        st.info("Attempting to fetch graph data from Nuclia API as fallback...")
        logging.info("Attempting API fallback for graph data.")
        if not kb_url_fallback:
             st.error("Nuclia KB URL not configured for API fallback. Cannot fetch graph.")
             logging.error("API fallback failed: NUCLIA_KB_URL is not set.")
             return []
        nuclia_api_key_for_fallback = os.getenv('NUCLIA_API_KEY')
        if not nuclia_api_key_for_fallback:
            st.error("Nuclia API Key not configured for API fallback. Cannot fetch graph.")
            logging.error("API fallback failed: NUCLIA_API_KEY is not set.")
            return []
        try:
            logging.info(f"Calling extract_triplets for URL: {kb_url_fallback}")
            graph_data = extract_triplets(kb_url_fallback) # This function uses NUCLIA_API_KEY internally
            if not graph_data:
                 st.warning("Fetched graph data from Nuclia is empty.")
                 logging.warning("extract_triplets (API fallback) returned empty data.")
                 return []
            st.success("Successfully fetched graph from Nuclia API.")
            logging.info("Successfully fetched graph data via API fallback.")
            return graph_data
        except Exception as e:
            st.error(f"Failed to fetch graph from Nuclia API: {e}")
            logging.error(f"API fallback (extract_triplets) failed: {e}", exc_info=True)
            return []
    logging.error("load_or_fetch_graph failed to load or fetch any graph data.")
    return []


@st.cache_resource(show_spinner="Initializing Baseline Model...")
def get_baseline_model() -> Optional[CustomWrapper]:
    """Initializes and caches the baseline GNN model wrapper."""
    # ... (Implementation from your provided code - assumed correct and tested) ...
    try:
        logging.info("Initializing baseline model (CustomWrapper)...")
        model = CustomWrapper()
        logging.info("Baseline model initialized successfully.")
        return model
    except Exception as e:
        st.error(f"Fatal Error: Failed to initialize the baseline model: {e}")
        logging.error(f"Failed to initialize CustomWrapper: {e}", exc_info=True)
        st.stop()
        return None

@st.cache_data(show_spinner="Processing Graph Data for GNN...")
def ingest_graph_data(_model: CustomWrapper, graph_data: List[Dict[str, Any]]) -> bool:
    """Ingests the loaded graph data into the baseline model."""
    # ... (Implementation from your provided code - assumed correct and tested) ...
    if not graph_data:
        st.warning("Cannot ingest empty graph data into the model.")
        logging.warning("Skipping model data ingestion: graph_data is empty.")
        return False
    if not _model:
         st.error("Cannot ingest data: Baseline model is not available.")
         logging.error("Skipping model data ingestion: Model instance is None.")
         return False
    try:
        logging.info(f"Ingesting graph data ({len(graph_data)} relations) into baseline model...")
        _model.ingest_data(graph_data)
        st.success("Graph data processed by baseline model.")
        logging.info("Graph data ingestion successful.")
        return True
    except Exception as e:
        st.error(f"Error ingesting graph data into the model: {e}")
        logging.error(f"Failed during model.ingest_data: {e}", exc_info=True)
        return False

@st.cache_data
def load_benchmark_data_for_app(filepath: str) -> List[Dict[str, str]]: # Renamed for clarity
    """Loads benchmark QA pairs from the specified JSON file for the app."""
    try:
        logging.info(f"Loading benchmark QA pairs from: {filepath}")
        st.info(f"Loading benchmark data from: {os.path.basename(filepath)}")
        # --- > Use new function that returns list of dicts < ---
        qa_pairs = load_benchmark_qa_pairs(filepath)
        if not qa_pairs:
            st.warning("No QA pairs loaded from benchmark file. Random question feature might be limited.")
            logging.warning(f"load_benchmark_qa_pairs returned empty list from {filepath}")
            return [] # Return empty list
        st.success(f"Loaded {len(qa_pairs)} benchmark QA pairs.")
        logging.info(f"Loaded {len(qa_pairs)} QA pairs from {filepath}")
        return qa_pairs
    except FileNotFoundError:
        st.error(f"Fatal Error: Benchmark file not found at {filepath}.")
        logging.error(f"Benchmark file not found: {filepath}")
        return []
    except Exception as e:
        st.error(f"Fatal Error loading benchmark QA pairs: {e}")
        logging.error(f"Failed to load QA pairs from {filepath}: {e}", exc_info=True)
        return []


# --- Streamlit App UI ---
st.set_page_config(
    page_title="RAG Relevance Explorer & Generator", # Updated title
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.title("🕵️‍♀️ Privacy Policy RAG Explorer & Answer Generator 🕵️‍♂️") # Updated title
st.markdown("Enter a question or load a random one from the benchmark. "
            "The system will identify relevant entities and relations, "
            "retrieve context, and generate an answer using an LLM.")

# --- Initialize Session State ---
if 'current_question' not in st.session_state:
    st.session_state.current_question = ""
    logging.debug("Initialized st.session_state.current_question")
if 'model_ready' not in st.session_state:
     st.session_state.model_ready = False
     logging.debug("Initialized st.session_state.model_ready")

# --- Load Data, Model, and Questions ---
benchmark_qa_data = load_benchmark_data_for_app(BENCHMARK_FILE_PATH)
graph_data = load_or_fetch_graph(LOCAL_GRAPH_FILE, NUCLIA_KB_URL)

if graph_data:
    baseline_model = get_baseline_model()
    if baseline_model:
        ingestion_successful = ingest_graph_data(baseline_model, graph_data)
        if ingestion_successful:
            st.session_state.model_ready = True
            logging.info("System ready: Model and data loaded successfully.")
        else:
             st.error("Model could not process the graph data. Application cannot proceed.")
             st.stop() # Stop if ingestion fails
    else:
         st.error("Baseline model failed to load. Application cannot proceed.")
         st.stop() # Stop if model loading fails
else:
    st.error("Failed to load graph data. Application cannot proceed.")
    st.stop() # Stop if graph data fails to load


# --- User Input Section ---
st.subheader("Enter Your Question:")
random_button_clicked = st.button(
    label="🎲 Get Random Benchmark Question & Answer", # Updated label
    help="Load a random question and its ground truth answer from the benchmark.",
    disabled=(not benchmark_qa_data or not st.session_state.model_ready)
)
if random_button_clicked:
     logging.info("Random QA pair button clicked.")
     if benchmark_qa_data:
         # --- > Get a random QA pair (dictionary) < ---
         random_qa_item = get_random_qa_pair(benchmark_qa_data)
         if random_qa_item:
             st.session_state.current_question_text = random_qa_item["question"]
             st.session_state.current_ground_truth_answer = random_qa_item["ground_truth_answer"] # Store GT
             logging.info(f"Loaded random question: {random_qa_item['question'][:50]}...")
             st.info("Random benchmark question (and its ground truth for evaluation) loaded!")
         else:
             st.warning("Failed to get a random QA pair from benchmark data.")
             logging.warning("get_random_qa_pair returned None.")
     else:
          st.warning("Cannot load random QA pair: Benchmark data is empty or failed to load.")
          logging.warning("Random QA pair requested, but no benchmark data available.")

st.text_area(
    label="Question Input:",
    value=st.session_state.get('current_question_text', ''), # Display current question
    key="question_input_widget_key",
    height=100,
    placeholder="Ask a question about the privacy policies...",
    help="Type your question here, or use the 'Random Question' button.",
    disabled=not st.session_state.model_ready,
    # When user types, this updates question_input_widget_key.
    # If they then click "Analyze", current_ground_truth_answer might be from a *previous* random load.
    # We need to clear current_ground_truth_answer if the text area is manually edited.
    on_change=lambda: st.session_state.update(current_ground_truth_answer=None, current_question_text=st.session_state.question_input_widget_key)
)

analyze_button = st.button(
    label="💡 Analyze, Generate & Evaluate Answer", # Updated label
    help="Find context, generate an answer using LLM, and evaluate it if ground truth is available.",
    type="primary",
    disabled=not st.session_state.model_ready
)


# --- Analysis and Output Section ---
if analyze_button and st.session_state.model_ready:
    logging.info("'Analyze & Generate Answer' button clicked.")

    # Retrieve the question to analyze from the text input widget's state.
    question_to_analyze = st.session_state.question_input_widget_key.strip()
    logging.info(f"Question for analysis: '{question_to_analyze[:100]}...'") # Log a snippet

    # Retrieve the corresponding ground truth answer, if one was loaded (e.g., from a random benchmark question).
    # It will be None if the user typed a custom question or if no benchmark data is loaded.
    ground_truth_for_eval = st.session_state.get('current_ground_truth_answer')

    if ground_truth_for_eval:
        logging.info(f"Ground truth answer is available for this question (first 50 chars): {ground_truth_for_eval[:50]}...")
    else:
        logging.info("No ground truth answer available for this question (e.g., it was manually typed or not from benchmark). Evaluation against ground truth will be skipped.")

    # Proceed only if there's a question to analyze.
    if question_to_analyze:
        st.markdown("---") # Visual separator
        st.subheader(f"Processing Results for Question:") # Main heading for this section
        st.markdown(f"> {question_to_analyze}") # Display the question being processed

        # Use a spinner to indicate activity during the multi-step RAG pipeline.
        with st.spinner("🧠 Analyzing question, retrieving context, generating answer, and evaluating..."):
            generated_answer_text: Optional[str] = None # Initialize variable to hold LLM's generated answer

            try:
                # === Step 1: Get Top Relevant Entities from Baseline Model ===
                logging.info("Step 1: Getting top relevant entities...")
                top_entities_info: Optional[List[tuple]] = baseline_model.input_to_entities(question_to_analyze)

                if not top_entities_info:
                    st.warning("The baseline GNN model did not return any relevant entities for this question.")
                    logging.warning("Baseline model (input_to_entities) returned no entities.")
                    # Note: We can still proceed to LLM answer generation without entities/relations/paragraphs;
                    # the LLM will then answer based on its general knowledge or the prompt's instructions.
                else:
                    logging.info(f"Baseline model found {len(top_entities_info)} top entities.")
                    # === Step 1b: Display Top Entities Table ===
                    st.subheader("🏆 Top Relevant Entities Found:")
                    try:
                        entity_display_data = []
                        for idx, node_info, similarity in top_entities_info:
                             entity_display_data.append({
                                 "Entity Value": node_info.get("value", "N/A"),
                                 "Entity Type (Group)": node_info.get("group", "N/A"),
                                 "Similarity Score": f"{similarity:.4f}",
                             })
                        df_entities = pd.DataFrame(entity_display_data)
                        st.dataframe(df_entities, use_container_width=True, hide_index=True)
                        logging.info("Displayed entity dataframe.")
                    except Exception as df_err:
                         st.error(f"Error occurred while displaying the entity table: {df_err}")
                         logging.error(f"Error creating/displaying entity DataFrame: {df_err}", exc_info=True)
                    st.divider() # Visual separator

                # === Step 2: Find Relations Involving Top Entities ===
                # This step proceeds even if top_entities_info is None/empty, passing it along.
                # find_relations_for_entities should handle empty top_entities_info gracefully.
                logging.info(f"Step 2: Finding relations (using top {TOP_N_ENTITIES_FOR_RELATIONS} entities if available).")
                relevant_relations: List[Dict[str, Any]] = find_relations_for_entities(
                    top_entities_info=(top_entities_info if top_entities_info else []), # Pass empty list if no entities
                    full_graph_data=graph_data,
                    top_n_entities=TOP_N_ENTITIES_FOR_RELATIONS
                )
                logging.info(f"Found {len(relevant_relations)} relevant relations.")

                # Initialize dictionary to store retrieved paragraph texts.
                retrieved_paragraphs_dict: Dict[str, str] = {}

                if relevant_relations:
                    # === Step 3: Retrieve Paragraph Texts ===
                    unique_para_ids = set()
                    for rel in relevant_relations:
                        para_id = rel.get("metadata", {}).get("paragraph_id")
                        if para_id and isinstance(para_id, str):
                            unique_para_ids.add(para_id)
                    logging.info(f"Found {len(unique_para_ids)} unique paragraph IDs from relations for retrieval.")

                    if unique_para_ids:
                        # NUCLIA_API_KEY is loaded at the top of app.py
                        if NUCLIA_API_KEY and NUCLIA_KB_URL:
                            st.info(f"Retrieving text for {len(unique_para_ids)} unique paragraphs from Nuclia API...")
                            logging.info(f"Step 3: Retrieving {len(unique_para_ids)} paragraphs using Nuclia API.")
                            retrieved_paragraphs_dict = retrieve_paragraphs(
                                paragraph_ids=list(unique_para_ids),
                                kb_url=NUCLIA_KB_URL,
                                api_key=NUCLIA_API_KEY
                            )
                            st.success(f"Paragraph retrieval attempt complete. Found text for {len(retrieved_paragraphs_dict)} paragraphs.")
                            if len(retrieved_paragraphs_dict) < len(unique_para_ids):
                                logging.warning(f"Could not retrieve text for all paragraph IDs ({len(retrieved_paragraphs_dict)}/{len(unique_para_ids)} successful).")
                                st.warning("Could not retrieve text for all relevant paragraphs. Check logs for details.")
                        else:
                            st.warning("Cannot retrieve paragraphs: Nuclia URL or API Key is missing from environment configuration.")
                            logging.warning("Paragraph retrieval skipped: Nuclia configuration (URL or API Key) missing.")
                    else:
                         st.info("No paragraph IDs found in the relevant relations, so no paragraph texts to retrieve.")
                         logging.info("No paragraph IDs found in relations for text retrieval.")

                    # === Step 4: Display Relations WITH Paragraph Text ===
                    st.subheader(f"🔗 Relevant Relations & Retrieved Paragraphs:")
                    logging.info(f"Displaying {len(relevant_relations)} relations with their paragraph context (if available).")
                    if not relevant_relations: # Should be caught by outer if, but good check
                        st.info("No relations were found to display.")
                    for i, rel in enumerate(relevant_relations):
                        try:
                            from_val = rel.get("from", {}).get("value", "?"); from_group = rel.get("from", {}).get("group", "?")
                            to_val = rel.get("to", {}).get("value", "?"); to_group = rel.get("to", {}).get("group", "?")
                            label = rel.get("label", "?"); para_id = rel.get("metadata", {}).get("paragraph_id")

                            st.markdown(f"**Relation {i+1}:**")
                            st.markdown(f"- **From:** `{from_val}` (`{from_group}`)")
                            st.markdown(f"- **Relation:** `{label}`")
                            st.markdown(f"- **To:** `{to_val}` (`{to_group}`)")
                            if para_id:
                                st.markdown(f"- **Source Paragraph ID:** `{para_id}`")
                                paragraph_text = retrieved_paragraphs_dict.get(para_id)
                                if paragraph_text:
                                     with st.expander("Show Retrieved Paragraph Text", expanded=False): # Default to collapsed
                                         st.markdown(f"```text\n{paragraph_text}\n```") # Display as code block for clarity
                                elif para_id in unique_para_ids: # Text was expected but not found
                                     st.caption("   (Paragraph text could not be retrieved or was empty for this ID)")
                            else:
                                st.caption("   (Paragraph ID missing in relation metadata)")
                            st.divider() # Visual separator
                        except Exception as display_err:
                            logging.warning(f"Error displaying relation {i+1}: {display_err}. Data: {rel}", exc_info=True)
                            st.warning(f"Could not properly display relation {i+1}. Please check application logs.")
                    logging.info("Finished displaying relations and paragraph context.")
                else: # If no relevant_relations were found
                    if top_entities_info: # Only show this if entities were found but no relations linking them
                        st.warning(f"No specific relations found in the graph involving the top entities.")
                    logging.warning("No relations found for top entities to display or retrieve paragraphs from.")

                # === Step 5: Generate Answer with LLM (OpenAI GPT-4.1) ===
                st.divider() # Separator before LLM answer
                st.subheader("🤖 Generated Answer (Using OpenAI GPT-4.1):")
                logging.info("Step 5: Generating answer with OpenAI LLM.")

                if not OPENAI_API_KEY: # Check if the OpenAI API key is available
                    st.error("OpenAI API Key not configured. Cannot generate answer. "
                             "Please set the OPENAI_API_KEY in your .env file.")
                    logging.error("LLM answer generation skipped: OPENAI_API_KEY not found.")
                    generated_answer_text = "Error: OpenAI API Key not configured." # Set error for eval skip
                else:
                    # Prepare the list of paragraph texts for the LLM context.
                    context_texts_for_llm = list(retrieved_paragraphs_dict.values())
                    if not context_texts_for_llm:
                        st.warning("No paragraph texts were retrieved to provide as context to the LLM. "
                                   "The LLM will attempt to answer based on its general knowledge.")
                        logging.warning("LLM Generation: No context paragraphs available to send to OpenAI.")
                    
                    logging.info(f"Calling OpenAI LLM with {len(context_texts_for_llm)} context paragraphs.")
                    generated_answer_text = generate_answer_with_openai(
                        api_key=OPENAI_API_KEY,
                        question=question_to_analyze,
                        context_paragraphs=context_texts_for_llm
                        # model_name="gpt-4.1" is the default in generate_answer_with_openai
                    )

                    if generated_answer_text:
                        if generated_answer_text.lower().startswith("error:"):
                            st.error(generated_answer_text) # Display error message from LLM util
                        else:
                            st.markdown(generated_answer_text) # Display the successful LLM answer
                        logging.info("LLM answer (or error from util) displayed.")
                    else:
                        # This case implies generate_answer_with_openai returned None
                        st.error("Failed to get a valid response or encountered an issue with the language model (OpenAI).")
                        logging.error("LLM Generation: generate_answer_with_openai returned None or empty string.")
                        generated_answer_text = "Error: Failed to get response from LLM." # Set error for eval skip

                # === Step 6: Evaluate Generated Answer (if ground truth and generated answer are available) ===
                # Only proceed with evaluation if an answer was successfully generated (not an error)
                # AND if a ground truth answer is available for the current question.
                if generated_answer_text and not generated_answer_text.lower().startswith("error:") and ground_truth_for_eval:
                    st.divider()
                    st.subheader("💯 LLM-based Evaluation of Generated Answer:")
                    logging.info(f"Step 6: Attempting LLM-based evaluation.")
                    
                    if not OPENAI_API_KEY: # Should have been caught above, but good check
                        st.error("OpenAI API Key not configured. Cannot perform LLM-based evaluation.")
                    else:
                        # --- > Call the updated evaluation function < ---
                        evaluation_result: Optional[Dict[str, Any]] = evaluate_answer_with_openai(
                            api_key=OPENAI_API_KEY,
                            question=question_to_analyze,
                            generated_answer=generated_answer_text,
                            ground_truth_answer=ground_truth_for_eval
                        )

                        # --- > Display score and explanation from the dictionary < ---
                        if evaluation_result:
                            score = evaluation_result.get("score")
                            explanation = evaluation_result.get("explanation", "No explanation provided by evaluator.")
                            raw_llm_eval_output = evaluation_result.get("raw_output", "") # For debugging

                            if score is not None:
                                st.metric(label="Answer Quality Score (by LLM)", value=f"{score}/10")
                                logging.info(f"LLM Evaluation Score: {score}/10")
                                with st.expander("Show Evaluation Explanation", expanded=True):
                                    st.markdown(explanation)
                                logging.info(f"LLM Evaluation Explanation: {explanation}")
                            else:
                                # Score parsing failed, but we might have an explanation or raw output
                                st.warning("Could not obtain a numerical evaluation score from the LLM.")
                                st.markdown("**Evaluator's Feedback (score parsing failed):**")
                                st.markdown(explanation if explanation else "No specific feedback provided.")
                                logging.warning(f"LLM evaluation did not yield a valid score. Raw output was: '{raw_llm_eval_output}'")
                        else: # Should be rare, evaluate_answer_with_openai aims to return a dict
                            st.error("LLM Evaluation process failed to return any result.")
                            logging.error("evaluate_answer_with_openai returned None (critical pre-API call failure).")
                
                elif ground_truth_for_eval and (not generated_answer_text or generated_answer_text.lower().startswith("error:")) :
                    # Case where ground truth was available, but answer generation failed.
                    st.info("LLM-based evaluation skipped as answer generation did not produce a valid result.")
                    logging.info("Evaluation skipped: Answer generation failed or resulted in an error.")
                elif not ground_truth_for_eval:
                    # Case where no ground truth was available (e.g., user typed a custom question).
                    st.info("LLM-based evaluation skipped: No ground truth answer available for this question.")
                    logging.info("Evaluation skipped: No ground truth answer was available.")


            # --- Handle Errors During Overall Analysis, Generation & Evaluation Steps ---
            except ValueError as ve:
                 st.error(f"Input Error encountered during the RAG & Evaluation pipeline: {ve}")
                 logging.error(f"ValueError during RAG & Eval pipeline: {ve}", exc_info=True)
            except Exception as e:
                st.error(f"An unexpected error occurred during the RAG & Evaluation pipeline: {e}")
                logging.error(f"Unexpected RAG & Eval pipeline error: {e}", exc_info=True)
    else:
        # User clicked 'Analyze & Generate Answer' but the text input area was empty.
        st.warning("Please enter a question in the text area above before analyzing.")
        logging.warning("'Analyze & Generate Answer' clicked with empty input.")

# Handle case where the 'Analyze' button is clicked but the model isn't ready
# (This should ideally be prevented by the button's 'disabled' state, but serves as a fallback)
elif analyze_button and not st.session_state.model_ready:
     st.warning("The system is not yet ready. Please wait for data and model loading to complete.")
     logging.warning("'Analyze & Generate Answer' clicked but st.session_state.model_ready is False.")

# --- Footer ---
st.markdown("---")
st.caption("RAG System | Entities: GNN | Context: Nuclia KG | Answer Generation: OpenAI GPT-4.1")