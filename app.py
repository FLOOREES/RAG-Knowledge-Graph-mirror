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
    from src.benchmark_utils import load_benchmark_questions, get_random_question
    from benchmarking.wrappers import CustomWrapper
    from benchmarking.general_utils import extract_triplets # Used for Nuclia API fallback
    from src.graph_processing_utils import find_relations_for_entities, retrieve_paragraphs
    from src.llm_utils import generate_answer_with_openai # For generating answers with OpenAI
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
def load_questions_from_file(filepath: str) -> List[str]:
    """Loads benchmark questions from the specified JSON file."""
    # ... (Implementation from your provided code - assumed correct and tested) ...
    try:
        logging.info(f"Loading benchmark questions from: {filepath}")
        st.info(f"Loading benchmark questions from: {os.path.basename(filepath)}")
        questions = load_benchmark_questions(filepath)
        if not questions:
            st.warning("No questions loaded from benchmark file.")
            logging.warning(f"load_benchmark_questions returned empty list from {filepath}")
            return []
        st.success(f"Loaded {len(questions)} benchmark questions.")
        logging.info(f"Loaded {len(questions)} questions from {filepath}")
        return questions
    except FileNotFoundError:
        st.error(f"Fatal Error: Benchmark file not found at {filepath}.")
        logging.error(f"Benchmark file not found: {filepath}")
        return []
    except Exception as e:
        st.error(f"Fatal Error loading benchmark questions: {e}")
        logging.error(f"Failed to load questions from {filepath}: {e}", exc_info=True)
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
benchmark_questions = load_questions_from_file(BENCHMARK_FILE_PATH)
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
    label="🎲 Get Random Benchmark Question",
    help="Load a random question from the privacy_qa benchmark.",
    disabled=(not benchmark_questions or not st.session_state.model_ready)
)
if random_button_clicked:
     logging.info("Random question button clicked.")
     if benchmark_questions:
         random_q = get_random_question(benchmark_questions)
         st.session_state.current_question = random_q
         logging.info(f"Loaded random question (first 50 chars): {random_q[:50]}...")
     else:
          st.warning("Cannot load random question: Benchmark questions list is empty or failed to load.")
          logging.warning("Random question requested, but no benchmark questions are available.")

st.text_area(
    label="Question Input:",
    value=st.session_state.get('current_question', ''),
    key="question_input_widget_key",
    height=100,
    placeholder="Ask a question about the privacy policies...",
    help="Type your question here, or use the 'Random Question' button.",
    disabled=not st.session_state.model_ready,
)
analyze_button = st.button(
    label="💡 Analyze & Generate Answer", # Updated button label
    help="Find relevant context and generate an answer using the LLM.",
    type="primary",
    disabled=not st.session_state.model_ready
)


# --- Analysis and Output Section ---
if analyze_button and st.session_state.model_ready:
    logging.info("'Analyze & Generate Answer' button clicked.")
    question_to_analyze = st.session_state.question_input_widget_key.strip()
    logging.info(f"Question for analysis: '{question_to_analyze[:100]}...'")

    if question_to_analyze:
        st.markdown("---")
        st.subheader(f"Analysis & Generation for Question:")
        st.markdown(f"> {question_to_analyze}")

        # Updated spinner text for all stages
        with st.spinner("🧠 Analyzing question, retrieving context, and generating answer with LLM..."):
            try:
                # === Step 1: Get Top Relevant Entities ===
                logging.info("Step 1: Getting top relevant entities...")
                top_entities_info: Optional[List[tuple]] = baseline_model.input_to_entities(question_to_analyze)

                if not top_entities_info:
                     st.warning("The baseline model did not return any relevant entities for this question.")
                     logging.warning("Baseline model returned no entities.")
                     # We can still proceed to LLM if desired, it will answer without specific context.
                else:
                    logging.info(f"Found {len(top_entities_info)} top entities from baseline.")
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
                         st.error(f"Error displaying entity table: {df_err}")
                         logging.error(f"Error creating/displaying entity DataFrame: {df_err}", exc_info=True)
                    st.divider()

                # === Step 2: Find Relations (even if no entities, pass empty list) ===
                relevant_relations: List[Dict[str, Any]] = []
                if top_entities_info: # Only search for relations if we have entities
                    st.info(f"Searching graph for relations involving the top {TOP_N_ENTITIES_FOR_RELATIONS} entities...")
                    logging.info(f"Step 2: Finding relations for top {TOP_N_ENTITIES_FOR_RELATIONS} entities.")
                    relevant_relations = find_relations_for_entities(
                        top_entities_info=top_entities_info,
                        full_graph_data=graph_data,
                        top_n_entities=TOP_N_ENTITIES_FOR_RELATIONS
                    )
                    logging.info(f"Found {len(relevant_relations)} relevant relations.")

                # Initialize paragraph dictionary
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
                        # NUCLIA_API_KEY is already loaded at the top
                        if NUCLIA_API_KEY and NUCLIA_KB_URL:
                            st.info(f"Retrieving text for {len(unique_para_ids)} unique paragraphs from Nuclia API...")
                            logging.info(f"Step 3: Retrieving {len(unique_para_ids)} paragraphs.")
                            retrieved_paragraphs_dict = retrieve_paragraphs(
                                paragraph_ids=list(unique_para_ids),
                                kb_url=NUCLIA_KB_URL,
                                api_key=NUCLIA_API_KEY
                            )
                            st.success(f"Attempted paragraph retrieval. Found text for {len(retrieved_paragraphs_dict)} paragraphs.")
                            if len(retrieved_paragraphs_dict) < len(unique_para_ids):
                                logging.warning(f"Could not retrieve text for all paragraph IDs.")
                                st.warning("Could not retrieve text for all relevant paragraphs. Check logs.")
                        else:
                            st.warning("Cannot retrieve paragraphs: Nuclia URL or API Key missing.")
                            logging.warning("Paragraph retrieval skipped: Nuclia config missing.")
                    else:
                         st.info("No paragraph IDs found in relevant relations to retrieve text for.")
                         logging.info("No paragraph IDs in relations for text retrieval.")

                    # === Step 4: Display Relations WITH Paragraph Text ===
                    st.subheader(f"🔗 Relevant Relations & Paragraphs:")
                    logging.info(f"Displaying {len(relevant_relations)} relations with context.")
                    for i, rel in enumerate(relevant_relations):
                        # ... (same display logic for relations and paragraphs as before) ...
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
                                 with st.expander("Show Retrieved Paragraph Text", expanded=False):
                                     st.markdown(f"```text\n{paragraph_text}\n```")
                            elif para_id in unique_para_ids:
                                 st.caption("   (Paragraph text could not be retrieved or was empty)")
                        else: st.caption("   (Paragraph ID missing in relation metadata)")
                        st.divider()
                    logging.info("Finished displaying relations and paragraphs.")
                else: # If no relevant_relations found
                    st.warning(f"No specific relations found in the graph involving the top entities.")
                    logging.warning("No relations found for top entities.")

                # === Step 5: Generate Answer with LLM (OpenAI GPT-4.1) ===
                st.divider()
                st.subheader("🤖 Generated Answer (Using OpenAI GPT-4.1):")
                if not OPENAI_API_KEY: # Check if the OpenAI key is loaded
                    st.error("OpenAI API Key not configured. Cannot generate answer. "
                             "Please set OPENAI_API_KEY in your .env file.")
                    logging.error("LLM Generation Attempt Failed: OPENAI_API_KEY not found.")
                else:
                    # Prepare context for LLM from the retrieved paragraphs
                    context_texts_for_llm = list(retrieved_paragraphs_dict.values())
                    if not context_texts_for_llm:
                        st.warning("No paragraph texts were retrieved to provide as context to the LLM. "
                                   "The LLM will attempt to answer based on its general knowledge.")
                        logging.warning("LLM Generation: No context paragraphs available to send to OpenAI.")

                    logging.info(f"Calling OpenAI LLM with {len(context_texts_for_llm)} context paragraphs.")
                    # Call the function from llm_utils to get the answer from OpenAI
                    generated_answer = generate_answer_with_openai(
                        api_key=OPENAI_API_KEY,
                        question=question_to_analyze,
                        context_paragraphs=context_texts_for_llm
                        # model_name can be passed if you want to use a different one than the default in llm_utils
                    )

                    if generated_answer:
                        # Display the answer, checking if it's an error message from the util
                        if generated_answer.lower().startswith("error:"):
                            st.error(generated_answer)
                        else:
                            st.markdown(generated_answer)
                        logging.info("LLM answer displayed (or error from LLM util).")
                    else:
                        # Handle case where LLM util returns None (should be rare if it returns error strings)
                        st.error("Failed to get a response from the language model (OpenAI).")
                        logging.error("LLM Generation: generate_answer_with_openai returned None or empty.")

            # --- Handle Errors During Overall Analysis Steps ---
            except ValueError as ve:
                 st.error(f"Input Error during analysis: {ve}")
                 logging.error(f"ValueError during analysis pipeline: {ve}", exc_info=True)
            except Exception as e:
                st.error(f"An unexpected error occurred during the RAG pipeline: {e}")
                logging.error(f"Unexpected RAG pipeline error: {e}", exc_info=True)
    else:
        # User clicked 'Analyze' but the text area was empty.
        st.warning("Please enter a question in the text area above before analyzing.")
        logging.warning("'Analyze & Generate Answer' clicked with empty input.")
elif analyze_button and not st.session_state.model_ready:
     st.warning("System is not ready. Please wait for data and model loading to complete.")
     logging.warning("'Analyze & Generate Answer' clicked but model_ready is False.")

# --- Footer ---
st.markdown("---")
st.caption("RAG System | Entities: GNN | Context: Nuclia KG | Answer Generation: OpenAI GPT-4.1")