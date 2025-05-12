# app.py
# --- Core Imports ---
import streamlit as st
import pandas as pd
import os
import json
from dotenv import load_dotenv
import random
import logging
import sys
from typing import List, Dict, Any # Added for type hinting clarity

# --- Configuration for Logging ---
# Provides informative messages during execution, helpful for debugging.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Ensure Application Modules are Discoverable ---
# Adds the project's root directory to the Python path to allow importing local modules.
# Adjust './' if your app.py is not in the root relative to 'src' and 'benchmarking'.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
logging.info(f"Appended to sys.path: {os.path.abspath(os.path.join(os.path.dirname(__file__), './'))}")

# --- Import Custom Application Modules ---
# Loads utility functions and the baseline model wrapper.
try:
    from src.benchmark_utils import load_benchmark_questions, get_random_question
    from benchmarking.wrappers import CustomWrapper
    from benchmarking.general_utils import extract_triplets # Still needed for API fallback
    from src.graph_processing_utils import find_relations_for_entities # Finds relations for entities
    logging.info("Successfully imported custom modules.")
except ImportError as e:
    # Display error prominently in the app and stop execution if imports fail.
    st.error(f"Fatal Error: Failed to import required application modules: {e}. "
             "Please ensure file structure is correct ('src', 'benchmarking' folders) "
             "and '__init__.py' files exist where needed.")
    logging.error(f"Import error on startup: {e}", exc_info=True)
    st.stop()
except Exception as e:
    st.error(f"Fatal Error: An unexpected error occurred during initial imports: {e}")
    logging.error(f"Unexpected import error: {e}", exc_info=True)
    st.stop()


# --- Configuration & Constants ---
# Load environment variables (API keys, URLs) from a .env file in the root directory.
if load_dotenv():
    logging.info(".env file loaded successfully.")
else:
    logging.warning(".env file not found or failed to load. Relying on environment variables.")

# Retrieve Nuclia KB URL for API fallback (if local graph file is missing/invalid).
NUCLIA_KB_URL = os.getenv('NUCLIA_KB_URL')
if not NUCLIA_KB_URL:
    logging.warning("NUCLIA_KB_URL not found in environment variables. API fallback will fail if needed.")

# Define path for the benchmark questions JSON file.
# Assumes 'data' folder is in the same directory as app.py.
BENCHMARK_FILE_PATH = os.path.join(os.path.dirname(__file__), "data", "LegalBench-RAG", "benchmarks", "privacy_qa.json")
logging.info(f"Benchmark questions path set to: {BENCHMARK_FILE_PATH}")

# Define path for the pre-extracted local Knowledge Graph JSON file.
# Assumes 'data' folder is in the same directory as app.py.
LOCAL_GRAPH_FILE = os.path.join(os.path.dirname(__file__), "data", "legal_graph.json")
logging.info(f"Local graph path set to: {LOCAL_GRAPH_FILE}")

# Configuration: How many top entities returned by the GNN model
# should be used to search for relevant relations in the graph.
TOP_N_ENTITIES_FOR_RELATIONS: int = 5
logging.info(f"Will search for relations involving the top {TOP_N_ENTITIES_FOR_RELATIONS} entities.")


# --- Caching Functions ---
# These functions use Streamlit's caching to avoid expensive re-computation
# or data loading on every interaction or script rerun.

@st.cache_data(show_spinner="Loading Knowledge Graph...")
def load_or_fetch_graph(local_graph_path: str, kb_url_fallback: str) -> List[Dict[str, Any]]:
    """
    Loads graph data preferentially from a local JSON file.
    If the file is not found or fails to load, falls back to fetching
    from the Nuclia API via extract_triplets.

    Args:
        local_graph_path: Path to the local JSON graph file.
        kb_url_fallback: The Nuclia KB URL for the API fallback.

    Returns:
        A list of relation dictionaries representing the graph, or empty list on failure.
    """
    graph_data = []
    loaded_from_local = False
    logging.info(f"Attempting to load graph from local path: '{local_graph_path}'")
    st.info(f"Checking for local graph file at: '{os.path.basename(local_graph_path)}'") # User-friendly path

    if os.path.exists(local_graph_path):
        st.success(f"Found local graph file!")
        logging.info("Local graph file found.")
        try:
            with open(local_graph_path, 'r', encoding='utf-8') as f:
                graph_data = json.load(f)
            st.success(f"Successfully loaded graph from {os.path.basename(local_graph_path)}!")
            logging.info(f"Successfully loaded graph from {local_graph_path}.")
            if not isinstance(graph_data, list):
                 st.warning("Warning: Loaded graph data is not a list. Processing might fail.")
                 logging.warning("Loaded graph data is not a list.")
            elif not graph_data:
                 st.warning("Warning: Loaded graph data is empty.")
                 logging.warning("Loaded graph data is empty.")
            loaded_from_local = True
            return graph_data # Success - return locally loaded data
        except json.JSONDecodeError as json_err:
            st.error(f"Error: Failed to decode JSON from {local_graph_path}. File might be corrupted.")
            st.warning("Will attempt to fetch from Nuclia API instead.")
            logging.error(f"JSONDecodeError reading {local_graph_path}: {json_err}", exc_info=True)
        except Exception as e:
             st.error(f"Error reading local graph file {local_graph_path}: {e}")
             st.warning("Will attempt to fetch from Nuclia API instead.")
             logging.error(f"Error reading local file {local_graph_path}: {e}", exc_info=True)
    else:
         st.info(f"Local graph file not found.")
         logging.info(f"Local graph file not found at {local_graph_path}.")

    # --- Fallback to Nuclia API ---
    if not loaded_from_local:
        st.info("Attempting to fetch graph data from Nuclia API...")
        logging.info("Attempting API fallback to fetch graph data.")
        if not kb_url_fallback:
             st.error("Nuclia KB URL not configured for API fallback.")
             logging.error("API fallback failed: NUCLIA_KB_URL is not set.")
             return []
        if not os.getenv('NUCLIA_API_KEY'):
            st.error("Nuclia API Key not configured for API fallback.")
            logging.error("API fallback failed: NUCLIA_API_KEY is not set.")
            return []

        try:
            logging.info(f"Calling extract_triplets for URL: {kb_url_fallback}")
            graph_data = extract_triplets(kb_url_fallback) # Assume this handles its own logging/errors somewhat
            if not graph_data:
                 st.warning("Fetched graph data from Nuclia is empty.")
                 logging.warning("extract_triplets returned empty data.")
                 return []
            st.success("Successfully fetched graph from Nuclia API.")
            logging.info("Successfully fetched graph data via API fallback.")
            # Optional: Save fetched graph to local file
            # try: ... save logic ... except ... log error ...
            return graph_data
        except Exception as e:
            st.error(f"Failed to fetch graph from Nuclia API: {e}")
            logging.error(f"API fallback failed: Error during extract_triplets call: {e}", exc_info=True)
            return [] # Return empty list on API failure

    logging.error("load_or_fetch_graph failed to load or fetch graph data.")
    return [] # Should ideally not be reached if logic above is sound


@st.cache_resource(show_spinner="Initializing Baseline Model...")
def get_baseline_model() -> CustomWrapper | None:
    """Initializes and caches the baseline GNN model wrapper."""
    try:
        logging.info("Initializing baseline model (CustomWrapper)...")
        model = CustomWrapper()
        logging.info("Baseline model initialized successfully.")
        return model
    except Exception as e:
        st.error(f"Fatal Error: Failed to initialize the baseline model: {e}")
        logging.error(f"Failed to initialize CustomWrapper: {e}", exc_info=True)
        st.stop() # Stop app execution if model fails to load
        return None # Explicitly return None although st.stop() halts

@st.cache_data(show_spinner="Processing Graph Data for GNN...")
def ingest_graph_data(_model: CustomWrapper, graph_data: List[Dict[str, Any]]) -> bool:
    """
    Ingests the loaded graph data into the baseline model.
    Uses Streamlit's data caching; depends on the hash of graph_data.

    Args:
        _model: The baseline model instance (used conceptually for caching).
        graph_data: The list of relation dictionaries.

    Returns:
        True if ingestion was successful, False otherwise.
    """
    if not graph_data:
        st.warning("Cannot ingest empty graph data into the model.")
        logging.warning("Skipping model data ingestion: graph_data is empty.")
        return False # Indicate failure clearly
    if not _model:
         st.error("Cannot ingest data: Baseline model is not available.")
         logging.error("Skipping model data ingestion: Model instance is None.")
         return False

    try:
        logging.info(f"Ingesting graph data ({len(graph_data)} relations) into baseline model...")
        _model.ingest_data(graph_data) # This performs GNN data prep and embedding
        st.success("Graph data processed by baseline model.")
        logging.info("Graph data ingestion successful.")
        return True
    except Exception as e:
        st.error(f"Error ingesting graph data into the model: {e}")
        logging.error(f"Failed during model.ingest_data: {e}", exc_info=True)
        return False

@st.cache_data # Cache the loaded questions list
def load_questions_from_file(filepath: str) -> List[str]:
    """Loads benchmark questions from the specified JSON file."""
    try:
        logging.info(f"Loading benchmark questions from: {filepath}")
        st.info(f"Loading benchmark questions from: {os.path.basename(filepath)}")
        questions = load_benchmark_questions(filepath) # Assumes this func handles file errors
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
        return [] # Return empty list on critical error
    except Exception as e:
        st.error(f"Fatal Error loading benchmark questions: {e}")
        logging.error(f"Failed to load questions from {filepath}: {e}", exc_info=True)
        return []


# --- Streamlit App UI ---
# Configure page settings (title, layout)
st.set_page_config(
    page_title="RAG Relevance Explorer",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Display application title and description
st.title("🕵️‍♀️ Privacy Policy QA - Baseline Relevance Explorer 🕵️‍♂️")
st.markdown("Enter a question or load a random one from the benchmark. "
            "The baseline GNN model will identify relevant entities, "
            "and related information from the Knowledge Graph will be displayed.")

# --- Initialize Session State ---
# Streamlit reruns the script on interactions. Session state preserves data between reruns.
if 'current_question' not in st.session_state:
    # Stores the question currently displayed in the text area or selected randomly.
    st.session_state.current_question = ""
    logging.debug("Initialized st.session_state.current_question")
if 'model_ready' not in st.session_state:
     # Flag indicating if the model and graph data loaded successfully. Controls UI element states.
     st.session_state.model_ready = False
     logging.debug("Initialized st.session_state.model_ready")

# --- Load Data, Model, and Questions ---
# These steps are performed once (or when cache expires) due to caching decorators.
benchmark_questions = load_questions_from_file(BENCHMARK_FILE_PATH)
# Load graph preferentially from local file, fallback to API
graph_data = load_or_fetch_graph(LOCAL_GRAPH_FILE, NUCLIA_KB_URL)

# Only proceed with model loading if graph data is available
if graph_data:
    baseline_model = get_baseline_model() # Load/get cached model instance
    if baseline_model:
        # Ingest graph data into the model (uses caching based on graph_data)
        ingestion_successful = ingest_graph_data(baseline_model, graph_data)
        if ingestion_successful:
            st.session_state.model_ready = True # Enable UI once model is ready
            logging.info("Model and data ready.")
        else:
             st.error("Model could not process the graph data. Cannot proceed.")
             # st.stop() is automatically handled by error in ingest function if critical
    else:
         # Error handled within get_baseline_model - app stops
         pass # App should have stopped if model loading failed critically
else:
    st.error("Failed to load graph data from local file or Nuclia API. Cannot proceed.")
    # st.stop() is automatically handled by error in load_or_fetch_graph if critical


# --- User Input Section ---
st.subheader("Enter Your Question:")

# --- Handle Random Question Button Action ---
# Check if button was clicked BEFORE rendering dependent widgets like text_area
random_button_clicked = st.button(
    label="🎲 Get Random Benchmark Question",
    help="Load a random question from the privacy_qa benchmark.",
    disabled=(not benchmark_questions or not st.session_state.model_ready) # Disable if questions/model aren't ready
)

if random_button_clicked:
     logging.info("Random question button clicked.")
     if benchmark_questions:
         random_q = get_random_question(benchmark_questions)
         st.session_state.current_question = random_q # Update state variable for text area value
         logging.info(f"Loaded random question: {random_q[:50]}...") # Log snippet
         # Rerun happens automatically, updating the text area below
     else:
          st.warning("Cannot load random question: Benchmark questions list is empty or failed to load.")
          logging.warning("Random question button clicked, but no benchmark questions available.")

# --- Render Question Input Text Area ---
# Uses 'current_question' state for its displayed value.
# Has its own unique key to read the potentially user-modified content.
st.text_area(
    label="Question Input:", # Added label for clarity
    value=st.session_state.get('current_question', ''), # Set display value from state
    key="question_input_widget_key", # Unique key for this specific widget instance
    height=100,
    placeholder="Ask a question about the privacy policies...",
    help="Type your question here, or use the 'Random Question' button.",
    disabled=not st.session_state.model_ready, # Disable if model isn't ready
)

# --- Render Analyze Button ---
analyze_button = st.button(
    label="🔍 Analyze Relevance",
    help="Run the baseline model to find relevant entities and relations for the entered question.",
    type="primary", # Make button visually prominent
    disabled=not st.session_state.model_ready # Disable if model isn't ready
)


# --- Analysis and Output Section ---
# This block executes only when the 'Analyze Relevance' button is clicked AND the model is ready.
if analyze_button and st.session_state.model_ready:
    logging.info(f"'Analyze Relevance' button clicked.")
    # Read the current text directly from the text area widget's state using its unique key.
    question_to_analyze = st.session_state.question_input_widget_key.strip()
    logging.info(f"Question to analyze: '{question_to_analyze[:100]}...'") # Log snippet

    if question_to_analyze:
        # Display the question being analyzed for clarity.
        st.markdown("---")
        st.subheader(f"Analysis Results for Question:")
        st.markdown(f"> {question_to_analyze}")

        # Show spinner during processing.
        with st.spinner("🧠 Running baseline model and finding relevant relations..."):
            try:
                # === Step 1: Get Top Relevant Entities ===
                logging.info("Calling baseline_model.input_to_entities...")
                top_entities_info: List[tuple] | None = baseline_model.input_to_entities(question_to_analyze)

                if not top_entities_info:
                     st.warning("The baseline model did not return any relevant entities for this question.")
                     logging.warning("baseline_model.input_to_entities returned no results.")
                else:
                    logging.info(f"Found {len(top_entities_info)} potential entities from baseline.")
                    # --- Display Top Entities Table ---
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

                    st.divider() # Visual separator

                    # === Step 2: Find Relations Involving Top Entities ===
                    st.info(f"Searching for relations involving the top {TOP_N_ENTITIES_FOR_RELATIONS} entities...")
                    logging.info(f"Calling find_relations_for_entities (top_n={TOP_N_ENTITIES_FOR_RELATIONS})...")
                    # Ensure 'graph_data' (loaded at the start) is passed correctly.
                    relevant_relations: List[Dict[str, Any]] = find_relations_for_entities(
                        top_entities_info=top_entities_info,
                        full_graph_data=graph_data,
                        top_n_entities=TOP_N_ENTITIES_FOR_RELATIONS
                    )
                    logging.info(f"find_relations_for_entities returned {len(relevant_relations)} relations.")

                    # === Step 3: Display Relevant Relations ===
                    if relevant_relations:
                        st.subheader(f"🔗 Relevant Relations Found (Involving Top {TOP_N_ENTITIES_FOR_RELATIONS} Entities):")

                        # Display relations in a structured format.
                        for i, rel in enumerate(relevant_relations):
                            try:
                                # Safely extract values using .get with defaults
                                from_val = rel.get("from", {}).get("value", "?")
                                from_group = rel.get("from", {}).get("group", "?")
                                to_val = rel.get("to", {}).get("value", "?")
                                to_group = rel.get("to", {}).get("group", "?")
                                label = rel.get("label", "?")
                                # Safely extract paragraph_id from potentially nested metadata
                                para_id = rel.get("metadata", {}).get("paragraph_id", "N/A")

                                # Display using markdown for better structure and formatting
                                st.markdown(f"**Relation {i+1}:**")
                                st.markdown(f"- **From:** `{from_val}` (`{from_group}`)")
                                st.markdown(f"- **Relation:** `{label}`")
                                st.markdown(f"- **To:** `{to_val}` (`{to_group}`)")
                                st.markdown(f"- **Source Paragraph ID:** `{para_id}`") # Key for context retrieval
                                st.divider() # Visual separator between relations

                            except Exception as display_e:
                                 # Log issues displaying specific relations but continue if possible.
                                 logging.warning(f"Could not display relation {i+1} due to error: {display_e}. Data: {rel}", exc_info=True)
                                 st.warning(f"Could not display relation {i+1}. Check logs.")
                        logging.info(f"Finished displaying {len(relevant_relations)} relations.")

                    else:
                        # Message if no relations were found for the top entities.
                        st.warning(f"No relations found in the graph involving the top {TOP_N_ENTITIES_FOR_RELATIONS} relevant entities.")
                        logging.warning(f"No relations found for top entities.")

            # Handle specific expected errors like ValueError from model input processing
            except ValueError as ve:
                 st.error(f"Input Error during analysis: {ve}")
                 logging.error(f"ValueError during analysis: {ve}", exc_info=True)
            # Catch any other unexpected errors during the analysis process
            except Exception as e:
                st.error(f"An unexpected error occurred during analysis: {e}")
                logging.error(f"Unexpected analysis error: {e}", exc_info=True)

    else:
        # User clicked 'Analyze' with an empty question input.
        st.warning("Please enter a question in the text area above before analyzing.")
        logging.warning("'Analyze Relevance' clicked with empty input.")

# Handle case where model isn't ready when analyze button is clicked
# (Should be prevented by disabled state, but good as a fallback check)
elif analyze_button and not st.session_state.model_ready:
     st.warning("Baseline model or data is not ready. Please wait for loading to complete.")
     logging.warning("'Analyze Relevance' clicked but model_ready is False.")


# --- Footer ---
st.markdown("---")
st.caption("Baseline Model: GNN-based RAG | Data Source: Local Cache / Nuclia KG | Benchmark: privacy_qa")