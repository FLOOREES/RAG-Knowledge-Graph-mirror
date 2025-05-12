# app.py
import streamlit as st
import pandas as pd
import os
import json
from dotenv import load_dotenv
import random

# --- Make sure 'benchmarking' module is discoverable ---
import sys
# Assuming app.py is in the root, and benchmark_utils is in 'src' subdir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))

# --- Import our utils and existing wrappers ---
try:
	# --- > Adjusted import path based on your code <---
	from src.benchmark_utils import load_benchmark_questions, get_random_question
	from benchmarking.wrappers import CustomWrapper
	from benchmarking.general_utils import extract_triplets # Still needed for fallback
except ImportError as e:
	st.error(f"Failed to import modules: {e}. Check file locations and __init__.py files.")
	st.stop()

# --- Configuration & Constants ---
load_dotenv()
NUCLIA_KB_URL = os.getenv('NUCLIA_KB_URL') # Still needed for fallback
# Define path relative to app.py (adjust if needed)
BENCHMARK_FILE_PATH = os.path.join(os.path.dirname(__file__), "data", "LegalBench-RAG", "benchmarks", "privacy_qa.json")
# --- > Path for your pre-extracted graph < ---
LOCAL_GRAPH_FILE = os.path.join(os.path.dirname(__file__), "data", "legal_graph.json")


# --- Caching Functions ---
@st.cache_data(show_spinner="Loading Knowledge Graph...")
def load_or_fetch_graph(local_graph_path: str, kb_url_fallback: str) -> list:
	"""
	Loads graph data preferentially from a local JSON file ('local_graph_path').
	If the file is not found or fails to load, falls back to fetching
	from the Nuclia API ('kb_url_fallback').

	Args:
		local_graph_path: Path to the local JSON graph file.
		kb_url_fallback: The Nuclia KB URL to use if fetching from API is needed.

	Returns:
		A list of relation dictionaries representing the graph, or empty list on failure.
	"""
	graph_data = []
	loaded_from_local = False

	# --- Step 1: Try loading from local JSON file first ---
	st.info(f"Checking for local graph file at: '{local_graph_path}'")
	if os.path.exists(local_graph_path):
		st.success(f"Found local graph file!")
		try:
			with open(local_graph_path, 'r', encoding='utf-8') as f:
				graph_data = json.load(f)
			st.success(f"Successfully loaded graph from {os.path.basename(local_graph_path)}!")
			if not isinstance(graph_data, list): # Basic validation
				st.warning("Warning: Loaded graph data is not a list. Processing might fail.")
			elif not graph_data:
				st.warning("Warning: Loaded graph data is empty.")

			loaded_from_local = True # Mark success
			return graph_data # Return data loaded from local file

		except json.JSONDecodeError:
			st.error(f"Error: Failed to decode JSON from {local_graph_path}. File might be corrupted.")
			st.warning("Will attempt to fetch from Nuclia API instead.")
		except Exception as e:
			st.error(f"Error reading local graph file {local_graph_path}: {e}")
			st.warning("Will attempt to fetch from Nuclia API instead.")
	else:
		st.info(f"Local graph file not found.")

	# --- Step 2: Fallback to Nuclia API if local loading failed or file missing ---
	if not loaded_from_local:
		st.info("Attempting to fetch graph data from Nuclia API...")
		if not kb_url_fallback or not os.getenv('NUCLIA_API_KEY'):
			st.error("NUCLIA_KB_URL or NUCLIA_API_KEY environment variable not set for API fallback.")
			return []

		try:
			graph_data = extract_triplets(kb_url_fallback)
			if not graph_data:
				st.warning("Fetched graph data from Nuclia is empty.")
				return []
			st.success("Successfully fetched graph from Nuclia API.")
			# Optional: Save fetched graph
			# try: ... save logic ...
			return graph_data
		except Exception as e:
			st.error(f"Failed to fetch graph from Nuclia API: {e}")
			return []

	return []


@st.cache_resource(show_spinner="Initializing Baseline Model...")
def get_baseline_model() -> CustomWrapper:
	try:
		model = CustomWrapper()
		return model
	except Exception as e:
		st.error(f"Failed to initialize the baseline model: {e}")
		st.stop()
		return None

@st.cache_data(show_spinner="Processing Graph Data for GNN...")
def ingest_graph_data(_model: CustomWrapper, graph_data: list):
	if not graph_data:
		st.warning("Cannot ingest empty graph data.")
		return False
	try:
		_model.ingest_data(graph_data)
		st.success("Graph data processed by baseline model.")
		return True
	except Exception as e:
		st.error(f"Failed to ingest graph data into the model: {e}")
		return False

@st.cache_data
def load_questions_from_file(filepath: str) -> list:
	try:
		st.info(f"Loading benchmark questions from: {filepath}")
		questions = load_benchmark_questions(filepath)
		if not questions:
			st.warning("No questions loaded from benchmark file.")
			return []
		st.success(f"Loaded {len(questions)} benchmark questions.")
		return questions
	except FileNotFoundError:
		st.error(f"Benchmark file not found: {filepath}. Please ensure the path is correct.")
		return []
	except (ValueError, KeyError) as e:
		st.error(f"Error loading benchmark questions: {e}")
		return []
	except Exception as e:
		st.error(f"An unexpected error occurred while loading questions: {e}")
		return []


# --- Streamlit App UI ---
st.set_page_config(layout="wide")
st.title("🕵️‍♀️ Privacy Policy QA - Baseline Relevance Explorer 🕵️‍♂️")
st.markdown("Type a question or load a random one from the benchmark, then analyze its relevance using the GNN baseline.")

# --- Initialize Session State ---
# --- > Use a different state variable name < ---
if 'current_question' not in st.session_state:
	st.session_state.current_question = "" # Holds the question to display/analyze
if 'model_ready' not in st.session_state:
	 st.session_state.model_ready = False

# --- Load Data, Model, and Questions ---
benchmark_questions = load_questions_from_file(BENCHMARK_FILE_PATH)
graph_data = load_or_fetch_graph(LOCAL_GRAPH_FILE, NUCLIA_KB_URL)

if graph_data:
	baseline_model = get_baseline_model()
	if baseline_model:
		ingestion_successful = ingest_graph_data(baseline_model, graph_data)
		if ingestion_successful:
			st.session_state.model_ready = True
		else:
			st.error("Model could not process the graph data. Cannot proceed.")
			st.stop()
	else:
		st.error("Baseline model failed to load. Cannot proceed.")
		st.stop()
else:
	st.error("Failed to load graph data from local file or Nuclia API. Cannot proceed.")
	st.stop()


# --- User Input Area ---
st.subheader("Enter Your Question:")

# --- Handle Random Button Action FIRST ---
random_button_clicked = st.button(
	"🎲 Get Random Benchmark Question",
	disabled=(not benchmark_questions or not st.session_state.model_ready)
)

if random_button_clicked:
	if benchmark_questions:
		random_q = get_random_question(benchmark_questions)
		# --- > Update the NEW state variable < ---
		st.session_state.current_question = random_q
		# Text area below will update on the script rerun triggered by the button click
	else:
		st.warning("Cannot load random question: Benchmark questions list is empty or failed to load.")

# --- Render Text Area ---
# Use the new state variable for the 'value'.
# Use a NEW UNIQUE key for the widget itself.
st.text_area(
	"Type your question here, or use the button above to load a random one.",
	value=st.session_state.get('current_question', ''), # Set display value
	key="question_input_widget_key", # <--- Unique key for the widget
	height=100,
	disabled=not st.session_state.model_ready,
)

# --- Render Analyze Button ---
analyze_button = st.button(
	"🔍 Analyze Relevance",
	type="primary",
	disabled=not st.session_state.model_ready
)

# --- Analysis and Output ---
if analyze_button and st.session_state.model_ready:
	# --- > Read the value from the text_area's UNIQUE state key < ---
	question_to_analyze = st.session_state.question_input_widget_key.strip()

	if question_to_analyze:
		st.markdown("---")
		st.subheader(f"Analyzing Question:")
		st.markdown(f"> {question_to_analyze}")
		with st.spinner("🧠 Running baseline model to find relevant entities..."):
			try:
				# Use the question read from the widget's state
				top_nodes_info = baseline_model.input_to_entities(question_to_analyze)
				if top_nodes_info:
					st.subheader("🏆 Top Relevant Entities Found:")
					display_data = []
					for idx, node_info, similarity in top_nodes_info:
						display_data.append({
							"Entity Value": node_info.get("value", "N/A"),
							"Entity Type (Group)": node_info.get("group", "N/A"),
							"Similarity Score": f"{similarity:.4f}",
						})
					df = pd.DataFrame(display_data)
					st.dataframe(df, use_container_width=True)
					st.markdown("---")
					st.markdown("**Top 5 Entities:**")
					for item in display_data[:5]:
						st.markdown(f"- **{item['Entity Value']}** (Type: {item['Entity Type (Group)']}) - Score: {item['Similarity Score']}")
				else:
					st.warning("The baseline model did not return any relevant entities for this question.")
			except ValueError as ve:
				st.error(f"Input Error: {ve}") # Handles 'ingest data first' error etc.
			except Exception as e:
				st.error(f"An error occurred during analysis: {e}")
				# import traceback # Uncomment for detailed debug info
				# st.code(traceback.format_exc()) # Uncomment for detailed debug info
	else:
		st.warning("Please enter a question in the text area above before analyzing.")
elif not st.session_state.model_ready:
	 st.warning("Baseline model or data is not ready. Please check logs/errors above.")

st.markdown("---")
st.caption("Baseline Model: GNN-based RAG | Data Source: Local Cache / Nuclia KG | Benchmark: privacy_qa")