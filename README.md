# Advanced Graph-RAG for Legal Q&A

> **Note:** This repository is a public mirror of a private project developed for Nuclia. It is shared for portfolio and demonstration purposes to showcase the methodologies and architecture implemented.

## 📖 Project Overview

This project explores and enhances Retrieval-Augmented Generation (RAG) for Question-Answering over complex legal documents, specifically privacy policies. The core innovation lies in leveraging a Knowledge Graph (KG) to provide highly relevant and structured context to a Large Language Model (LLM), moving beyond traditional vector-based similarity search.

The system was developed to improve upon existing graph retrieval methods. It compares two distinct Graph Neural Network (GNN) based approaches:
1.  **Baseline Entity Retriever**: A simple Message Passing Neural Network (MPNN) that identifies the most relevant entities in the graph based on the user's query.
2.  **Enhanced Path Retriever**: A more sophisticated approach using a Relational Graph Convolutional Network (R-GCN) methodology to identify and score entire reasoning paths within the KG, providing richer, more interconnected context to the LLM.

Both systems were successfully implemented and demonstrated the viability and advantages of using structured graph-based context for RAG in specialized domains.

---

## 🏛️ Architecture and Components

The project is structured into several key components that work together to form the complete RAG pipeline.

1.  **Knowledge Graph (KG) Construction**
    -   The foundation of the system is a knowledge graph built from legal documents.
    -   The initial graph, containing entities and their relationships, is extracted using Nuclia's powerful data processing capabilities.
    -   This graph is saved locally as `data/legal_graph.json` and serves as the single source of truth for all subsequent components.

2.  **Retrieval Models**
    -   **Baseline GNN (`app.py`)**: Implemented in `benchmarking/wrappers.py` using `SimpleMPNN`. This model takes an embedded user query, identifies the top-K most similar entities (nodes) in the KG, and retrieves all direct relations involving these entities.
    -   **Enhanced Path-Finding GNN (`app_v2.py`)**: Implemented in `src/rgcn.py` and related modules. This model first identifies "seed entities" from the query, extracts a multi-hop bidirectional subgraph around them, and then intelligently scores paths within this subgraph to find the most relevant reasoning chains for answering the query.

3.  **Generation & Evaluation**
    -   **LLM Integration (`src/llm_utils.py`)**: Both retrieval methods pass their collected context (relations, entities, and original text paragraphs) to an OpenAI model (e.g., GPT-4o-mini) to generate a natural language answer.
    -   **LLM-as-a-Judge Evaluation**: For benchmarking, a separate LLM call evaluates the generated answer against a ground-truth answer, providing a quantitative score (1-10) and a qualitative explanation.

4.  **Interactive Applications (`app.py`, `app_v2.py`)**
    -   Two separate Streamlit applications are provided to demonstrate and compare the two retrieval methodologies in real-time. They visualize the retrieved context and the final generated answer.

5.  **Benchmarking Framework (`src/nuclia_eval/`)**
    -   A robust evaluation script (`run_nuclia_eval_mq.py`) allows for running a full benchmark (e.g., `privacy_qa.json`) against different retrieval solutions (`nuclia`, `GNN`, `Similarity`, `MSPN`) and saves the detailed results, including context, answers, and evaluation scores, to a `.jsonl` file.

---

## 🚀 Getting Started

### Prerequisites

-   Python 3.9+
-   `pip` and `virtualenv`

### Setup and Installation

1.  **Clone the Repository**
    ```bash
    git clone <https://github.com/FLOOREES/RAG-Knowledge-Graph-mirror>
    cd <RAG-Knowledge-Graph-mirror>
    ```

2.  **Create a Virtual Environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    The `requirements.txt` file contains all necessary packages.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set Up Environment Variables**
    Create a `.env` file in the root directory of the project by copying the example file:
    ```bash
    cp .env.example .env
    ```
    Now, open the `.env` file and add your credentials:
    ```env
    # .env
    NUCLIA_KB_URL="your_nuclia_kb_url"
    NUCLIA_API_KEY="your_nuclia_api_key"
    OPENAI_API_KEY="your_openai_api_key"
    ```

---

## 💻 Usage

There are three main ways to run this project: generating the KG, running the interactive apps, or executing the benchmark evaluation.

### 1. Generate the Knowledge Graph (Optional)

The pre-generated knowledge graph (`legal_graph.json`) is already included in the `data/` directory. However, if you need to regenerate it from the Nuclia Knowledge Box, you can use the `extract_triplets` function found in `benchmarking/general_utils.py`.

*This step is typically not required unless the source data in Nuclia has changed.*

### 2. Run the Interactive RAG Applications

The project includes two Streamlit applications to demonstrate the different retrieval models.

-   **To run the Baseline (Entity-Centric) Model:**
    ```bash
    streamlit run app.py
    ```

-   **To run the Enhanced (Path-Based) Model:**
    ```bash
    streamlit run app_v2.py
    ```

Open your browser to the local URL provided by Streamlit to interact with the applications.

### 3. Run the Benchmark Evaluation

The `run_nuclia_eval_mq.py` script allows you to systematically evaluate the performance of different RAG solutions on a benchmark dataset.

The main argument is `--solution`, which can be one of:
-   `nuclia`: Nuclia's native RAG pipeline.
-   `Similarity`: The baseline GNN model that retrieves top entities by similarity (`app.py`).
-   `MSPN`: The enhanced Message-Passing Semantic Network that retrieves relevant paths (`app_v2.py`).

**Example Command:**

To run the `MSPN` solution on the `privacy_qa` benchmark and save the results in a directory named `eval_results/`:

```bash
python -m src.nuclia_eval.run_nuclia_eval_mq \
    --benchmarks privacy_qa \
    --solution MSPN \
    --output_dir eval_results/ \
    --log_level INFO
```

The script will generate a .jsonl file in the output directory containing detailed results for each question in the benchmark.

## 📦 Dependencies

The main dependencies for this project are listed below. Please refer to `requirements.txt` for the full list of packages and their versions.

-   `streamlit`: For the interactive web applications.
-   `torch` & `torch_geometric`: For building and operating the GNN models.
-   `sentence-transformers`: For encoding text into vector embeddings.
-   `openai`: For interacting with the GPT models for generation and evaluation.
-   `nuclia-sdk`: For interacting with the Nuclia API.
-   `pandas`: For data manipulation and display.
-   `python-dotenv`: For managing environment variables.
-   `spacy` & `rapidfuzz`: For Named Entity Recognition and fuzzy string matching.