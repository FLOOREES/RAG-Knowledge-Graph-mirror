"""
EVALUATION AS IN RAGAS DOCS
https://docs.ragas.io/en/stable/howtos/applications/gemini_benchmarking/#about-the-dataset
"""

from datasets import load_dataset
from utils import preprocess_hf_dataset
import logging
from wrappers import NucliaWrapper,CustomWrapper

# Load Dataset
dataset = load_dataset("allenai/qasper", split="validation[:10]")

# Preprocess dataset
preprocessed_dataset = preprocess_hf_dataset(dataset)

# Define common prompt
qa_prompt = (
    "Given the context information and not prior knowledge, "
    "answer the query.\n"
    "If you cannot find answer to the query, just say that it cannot be answered.\n"
    "Query: {query_str}\n"
    "Answer: "
)

# Initialize Services
nuclia_client = NucliaWrapper(qa_prompt=qa_prompt)
custom_wrapper = CustomWrapper(qa_prompt=qa_prompt)

# Ingest Data and retrieve graph
nuclia_client.ingest_context(context=context)

# 






















logging.info(preprocessed_dataset['full_text'].iloc[1])
