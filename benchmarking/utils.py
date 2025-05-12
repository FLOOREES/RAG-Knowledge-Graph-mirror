from datasets import load_dataset
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

def convert_full_text_to_markdown(full_text_dict):
    """
    Converts a full_text dictionary into a markdown-formatted string.

    Expected keys:
      - "section_name": list of section titles.
      - "paragraphs": list of lists of paragraphs corresponding to each section.

    Each section becomes a markdown header (##) followed by its paragraphs.
    """
    sections = full_text_dict.get("section_name", [])
    paragraphs = full_text_dict.get("paragraphs", [])

    markdown_lines = []
    for section, paragraph in zip(sections, paragraphs):
        markdown_lines.append(f"## {section}")
        markdown_lines.append("")  # Blank line
        markdown_lines.append("\n".join(map(str, paragraph)))
        markdown_lines.append("")  # End of section
        markdown_lines.append("")  # Extra blank line for separation
    return "\n".join(markdown_lines)

def combine_responses(row):
    """
    Combines 'extractive_spans', 'yes_no', and 'free_form_answer'
    into one single string. Skips components that are missing.
    """
    responses = []
    if pd.notna(row.get("extractive_spans")):
        if isinstance(row["extractive_spans"], list):
            responses.append(" ".join(map(str, row["extractive_spans"])))
        else:
            responses.append(str(row["extractive_spans"]))
    if pd.notna(row.get("yes_no")):
        responses.append(str(row["yes_no"]))
    if pd.notna(row.get("free_form_answer")):
        responses.append(str(row["free_form_answer"]))
    return "\n".join(responses) if responses else np.nan

def preprocess_hf_dataset(hf_ds):
    """
    Processes a HuggingFace dataset split into a cleaned Pandas DataFrame.

    Steps:
      1. For each sample, convert 'full_text' to a markdown string.
      2. For every QA pair in the sample, extract the question and first answer.
      3. Build lists for answers, questions, and full_text (duplicated per question).
      4. Create a DataFrame from the collected data.
      5. Clean columns by replacing empty lists/strings with NaN and joining lists.
      6. Combine the answer components into a single 'golden response'.

    The function uses nested tqdm progress bars for real-time feedback.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    answers_list = []  # Stores the first answer for each question
    questions_list = []  # Stores each question text
    full_text_list = []  # Stores the formatted full text per QA pair

    # Outer loop: iterate over samples with progress bar
    for sample in tqdm(hf_ds, desc="Processing samples", unit="sample"):
        # Convert full text once per sample
        formatted_text = convert_full_text_to_markdown(sample["full_text"])
        # Create a list of QA pairs
        qa_pairs = list(zip(sample["qas"]["question"], sample["qas"]["answers"]))

        # Inner loop: iterate over each QA pair with its own progress bar
        for question, answer_set in tqdm(
            qa_pairs, desc="Processing QAs", total=len(qa_pairs), leave=False, unit="qa"
        ):
            answers_list.append(answer_set["answer"][0])
            questions_list.append(question)
            full_text_list.append(formatted_text)

    # Create DataFrame from the collected data
    df = pd.DataFrame(answers_list)
    df["question"] = questions_list
    df["full_text"] = full_text_list

    # Data Cleaning: Replace empty lists/strings with NaN and join lists if needed
    df["extractive_spans"] = df["extractive_spans"].apply(
        lambda x: np.nan if isinstance(x, list) and len(x) == 0 else x
    )
    df["free_form_answer"] = df["free_form_answer"].apply(
        lambda x: np.nan if isinstance(x, str) and x.strip() == "" else x
    )
    df["yes_no"] = df["yes_no"].apply(lambda x: np.nan if x is None else x)
    df["extractive_spans"] = df["extractive_spans"].apply(
        lambda x: "\n".join(x) if isinstance(x, list) else x
    )

    # Combine the answer components into a single 'golden response'
    df["golden response"] = df.apply(lambda row: combine_responses(row), axis=1)

    return df