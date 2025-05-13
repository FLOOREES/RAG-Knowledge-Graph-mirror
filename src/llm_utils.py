# src/llm_utils.py
import logging
from typing import List, Optional, Dict, Any # Keep Optional
import openai # Import the OpenAI library

# Configure logging for this module
logger = logging.getLogger(__name__)
# Ensure basicConfig is called in app.py or this won't output unless app also configures root logger
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

import re


# --- OpenAI Function ---
def generate_answer_with_openai(
        api_key: str,
        question: str,
        context_paragraphs: List[str],
        # --- > Using "gpt-4.1" based on your example. < ---
        # If "GPT-4.1-nano" has a different specific ID, use that instead.
        model_name: str = "gpt-4.1-nano"
    ) -> Optional[str]:
    """
    Generates an answer to a given question based on the provided context
    paragraphs using the specified OpenAI GPT model.

    Args:
        api_key: The OpenAI API key.
        question: The user's question.
        context_paragraphs: A list of text strings, each representing a retrieved paragraph.
        model_name: The OpenAI model identifier string (e.g., "gpt-4.1", "gpt-4o").
                    Verify this ID if "GPT-4.1-nano" is a more specific variant.

    Returns:
        The generated answer as a string, or None if an error occurs.
    """
    if not api_key:
        logger.error("OpenAI API key is missing. Cannot generate answer.")
        return "Error: OpenAI API key not provided. Please set OPENAI_API_KEY in your .env file."
    if not question:
        logger.error("Question is missing for OpenAI. Cannot generate answer.")
        return "Error: Question not provided."

    try:
        # Initialize the OpenAI client with the provided API key.
        client = openai.OpenAI(api_key=api_key)
        logger.info(f"Initialized OpenAI client. Attempting to use model: {model_name}")

        # --- Construct the messages for the Chat Completions API ---
        # System message to guide the LLM's behavior for RAG.
        system_message = (
            "You are a helpful AI assistant. Based *only* on the following context extracted from privacy policy documents, "
            "please answer the user's question. If the context does not contain the answer, "
            "state that you cannot answer based on the provided information or that the information is not available in the context. "
            "Do not use any external knowledge or make assumptions beyond the provided text."
        )

        # Combine context paragraphs into a single string.
        context_string = "\n---\n".join(
            f"Context Paragraph {i+1}:\n{para_text.strip()}" for i, para_text in enumerate(context_paragraphs)
        )

        # Construct the user message part of the prompt.
        if not context_paragraphs:
            user_prompt_content = (
                "No specific context paragraphs were retrieved from the documents. "
                "Please answer the following question based on your general knowledge. "
                "However, clearly state if your answer is not based on specific document excerpts provided now.\n\n"
                f"Question: {question}"
            )
            logger.warning("No context paragraphs provided to OpenAI. LLM will use general knowledge with a disclaimer.")
        else:
            user_prompt_content = (
                "Here is the context from the privacy policy documents:\n"
                "---BEGIN CONTEXT---\n"
                f"{context_string}\n"
                "---END CONTEXT---\n\n"
                f"Based on the context provided above, answer this question: {question}"
            )
            logger.info(f"Prepared context with {len(context_paragraphs)} paragraphs for OpenAI.")

        logger.debug(f"System Message for OpenAI: {system_message}")
        logger.debug(f"User Prompt for OpenAI (first 300 chars of content):\n{user_prompt_content[:300]}...")

        # --- Make the API call using the Chat Completions endpoint ---
        completion = client.chat.completions.create(
            model=model_name, # Uses the model ID like "gpt-4.1"
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt_content}
            ],
            # Optional parameters for more control:
            # temperature=0.5,  # Lower for more factual, higher for more creative. Default is often 0.7-1.0
            # max_tokens=1024,  # Adjust based on expected answer length and model limits
        )

        logger.info("Received response from OpenAI API.")

        # Extract the answer from the first choice.
        if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
            answer = completion.choices[0].message.content
            logger.info(f"Successfully extracted answer from OpenAI: {answer[:100]}...") # Log snippet
            return answer.strip()
        else:
            # This case should be rare if the API call itself was successful.
            logger.error(f"Could not extract answer content from OpenAI response. Response object: {completion}")
            return "Error: The language model returned a response, but the answer content was empty or in an unexpected format."

    except openai.APIConnectionError as e:
        logger.error(f"OpenAI API Connection Error: {e}", exc_info=True)
        return f"Error: Could not connect to OpenAI API. Please check your network connection. Details: {e}"
    except openai.RateLimitError as e:
        logger.error(f"OpenAI API Rate Limit Exceeded: {e}", exc_info=True)
        return f"Error: OpenAI API rate limit exceeded. Please try again later or check your usage limits. Details: {e}"
    except openai.AuthenticationError as e:
        logger.error(f"OpenAI API Authentication Error: {e}", exc_info=True)
        return f"Error: OpenAI API authentication failed. Please check your API key. Details: {e}"
    except openai.APIStatusError as e: # More general API error
        logger.error(f"OpenAI API Status Error: Status Code {e.status_code}, Response: {e.response}", exc_info=True)
        return f"Error: OpenAI API returned an error (Status {e.status_code}). Details: {e.message}"
    except Exception as e: # Catch any other unexpected exceptions
        logger.error(f"An unexpected error occurred while generating answer with OpenAI: {e}", exc_info=True)
        return f"Error: An unexpected error occurred while contacting the OpenAI API: {str(e)}"

def evaluate_answer_with_openai(
        api_key: str,
        question: str,
        generated_answer: str,
        ground_truth_answer: str,
        model_name: str = "gpt-4.1" # Or your confirmed model ID
    ) -> Optional[Dict[str, Any]]: # Now returns a dictionary
    """
    Evaluates a generated answer against a ground truth answer using OpenAI,
    aiming for a score (1-10) and a brief explanation.

    Args:
        api_key, question, generated_answer, ground_truth_answer, model_name.

    Returns:
        A dictionary e.g., {"score": int, "explanation": str, "raw_output": str}.
        Score or explanation might be None if parsing fails.
        Returns None for critical pre-API call errors (e.g., missing args).
    """
    if not all([api_key, question, generated_answer, ground_truth_answer]):
        logger.error("evaluate_answer_with_openai: Missing one or more required arguments.")
        return None # Critical argument failure

    try:
        client = openai.OpenAI(api_key=api_key)
        logger.info(f"Initializing OpenAI client for evaluation with model: {model_name}")

        # Updated system prompt to ask for score AND explanation in a structured way
        system_message = (
            "You are an impartial and meticulous AI evaluator. Your task is to assess the quality of a 'Generated Answer' "
            "in response to an 'Original Question', by comparing it against a 'Ground Truth Answer'.\n"
            "Consider the following criteria for your evaluation:\n"
            "1.  **Relevance:** Is the Generated Answer directly and fully relevant to the Original Question?\n"
            "2.  **Faithfulness/Accuracy:** Is the Generated Answer factually correct and consistent with the Ground Truth Answer? It must not contradict the Ground Truth or introduce unverifiable external information.\n"
            "3.  **Completeness:** Does the Generated Answer comprehensively address all key aspects of the Original Question that are covered by the Ground Truth Answer?\n"
            "4.  **Conciseness:** Is the Generated Answer to the point, avoiding unnecessary verbosity or redundant information not present or implied by the Ground Truth?\n\n"
            "Please provide your evaluation in the following format:\n"
            "Score: [A single integer from 1 to 10]\n"
            "Explanation: [A brief (1-3 sentences) explanation for your score, referencing the criteria above.]\n\n"
            "Example:\n"
            "Score: 7\n"
            "Explanation: The generated answer is relevant and mostly accurate but misses a subtle detail present in the ground truth regarding user consent. It is otherwise concise."
        )

        user_prompt_content = (
            f"Original Question:\n```\n{question}\n```\n\n"
            f"Ground Truth Answer:\n```\n{ground_truth_answer}\n```\n\n"
            f"Generated Answer to Evaluate:\n```\n{generated_answer}\n```\n\n"
            "Please provide your evaluation in the specified format (Score line, then Explanation line):"
        )

        logger.debug(f"Evaluation - System Message for LLM:\n{system_message}")
        # logger.debug(f"Evaluation - User Prompt for LLM:\n{user_prompt_content}") # Can be verbose

        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt_content}
            ],
            temperature=0.0,  # Lowest temperature for consistency in evaluation
            max_tokens=200,   # Increased to allow for a good explanation
            n=1
        )
        logger.info("Received evaluation response from OpenAI API.")

        parsed_score: Optional[int] = None
        explanation: str = "Evaluation explanation could not be parsed." # Default
        raw_output_text: str = ""

        if completion.choices and completion.choices[0].message and completion.choices[0].message.content:
            raw_output_text = completion.choices[0].message.content.strip()
            logger.info(f"Raw evaluation output from LLM:\n'''\n{raw_output_text}\n'''")

            # Attempt to parse score using regex, looking for "Score: X"
            score_match = re.search(r"Score:\s*(\b(?:10|[1-9])\b)", raw_output_text, re.IGNORECASE)
            if score_match:
                try:
                    parsed_score = int(score_match.group(1))
                    logger.info(f"Parsed evaluation score: {parsed_score}")
                except ValueError:
                    logger.warning(f"Could not convert parsed score digits '{score_match.group(1)}' to an integer.")
            else: # Fallback: try to find just a number if "Score:" prefix is missing
                first_number_match = re.search(r"\b(10|[1-9])\b", raw_output_text)
                if first_number_match:
                    try:
                        parsed_score = int(first_number_match.group(1))
                        logger.info(f"Parsed evaluation score (fallback, found number): {parsed_score}")
                    except ValueError:
                        logger.warning(f"Fallback: Could not convert found digits '{first_number_match.group(1)}' to int.")
                else:
                    logger.warning(f"Could not find 'Score: [1-10]' pattern or a standalone number (1-10) in LLM response: '{raw_output_text}'")


            # Attempt to parse explanation, looking for "Explanation: ..."
            explanation_match = re.search(r"Explanation:\s*(.+)", raw_output_text, re.IGNORECASE | re.DOTALL)
            if explanation_match:
                explanation = explanation_match.group(1).strip()
                logger.info(f"Parsed explanation: {explanation}")
            else:
                # If no "Explanation:" prefix, but there's text after a potential score line, use that.
                lines = raw_output_text.splitlines()
                if score_match and len(lines) > 1: # If score was found and there's more text
                    explanation_candidate = "\n".join(lines[1:]).strip()
                    if explanation_candidate: # If there's actually text on subsequent lines
                        explanation = explanation_candidate
                        logger.info(f"Parsed explanation (fallback, text after score line): {explanation}")
                elif not score_match and raw_output_text: # No score, use whole output as explanation
                    explanation = f"LLM did not provide a score in the expected format. Full response: {raw_output_text}"


            return {"score": parsed_score, "explanation": explanation, "raw_output": raw_output_text}
        else:
            logger.error(f"Could not extract evaluation content from OpenAI response object: {completion}")
            return {"score": None, "explanation": "Error: OpenAI response was empty or malformed.", "raw_output": str(completion)}

    except openai.APIError as e:
        logger.error(f"OpenAI API Error during evaluation: {e.status_code} - {e.message}", exc_info=True)
        return {"score": None, "explanation": f"OpenAI API Error: {e.message}", "raw_output": str(e)}
    except Exception as e:
        logger.error(f"An unexpected error occurred during LLM evaluation: {e}", exc_info=True)
        return {"score": None, "explanation": f"Unexpected error during evaluation: {str(e)}", "raw_output": str(e)}