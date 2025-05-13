# src/llm_utils.py
import logging
from typing import List, Optional
import openai # Import the OpenAI library

# Configure logging for this module
logger = logging.getLogger(__name__)
# Ensure basicConfig is called in app.py or this won't output unless app also configures root logger
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')


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
