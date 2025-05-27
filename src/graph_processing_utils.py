# src/graph_processing_utils.py
from typing import List, Dict, Any, Set
import logging # Use logging instead of print for better practice
import requests
import json
from typing import Optional
import streamlit as st
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def find_relations_for_entities(
		top_entities_info: List[tuple],
		full_graph_data: List[Dict[str, Any]],
		top_n_entities: int = 5
	) -> List[Dict[str, Any]]:
	"""
	Finds all relations from the full graph that involve the top N entities.

	Args:
		top_entities_info: A list of tuples, where each tuple contains
						   (node_index, node_info_dict, similarity_score),
						   as returned by get_relevant_nodes. node_info_dict
						   should have a 'value' key for the entity text.
		full_graph_data: The complete knowledge graph, represented as a list
						 of relation dictionaries (e.g., loaded from JSON).
		top_n_entities: The number of top entities to consider from top_entities_info.

	Returns:
		A list of relation dictionaries from full_graph_data that involve
		at least one of the top N entities as 'from' or 'to'.
		Includes basic deduplication based on (from_val, label, to_val).
		Returns an empty list if no relevant relations are found or inputs are invalid.
	"""
	if not top_entities_info or not full_graph_data:
		logging.warning("find_relations_for_entities: Invalid input (empty top_entities or graph_data).")
		return []

	# Extract the string values of the top N entities more robustly
	top_entity_values: Set[str] = set()
	try:
		count = 0
		for entity_info_tuple in top_entities_info:
			if count >= top_n_entities:
				break
			# Check tuple structure and dictionary type
			if len(entity_info_tuple) >= 2 and isinstance(entity_info_tuple[1], dict):
				node_info = entity_info_tuple[1]
				entity_value = node_info.get("value")
				# Ensure value is a non-empty string
				if entity_value and isinstance(entity_value, str):
					top_entity_values.add(entity_value)
					count += 1
			else:
				logging.warning(f"Skipping invalid item in top_entities_info: {entity_info_tuple}")

	except Exception as e:
		logging.error(f"Error processing top_entities_info: {e}", exc_info=True)
		return [] # Return empty if processing fails

	if not top_entity_values:
		logging.warning("Could not extract valid entity values from top_entities_info.")
		return []

	logging.info(f"Searching for relations involving top {len(top_entity_values)} entities: {top_entity_values}")

	relevant_relations: List[Dict[str, Any]] = []
	# Use tuple(from_val, label, to_val) for deduplication
	found_relation_signatures: Set[tuple] = set()

	for i, relation_dict in enumerate(full_graph_data):
		try:
			# Ensure basic structure exists before accessing deeply nested keys
			if not isinstance(relation_dict, dict) or "from" not in relation_dict or "to" not in relation_dict:
				logging.warning(f"Skipping relation at index {i} due to missing 'from' or 'to' key. Data: {relation_dict}")
				continue

			from_node = relation_dict.get("from", {})
			to_node = relation_dict.get("to", {})

			if not isinstance(from_node, dict) or not isinstance(to_node, dict):
				logging.warning(f"Skipping relation at index {i} due to invalid 'from'/'to' node type. Data: {relation_dict}")
				continue

			from_value = from_node.get("value")
			to_value = to_node.get("value")
			label = relation_dict.get("label") # Label can be missing or None

			# Check if either 'from' or 'to' entity value is in our top set
			# Ensure values are strings before checking containment
			is_relevant = False
			if isinstance(from_value, str) and from_value in top_entity_values:
				is_relevant = True
			if isinstance(to_value, str) and to_value in top_entity_values:
				is_relevant = True

			if is_relevant:
				# Create a signature for deduplication
				relation_signature = (str(from_value), str(label), str(to_value))

				if relation_signature not in found_relation_signatures:
					relevant_relations.append(relation_dict)
					found_relation_signatures.add(relation_signature)

		except Exception as e:
			# Log unexpected errors during processing of a single relation dict
			logging.error(f"Error processing relation dict at index {i} due to error: {e}. Dict: {relation_dict}", exc_info=True)
			continue # Skip this problematic relation dict

	logging.info(f"Found {len(relevant_relations)} relevant relations.")
	return relevant_relations

def retrieve_paragraphs(
    paragraph_ids: List[str], 
    kb_url: str, 
    api_key: str,
    local_cache_filepath: Optional[Path] = None 
) -> Dict[str, str]:
    """
    Retrieves text for paragraph IDs. 
    If local_cache_filepath is provided and exists, it's loaded and checked first.
    Paragraphs not found in the cache are then fetched from the Nuclia API.
    """
    if not paragraph_ids:
        logging.info("retrieve_paragraphs: Received empty list of paragraph IDs.")
        return {}

    paragraph_texts: Dict[str, str] = {}
    ids_to_fetch_from_api: Set[str] = set()
    unique_paragraph_ids_input = set(paragraph_ids)
    
    logging.info(f"Attempting to retrieve text for {len(unique_paragraph_ids_input)} unique paragraph IDs.")

    # 1. Load and check local persistent cache if filepath is provided
    file_cache_content: Dict[str, str] = {}
    if local_cache_filepath:
        if local_cache_filepath.exists():
            try:
                with open(local_cache_filepath, 'r', encoding='utf-8') as f:
                    file_cache_content = json.load(f)
                if not isinstance(file_cache_content, dict):
                    logging.error(f"Local paragraph cache at {local_cache_filepath} is not a valid JSON dictionary. Will fetch all from API.")
                    file_cache_content = {} # Reset if invalid format
                else:
                    logging.info(f"Successfully loaded {len(file_cache_content)} paragraphs from local cache: {local_cache_filepath}")
            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON from local paragraph cache: {local_cache_filepath}. Will fetch all from API.", exc_info=True)
                file_cache_content = {} # Reset on error
            except Exception as e:
                logging.error(f"Error reading local paragraph cache {local_cache_filepath}: {e}. Will fetch all from API.", exc_info=True)
                file_cache_content = {} # Reset on error
        else:
            logging.info(f"Local paragraph cache file not found at: {local_cache_filepath}. Will fetch all from API if needed.")
            # No need to set file_cache_content to {} here, it's already the default

    # 2. Populate from cache and identify IDs still needing API fetch
    for para_id in unique_paragraph_ids_input:
        if para_id in file_cache_content: # Check the loaded content for the current call
            paragraph_texts[para_id] = file_cache_content[para_id]
            logging.debug(f"Found paragraph '{para_id}' in local file cache.")
        else:
            ids_to_fetch_from_api.add(para_id)
    
    cached_found_count = len(paragraph_texts)
    if local_cache_filepath: # Only log cache stats if a cache path was attempted
        logging.info(f"Found {cached_found_count} paragraphs in local cache. Need to fetch {len(ids_to_fetch_from_api)} from API.")
    else:
        logging.info(f"No local cache filepath provided. Need to fetch {len(ids_to_fetch_from_api)} from API.")


    if not ids_to_fetch_from_api: # All found in cache or no IDs were requested initially after filtering
        if local_cache_filepath and cached_found_count > 0 : # Only log this if cache was used and items were found
             logging.info("All requested paragraphs found in local cache. No API calls needed for this batch.")
        return paragraph_texts

    # 3. Fetch remaining IDs from Nuclia API (if any)
    if not kb_url or not api_key:
        logging.error("retrieve_paragraphs: Missing kb_url or api_key for API fetch. Cannot retrieve remaining paragraphs.")
        for api_para_id in ids_to_fetch_from_api:
            # Only add error for those not already found in cache (which is all of them if cache failed)
            if api_para_id not in paragraph_texts: 
                 paragraph_texts[api_para_id] = "Error: Nuclia credentials missing for API fetch."
        return paragraph_texts

    # In-memory cache for API responses for *resources* during *this specific function call*
    resource_api_call_cache: Dict[str, Optional[Dict]] = {} 
    headers = {"accept": "application/json", "X-NUCLIA-SERVICEACCOUNT": f"Bearer {api_key}"}
    # _logged_api_structures = set() # If you need the detailed API structure logging from before

    logging.info(f"Proceeding to fetch {len(ids_to_fetch_from_api)} paragraphs from Nuclia API.")
    for para_id in ids_to_fetch_from_api:
        resource_id = None 
        try:
            # --- Start of your existing, correct API fetching & parsing logic for a single para_id ---
            parts = para_id.split('/')
            if len(parts) != 4: logging.warning(f"API Fetch: Skipping invalid paragraph_id format: '{para_id}'"); continue
            resource_id, field_type_short, field_id, range_str = parts
            range_parts = range_str.split('-');
            if len(range_parts) != 2: logging.warning(f"API Fetch: Skipping invalid range format: '{range_str}' from '{para_id}'"); continue
            start, end = int(range_parts[0]), int(range_parts[1])
            if not resource_id or not field_id or start < 0 or end < start: logging.warning(f"API Fetch: Skipping invalid parsed values: '{para_id}'"); continue

            resource_data_from_api: Optional[Dict] = None
            if resource_id in resource_api_call_cache:
                resource_data_from_api = resource_api_call_cache[resource_id]
                if resource_data_from_api is None: logging.debug(f"API Fetch: Skipping '{para_id}', resource '{resource_id}' failed API fetch previously in this call."); continue
                logging.debug(f"API Fetch: Using API-call-cached content for resource '{resource_id}'.")
            else:
                logging.debug(f"API Fetch: Fetching content for resource '{resource_id}' with '?show=extracted'...")
                resource_url = f"{kb_url}/resource/{resource_id}?show=extracted"
                try:
                    response = requests.get(resource_url, headers=headers, timeout=20)
                    response.raise_for_status()
                    resource_data_from_api = response.json()
                    resource_api_call_cache[resource_id] = resource_data_from_api
                except requests.exceptions.RequestException as req_err:
                    logging.warning(f"API Fetch: Failed for resource '{resource_id}': {req_err}") 
                    resource_api_call_cache[resource_id] = None; continue
                except json.JSONDecodeError as json_err:
                    logging.warning(f"API Fetch: Failed JSON decode for resource '{resource_id}': {json_err}")
                    resource_api_call_cache[resource_id] = None; continue
            
            if resource_data_from_api is None: continue

            full_text: Optional[str] = None
            field_type_map = {'f': 'files', 'l': 'links', 't': 'texts'}
            mapped_field_type_key = field_type_map.get(field_type_short, field_type_short)
            
            text_container = resource_data_from_api.get('extracted', {}).get(mapped_field_type_key, {}).get(field_id, {}).get('text', {})
            full_text = text_container.get('text') if isinstance(text_container, dict) else None

            if not isinstance(full_text, str) or not full_text:
                logging.warning(f"API Fetch: Could not find valid text for field '{mapped_field_type_key}/{field_id}' in resource '{resource_id}' for para_id '{para_id}'.")
                continue

            text_len = len(full_text)
            if start >= text_len: logging.warning(f"API Fetch: Start offset {start} out of bounds ({text_len}) for '{para_id}'. Skipping."); continue
            if end > text_len: logging.debug(f"API Fetch: End offset {end} exceeds length {text_len} for '{para_id}'. Clamping."); end = text_len
            if start >= end: logging.warning(f"API Fetch: Start >= end ({start}>={end}) for '{para_id}'. Skipping."); continue
            
            paragraph_texts[para_id] = full_text[start:end]
            logging.debug(f"API Fetch: Extracted text for '{para_id}'.")
            # --- End of your existing API fetch logic ---
        except ValueError as parse_err: 
            logging.warning(f"API Fetch: Skipping '{para_id}' due to parsing error: {parse_err}")
            paragraph_texts[para_id] = f"Error: Parsing error - {parse_err}"
            continue
        except Exception as e: 
            logging.error(f"API Fetch: Error processing '{para_id}' (Resource: {resource_id}): {e}", exc_info=True)
            paragraph_texts[para_id] = f"Error: API processing - {e}"
            continue
    
    total_retrieved_count = len(paragraph_texts)
    # Count how many were actually successfully fetched and added from API
    api_actually_added_count = 0
    for id_in_api_list in ids_to_fetch_from_api:
        if id_in_api_list in paragraph_texts and not paragraph_texts[id_in_api_list].startswith("Error:"):
            api_actually_added_count +=1
            
    logging.info(f"retrieve_paragraphs: Total retrieved: {total_retrieved_count} / {len(unique_paragraph_ids_input)} requested. "
                 f"From local cache: {cached_found_count}. Newly fetched and added from API: {api_actually_added_count}.")
    return paragraph_texts