# src/graph_processing_utils.py
from typing import List, Dict, Any, Set
import logging # Use logging instead of print for better practice
import requests
import json
from typing import Optional
import streamlit as st

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

def retrieve_paragraphs(paragraph_ids: List[str], kb_url: str, api_key: str) -> Dict[str, str]:
    """
    Retrieves the text content for a list of paragraph IDs from Nuclia API.
    Modified to request 'extracted' data and use a revised path for text.
    """
    if not paragraph_ids: # ... (input validation same as before) ...
        logging.info("retrieve_paragraphs: Received empty list of paragraph IDs.")
        return {}
    if not kb_url or not api_key:
        logging.error("retrieve_paragraphs: Missing kb_url or api_key.")
        return {}

    unique_paragraph_ids = set(paragraph_ids)
    logging.info(f"Attempting to retrieve text for {len(unique_paragraph_ids)} unique paragraph IDs.")
    resource_content_cache: Dict[str, Optional[Dict]] = {}
    paragraph_texts: Dict[str, str] = {}
    headers = {
        "accept": "application/json",
        "X-NUCLIA-SERVICEACCOUNT": f"Bearer {api_key}",
    }
    _logged_structures = set() # To log new structure only once per resource_id

    for para_id in unique_paragraph_ids:
        resource_id = None # Initialize for each loop
        try:
            # 1. Parse Paragraph ID
            parts = para_id.split('/')
            if len(parts) != 4: logging.warning(f"Skipping invalid paragraph_id format: '{para_id}'"); continue
            resource_id, field_type_short, field_id, range_str = parts # e.g., field_type_short = 'f'
            range_parts = range_str.split('-');
            if len(range_parts) != 2: logging.warning(f"Skipping invalid range format: '{range_str}' from '{para_id}'"); continue
            start, end = int(range_parts[0]), int(range_parts[1])
            if not resource_id or not field_id or start < 0 or end < start: logging.warning(f"Skipping invalid parsed values: '{para_id}'"); continue

            # 2. Fetch Resource Content (use cache)
            resource_data: Optional[Dict] = None
            if resource_id in resource_content_cache:
                resource_data = resource_content_cache[resource_id]
                if resource_data is None: logging.warning(f"Skipping '{para_id}', resource '{resource_id}' failed previously."); continue
                logging.debug(f"Using cached content for resource '{resource_id}'.")
            else:
                logging.debug(f"Fetching content for resource '{resource_id}' with '?show=extracted'...")
                # --- > MODIFIED URL to include ?show=extracted < ---
                resource_url = f"{kb_url}/resource/{resource_id}?show=extracted"
                # You might also consider "&show=origin" if 'extracted' doesn't have the full fidelity text.
                # Or "&show=text" if that's a supported flag. Start with "extracted".
                try:
                    response = requests.get(resource_url, headers=headers, timeout=20)
                    response.raise_for_status()
                    resource_data = response.json()
                    resource_content_cache[resource_id] = resource_data
                    logging.debug(f"Fetched resource '{resource_id}' using ?show=extracted.")
                except requests.exceptions.RequestException as req_err:
                    logging.error(f"Failed fetch for resource '{resource_id}' with ?show=extracted: {req_err}", exc_info=True)
                    resource_content_cache[resource_id] = None; continue
                except json.JSONDecodeError as json_err:
                     logging.error(f"Failed JSON decode for resource '{resource_id}' (with ?show=extracted): {json_err}", exc_info=True)
                     resource_content_cache[resource_id] = None; continue

            if resource_data is None: continue

            # 3. Extract Field Text
            full_text: Optional[str] = None
            # Map short field_type from paragraph_id (like 'f') to the longer key used in the JSON (like 'files')
            field_type_map = {'f': 'files', 'l': 'links', 't': 'texts'}
            mapped_field_type_key = field_type_map.get(field_type_short, field_type_short) # Use original if no mapping

            # --- > REVISED PATH ASSUMPTION for text extraction from "extracted" data < ---
            # Assumes structure like: resource_data -> 'data' -> 'files' (mapped_field_type_key) -> field_id -> 'extracted' -> 'text' -> 'body'
            current_data_node = resource_data.get('data', {}).get(mapped_field_type_key, {}).get(field_id, {})
            if current_data_node: # Check if path to field_id exists
                full_text = current_data_node.get('extracted', {}).get('text', {}).get('text')

            if not isinstance(full_text, str) or not full_text:
                 logging.warning(f"Could not find valid text body for field '{mapped_field_type_key}/{field_id}' "
                                 f"in resource '{resource_id}' for paragraph '{para_id}' using path via 'data.{mapped_field_type_key}.{field_id}.extracted.text.text'. "
                                 f"Found type: {type(full_text)}")
                 # --- Keep the structural logging for diagnostics if the new path also fails ---
                 if resource_id not in _logged_structures:
                     logging.info(f"--- Logging structure for resource '{resource_id}' (after ?show=extracted) ---")
                     logging.info(f"  Resource top-level keys: {list(resource_data.keys())}")
                     data_section = resource_data.get('data', {})
                     if isinstance(data_section, dict):
                          logging.info(f"  Resource 'data' keys: {list(data_section.keys())}")
                          if mapped_field_type_key in data_section and isinstance(data_section[mapped_field_type_key], dict):
                              logging.info(f"    'data.{mapped_field_type_key}' keys: {list(data_section[mapped_field_type_key].keys())}")
                              if field_id in data_section[mapped_field_type_key] and isinstance(data_section[mapped_field_type_key][field_id], dict):
                                  logging.info(f"      'data.{mapped_field_type_key}.{field_id}' keys: {list(data_section[mapped_field_type_key][field_id].keys())}")
                                  if 'extracted' in data_section[mapped_field_type_key][field_id]:
                                      logging.info(f"        '...{field_id}.extracted' keys: {list(data_section[mapped_field_type_key][field_id]['extracted'].keys())}")
                                      if 'text' in data_section[mapped_field_type_key][field_id]['extracted']:
                                          logging.info(f"        '...{field_id}.extracted.text' keys: {list(data_section[mapped_field_type_key][field_id]['extracted']['text'].keys())}")


                     extracted_section = resource_data.get('extracted', {}) # Also log top-level 'extracted' just in case
                     if isinstance(extracted_section, dict): logging.info(f"  Resource 'extracted' (top-level) keys: {list(extracted_section.keys())}")

                     _logged_structures.add(resource_id)
                 continue # Skip this paragraph if text wasn't found

            # 4. Extract Substring (Paragraph)
            # ... (substring extraction logic same as before) ...
            text_len = len(full_text)
            if start >= text_len: logging.warning(f"Start offset {start} out of bounds ({text_len}) for '{para_id}'. Skipping."); continue
            if end > text_len: logging.debug(f"End offset {end} exceeds length {text_len} for '{para_id}'. Clamping."); end = text_len
            if start >= end: logging.warning(f"Start >= end ({start}>={end}) for '{para_id}'. Skipping."); continue
            paragraph_text = full_text[start:end]
            paragraph_texts[para_id] = paragraph_text
            logging.debug(f"Extracted text for '{para_id}'. Length: {len(paragraph_text)}")

        except ValueError as parse_err: logging.warning(f"Skipping '{para_id}' due to parsing error: {parse_err}", exc_info=True); continue
        except Exception as e: logging.error(f"Error processing '{para_id}' (Resource: {resource_id}): {e}", exc_info=True); continue

    logging.info(f"Retrieved text for {len(paragraph_texts)} out of {len(unique_paragraph_ids)} unique IDs.")
    return paragraph_texts