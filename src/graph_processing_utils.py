# src/graph_processing_utils.py
from typing import List, Dict, Any, Set
import logging # Use logging instead of print for better practice

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