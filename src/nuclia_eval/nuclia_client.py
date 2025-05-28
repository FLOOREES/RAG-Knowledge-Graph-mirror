# PROJECT_ROOT/nuclia_eval/nuclia_client.py
import logging
from typing import Dict, Any, List, Optional

from nuclia import sdk
# NucliaSearch.ask returns AskAnswer, not a stream, based on user-provided source
from nuclia.sdk.search import AskAnswer
from nucliadb_models.search import (
    AskRequest, 
    Relations, 
    EntitySubgraph,        # For values in relations_result.entities
    DirectionalRelation,   # For items in EntitySubgraph.related_to
    KnowledgeboxFindResults, # Type of find_result
    FindResource,          # For values in find_result.resources
    FindField,             # For values in FindResource.fields
    FindParagraph          # For values in FindField.paragraphs
)

from src.utils.logger_setup import setup_logger

# Configure a module-specific logger
logger = setup_logger(__name__)

class NucliaClientWrapper:
    """
    A wrapper class for interacting with the Nuclia SDK, focusing on
    Knowledge Graph queries using the synchronous client.
    """

    def __init__(self, kb_url: str, api_key: str):
        """
        Initializes the NucliaClientWrapper and authenticates with Nuclia.

        Args:
            kb_url (str): The URL of the Nuclia Knowledge Box.
            api_key (str): The API key for the Nuclia Knowledge Box.
        """
        self.kb_url = kb_url
        self.api_key = api_key
        self._search_client: Optional[sdk.NucliaSearch] = None
        self._authenticate()

    def _authenticate(self) -> None:
        """
        Authenticates with Nuclia using the provided KB URL and API key.
        """
        try:
            logger.info(f"Attempting to authenticate with Nuclia KB: {self.kb_url}")
            auth = sdk.NucliaAuth()
            auth.kb(url=self.kb_url, token=self.api_key)
            self._search_client = sdk.NucliaSearch() # Initialize the synchronous search client
            logger.info("Successfully authenticated with Nuclia and initialized search client.")
        except Exception as e:
            logger.error(f"Nuclia authentication failed: {e}", exc_info=True)
            raise

    def query_knowledge_graph(self, question: str, generative_model_override: Optional[str] = None) -> Dict[str, Any]:
        """
        Sends a query to the Nuclia Knowledge Box using the synchronous ask method.
        Parses relations from EntitySubgraph.related_to and paragraphs from the
        nested structure within KnowledgeboxFindResults.
        """
        if not self._search_client:
            logger.error("Search client not initialized. Authentication might have failed.")
            raise ConnectionError("Nuclia search client not available. Authentication failed.")

        logger.info(f"Querying Nuclia KG with question: '{question}'")
        ask_query_request = AskRequest(
            query=question,
            features=["semantic","keyword","relations"],
            citations=True,
            generative_model=generative_model_override 
        )

        full_answer_text = ""
        retrieved_relations_list: List[Dict[str, Any]] = []
        source_citations_list: List[Dict[str, Any]] = []
        retrieved_context_paragraphs_text: List[str] = []

        try:
            logger.debug(f"Sending AskRequest to Nuclia: {ask_query_request.model_dump_json(indent=2)}")
            response_object: AskAnswer = self._search_client.ask(query=ask_query_request)
            
            logger.debug(f"Received AskAnswer object of type: {type(response_object)}")
            # Log raw object only if absolutely needed for further deep dive, can be verbose
            # if hasattr(response_object, '__dict__'): logger.debug(f"Raw AskAnswer (vars): {vars(response_object)}")
            # else: logger.debug(f"Raw AskAnswer (raw): {response_object}")

            # 1. Process Answer
            if hasattr(response_object, 'answer') and response_object.answer is not None:
                full_answer_text = response_object.answer.decode().strip()
                logger.info(f"Decoded answer (snippet): {full_answer_text[:200]}...")
            else:
                logger.warning("No 'answer' attribute or it's None in AskAnswer object.")

            # 2. Process Relations
            logger.debug("Attempting to process relations...")
            relations_data_source: Optional[Relations] = getattr(response_object, 'relations_result', getattr(response_object, 'relations', None))
            
            if relations_data_source and hasattr(relations_data_source, 'entities') and \
               isinstance(relations_data_source.entities, dict):
                logger.info("Found 'entities' dictionary in relations_data_source. Processing...")
                if not relations_data_source.entities: logger.info("'entities' dictionary is empty.")
                
                for entity_uri, entity_subgraph_obj in relations_data_source.entities.items():
                    # entity_subgraph_obj is an EntitySubgraph
                    if not isinstance(entity_subgraph_obj, EntitySubgraph):
                        logger.warning(f"Item for entity URI '{entity_uri}' is not an EntitySubgraph. Type: {type(entity_subgraph_obj)}. Skipping.")
                        continue
                    
                    logger.debug(f"Processing EntitySubgraph for source entity URI: '{entity_uri}'")
                    if hasattr(entity_subgraph_obj, 'related_to') and isinstance(entity_subgraph_obj.related_to, list):
                        if not entity_subgraph_obj.related_to:
                            logger.debug(f"  'related_to' list is empty for EntitySubgraph of '{entity_uri}'.")
                        
                        for rel_detail in entity_subgraph_obj.related_to:
                            # rel_detail is a DirectionalRelation object
                            if not isinstance(rel_detail, DirectionalRelation):
                                logger.warning(f"  Item in 'related_to' list is not a DirectionalRelation. Type: {type(rel_detail)}. Skipping.")
                                continue

                            relation_entry = {"source_entity_uri": entity_uri}
                            if hasattr(rel_detail, 'entity'): relation_entry["target_entity_uri"] = rel_detail.entity
                            if hasattr(rel_detail, 'relation_label'): relation_entry["relation_label"] = rel_detail.relation_label
                            if hasattr(rel_detail, 'direction'): relation_entry["direction"] = str(rel_detail.direction) # Convert enum to str
                            if hasattr(rel_detail, 'resource_id'): relation_entry["resource_id"] = rel_detail.resource_id
                            if hasattr(rel_detail, 'entity_type'): relation_entry["target_entity_type"] = str(rel_detail.entity_type) # Convert enum to str
                            # Example of accessing metadata if it exists and has a specific field
                            if hasattr(rel_detail, 'metadata') and rel_detail.metadata and hasattr(rel_detail.metadata, 'paragraph_id'):
                                relation_entry["metadata_paragraph_id"] = rel_detail.metadata.paragraph_id
                            
                            if len(relation_entry) > 1 and "target_entity_uri" in relation_entry:
                                retrieved_relations_list.append(relation_entry)
                                logger.debug(f"    Added relation: {relation_entry}")
                    else:
                        logger.debug(f"  EntitySubgraph for '{entity_uri}' does not have a 'related_to' list attribute, or it's not a list.")
            else:
                logger.info("No 'relations_result' or 'relations' attribute found, or 'entities' not found/not a dict.")
            logger.info(f"Total relations extracted: {len(retrieved_relations_list)}")
            if retrieved_relations_list: logger.debug(f"Sample extracted relations: {retrieved_relations_list[:min(2, len(retrieved_relations_list))]}")

            # 3. Process Citations
            logger.debug("Attempting to process citations...")
            if hasattr(response_object, 'citations') and response_object.citations is not None:
                raw_citations = response_object.citations
                logger.debug(f"Raw citations data: {raw_citations} (type: {type(raw_citations)})")
                if isinstance(raw_citations, list): source_citations_list.extend(raw_citations)
                elif isinstance(raw_citations, dict): source_citations_list.append(raw_citations)
                else:
                    logger.warning(f"Citations data is of unhandled type: {type(raw_citations)}. Appending raw.")
                    source_citations_list.append(raw_citations) 
                logger.info(f"Citation items processed: {len(source_citations_list)}")
            else:
                logger.info("No 'citations' attribute or it's None.")

            # 4. Process Context Paragraphs
            logger.debug("Attempting to process context paragraphs...")
            find_result_data: Optional[KnowledgeboxFindResults] = getattr(response_object, 'find_result', None)
            
            if find_result_data and hasattr(find_result_data, 'resources') and \
               isinstance(find_result_data.resources, dict):
                logger.info("Found 'resources' dictionary in find_result. Processing...")
                if not find_result_data.resources: logger.info("'resources' dictionary is empty.")

                for resource_id, resource_obj in find_result_data.resources.items():
                    # resource_obj is a FindResource
                    if not isinstance(resource_obj, FindResource):
                        logger.warning(f"Item for resource_id '{resource_id}' is not a FindResource. Type: {type(resource_obj)}. Skipping.")
                        continue
                    logger.debug(f"  Processing FindResource: {resource_id}")
                    
                    if hasattr(resource_obj, 'fields') and isinstance(resource_obj.fields, dict):
                        if not resource_obj.fields: logger.debug(f"    'fields' dictionary is empty for resource {resource_id}.")
                        for field_id, field_obj in resource_obj.fields.items():
                            # field_obj is a FindField
                            if not isinstance(field_obj, FindField):
                                logger.warning(f"Item for field_id '{field_id}' is not a FindField. Type: {type(field_obj)}. Skipping.")
                                continue
                            logger.debug(f"    Processing FindField: {field_id}")

                            if hasattr(field_obj, 'paragraphs') and isinstance(field_obj.paragraphs, dict):
                                if not field_obj.paragraphs: logger.debug(f"      'paragraphs' dictionary is empty for field {field_id}.")
                                for para_id, para_obj in field_obj.paragraphs.items():
                                    # para_obj is a FindParagraph
                                    if not isinstance(para_obj, FindParagraph):
                                        logger.warning(f"Item for para_id '{para_id}' is not a FindParagraph. Type: {type(para_obj)}. Skipping.")
                                        continue
                                    
                                    if hasattr(para_obj, 'text') and isinstance(para_obj.text, str):
                                        retrieved_context_paragraphs_text.append(para_obj.text)
                                        logger.debug(f"        Extracted paragraph text (id: {para_id}, snippet): {para_obj.text[:100]}...")
                                    else:
                                        logger.warning(f"        FindParagraph (id: {para_id}) has no 'text' attribute or it's not a string.")
                            else:
                                logger.debug(f"      FindField '{field_id}' has no 'paragraphs' dict or it's not a dict.")
                    else:
                        logger.debug(f"    FindResource '{resource_id}' has no 'fields' dict or it's not a dict.")
            else:
                logger.info("No 'find_result' or 'find_result.resources' not found/not a dict (for context paragraphs).")
            logger.info(f"Total context paragraphs extracted: {len(retrieved_context_paragraphs_text)}")

            # 5. Check for errors reported by Nuclia
            if hasattr(response_object, 'error_details') and response_object.error_details:
                logger.error(f"Nuclia AskAnswer reported an error: {response_object.error_details}")
            
            logger.info(f"Successfully processed Nuclia AskAnswer for question: '{question}'")

        except Exception as e:
            logger.error(f"CRITICAL ERROR during Nuclia query or AskAnswer processing: {e}", exc_info=True)
            raise

        return {
            "question": question,
            "answer": full_answer_text,
            "relations": retrieved_relations_list,
            "citations": source_citations_list,
            "retrieved_context_paragraphs": retrieved_context_paragraphs_text
        }
