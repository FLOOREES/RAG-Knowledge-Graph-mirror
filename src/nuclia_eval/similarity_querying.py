# src/path_rgcn_retriever.py
import json
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
from torch_geometric.data import HeteroData
# RGCNConv might not be needed if we remove per-query training, but keep for potential future pre-trained model
from torch_geometric.nn import RGCNConv
from typing import List, Dict, Any, Optional, Tuple

import spacy
from rapidfuzz import process # For fuzzy matching

# --- Assuming these are in your project's PYTHONPATH ---
from model2vec import StaticModel
# from src.utils import extract_k_hop_subgraph_bidirectional
from src.utils_max import extract_k_hop_subgraph_bidirectional
from src.llm_utils import generate_answer_with_openai
from src.graph_processing_utils import retrieve_paragraphs
import os 
from dotenv import load_dotenv


import logging
logger = logging.getLogger(__name__)

# --- Configuration & Constants ---
# Load environment variables from a .env file in the project root.
# This file should contain API keys and other sensitive configurations.
if load_dotenv(override=True):
    logging.info(".env file found and loaded successfully.")
else:
    logging.warning(".env file not found. Relying on pre-set environment variables if available.")

# Nuclia configurations (used for API fallback if local graph fails or for paragraph retrieval)
NUCLIA_KB_URL = os.getenv('NUCLIA_KB_URL')
if not NUCLIA_KB_URL:
    logging.warning("NUCLIA_KB_URL not found in environment. Defaulting to 'https://api.nuclia.cloud'.")
    NUCLIA_KB_URL = "https://europe-1.nuclia.cloud/api/v1/kb/3aa88834-0641-4367-8994-985a86f01e55"  # Default URL if not set
print(f"NUCLIA_KB_URL loaded from environment: '{NUCLIA_KB_URL}'")
NUCLIA_API_KEY = os.getenv('NUCLIA_API_KEY') # Used by retrieve_paragraphs and extract_triplets
logging.info(f"NUCLIA_KB_URL loaded from environment: '{NUCLIA_KB_URL}'")
if not NUCLIA_KB_URL: logging.warning("NUCLIA_KB_URL not found. API fallback/paragraph retrieval might fail.")
if not NUCLIA_API_KEY: logging.warning("NUCLIA_API_KEY not found. API fallback/paragraph retrieval might fail.")

# OpenAI API Key for LLM-based answer generation
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    logging.warning("OPENAI_API_KEY not found in environment. LLM answer generation will be disabled.")
    # User will be notified in the UI if they try to generate an answer.


RELATIONS_PATH = "data/legal_graph.json" # Path to your relations file

class PathRGCNRetriever:
    def __init__(self, embedding_model_name: str = "minishlab/potion-base-8M"):
        logger.info(f"Initializing PathRGCNRetriever with graph and model: {embedding_model_name}")
        self.embedding_model = StaticModel.from_pretrained(embedding_model_name)
        self.nlp = spacy.load("en_core_web_trf") # Load NER model

        self.entity2id: Dict[str, int] = {}
        self.id2entity: Dict[int, str] = {}
        self.relation2id: Dict[str, int] = {} # Stores original relation labels to their IDs (0 to N-1)
        self.id2relation: Dict[int, str] = {} # Inverse of relation2id
        self.num_original_relations: int = 0
        self.relation_texts_full: List[str] = [] # Stores text for original (0 to N-1) and inverse (N to 2N-1) relations
        self.entity_embeddings_full: Optional[torch.Tensor] = None
        self.rel_embs_full: Optional[torch.Tensor] = None
        self.embedding_dim: int = 0
        self.full_graph_data: Optional[HeteroData] = None
        self.relation_metadata: Dict[int, Dict[str, Any]] = {}
        self.para_id_dict: Dict[Tuple[str,str,str], str] = {}

    def ingest_data(self, relations: List[Dict[str, Any]]):
        logger.info(f"Ingesting {len(relations)} relations into the knowledge graph.")
        self.entity2id = {}
        self.relation2id = {} # Original relation labels to ID (0 to N-1)
        edges = []
        edge_types_original = [] # Store original edge type IDs (0 to N-1)

        for idx, rel in enumerate(relations):
            src_val, dst_val, label_val = rel['from']['value'], rel['to']['value'], rel['label']
            self.para_id_dict[(src_val, dst_val, label_val)] = rel['metadata']['paragraph_id'] #if 'metadata' in rel and 'para_id' in rel['metadata'] else None
            self.relation_metadata[idx] = rel

            for node in (src_val, dst_val):
                if node not in self.entity2id:
                    self.entity2id[node] = len(self.entity2id)
            if label_val not in self.relation2id:
                self.relation2id[label_val] = len(self.relation2id)

            edges.append((self.entity2id[src_val], self.entity2id[dst_val]))
            edge_types_original.append(self.relation2id[label_val])

        self.id2entity = {i: e for e, i in self.entity2id.items()}
        self.num_original_relations = len(self.relation2id)
        self.id2relation = {i: r for r, i in self.relation2id.items()}

        # Prepare texts for original and inverse relations
        self.relation_texts_full = [""] * (2 * self.num_original_relations)
        for label, r_id in self.relation2id.items():
            self.relation_texts_full[r_id] = label
            self.relation_texts_full[r_id + self.num_original_relations] = f"inverse of {label}"

        logger.info(f"KG loaded: {len(self.entity2id)} entities, {self.num_original_relations} original relation types (total {2*self.num_original_relations} with inverses).")

        # Entity Embeddings
        entity_texts = [self.id2entity[i] for i in range(len(self.id2entity))]
        if entity_texts: # Ensure there are entities before encoding
            # Potentially batch encode if self.embedding_model.encode supports list input
            # For now, assuming individual encoding as in original
            encoded_entities = [self.embedding_model.encode(text) for text in entity_texts]
            if encoded_entities and isinstance(encoded_entities[0], np.ndarray) and encoded_entities[0].ndim == 2:
                self.entity_embeddings_full = torch.tensor(np.concatenate(encoded_entities, axis=0), dtype=torch.float)
            else:
                self.entity_embeddings_full = torch.tensor(np.array(encoded_entities), dtype=torch.float)
            self.embedding_dim = self.entity_embeddings_full.size(1)
        else:
            logger.warning("No entities found to embed.")
            self.entity_embeddings_full = torch.empty(0,0, dtype=torch.float) # Handle empty graph case
            self.embedding_dim = 0 # Or a default, but likely indicates an issue

        # Relation Embeddings (Original + Inverse)
        if self.relation_texts_full and self.embedding_dim > 0 :
            encoded_relations = []
            for text_rel in self.relation_texts_full:
                emb = self.embedding_model.encode(text_rel)
                if isinstance(emb, np.ndarray) and emb.ndim == 2:
                    encoded_relations.append(torch.tensor(emb[0], dtype=torch.float))
                else: # Fallback for 1D arrays
                    encoded_relations.append(torch.tensor(emb, dtype=torch.float))
            self.rel_embs_full = torch.stack(encoded_relations) # [2*R_original, D]
        elif self.embedding_dim == 0 and self.relation_texts_full:
            logger.warning("Cannot generate relation embeddings as entity embedding dimension is 0 (likely no entities).")
            self.rel_embs_full = torch.empty(0,0, dtype=torch.float)
        else:
            self.rel_embs_full = torch.empty(0,0, dtype=torch.float)


        self.full_graph_data = HeteroData()
        if self.entity_embeddings_full.numel() > 0:
            self.full_graph_data['entity'].x = self.entity_embeddings_full
        else: # Handle case with no entities
            self.full_graph_data['entity'].x = torch.empty((0, self.embedding_dim if self.embedding_dim > 0 else 1), dtype=torch.float)


        self.full_graph_data['entity', 'to', 'entity'].edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        # Edge types stored in the graph are the *original* relation IDs (0 to N-1)
        # The extract_k_hop_subgraph_bidirectional utility is responsible for creating appropriate
        # edge types for the subgraph, including original and inverse.
        self.full_graph_data['entity', 'to', 'entity'].edge_type = torch.tensor(edge_types_original, dtype=torch.long)
        logger.info("Full graph HeteroData object created.")

    # --- RGCN Model (Commented out as we are not training per query for RAG adaptation) ---
    # If you have a PRE-TRAINED RGCN, you would load it and use it for inference on the subgraph.
    # class _RGCN(torch.nn.Module):
    #     def __init__(self, in_c, hid_c, out_c, num_rels_in_subgraph):
    #         super().__init__()
    #         self.num_rels_actual = max(1, num_rels_in_subgraph) # Should be 2*num_original_relations for a pre-trained model
    #         self.conv1 = RGCNConv(in_c, hid_c, self.num_rels_actual)
    #         self.conv2 = RGCNConv(hid_c, out_c, self.num_rels_actual)
    #         self.out_channels_final = out_c
    #     def forward(self, x, edge_index, edge_type):
    #         # ... (forward pass logic, similar to original) ...
    #         # This would be used as:
    #         #   node_embs_from_gnn = self.pretrained_rgcn_model(x_sub, edge_index_sub, edge_type_sub)
    #         pass


    def _bfs_path_rels(self, adj_list_subgraph: List[List[Tuple[int, int]]],
                       seed_local_idx: int, target_local_idx: int) -> Tuple[List[int], List[int]]:
        queue_bfs = deque([(seed_local_idx, [])]) # (node, path_to_node_relations)
        # To store path_nodes, we need parent map: {node: (parent, rel_type_from_parent)}
        parent_map: Dict[int, Optional[Tuple[int, int]]] = {seed_local_idx: None}
        
        path_found_for_target = False
        
        # Temporary queue for BFS node exploration, separate from path reconstruction needs
        explore_q = deque([seed_local_idx])
        visited_for_bfs = {seed_local_idx}

        while explore_q:
            u = explore_q.popleft()
            if u == target_local_idx:
                path_found_for_target = True
                break # Found shortest path to target in terms of hops

            if u < len(adj_list_subgraph): # Check bounds
                for v_local, r_type_in_subgraph in adj_list_subgraph[u]:
                    if v_local not in visited_for_bfs:
                        visited_for_bfs.add(v_local)
                        parent_map[v_local] = (u, r_type_in_subgraph)
                        explore_q.append(v_local)
            else:
                logger.warning(f"BFS: Node index {u} out of bounds for adj_list of length {len(adj_list_subgraph)}")


        final_path_rel_types: List[int] = []
        final_path_nodes_local: List[int] = [] # Path from seed to target, target is last

        if not path_found_for_target or target_local_idx not in parent_map:
            return [], []

        curr = target_local_idx
        # Reconstruct path from target back to seed
        while curr != seed_local_idx and parent_map.get(curr) is not None:
            p_local, r_type = parent_map[curr] # type: ignore
            final_path_rel_types.append(r_type)
            final_path_nodes_local.append(curr) # Add current node to path
            curr = p_local
        
        # Add the seed node at the beginning if path exists and is not just the seed itself
        if final_path_nodes_local or target_local_idx == seed_local_idx :
             if target_local_idx != seed_local_idx : # if path exists beyond seed
                 final_path_nodes_local.append(seed_local_idx)
             # else: if target is seed, path_nodes_local is empty, path_rels is empty. Handled later.
        
        # Reverse to get path from seed to target
        return list(reversed(final_path_rel_types)), list(reversed(final_path_nodes_local))


    def extract_entities(self, query: str) -> List[str]:
        doc = self.nlp(query)
        entities = []
        if not self.entity2id: # No entities loaded
            logger.warning("Cannot extract entities, entity2id is empty. Call ingest_data first.")
            return []
        node_names = list(self.entity2id.keys()) # rapidfuzz needs a list
        for ent in doc.ents:
            # Fuzzy match the entity text to the node names
            best_match, score, _ = process.extractOne(ent.text, node_names)
            if score > 80 and best_match not in entities: # Adjust threshold as needed
                entities.append(best_match)
        return entities

    def query_knowledge_graph(self, query: str, 
                                k_hops: int = 7, top_n_results: int = 5,
                                alpha_target_node: float = 0.5, alpha_path_relations: float = 0.5, 
                               ) -> List[Dict[str, Any]]:
        logger.info(f"Running single query evaluation for: '{query}' with k_hops={k_hops}, top_n_results={top_n_results}")
        logger.info("Starting to retrieve paths from the knowledge graph...")
        try:
            with open(RELATIONS_PATH, "r", encoding="utf-8") as f:
                loaded_relations = json.load(f)
            logger.info(f"Successfully read {len(loaded_relations)} relations from {RELATIONS_PATH}.")
        except Exception as e:
            logger.error(f"Failed to read relations from {RELATIONS_PATH}: {e}", exc_info=True)
        self.ingest_data(loaded_relations) # Ensure data is ingested before running queries

        seed_entity_value = self.extract_entities(query)
        if not seed_entity_value:
            logger.error("No entities extracted from the query. Cannot retrieve paths.")
            return {
            "question": query,
            "answer": "",
            "relations": "",
            "retrieved_context_paragraphs": {},
            "citations": []
        }
        seed_entity_value = seed_entity_value[0]  # Use the first extracted entity
        logger.info(f"Retrieving paths for query: '{query[:50]}...', seed: '{seed_entity_value}', k={k_hops}")

        if not self.full_graph_data or self.entity_embeddings_full is None or self.rel_embs_full is None:
            logger.error("Graph data or embeddings not ingested. Call ingest_data() first.")
            return {
            "question": query,
            "answer": "",
            "relations": "",
            "retrieved_context_paragraphs": {},
            "citations": []
        }
        if self.entity_embeddings_full.numel() == 0:
            logger.warning("No entity embeddings available. Cannot retrieve paths.")
            return {
            "question": query,
            "answer": "",
            "relations": "",
            "retrieved_context_paragraphs": {},
            "citations": []
        }

        q_emb_np = self.embedding_model.encode(query)
        if q_emb_np.ndim == 2 and q_emb_np.shape[0] == 1: q_emb_np = q_emb_np[0]
        q_emb = torch.tensor(q_emb_np, dtype=torch.float).unsqueeze(0) # [1, D]
        q_emb_norm = F.normalize(q_emb, dim=1)

        if seed_entity_value not in self.entity2id:
            logger.error(f"Seed entity '{seed_entity_value}' not found in knowledge graph.")
            return []
        seed_id_global = self.entity2id[seed_entity_value]

        sub_data, mapping_subset_orig_to_local, subset_original_indices = extract_k_hop_subgraph_bidirectional(
            seed_id=seed_id_global,
            K=k_hops,
            data=self.full_graph_data,
            num_rels_total=self.num_original_relations # This is key for inverse relation indexing by the utility
        )

        if sub_data['entity'].x.numel() == 0 or subset_original_indices.numel() == 0:
             logger.warning(f"Bidirectional subgraph for '{seed_entity_value}' is empty or has no nodes with original indices.")
             return []

        seed_id_local_candidates = (subset_original_indices == seed_id_global).nonzero(as_tuple=True)[0]
        if seed_id_local_candidates.numel() == 0:
            logger.error(f"Seed entity '{seed_entity_value}' (ID {seed_id_global}) not found in its subgraph.")
            return []
        seed_id_local = seed_id_local_candidates[0].item()

        logger.info(f"Bidirectional subgraph: {sub_data['entity'].x.size(0)} nodes, "
                    f"{sub_data['entity','to','entity'].edge_index.size(1)} edges. "
                    f"Seed '{seed_entity_value}' (global {seed_id_global}) is local ID {seed_id_local}.")

        # --- Use initial embeddings from the subgraph (NO PER-QUERY GNN TRAINING) ---
        # If you had a pre-trained GNN, you'd use it here:
        # node_embs_from_gnn = self.pretrained_gnn_model(sub_data['entity'].x,
        #                                              sub_data['entity','to','entity'].edge_index,
        #                                              sub_data['entity','to','entity'].edge_type)
        node_embs_in_subgraph = sub_data['entity'].x
        edge_index_sub = sub_data['entity','to','entity'].edge_index
        edge_type_sub = sub_data['entity','to','entity'].edge_type # These types include original + inverse

        if node_embs_in_subgraph.numel() == 0:
            logger.warning("No node features in subgraph. Cannot proceed.")
            return []
        
        node_embs_norm = F.normalize(node_embs_in_subgraph, dim=1)
        N_sub = node_embs_norm.size(0)

        # Build adjacency list for BFS on the subgraph
        adj_list_sub: List[List[Tuple[int, int]]] = [[] for _ in range(N_sub)]
        for i in range(edge_index_sub.size(1)):
            u, v = edge_index_sub[0, i].item(), edge_index_sub[1, i].item()
            r_type = edge_type_sub[i].item() # This is already 0 to 2N-1
            if 0 <= u < N_sub and 0 <= v < N_sub : # Check bounds before appending
                adj_list_sub[u].append((v, r_type))
            else:
                logger.warning(f"Edge ({u},{v}) with type {r_type} has out-of-bounds node index for subgraph size {N_sub}. Skipping this edge for BFS.")


        scores_and_paths_data = []
        for target_node_local_idx in range(N_sub):
            # Path from seed_id_local to target_node_local_idx in the subgraph
            # _bfs_path_rels returns: path_rel_types (list of rel_ids), path_nodes_local (list of node_ids in subgraph, from seed to target)
            path_rel_types_on_path, path_nodes_local_indices_on_path = self._bfs_path_rels(
                adj_list_sub, seed_id_local, target_node_local_idx
            )
            
            # If target is seed, path_nodes_local will be empty from _bfs_path_rels's old logic, or just [seed_id_local]
            # We want to score the seed node itself if it's the target, path_rels is empty.
            if target_node_local_idx == seed_id_local and not path_nodes_local_indices_on_path:
                 path_nodes_local_indices_on_path = [seed_id_local] # Path to self is just self

            if not path_nodes_local_indices_on_path and target_node_local_idx != seed_id_local: # No path found
                continue

            h_target_node_sub = node_embs_norm[target_node_local_idx]
            h_rel_path_avg = torch.zeros(self.embedding_dim, device=h_target_node_sub.device)

            if path_rel_types_on_path:
                # Ensure rel_embs_full is not None and has embeddings
                if self.rel_embs_full is not None and self.rel_embs_full.numel() > 0:
                    # These r_types are already 0 to 2N-1 from subgraph extraction
                    valid_rel_indices_on_path = [r_type for r_type in path_rel_types_on_path if 0 <= r_type < self.rel_embs_full.size(0)]
                    if valid_rel_indices_on_path:
                        rel_vecs_on_path = self.rel_embs_full[valid_rel_indices_on_path]
                        h_rel_path_avg = F.normalize(rel_vecs_on_path.mean(dim=0), dim=0)
                else:
                    logger.warning("self.rel_embs_full is not available for path embedding calculation.")

            # Combine target node embedding and path embedding
            if h_rel_path_avg.numel() == 0 or torch.all(h_rel_path_avg == 0): # No path relations or zero vector
                h_combined = F.normalize(h_target_node_sub, dim=0) # Score based on target node only
            elif h_target_node_sub.numel() == 0: # Should not happen if node_embs_norm is populated
                h_combined = F.normalize(h_rel_path_avg, dim=0)
            else:
                h_combined = F.normalize(alpha_target_node * h_target_node_sub + alpha_path_relations * h_rel_path_avg, dim=0)

            similarity_score = F.cosine_similarity(q_emb_norm.squeeze(0), h_combined, dim=0).item()

            # --- Format path for structured output ---
            # Convert local subgraph node indices in path to global original indices, then to names
            path_nodes_global_ids = [subset_original_indices[node_local_idx].item() for node_local_idx in path_nodes_local_indices_on_path]
            path_nodes_display_names = [self.id2entity[global_id] for global_id in path_nodes_global_ids]
            
            path_relation_labels_display = []
            for r_type_on_path in path_rel_types_on_path:
                if 0 <= r_type_on_path < len(self.relation_texts_full):
                    path_relation_labels_display.append(self.relation_texts_full[r_type_on_path])
                else:
                    path_relation_labels_display.append(f"UNKNOWN_REL_TYPE_{r_type_on_path}")

            target_node_global_id = subset_original_indices[target_node_local_idx].item()
            target_node_global_name = self.id2entity[target_node_global_id]

            # Create human-readable path string
            path_str = ""
            if path_nodes_display_names:
                path_str = f"({path_nodes_display_names[0]})" # Starts with seed or first node in path
                for i, rel_label in enumerate(path_relation_labels_display):
                    # Path nodes include start and end, relations are between them
                    if (i + 1) < len(path_nodes_display_names):
                        node_name_in_path = path_nodes_display_names[i+1]
                        path_str += f" --[{rel_label}]--> ({node_name_in_path})"
                    else: # Should not happen if path_nodes has one more element than path_relations
                        path_str += f" --[{rel_label}]--> (ERROR: UNKNOWN NEXT NODE)"
                    # Collect paragraph IDs for each edge in the path (head, rel, tail)
                para_ids = []
                for i in range(len(path_nodes_display_names) - 1):
                    head = path_nodes_display_names[i]
                    tail = path_nodes_display_names[i + 1]
                    rel = path_relation_labels_display[i] if i < len(path_relation_labels_display) else None
                    para_id = self.para_id_dict.get((head, tail, rel))
                    para_ids.append(para_id)
        

            # If path is just to the seed node itself
            if not path_relation_labels_display and len(path_nodes_display_names) == 1:
                path_str = f"({path_nodes_display_names[0]})" # Just the seed node

            # "answer","relations",("retrieved_context_paragraphs","citations"


            scores_and_paths_data.append({
                "score": similarity_score,
                "para_ids": para_ids, # List of paragraph IDs for each edge in the path
                "target_node_global_id": target_node_global_id,
                "target_node_name": target_node_global_name,
                "path_nodes_global_ids": path_nodes_global_ids,
                "path_nodes_names": path_nodes_display_names,
                "path_relation_global_ids": path_rel_types_on_path, # these are already 0 to 2N-1
                "path_relation_labels": path_relation_labels_display,
                "path_string_formatted": f"{path_str} (Score: {similarity_score:.4f})"
            })

        scores_and_paths_data.sort(key=lambda x: x["score"], reverse=True)
        
        # Log top results
        for item in scores_and_paths_data[:top_n_results]:
            logger.info(item["path_string_formatted"])

        top_scores = scores_and_paths_data[:top_n_results] 

        if not top_scores:
            logger.warning("No relevant paths found for the given query and seed entity.")
            return {
            "question": query,
            "answer": "",
            "relations": "",
            "retrieved_context_paragraphs": {},
            "citations": []
        }
        # Collect unique paragraph IDs from all paths
        unique_para_ids = set()
        formatted_paths = []
        for item in top_scores:
            unique_para_ids.update(item["para_ids"])
            formatted_paths.append(item['path_string_formatted'])

        retrieved_paragraphs_dict = retrieve_paragraphs(
                                paragraph_ids=list(unique_para_ids),
                                kb_url=NUCLIA_KB_URL,
                                api_key=NUCLIA_API_KEY
                            )
        context_texts_for_llm = list(retrieved_paragraphs_dict.values())
        if not context_texts_for_llm:
            logger.warning("No context paragraphs retrieved for the paths.")
        context_texts_for_llm.extend(formatted_paths)
        # Generate answer using OpenAI LLM
        if OPENAI_API_KEY:
            answer = generate_answer_with_openai(
                api_key=OPENAI_API_KEY,
                question=query,
                context_paragraphs=context_texts_for_llm,
              )
            logger.info(f"Generated answer: {answer}")
        else:
            logger.warning("OpenAI API key not set. Skipping LLM answer generation.")
            answer = "LLM answer generation skipped due to missing OpenAI API key."
        # Return structured data with paths and answer
        
        return {
            "question": query,
            "answer": answer,
            "relations": formatted_paths,
            "retrieved_context_paragraphs": retrieved_paragraphs_dict,
            "citations": []
        }
        


# # Example usage:
# if __name__ == '__main__':
#     # Configure logging for better output
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

#     retriever = PathRGCNRetriever()
#     try:
#         with open(RELATIONS_PATH, "r", encoding="utf-8") as f:
#             loaded_relations = json.load(f)
#         logger.info(f"Successfully read {len(loaded_relations)} relations from {RELATIONS_PATH}.")
#     except Exception as e:
#         logger.error(f"Failed to read relations from {RELATIONS_PATH}: {e}", exc_info=True)
#     sample_relations = [
#         {"from": {"value": "Alice"}, "to": {"value": "Project X"}, "label": "works_on"},
#         {"from": {"value": "Project X"}, "to": {"value": "Topic A"}, "label": "related_to"},
#         {"from": {"value": "Bob"}, "to": {"value": "Project X"}, "label": "manages"},
#         {"from": {"value": "Alice"}, "to": {"value": "Topic B"}, "label": "interested_in"},
#         {"from": {"value": "Topic A"}, "to": {"value": "Field Z"}, "label": "subfield_of"},
#     ]
#     sample_relations = loaded_relations if loaded_relations else sample_relations
#     retriever.ingest_data(sample_relations)
#     if loaded_relations:
#         query = "What is the relationship between David Zinsner and Jen-Hsun Huang?"
#         entities = retriever.extract_entities(query)
#         logger.info(f"Query: '{query}' -> Extracted Entities: {entities}")
        
#         for entity in entities:
#             print(f"Entity '{entity}' ID: {retriever.entity2id.get(entity, 'Not found')}")

#             retrieved_paths = retriever.retrieve_relevant_paths(
#                 query=query,
#                 seed_entity_value=entity if entity else "Intel", # Default to Alice if no entities found
#                 k_hops=7,
#                 top_n_results=5
#             )
#         print("\n--- Retrieved Paths ---")
#         for path_data in retrieved_paths:
#             print(path_data["path_string_formatted"])
#             # print(json.dumps(path_data, indent=2)) # For full structured output

#     else:
#         print("\n--- Testing Entity Extraction ---")
#         query1 = "What is Alice working on related to Topic A?"
#         entities = retriever.extract_entities(query1)
#         print(f"Query: '{query1}' -> Extracted Entities: {entities}") # Expected: ['Alice', 'Topic A'] (order may vary)

#         if "Alice" in retriever.entity2id:
#             print("\n--- Testing Path Retrieval for Alice ---")
#             retrieved_paths_alice = retriever.retrieve_relevant_paths(
#                 query="Alice's projects and related topics",
#                 seed_entity_value="Alice",
#                 k_hops=2,
#                 top_n_results=5
#             )
#             # print("\nRetrieved paths for Alice:")
#             # for path_data in retrieved_paths_alice:
#             #     print(path_data["path_string_formatted"])
#             #     # print(json.dumps(path_data, indent=2)) # For full structured output
#         else:
#             print("Seed entity 'Alice' not found for retrieval test.")

#         if "Project X" in retriever.entity2id:
#             print("\n--- Testing Path Retrieval for Project X ---")
#             retrieved_paths_project_x = retriever.retrieve_relevant_paths(
#                 query="Who manages Project X and what is it related to?",
#                 seed_entity_value="Project X",
#                 k_hops=1, # Smaller k-hop for different test
#                 top_n_results=3,
#                 alpha_target_node=0.3, # Example of different weights
#                 alpha_path_relations=0.7
#             )
#             # print("\nRetrieved paths for Project X:")
#             # for path_data in retrieved_paths_project_x:
#             #     print(path_data["path_string_formatted"])
#         else:
#             print("Seed entity 'Project X' not found for retrieval test.")

#         print("\n--- Testing with a non-existent seed ---")
#         retrieved_paths_non_existent = retriever.retrieve_relevant_paths(
#             query="About foobar",
#             seed_entity_value="Foobar",
#             k_hops=2
#         )
#         print(f"Paths for 'Foobar': {retrieved_paths_non_existent}")


#         print("\n--- Testing with empty relations ---")
#         empty_retriever = PathRGCNRetriever()
#         empty_retriever.ingest_data([])
#         retrieved_paths_empty = empty_retriever.retrieve_relevant_paths(
#             query="Anything",
#             seed_entity_value="Anything" # Won't be found
#         )
#         print(f"Paths from empty KG: {retrieved_paths_empty}")

#         print("\n--- Testing with relations but query for non-existent seed ---")
#         retriever_with_data_no_seed_in_query = PathRGCNRetriever()
#         retriever_with_data_no_seed_in_query.ingest_data(sample_relations)
#         retrieved_paths_no_seed = retriever_with_data_no_seed_in_query.retrieve_relevant_paths(
#             query="Information about Gamma", # Gamma is not in KG
#             seed_entity_value="Gamma",
#             k_hops=2
#         )
#         print(f"Paths for 'Gamma' (not in KG): {retrieved_paths_no_seed}")