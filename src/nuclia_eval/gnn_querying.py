# src/path_rgcn_retriever.py
import json
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
from torch_geometric.data import HeteroData
from typing import List, Dict, Any, Optional, Tuple

import spacy
from rapidfuzz import process

# from model2vec import StaticModel # If you switched completely to SentenceTransformer
from sentence_transformers import SentenceTransformer # Assuming this is your primary text embedder now

from src.utils_max import extract_k_hop_subgraph_bidirectional
from src.llm_utils import generate_answer_with_openai
from src.graph_processing_utils import retrieve_paragraphs
from dotenv import load_dotenv
import logging
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')

RELATIONS_PATH = "data/legal_graph.json" # Same as train.py KG_DATA_PATH
# --- These paths should match OUTPUT paths from your train_kg_gnn.py script ---
PRETRAINED_ENTITY_EMBS_PATH = "legal_pretrained_entity_embeddings_distmult.pt"
PRETRAINED_RELATION_EMBS_PATH = "legal_pretrained_relation_embeddings_distmult.pt" # NEW
KG_METADATA_PATH = "legal_kg_metadata_distmult.json"
FALLBACK_ANSWER = {"question": "query",
            "answer": "Not enough information to generate an answer.",
            "relations": [],
            "retrieved_context_paragraphs": {},
            "citations": [{}]
        }


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

class PathRGCNRetrieverTrained:
    def __init__(self,
                 text_embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 pretrained_entity_embeddings_path: Optional[str] = PRETRAINED_ENTITY_EMBS_PATH,
                 pretrained_relation_embeddings_path: Optional[str] = PRETRAINED_RELATION_EMBS_PATH, # NEW
                 kg_metadata_path: Optional[str] = KG_METADATA_PATH):
        
        logger.info(f"Initializing PathRGCNRetriever. Text embedding model: {text_embedding_model_name}")
        try:
            self.text_embedding_model = SentenceTransformer(text_embedding_model_name)
            self.query_embedding_dim = self.text_embedding_model.get_sentence_embedding_dimension()
        except Exception as e:
            logger.error(f"Failed to load text_embedding_model {text_embedding_model_name}: {e}", exc_info=True)
            raise

        self.nlp = spacy.load("en_core_web_trf")

        self.entity2id: Dict[str, int] = {}
        self.id2entity: Dict[int, str] = {}
        self.relation2id_original: Dict[str, int] = {} # From KGE metadata (original rel_label -> 0 to N-1)
        self.id2relation_original: Dict[int, str] = {}
        self.num_original_relations: int = 0
        self.relation_texts_full: List[str] = [] # For display: label for 0..2N-1
        self.para_id_dict: Dict[Tuple[str,str,str], str] = {}
        self.entity_embeddings_full: Optional[torch.Tensor] = None
        self.rel_embs_full: Optional[torch.Tensor] = None # Will hold KGE-learned or text-based relation embeddings
        
        # This will be the GNN_EMBEDDING_DIM if KGE embeddings are loaded,
        # otherwise, it might default to query_embedding_dim or be 0 initially.
        self.kg_embedding_dim: int = 0 
        
        self.full_graph_data: Optional[HeteroData] = None
        self.relation_metadata_store: Dict[int, Dict[str, Any]] = {}

        self.pretrained_entity_embeddings_path = pretrained_entity_embeddings_path
        self.pretrained_relation_embeddings_path = pretrained_relation_embeddings_path # NEW
        self.kg_metadata_path = kg_metadata_path

        if self.kg_metadata_path:
            if self.pretrained_entity_embeddings_path:
                logger.info(f"Attempting to load pre-trained entity data...")
                self._load_pretrained_entity_data()
            if self.pretrained_relation_embeddings_path: # NEW
                logger.info(f"Attempting to load pre-trained relation data...")
                self._load_pretrained_relation_data() # NEW
        # (Some warning logic if paths are partially provided could be added)
        try:
            with open(RELATIONS_PATH, "r", encoding="utf-8") as f:
                loaded_relations = json.load(f)
            logger.info(f"Successfully read {len(loaded_relations)} relations from {RELATIONS_PATH}.")
        except Exception as e:
            logger.error(f"Failed to read relations from {RELATIONS_PATH}: {e}", exc_info=True)
        self.ingest_data(loaded_relations) # Ensure data is ingested before running queries


    def _load_kg_metadata(self) -> Optional[Dict[str, Any]]:
        """Loads metadata if not already loaded, or if entity2id is empty."""
        if not self.entity2id and self.kg_metadata_path and os.path.exists(self.kg_metadata_path):
            try:
                with open(self.kg_metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                self.entity2id = metadata.get('entity2id', {})
                self.id2entity = {i: e for e, i in self.entity2id.items()}
                # Load relation2id_original from KGE training metadata
                self.relation2id_original = metadata.get('relation2id_original', {})
                self.id2relation_original = {i: r for r, i in self.relation2id_original.items()}
                self.num_original_relations = len(self.relation2id_original)
                
                # This is the embedding dimension of the GNN/KGE model's output
                self.kg_embedding_dim = metadata.get('embedding_dim', 0) 
                logger.info(f"KG metadata loaded. Entity count: {len(self.entity2id)}, Original relation count: {self.num_original_relations}, KG Embedding Dim: {self.kg_embedding_dim}")
                return metadata
            except Exception as e:
                logger.error(f"Error loading KG metadata from {self.kg_metadata_path}: {e}", exc_info=True)
        elif self.entity2id: # Metadata likely loaded
             return {'entity2id': self.entity2id, 'relation2id_original': self.relation2id_original, 'embedding_dim': self.kg_embedding_dim } # Return already loaded
        return None


    def _load_pretrained_entity_data(self):
        metadata = self._load_kg_metadata() # Ensures entity2id and kg_embedding_dim are populated
        if not metadata or not self.pretrained_entity_embeddings_path or not os.path.exists(self.pretrained_entity_embeddings_path):
            logger.warning("Cannot load pre-trained entity embeddings: metadata or embeddings file missing.")
            return

        try:
            loaded_embeddings = torch.load(self.pretrained_entity_embeddings_path, map_location=torch.device('cpu'))
            if len(self.entity2id) != loaded_embeddings.size(0):
                logger.error(f"Mismatch in entity count from metadata ({len(self.entity2id)}) and loaded entity embeddings ({loaded_embeddings.size(0)}). Cannot use pre-trained entity embeddings.")
                self.entity2id = {} # Invalidate to force re-init
                self.id2entity = {}
                return

            self.entity_embeddings_full = loaded_embeddings
            if self.kg_embedding_dim == 0: # If not set by metadata (older metadata file)
                self.kg_embedding_dim = loaded_embeddings.size(1)
            elif self.kg_embedding_dim != loaded_embeddings.size(1):
                logger.warning(f"Entity embedding dim from metadata ({self.kg_embedding_dim}) differs from loaded tensor ({loaded_embeddings.size(1)}). Using loaded tensor's dim.")
                self.kg_embedding_dim = loaded_embeddings.size(1)
            
            logger.info(f"Successfully loaded pre-trained entity embeddings. Shape: {self.entity_embeddings_full.shape}")

        except Exception as e:
            logger.error(f"Error loading pre-trained entity embeddings: {e}", exc_info=True)
            self.entity_embeddings_full = None


    def _load_pretrained_relation_data(self): # NEW METHOD
        metadata = self._load_kg_metadata() # Ensures relation2id_original and kg_embedding_dim are populated
        if not metadata or not self.pretrained_relation_embeddings_path or not os.path.exists(self.pretrained_relation_embeddings_path):
            logger.warning("Cannot load pre-trained relation embeddings: metadata or embeddings file missing.")
            return
        
        if not self.relation2id_original:
            logger.warning("relation2id_original not found in metadata. Cannot map pre-trained relation embeddings.")
            return

        try:
            loaded_rel_embeddings = torch.load(self.pretrained_relation_embeddings_path, map_location=torch.device('cpu'))
            
            # KGE training saves embeddings for 2*num_original_relations
            expected_num_rel_embs = 2 * self.num_original_relations
            if expected_num_rel_embs != loaded_rel_embeddings.size(0):
                logger.error(f"Mismatch in expected relation embedding count ({expected_num_rel_embs}) "
                               f"and loaded relation embeddings ({loaded_rel_embeddings.size(0)}). Cannot use.")
                return

            if self.kg_embedding_dim == 0: # If not set by metadata/entity_embs
                self.kg_embedding_dim = loaded_rel_embeddings.size(1)
            elif self.kg_embedding_dim != loaded_rel_embeddings.size(1):
                logger.warning(f"Relation embedding dim from loaded tensor ({loaded_rel_embeddings.size(1)}) "
                               f"differs from established kg_embedding_dim ({self.kg_embedding_dim}). This might cause issues.")
                # Decide on a strategy: error, use relation dim, or project. For now, proceed with caution.
                # self.kg_embedding_dim = loaded_rel_embeddings.size(1) # Or choose to keep the entity one

            self.rel_embs_full = loaded_rel_embeddings
            logger.info(f"Successfully loaded pre-trained relation embeddings. Shape: {self.rel_embs_full.shape}")

        except Exception as e:
            logger.error(f"Error loading pre-trained relation embeddings: {e}", exc_info=True)
            self.rel_embs_full = None


    def ingest_data(self, relations_input: List[Dict[str, Any]]): # Renamed 'relations' to 'relations_input'
        logger.info(f"Ingesting {len(relations_input)} relations into the knowledge graph.")
        
        # Ensure metadata (and potentially pre-trained entity/relation data) is loaded first
        self._load_kg_metadata() 
        # If _load_pretrained_entity_data or _load_pretrained_relation_data were called in __init__,
        # self.entity_embeddings_full and self.rel_embs_full might already be populated.

        # If entity2id is still empty, it means no pre-trained metadata or it failed. Build from scratch.
        if not self.entity2id:
            logger.info("No pre-loaded entity2id. Building from input relations.")
            temp_entity2id = {}
            for rel_item in relations_input:
                for node_val in (rel_item['from']['value'], rel_item['to']['value']):
                    if node_val not in temp_entity2id:
                        temp_entity2id[node_val] = len(temp_entity2id)
            self.entity2id = temp_entity2id
            self.id2entity = {i: e for e, i in self.entity2id.items()}
        
        # Process relations based on the now established self.entity2id
        # And determine self.relation2id_original if not loaded from KGE metadata
        if not self.relation2id_original: # If not loaded from KGE metadata
            logger.info("No pre-loaded relation2id_original. Building from input relations.")
            temp_relation2id_orig = {}
            for rel_item in relations_input:
                label_val = rel_item['label']
                if label_val not in temp_relation2id_orig:
                    temp_relation2id_orig[label_val] = len(temp_relation2id_orig)
            self.relation2id_original = temp_relation2id_orig
            self.id2relation_original = {i: r for r, i in self.relation2id_original.items()}
            self.num_original_relations = len(self.relation2id_original)
            if self.num_original_relations > 0 and self.kg_embedding_dim == 0:
                 # Try to get a default dim if not set (e.g. from query model)
                 self.kg_embedding_dim = self.query_embedding_dim
                 logger.info(f"kg_embedding_dim was 0, set to query_embedding_dim: {self.kg_embedding_dim} for scratch mode.")


        edges = []
        edge_types_original_ids = [] # Store original relation IDs (0 to N-1 based on self.relation2id_original)

        for idx, rel_item in enumerate(relations_input):
            src_val, dst_val, label_val = rel_item['from']['value'], rel_item['to']['value'], rel_item['label']
            self.para_id_dict[(src_val, dst_val, label_val)] = rel_item['metadata']['paragraph_id']
            self.relation_metadata_store[idx] = rel_item
            
            if src_val in self.entity2id and dst_val in self.entity2id and label_val in self.relation2id_original:
                edges.append((self.entity2id[src_val], self.entity2id[dst_val]))
                edge_types_original_ids.append(self.relation2id_original[label_val])
            else:
                logger.warning(f"Skipping relation {rel_item} due to missing entity/relation mapping in established dictionaries.")

        # Prepare textual labels for all relations (original + inverse) for display purposes
        self.relation_texts_full = [""] * (2 * self.num_original_relations if self.num_original_relations > 0 else 0)
        for label, r_id in self.relation2id_original.items():
            self.relation_texts_full[r_id] = label
            if (r_id + self.num_original_relations) < len(self.relation_texts_full):
                 self.relation_texts_full[r_id + self.num_original_relations] = f"inverse of {label}"

        logger.info(f"KG structure processed: {len(self.entity2id)} entities, {self.num_original_relations} original relation types.")

        # --- Entity Embeddings ---
        if self.entity_embeddings_full is None:
            logger.info("Generating entity embeddings using text_embedding_model (no pre-trained GNN entity embs found/loaded).")
            if self.entity2id:
                entity_texts = [self.id2entity.get(i, f"UNKNOWN_ENTITY_{i}") for i in range(len(self.entity2id))]
                encoded_entities_np = self.text_embedding_model.encode(entity_texts, convert_to_numpy=True, show_progress_bar=True)
                self.entity_embeddings_full = torch.tensor(encoded_entities_np, dtype=torch.float)
                if self.kg_embedding_dim == 0: self.kg_embedding_dim = self.entity_embeddings_full.size(1)
                logger.info(f"Generated entity embeddings of shape: {self.entity_embeddings_full.shape}")
            else:
                self.entity_embeddings_full = torch.empty(0, self.query_embedding_dim if self.query_embedding_dim > 0 else 1, dtype=torch.float)
        
        # --- Relation Embeddings ---
        if self.rel_embs_full is None: # Only generate if not pre-loaded from KGE training
            logger.info("Generating relation embeddings using text_embedding_model (no pre-trained GNN relation embs found/loaded).")
            if self.num_original_relations > 0 and self.relation_texts_full:
                # We need embeddings for original (0..N-1) and inverse (N..2N-1)
                # The self.relation_texts_full should be correctly sized (2*N)
                encoded_relations_np = self.text_embedding_model.encode(self.relation_texts_full, convert_to_numpy=True, show_progress_bar=True)
                self.rel_embs_full = torch.tensor(encoded_relations_np, dtype=torch.float)
                # Ensure kg_embedding_dim is consistent if this is the first embedding set
                if self.kg_embedding_dim == 0: self.kg_embedding_dim = self.rel_embs_full.size(1)
                elif self.kg_embedding_dim != self.rel_embs_full.size(1):
                     logger.warning(f"Text-generated relation embedding dim ({self.rel_embs_full.size(1)}) "
                                   f"differs from established kg_embedding_dim ({self.kg_embedding_dim}).")
                logger.info(f"Generated relation embeddings of shape: {self.rel_embs_full.shape}")

            else: # No relations to embed
                # Ensure kg_embedding_dim has a fallback if still 0
                fallback_dim = self.kg_embedding_dim if self.kg_embedding_dim > 0 else (self.query_embedding_dim if self.query_embedding_dim > 0 else 1)
                self.rel_embs_full = torch.empty(0, fallback_dim, dtype=torch.float)
        
        # Final check on kg_embedding_dim
        if self.kg_embedding_dim == 0:
            if self.entity_embeddings_full is not None and self.entity_embeddings_full.numel() > 0:
                self.kg_embedding_dim = self.entity_embeddings_full.size(1)
            elif self.rel_embs_full is not None and self.rel_embs_full.numel() > 0:
                self.kg_embedding_dim = self.rel_embs_full.size(1)
            elif self.query_embedding_dim > 0:
                self.kg_embedding_dim = self.query_embedding_dim
            else: # Absolute fallback
                self.kg_embedding_dim = 1 
            logger.info(f"Final kg_embedding_dim set to: {self.kg_embedding_dim}")


        self.full_graph_data = HeteroData()
        if self.entity_embeddings_full is not None and self.entity_embeddings_full.numel() > 0:
            self.full_graph_data['entity'].x = self.entity_embeddings_full
        else:
            self.full_graph_data['entity'].x = torch.empty((0, self.kg_embedding_dim), dtype=torch.float)

        if edges:
            self.full_graph_data['entity', 'to', 'entity'].edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            self.full_graph_data['entity', 'to', 'entity'].edge_type = torch.tensor(edge_types_original_ids, dtype=torch.long)
        else:
            self.full_graph_data['entity', 'to', 'entity'].edge_index = torch.empty((2,0), dtype=torch.long)
            self.full_graph_data['entity', 'to', 'entity'].edge_type = torch.empty(0, dtype=torch.long)

        logger.info("Full graph HeteroData object created for retriever.")


    # ... _bfs_path_rels and extract_entities remain the same ...
    def _bfs_path_rels(self, adj_list_subgraph: List[List[Tuple[int, int]]],
                       seed_local_idx: int, target_local_idx: int) -> Tuple[List[int], List[int]]:
        queue_bfs = deque([(seed_local_idx, [])]) 
        parent_map: Dict[int, Optional[Tuple[int, int]]] = {seed_local_idx: None}
        path_found_for_target = False
        explore_q = deque([seed_local_idx])
        visited_for_bfs = {seed_local_idx}

        while explore_q:
            u = explore_q.popleft()
            if u == target_local_idx:
                path_found_for_target = True
                break
            if u < len(adj_list_subgraph):
                for v_local, r_type_in_subgraph in adj_list_subgraph[u]:
                    if v_local not in visited_for_bfs:
                        visited_for_bfs.add(v_local)
                        parent_map[v_local] = (u, r_type_in_subgraph)
                        explore_q.append(v_local)
            else:
                logger.warning(f"BFS: Node index {u} out of bounds for adj_list of length {len(adj_list_subgraph)}")

        final_path_rel_types: List[int] = []
        final_path_nodes_local: List[int] = [] 

        if not path_found_for_target or target_local_idx not in parent_map:
            return [], []

        curr = target_local_idx
        while curr != seed_local_idx and parent_map.get(curr) is not None:
            p_local, r_type = parent_map[curr] 
            final_path_rel_types.append(r_type)
            final_path_nodes_local.append(curr)
            curr = p_local
        
        if final_path_nodes_local or target_local_idx == seed_local_idx :
             if target_local_idx != seed_local_idx or not final_path_nodes_local : 
                 final_path_nodes_local.append(seed_local_idx)
        
        return list(reversed(final_path_rel_types)), list(reversed(final_path_nodes_local))


    def extract_entities(self, query: str) -> List[str]:
        doc = self.nlp(query)
        entities = []
        if not self.entity2id: 
            logger.warning("Cannot extract entities, entity2id is empty.")
            return []
        node_names = list(self.entity2id.keys())
        for ent in doc.ents:
            if not node_names: # handles case where entity2id initialized but empty
                logger.warning("Entity list for fuzzy matching is empty.")
                break
            best_match, score, _ = process.extractOne(ent.text, node_names)
            if score > 80 and best_match not in entities: 
                entities.append(best_match)
        return entities

    def query_knowledge_graph(self, query: str, 
                                k_hops: int = 2, top_n_results: int = 5,
                                alpha_target_node: float = 0.5, alpha_path_relations: float = 0.5, model_override: Optional[str] = None,
                               ) -> List[Dict[str, Any]]:
        
        logger.info(f"Retrieving paths for query: '{query[:50]}...', k={k_hops}")
        entities = self.extract_entities(query)
        seed_entity_value = self.extract_entities(query)
        if not seed_entity_value:
            logger.error("No entities extracted from the query. Cannot retrieve paths.")
            return {
            "question": query,
            "answer": "Not enough information to generate an answer.",
            "relations": [],
            "retrieved_context_paragraphs": {},
            "citations": [{}]
        }
        seed_entity_value = seed_entity_value[0]  # Use the first extracted entity
        if not entities:
            logger.warning(f"No entities extracted from query '{query}'. Cannot retrieve paths.")
            FALLBACK_ANSWER["answer"] = query
            return FALLBACK_ANSWER
        # --- Essential Checks ---
        if self.full_graph_data is None:
            logger.error("Full graph data not available. Call ingest_data() first.")
            FALLBACK_ANSWER["answer"] = query
            return FALLBACK_ANSWER
        if self.entity_embeddings_full is None or self.entity_embeddings_full.numel() == 0:
            logger.error("Entity embeddings not available. Cannot retrieve paths.")
            FALLBACK_ANSWER["answer"] = query
            return FALLBACK_ANSWER
        if self.rel_embs_full is None or self.rel_embs_full.numel() == 0 :
            logger.warning("Relation embeddings not available. Path scoring will only use target nodes.")
            # Allow proceeding, h_rel_path_avg will be zero.

        if self.kg_embedding_dim == 0:
            logger.error("KG embedding dimension is 0. Cannot reliably score paths.")
            FALLBACK_ANSWER["answer"] = query
            return FALLBACK_ANSWER


        # --- Query Embedding ---
        # Query embedding dimension should ideally match self.kg_embedding_dim
        q_emb_np = self.text_embedding_model.encode(query, convert_to_numpy=True)
        q_emb = torch.tensor(q_emb_np, dtype=torch.float).unsqueeze(0) # [1, D_query]
        
        # Projection if query dim != kg_embedding_dim (conceptual)
        # For now, we assume they should match or the user is aware of potential issues.
        # A learnable projection layer would be ideal here if dims consistently mismatch.
        # self.query_projection = torch.nn.Linear(self.query_embedding_dim, self.kg_embedding_dim)
        # q_emb_projected = self.query_projection(q_emb)
        # q_emb_norm = F.normalize(q_emb_projected, dim=1)
        
        if q_emb.size(1) != self.kg_embedding_dim:
            logger.warning(
                f"Query embedding dim ({q_emb.size(1)}) != KG embedding dim ({self.kg_embedding_dim}). "
                "Path scoring quality may be affected. Consider aligning dimensions or adding a projection layer for the query."
            )
            # If you want to force stop: return []
            # If you want to proceed (e.g. for testing, or if a simple projection isn't available):
            # Option: Pad/truncate q_emb or kg_embs (not ideal)
            # Option: Use a different similarity if dims mismatch (e.g. if one is much larger)
            # For now, we will proceed, but normalization is on original q_emb
        q_emb_norm = F.normalize(q_emb, dim=1)


        # --- Seed Entity and Subgraph ---
        if seed_entity_value not in self.entity2id:
            logger.error(f"Seed entity '{seed_entity_value}' not found in knowledge graph.")
            return []
        seed_id_global = self.entity2id[seed_entity_value]

        sub_data, mapping_subset_orig_to_local, subset_original_indices = extract_k_hop_subgraph_bidirectional(
            seed_id=seed_id_global, K=k_hops, data=self.full_graph_data,
            num_rels_total=self.num_original_relations # This num_original_relations is key
        )

        if sub_data['entity'].x.numel() == 0 or subset_original_indices.numel() == 0:
             logger.warning(f"Bidirectional subgraph for '{seed_entity_value}' is empty.")
             return []

        seed_id_local_candidates = (subset_original_indices == seed_id_global).nonzero(as_tuple=True)[0]
        if seed_id_local_candidates.numel() == 0:
            logger.error(f"Seed entity '{seed_entity_value}' not found in its own subgraph. This is unexpected.")
            return []
        seed_id_local = seed_id_local_candidates[0].item()
        
        node_embs_in_subgraph = sub_data['entity'].x # These are from KGE (or text model if fallback)
        edge_index_sub = sub_data['entity','to','entity'].edge_index
        edge_type_sub = sub_data['entity','to','entity'].edge_type # These are 0 to 2*N-1

        if node_embs_in_subgraph.numel() == 0: return [] # Should be caught by earlier check
        
        # Normalize node embeddings from the subgraph
        # Ensure these embeddings are in self.kg_embedding_dim
        if node_embs_in_subgraph.size(1) != self.kg_embedding_dim:
            logger.error(f"Subgraph node embedding dim ({node_embs_in_subgraph.size(1)}) != "
                         f"expected KG embedding dim ({self.kg_embedding_dim}). Halting.")
            return []
        node_embs_norm = F.normalize(node_embs_in_subgraph, dim=1)


        N_sub = node_embs_norm.size(0)
        adj_list_sub: List[List[Tuple[int, int]]] = [[] for _ in range(N_sub)]
        # Build adj list from subgraph edges
        for i in range(edge_index_sub.size(1)):
            u, v = edge_index_sub[0, i].item(), edge_index_sub[1, i].item()
            r_type = edge_type_sub[i].item() # This is 0..(2N-1)
            if 0 <= u < N_sub and 0 <= v < N_sub :
                adj_list_sub[u].append((v, r_type))

        scores_and_paths_data = []
        for target_node_local_idx in range(N_sub):
            path_rel_types_on_path, path_nodes_local_indices_on_path = self._bfs_path_rels(
                adj_list_sub, seed_id_local, target_node_local_idx
            )
            
            if target_node_local_idx == seed_id_local and not path_nodes_local_indices_on_path:
                 path_nodes_local_indices_on_path = [seed_id_local]

            if not path_nodes_local_indices_on_path and target_node_local_idx != seed_id_local:
                continue

            h_target_node_sub = node_embs_norm[target_node_local_idx] # Dim: kg_embedding_dim
            
            # Initialize path embedding (must also be in kg_embedding_dim)
            h_rel_path_avg = torch.zeros(self.kg_embedding_dim, device=h_target_node_sub.device)

            if path_rel_types_on_path and self.rel_embs_full is not None and self.rel_embs_full.numel() > 0:
                # Ensure loaded relation embeddings are used and have the correct dimension
                if self.rel_embs_full.size(1) != self.kg_embedding_dim:
                    logger.warning(f"Actual relation embedding dim ({self.rel_embs_full.size(1)}) != "
                                   f"expected KG embedding dim ({self.kg_embedding_dim}). "
                                   "Skipping relation contribution to path score.")
                else:
                    # These r_types are already 0 to 2N-1 from subgraph extraction utility
                    # which should correspond to rows in self.rel_embs_full if it's from KGE
                    valid_rel_indices_on_path = [r_type for r_type in path_rel_types_on_path if 0 <= r_type < self.rel_embs_full.size(0)]
                    if valid_rel_indices_on_path:
                        rel_vecs_on_path = self.rel_embs_full[valid_rel_indices_on_path].to(h_target_node_sub.device)
                        h_rel_path_avg = F.normalize(rel_vecs_on_path.mean(dim=0), dim=0)
            
            # Combine target node and path relation embeddings (both should be in kg_embedding_dim)
            target_contrib = alpha_target_node * h_target_node_sub
            path_contrib = alpha_path_relations * h_rel_path_avg # h_rel_path_avg is zero if no valid rels

            h_combined = F.normalize(target_contrib + path_contrib, dim=0)

            # Similarity score: q_emb_norm (D_query) vs h_combined (kg_embedding_dim)
            # This is where dim mismatch is critical.
            # If q_emb was projected to kg_embedding_dim, this is fine.
            # Otherwise, this cosine similarity might be comparing vectors of different lengths if not handled.
            # For now, we assume if q_emb_norm.size(1) != h_combined.size(0), the warning was issued.
            # The F.cosine_similarity will likely fail if dims don't match.
            # Let's add a check for safety:
            final_q_emb_for_scoring = q_emb_norm.squeeze(0).to(h_combined.device)
            if final_q_emb_for_scoring.size(0) != h_combined.size(0):
                logger.error(f"Dimension mismatch for cosine similarity: query_emb ({final_q_emb_for_scoring.size(0)}) "
                             f"vs combined_kg_emb ({h_combined.size(0)}). Cannot score path. Skipping.")
                similarity_score = -1.0 # Or handle error
                # Potentially skip adding this path if scoring fails
                # continue
            else:
                similarity_score = F.cosine_similarity(final_q_emb_for_scoring, h_combined, dim=0).item()

            # ... (rest of path formatting and appending to scores_and_paths_data remains same) ...
            path_nodes_global_ids = [subset_original_indices[node_local_idx].item() for node_local_idx in path_nodes_local_indices_on_path]
            path_nodes_display_names = [self.id2entity.get(global_id, f"UNKNOWN_ID_{global_id}") for global_id in path_nodes_global_ids]
            
            path_relation_labels_display = []
            for r_type_on_path in path_rel_types_on_path:
                # Use self.relation_texts_full which is based on self.relation2id_original for labels
                if 0 <= r_type_on_path < len(self.relation_texts_full):
                    path_relation_labels_display.append(self.relation_texts_full[r_type_on_path])
                else:
                    path_relation_labels_display.append(f"UNKNOWN_REL_TYPE_ID_{r_type_on_path}")

            target_node_global_id = subset_original_indices[target_node_local_idx].item()
            target_node_global_name = self.id2entity.get(target_node_global_id, f"UNKNOWN_ID_{target_node_global_id}")

            path_str = ""
            if path_nodes_display_names:
                path_str = f"({path_nodes_display_names[0]})"
                for i, rel_label in enumerate(path_relation_labels_display):
                    if (i + 1) < len(path_nodes_display_names):
                        node_name_in_path = path_nodes_display_names[i+1]
                        path_str += f" --[{rel_label}]--> ({node_name_in_path})"

                para_ids = []
                for i in range(len(path_nodes_display_names) - 1):
                    head = path_nodes_display_names[i]
                    tail = path_nodes_display_names[i + 1]
                    rel = path_relation_labels_display[i] if i < len(path_relation_labels_display) else None
                    para_id = self.para_id_dict.get((head, tail, rel))
                    if para_id is not None:
                        para_ids.append(para_id)

            if not path_relation_labels_display and len(path_nodes_display_names) == 1: # Path to self
                path_str = f"({path_nodes_display_names[0]})"

            scores_and_paths_data.append({
                "score": similarity_score,
                "para_ids": para_ids, # List of paragraph IDs for each edge in the path
                "target_node_global_id": target_node_global_id,
                "target_node_name": target_node_global_name,
                "path_nodes_global_ids": path_nodes_global_ids,
                "path_nodes_names": path_nodes_display_names,
                "path_relation_global_ids": path_rel_types_on_path,
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
            "answer": "Not enough information to generate an answer.",
            "relations": [],
            "retrieved_context_paragraphs": {},
            "citations": [{}]
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
                model_name=model_override
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
            "citations": [{}]
        }


# --- Example Usage Update ---
# if __name__ == '__main__':
    
#     # Save sample_relations_data to ./test_max/relations.json
    
#     # No need to write SIMULATED_RELATIONS_PATH here again if train.py already created it.
#     logger.info(f"Attempting to read relations from {SIMULATED_RELATIONS_PATH}...")
#     try:
#         with open(SIMULATED_RELATIONS_PATH, "r", encoding="utf-8") as f:
#             loaded_relations = json.load(f)
#         logger.info(f"Successfully read {len(loaded_relations)} relations from {SIMULATED_RELATIONS_PATH}.")
#     except Exception as e:
#         logger.error(f"Failed to read relations from {SIMULATED_RELATIONS_PATH}: {e}", exc_info=True)
#     logger.info(f"Ensure the following files exist from running train_kg_gnn.py:")
#     logger.info(f"  Entity Embeddings: {PRETRAINED_ENTITY_EMBS_PATH}")
#     logger.info(f"  Relation Embeddings: {PRETRAINED_RELATION_EMBS_PATH}")
#     logger.info(f"  KG Metadata: {KG_METADATA_PATH}")
    
#     if not all(os.path.exists(p) for p in [PRETRAINED_ENTITY_EMBS_PATH, PRETRAINED_RELATION_EMBS_PATH, KG_METADATA_PATH]):
#         logger.error("One or more pre-trained files are missing. Please run train_kg_gnn.py first to generate them. Exiting example.")
#         exit()
    
#     try:
#         retriever = PathRGCNRetriever(
#             text_embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", # For queries
#             pretrained_entity_embeddings_path=PRETRAINED_ENTITY_EMBS_PATH,
#             pretrained_relation_embeddings_path=PRETRAINED_RELATION_EMBS_PATH, # Pass path
#             kg_metadata_path=KG_METADATA_PATH
#         )
        
#         # Ingest the relations. Retriever will use pre-trained data if loaded successfully.
#         retriever.ingest_data(loaded_relations)

#         print("\n--- Testing Entity Extraction ---")
#         # query1 = "What is the relationship between David Zinsner and Jen-Hsun Huang?"
#         query1 = "Who is the CEO of NVIDA?"
#         entities = retriever.extract_entities(query1)
#         print(f"Query: '{query1}' -> Extracted Entities: {entities}")

#         for entity in entities:
#             print(f"\n--- Testing Path Retrieval for {entity} ---")
#             paths_intel = retriever.retrieve_relevant_paths(
#                 query=query1,
#                 seed_entity_value=entity, k_hops=7, top_n_results=5,
#             )


#     except Exception as e:
#         logger.error(f"An error occurred during the PathRGCNRetriever example: {e}", exc_info=True)