# src/path_rgcn_retriever.py
import json
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
from torch_geometric.data import HeteroData
from torch_geometric.nn import RGCNConv
from typing import List, Dict, Any, Optional # For type hinting clarity
import spacy
from rapidfuzz import process # For fuzzy matching

# --- Corrected Imports ---
from model2vec import StaticModel # Assuming this is in your project's PYTHONPATH
from src.utils import extract_k_hop_subgraph_bidirectional # From your utils.py

import logging
logger = logging.getLogger(__name__)

class PathRGCNRetriever:
    def __init__(self, embedding_model_name: str = "minishlab/potion-base-8M"):
        logger.info(f"Initializing PathRGCNRetriever with graph and model: {embedding_model_name}")
        self.embedding_model = StaticModel.from_pretrained(embedding_model_name)
        self.nlp = spacy.load("en_core_web_sm") # Load NER model
        # with open(relations_json_path, 'r', encoding='utf-8') as f:
        #     relations = json.load(f)
    def ingest_data(self, relations: List[Dict[str, Any]]):
        
        self.entity2id = {}
        self.relation2id = {}
        edges = []
        edge_types = [] # Store original edge types here
        self.relation_metadata = {} # To store original relation dicts if needed for paragraph_id later

        for idx, rel in enumerate(relations):
            src_val, dst_val, label_val = rel['from']['value'], rel['to']['value'], rel['label']
            
            # Store original relation for potential metadata retrieval later
            # Create a unique ID for the original relation (e.g., its index or a hash)
            original_relation_id = idx # Or some other unique identifier from `rel` if available
            self.relation_metadata[original_relation_id] = rel # Store the whole dict

            for node in (src_val, dst_val):
                if node not in self.entity2id:
                    self.entity2id[node] = len(self.entity2id)
            if label_val not in self.relation2id:
                self.relation2id[label_val] = len(self.relation2id)
            
            edges.append((self.entity2id[src_val], self.entity2id[dst_val]))
            edge_types.append(self.relation2id[label_val]) # Store original relation type ID

        self.id2entity = {i: e for e, i in self.entity2id.items()}
        self.num_original_relations = len(self.relation2id) # Number of distinct original relation types
        self.relation_texts = [""] * self.num_original_relations # Initialize with correct size
        for label, r_id in self.relation2id.items():
            self.relation_texts[r_id] = label
        
        logger.info(f"KG loaded: {len(self.entity2id)} entities, {self.num_original_relations} original relation types.")

        entity_texts = [self.id2entity[i] for i in range(len(self.id2entity))]
        # Assuming encode returns a 2D array [1, D] for single text, so stack and squeeze
        encoded_entities = [self.embedding_model.encode(text) for text in entity_texts]
        if encoded_entities and isinstance(encoded_entities[0], np.ndarray) and encoded_entities[0].ndim == 2:
             self.entity_embeddings_full = torch.tensor(np.concatenate(encoded_entities, axis=0), dtype=torch.float)
        else: # Fallback if encode returns 1D or other formats
             self.entity_embeddings_full = torch.tensor(np.array(encoded_entities), dtype=torch.float)
        
        # Ensure correct shape for relation embeddings
        encoded_relations = []
        for lbl in self.relation_texts:
            emb = self.embedding_model.encode(lbl)
            if isinstance(emb, np.ndarray) and emb.ndim == 2:
                encoded_relations.append(torch.tensor(emb[0], dtype=torch.float))
            else:
                encoded_relations.append(torch.tensor(emb, dtype=torch.float))
        self.rel_embs_full = torch.stack(encoded_relations) # [R_original, D]

        self.embedding_dim = self.entity_embeddings_full.size(1) # Get embedding dimension

        self.full_graph_data = HeteroData()
        self.full_graph_data['entity'].x = self.entity_embeddings_full
        # --- Standardized edge type key ---
        self.full_graph_data['entity', 'to', 'entity'].edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        self.full_graph_data['entity', 'to', 'entity'].edge_type = torch.tensor(edge_types, dtype=torch.long) # Original types
        logger.info("Full graph HeteroData object created.")

    class _RGCN(torch.nn.Module):
        def __init__(self, in_c, hid_c, out_c, num_rels_in_subgraph):
            super().__init__()
            # num_rels_in_subgraph can include original + inverse types
            self.num_rels_actual = max(1, num_rels_in_subgraph)
            self.conv1 = RGCNConv(in_c, hid_c, self.num_rels_actual)
            self.conv2 = RGCNConv(hid_c, out_c, self.num_rels_actual)
            self.out_channels_final = out_c # Store for empty tensor creation

        def forward(self, x, edge_index, edge_type):
            if x is None or x.numel() == 0:
                logger.warning("RGCN: Input node features 'x' are empty.")
                return torch.empty(0, self.out_channels_final, device=x.device if x is not None else 'cpu')
            if edge_index is None or edge_index.numel() == 0 or edge_type is None or edge_type.numel() == 0:
                logger.warning("RGCN: Edge index or edge types are empty. Passing through input features.")
                # If model output dim is different from input, need a projection or return zeros
                if x.size(1) != self.out_channels_final:
                     # This case needs careful handling; for now, returning input if dims match, else error/zeros
                     logger.error(f"RGCN input dim {x.size(1)} != output dim {self.out_channels_final} and no edges to convolve.")
                     return torch.zeros(x.size(0), self.out_channels_final, device=x.device) # Or raise error
                return x # Pass through if no edges and dims match
            
            x = F.relu(self.conv1(x, edge_index, edge_type))
            return self.conv2(x, edge_index, edge_type)

    def _bfs_path_rels(self, adj_list_subgraph, seed_local_idx, target_local_idx):
        # adj_list_subgraph: list of lists, where adj_list_subgraph[u] = [(v1, rel_type1), (v2, rel_type2)...]
        # rel_type here can be original or inverse type ID
        # ... (BFS logic from your previous version of PathRGCNRetriever - it reconstructs one path) ...
        queue_bfs = deque([seed_local_idx])
        parent_map = {seed_local_idx: None}
        
        path_found_for_target = False
        while queue_bfs:
            u = queue_bfs.popleft()
            if u == target_local_idx:
                path_found_for_target = True
                break
            if u < len(adj_list_subgraph):
                for v_local, r_type_in_subgraph in adj_list_subgraph[u]:
                    if v_local not in parent_map:
                        parent_map[v_local] = (u, r_type_in_subgraph)
                        queue_bfs.append(v_local)
            else: logger.warning(f"BFS: Node index {u} out of bounds for adj_list of length {len(adj_list_subgraph)}")

        final_path_rel_types = [] # Will store relation types (can be original or inverse)
        final_path_nodes_local = [] # Local indices in subgraph

        if not path_found_for_target or target_local_idx not in parent_map or parent_map[target_local_idx] is None:
            return [], []

        curr = target_local_idx
        while curr != seed_local_idx and parent_map.get(curr) is not None:
            p_local, r_type = parent_map[curr]
            final_path_rel_types.append(r_type)
            final_path_nodes_local.append(curr)
            curr = p_local
        
        return list(reversed(final_path_rel_types)), list(reversed(final_path_nodes_local))

    def extract_entities(self, query: str):
        """
        Extract entities from the query using NER
        """
        doc = self.nlp(query)
        entities = []
        for ent in doc.ents:
            node_names = self.entity2id.keys()
            # Fuzzy match the entity text to the node names
            best_match, score, _ = process.extractOne(ent.text, node_names)
            if score > 80 and best_match not in entities: # Adjust threshold as needed
                entities.append(best_match)
        
        
        return entities
        

    def retrieve_relevant_paths(self, query: str, seed_entity_value: str,
                                k_hops: int = 2, num_epochs: int = 7, top_n_results: int = 3):
        logger.info(f"Retrieving paths for query: '{query[:50]}...', seed: '{seed_entity_value}', k={k_hops}")
        
        q_emb_np = self.embedding_model.encode(query)
        if q_emb_np.ndim == 2 and q_emb_np.shape[0] == 1: q_emb_np = q_emb_np[0] # Ensure 1D
        q_emb = torch.tensor(q_emb_np, dtype=torch.float).unsqueeze(0) # [1, D]
        q_emb_norm = F.normalize(q_emb, dim=1)

        if seed_entity_value not in self.entity2id:
            logger.error(f"Seed entity '{seed_entity_value}' not found in knowledge graph.")
            return []
        seed_id_global = self.entity2id[seed_entity_value]

        # --- Use your bidirectional subgraph extraction ---
        sub_data, mapping_subset_orig_to_local, subset_original_indices = extract_k_hop_subgraph_bidirectional(
            seed_id=seed_id_global,
            K=k_hops,
            data=self.full_graph_data, # Pass the full graph
            num_rels_total=self.num_original_relations # Number of ORIGINAL relation types
        )
        
        if sub_data['entity'].x.numel() == 0 or subset_original_indices.numel() == 0:
             logger.warning(f"Bidirectional subgraph for '{seed_entity_value}' is empty.")
             return []

        # Find local ID of the seed in the subgraph
        # subset_original_indices[local_idx] = global_idx
        # We need local_idx where subset_original_indices[local_idx] == seed_id_global
        seed_id_local_candidates = (subset_original_indices == seed_id_global).nonzero(as_tuple=True)[0]
        if seed_id_local_candidates.numel() == 0:
            logger.error(f"Seed entity '{seed_entity_value}' (ID {seed_id_global}) not found in its own {k_hops}-hop bidirectional subgraph. This is unexpected.")
            return []
        seed_id_local = seed_id_local_candidates[0].item()
        
        logger.info(f"Bidirectional subgraph: {sub_data['entity'].x.size(0)} nodes, "
                    f"{sub_data['entity','to','entity'].edge_index.size(1)} edges. "
                    f"Seed '{seed_entity_value}' (global {seed_id_global}) is local ID {seed_id_local}.")

        x_sub = sub_data['entity'].x
        edge_index_sub = sub_data['entity','to','entity'].edge_index
        edge_type_sub = sub_data['entity','to','entity'].edge_type # These types include original + inverse

        if x_sub.numel() == 0: return []
        
        # RGCN parameters
        # Output dimension of RGCN should match self.embedding_dim for the combination step
        rgcn_out_c = self.embedding_dim
        rgcn_model = self._RGCN(
            in_c=self.embedding_dim,
            hid_c=128, # As in your script
            out_c=rgcn_out_c, # Match embedding_dim for combination with rel_embs
            num_rels_in_subgraph=(int(edge_type_sub.max().item()) + 1) if edge_type_sub.numel() > 0 else 1
        )
        optimizer = torch.optim.Adam(rgcn_model.parameters(), lr=0.0001)

        if edge_index_sub.numel() > 0 and edge_type_sub.numel() > 0: # Only train if there are edges
            logger.info(f"Training RGCN on subgraph for {num_epochs} epochs...")
            rgcn_model.train()
            for epoch in range(1, num_epochs + 1):
                optimizer.zero_grad()
                emb_sub = rgcn_model(x_sub, edge_index_sub, edge_type_sub)
                if emb_sub.numel() > 0 :
                    src_sub, dst_sub = edge_index_sub
                    if src_sub.max() < emb_sub.size(0) and dst_sub.max() < emb_sub.size(0):
                        loss = F.mse_loss(emb_sub[src_sub], emb_sub[dst_sub])
                        loss.backward()
                        optimizer.step()
                        if epoch % 5 == 0 or epoch == num_epochs: logger.debug(f"RGCN Epoch {epoch:02d} — Loss: {loss.item():.4f}")
                    else: logger.warning(f"RGCN Epoch {epoch:02d} - Edge indices out of bounds. Skipping loss."); break
                else: logger.debug(f"RGCN Epoch {epoch:02d} - No embeddings from RGCN. Skipping training step."); break
        else:
            logger.warning("Subgraph has no edges or edge types. Skipping RGCN training.")

        logger.info("Evaluating paths in subgraph...")
        rgcn_model.eval()
        with torch.no_grad():
            node_embs_from_rgcn = rgcn_model(x_sub, edge_index_sub, edge_type_sub)
        
        if node_embs_from_rgcn.numel() == 0 and x_sub.numel() > 0 :
             node_embs_from_rgcn = x_sub
             logger.warning("RGCN output empty, falling back to initial subgraph node embeddings.")
        elif node_embs_from_rgcn.numel() == 0 and x_sub.numel() == 0: return []
        
        node_embs_from_rgcn_norm = F.normalize(node_embs_from_rgcn, dim=1)

        N_sub = node_embs_from_rgcn_norm.size(0)
        adj_list_sub = [[] for _ in range(N_sub)]
        src_sub_list, dst_sub_list = edge_index_sub
        for e_idx, (u_local, v_local) in enumerate(zip(src_sub_list.tolist(), dst_sub_list.tolist())):
            r_type_in_subgraph = edge_type_sub[e_idx].item() # This can be original or inverse
            adj_list_sub[u_local].append((v_local, r_type_in_subgraph))
            # Your BFS in the script implies an undirected graph for pathfinding, so add reverse edges if not already present
            # The bidirectional subgraph already contains both directions, so this might not be needed if BFS explores it
            # For simplicity, let's assume adj_list_sub from bidirectional edges is sufficient for BFS to find paths.
            # If your original script added inverse edges for BFS on a directed subgraph, this is different.
            # The extract_k_hop_subgraph_bidirectional handles the bidirectionality for k-hop search.
            # For BFS on the *resulting* subgraph, adj_list should represent its actual directed edges.
            # Let's build adj_list_sub from the directed edges of sub_data:
        adj_list_sub = [[] for _ in range(N_sub)] # Re-init for directed
        for i in range(edge_index_sub.size(1)):
            u, v = edge_index_sub[0, i].item(), edge_index_sub[1, i].item()
            r_type = edge_type_sub[i].item()
            adj_list_sub[u].append((v, r_type))


        scores_and_paths = []
        for target_node_local_idx in range(N_sub):
            # Path from seed_id_local to target_node_local_idx
            path_rel_types, path_nodes_local_indices = self._bfs_path_rels(adj_list_sub, seed_id_local, target_node_local_idx)
            
            h_target_node_sub = node_embs_from_rgcn_norm[target_node_local_idx]
            h_rel_path_avg = torch.zeros(self.embedding_dim, device=h_target_node_sub.device) # Default

            if path_rel_types:
                valid_rel_embeddings_on_path = []
                for r_type in path_rel_types:
                    if 0 <= r_type < self.num_original_relations: # Original relation
                        valid_rel_embeddings_on_path.append(self.rel_embs_full[r_type])
                    elif self.num_original_relations <= r_type < 2 * self.num_original_relations: # Inverse relation
                        original_r_type = r_type - self.num_original_relations
                        # For inverse, you might negate or use a learned transformation.
                        # Simplest: use original embedding, or mark as inverse.
                        # Your script uses self.rel_embs_full[rel_ids] - this would fail for inverse if rel_embs_full is only original.
                        # For now, let's assume rel_embs_full needs to be extended or inverse rels handled.
                        # Let's use original embeddings for inverse relations for simplicity in this step.
                        valid_rel_embeddings_on_path.append(self.rel_embs_full[original_r_type]) # Or a specific embedding for inverse
                    # Else: unknown relation type, skip or handle
                
                if valid_rel_embeddings_on_path:
                    rel_vecs_on_path = torch.stack(valid_rel_embeddings_on_path)
                    h_rel_path_avg = F.normalize(rel_vecs_on_path.mean(dim=0), dim=0)
            
            h_combined = F.normalize(0.5 * h_target_node_sub + 0.5 * h_rel_path_avg, dim=0)
            similarity_score = F.cosine_similarity(q_emb_norm.squeeze(0), h_combined, dim=0).item()
            
            # Format path for output
            current_path_display_nodes = [seed_entity_value]
            for node_local_idx_on_path in path_nodes_local_indices:
                current_path_display_nodes.append(self.id2entity[subset_original_indices[node_local_idx_on_path].item()])
            
            path_relation_labels_display = []
            for r_type in path_rel_types:
                if 0 <= r_type < self.num_original_relations:
                    path_relation_labels_display.append(self.relation_texts[r_type])
                elif self.num_original_relations <= r_type < 2 * self.num_original_relations:
                    path_relation_labels_display.append(f"{self.relation_texts[r_type - self.num_original_relations]}_inverse")
                else:
                    path_relation_labels_display.append(f"UNKNOWN_REL_TYPE_{r_type}")

            target_node_global_name = self.id2entity[subset_original_indices[target_node_local_idx].item()]

            scores_and_paths.append({
                "score": similarity_score,
                "target_node_name": target_node_global_name,
                "path_relation_labels": path_relation_labels_display, # Display labels
                "path_nodes_display": current_path_display_nodes, # Full path of node names for display
                "path_raw_rel_types": path_rel_types, # Original/inverse types
                "path_raw_nodes_local": path_nodes_local_indices # Local indices in subgraph
            })

        scores_and_paths.sort(key=lambda x: x["score"], reverse=True)
        
        output_paths_formatted = []
        for item in scores_and_paths[:top_n_results]:
            path_str = f"({item['path_nodes_display'][0]})" # Starts with seed
            for i, rel_label in enumerate(item["path_relation_labels"]):
                node_name_in_path = item['path_nodes_display'][i+1] # Node after this relation
                path_str += f" --[{rel_label}]--> ({node_name_in_path})"
            
            final_path_str_with_score = f"{path_str} (Score: {item['score']:.4f})"
            item["path_string_formatted"] = final_path_str_with_score # Add formatted string
            output_paths_formatted.append(item["path_string_formatted"])
            logger.info(final_path_str_with_score)

        return output_paths_formatted
    

# # Example usage:
# juan = PathRGCNRetriever()
# juan.ingest_data([
#     {"from": {"value": "Wordscapes"}, "to": {"value": "B"}, "label": "relation1"},
#     {"from": {"value": "B"}, "to": {"value": "C"}, "label": "relation2"},
#     {"from": {"value": "C"}, "to": {"value": "D"}, "label": "relation3"},
#     {"from": {"value": "A"}, "to": {"value": "D"}, "label": "relation4"}
# ])
# # print(juan.retrieve_relevant_paths("What is the relation between A and D?", "A", k_hops=2, num_epochs=5, top_n_results=3))
# # # Output: List of paths with their scores
# # print(juan.extract_entities("What is the relation between  and D?"))
# print(juan.extract_entities("Consider Wordscapes's privacy policy; does the app show targeted advertisements?"))
