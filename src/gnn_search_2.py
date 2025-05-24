import json
from sentence_transformers import SentenceTransformer
import torch
from torch_geometric.data import Data, HeteroData
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from rapidfuzz import process
import spacy
from torch_geometric.utils import k_hop_subgraph
from collections import deque
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

import logging
logger = logging.getLogger(__name__)

class GraphGATNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=1, dropout=0.2):
        super(GraphGATNet, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def forward(self, data: HeteroData):
        x, edge_index = data.x, data.edge_index

        # Primer layer de GAT con activación ReLU
        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Segundo layer de GAT sin activación (para tareas de clasificación, se aplica fuera)
        x = self.gat2(x, edge_index)

        return x
    
class SimpleMPNN(nn.Module):
    def __init__(self, aggregation='mean', num_layers=2, alpha=0.5):
        super(SimpleMPNN, self).__init__()
        assert aggregation in ['mean', 'sum'], "Only 'mean' or 'sum' aggregation supported"
        self.aggregation = aggregation
        self.num_layers = num_layers
        self.alpha = alpha  # peso del embedding original

    def forward(self, data: HeteroData, question_emb):
        x, edge_index, edge_attr = data['entity'].x, data['entity', 'to', 'entity'].edge_index, data['entity', 'to', 'entity'].edge_attr
        num_nodes = x.size(0)

        edge_sim = torch.matmul(
            F.normalize(edge_attr, p=2, dim=1),        # [num_edges, dim]
            F.normalize(question_emb.squeeze(0), p=2, dim=0)      # [dim]
        )
        # for i in range(edge_sim.size(0)):
        #     print(edge_sim[i],': ',data.edge_attr_texts[i])

        for _ in range(self.num_layers):
            x_new = torch.zeros_like(x)

            # Agregación de mensajes
            for i in range(edge_index.size(1)):
                src = edge_index[0, i]
                dst = edge_index[1, i]
                x_new[dst] += x[src] * edge_sim[i]

            if self.aggregation == 'mean':
                # Calcular el número de vecinos para el promedio
                degree = torch.zeros(num_nodes, device=x.device)
                for i in range(edge_index.size(1)):
                    dst = edge_index[1, i]
                    degree[dst] += 1
                degree = degree.clamp(min=1).unsqueeze(-1)
                x_new = x_new / degree

            # Soft update entre embeddings antiguos y nuevos
            x = self.alpha * x + (1 - self.alpha) * x_new

        return x


# def get_relevant_nodes(output, question_emb, graph_data, top_k=3):
#     output_norm = F.normalize(output, p=2, dim=1)
#     question_emb_norm = F.normalize(question_emb, p=2, dim=0)

#     similarities = torch.matmul(output_norm, question_emb_norm)
#     top_k_indices = torch.topk(similarities, k=top_k).indices.tolist()

#     top_nodes_info = []
#     for idx in top_k_indices:
#         node_info = graph_data.node_idx_to_info[idx]
#         top_nodes_info.append((idx, node_info, similarities[idx].item()))

#     return top_nodes_info

class gnn_search:
    def __init__(self, embedding_model: SentenceTransformer, gnn_model):
        self.nlp = spacy.load("en_core_web_sm")
        self.embedding_model = embedding_model
        self.gnn_model = gnn_model
        self.entity2id = {}
        self.relation2id = {}
        self.relation_metadata = {}
        self.id2entity = {}
        self.id2relation = {}
        self.num_original_relations = 0
        self.relation_texts_full = []
        self.entity_embeddings_full = None
        self.rel_embs_full = None
        self.embedding_dim = 0
        self.full_graph_data = HeteroData()

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

    def extract_k_hop_subgraph_bidirectional(self,seed_id,
                                            K: int,
                                            data: HeteroData,
                                            num_rels_total: int):
        """
        Extrae el subgrafo K-hop considerando aristas en ambas direcciones,
        pero SIN modificar `data` permanentemente.

        - seed_id: índice del nodo semilla (global).
        - K:       número de saltos.
        - data:    tu HeteroData original.
        - num_rels_total: número de tipos de relación ORIGINALES en todo el KG.
        """

        # 1) Sacamos los tensores originales
        edge_index = data['entity','to','entity'].edge_index      # [2, E]
        edge_type  = data['entity','to','entity'].edge_type       # [E]
        edge_attr  = data['entity','to','entity'].edge_attr       # [E, D] (D = embedding dim)

        # 2) Creamos las aristas inversas
        #    Invertimos filas de edge_index y desplazamos el ID de relación
        inv_edge_index = edge_index.flip(0)                       # [2, E]
        inv_edge_type  = edge_type + num_rels_total               # [E]
        inv_edge_attr  = edge_attr                                # [E, D] (usamos el mismo embedding para la inversa)

        # 3) Combinamos originales + inversas
        comb_edge_index = torch.cat([edge_index,    inv_edge_index], dim=1)  # [2, 2E]
        comb_edge_type  = torch.cat([edge_type,     inv_edge_type],  dim=0)  # [2E]
        comb_edge_attr  = torch.cat([edge_attr,     inv_edge_attr],  dim=0)  # [2E, D]

        # 4) Usamos k_hop_subgraph sobre este grafo bidireccional temporal
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx      = seed_id,
            num_hops      = K,
            edge_index    = comb_edge_index,
            relabel_nodes = True,
            num_nodes     = data['entity'].x.size(0)
        )

        # 5) Filtramos los tipos de arista y atributos con la máscara
        sub_edge_type = comb_edge_type[edge_mask]
        sub_edge_attr = comb_edge_attr[edge_mask]

        # 6) Construimos y devolvemos el subgrafo
        sub_data = HeteroData()
        sub_data['entity'].x = data['entity'].x[subset]
        sub_data['entity','to','entity'].edge_index = sub_edge_index
        sub_data['entity','to','entity'].edge_type  = sub_edge_type
        sub_data['entity','to','entity'].edge_attr  = sub_edge_attr

        return sub_data, mapping, subset

    # def graph_to_gnn_data(graph, model):
    #     node_dict = {}
    #     node_id = 0
    #     node_texts = []
    #     node_idx_to_info = {}  # ← añadimos esto

    #     for edge in graph:
    #         for node in [edge["from"], edge["to"]]:
    #             key = (node["value"], node["type"], node["group"])
    #             if key not in node_dict:
    #                 node_dict[key] = node_id
    #                 node_texts.append(node["value"])
    #                 node_idx_to_info[node_id] = node  # ← guardamos la info del nodo
    #                 node_id += 1

    #     node_embeddings = model.encode(node_texts, convert_to_tensor=True)

    #     edge_index = []
    #     edge_attr_texts = []
    #     edge_types = []
    #     for edge in graph:
    #         src_key = (edge["from"]["value"], edge["from"]["type"], edge["from"]["group"])
    #         dst_key = (edge["to"]["value"], edge["to"]["type"], edge["to"]["group"])
    #         src = node_dict[src_key]
    #         dst = node_dict[dst_key]
    #         edge_index.append([src, dst])
    #         edge_attr_texts.append(edge["label"])
    #         edge_types.append(edge["label"])

    #     edge_attr = model.encode(edge_attr_texts, convert_to_tensor=True)
    #     edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    #     data = HeteroData(x=node_embeddings, edge_index=edge_index, edge_attr=edge_attr)
    #     data.edge_types = edge_types
    #     data.node_idx_to_info = node_idx_to_info  # ← lo añadimos al objeto Data
    #     data.edge_attr_texts = edge_attr_texts


    #     entity2id = {}
    #     relation2id = {} # Original relation labels to ID (0 to N-1)
    #     edges = []
    #     edge_types_original = [] # Store original edge type IDs (0 to N-1)

    #     for idx, rel in enumerate(graph):
    #         src_val, dst_val, label_val = rel['from']['value'], rel['to']['value'], rel['label']

    #         for node in (src_val, dst_val):
    #             if node not in entity2id:
    #                 entity2id[node] = len(entity2id)
    #         if label_val not in relation2id:
    #             relation2id[label_val] = len(relation2id)

    #         edges.append((entity2id[src_val], entity2id[dst_val]))
    #         edge_types_original.append(relation2id[label_val])
    #     return data, entity2id, relation2id

    def ingest_data(self, relations: List[Dict[str, Any]]):
        print(f"Ingesting {len(relations)} relations into the knowledge graph.")
        self.entity2id = {}
        self.relation2id = {} # Original relation labels to ID (0 to N-1)
        edges = []
        edge_types_original = [] # Store original edge type IDs (0 to N-1)
        edges_attr = []

        for idx, rel in enumerate(relations):
            src_val, dst_val, label_val = rel['from']['value'], rel['to']['value'], rel['label']
            self.relation_metadata[idx] = rel

            for node in (src_val, dst_val):
                if node not in self.entity2id:
                    self.entity2id[node] = len(self.entity2id)
            if label_val not in self.relation2id:
                self.relation2id[label_val] = len(self.relation2id)

            edges_attr.append(rel['label'])  # Store relation labels for edge attributes
            edges.append((self.entity2id[src_val], self.entity2id[dst_val]))
            edge_types_original.append(self.relation2id[label_val])
        
        edges_attr = self.embedding_model.encode(edges_attr, convert_to_tensor=True)
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
        self.full_graph_data['entity', 'to', 'entity'].edge_attr = edges_attr
        logger.info("Full graph HeteroData object created.")

    def gnn_message_passing(self, sub_data, question_emb, alpha=0.5):
        gnn.eval()
        return self.gnn_model(sub_data, question_emb)

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

    def retrieve_relevant_paths(self,sub_data,subset_original_indices, q_emb_norm, entities, top_n_results: int = 5,
                                alpha_target_node: float = 0.5, alpha_path_relations: float = 0.5
                               ) -> List[Dict[str, Any]]:
        
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
        for seed_id_local in entities:
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


                # If path is just to the seed node itself
                if not path_relation_labels_display and len(path_nodes_display_names) == 1:
                    path_str = f"({path_nodes_display_names[0]})" # Just the seed node

                scores_and_paths_data.append({
                    "score": similarity_score,
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

        return scores_and_paths_data[:top_n_results]
    
    def retrieve(self, question: str, k_hops: int = 2, top_n_results: int = 5) -> List[Dict[str, Any]]:
        if not self.entity2id:
            logger.error("No entities loaded. Please call ingest_data first.")
            return []

        # Encode the question
        question_emb = self.embedding_model.encode(question, convert_to_tensor=True)
        question_emb = torch.tensor(question_emb, dtype=torch.float).unsqueeze(0)
        question_emb_norm = F.normalize(question_emb, dim=1)
        # Extract entities from the question
        entities = self.extract_entities(question)
        if not entities:
            logger.warning("No entities extracted from the question.")
            return []
        # Convert entity names to IDs
        entity_ids = [self.entity2id[ent] for ent in entities]
        if not entity_ids:
            logger.warning("No valid entities found in the graph for the extracted entities.")
            return []
        # Extract k-hop subgraph around the entities
        sub_graph, mapping, subset = self.extract_k_hop_subgraph_bidirectional(
            seed_id=entity_ids,  # Use the first entity as the seed
            K=k_hops,
            data=self.full_graph_data,
            num_rels_total=self.num_original_relations
        )
        # Run the GNN model on the subgraph
        self.gnn_model.eval()
        output = self.gnn_model(sub_graph, question_emb)

        sub_graph['entity'].x = output  # Update subgraph with GNN output embeddings

        seed_local_indices = mapping.tolist()

        # Retrieve relevant paths based on the GNN output
        retrieved_paths = self.retrieve_relevant_paths(sub_graph,subset,question_emb_norm, seed_local_indices)

        return retrieved_paths


# Ejemplo de uso
with open('C:/Users/alexn/Desktop/Uni/3r Q2/PIA/RAG-Knowledge-Graph/notebooks/relations.json', 'r') as file:
    graph = json.load(file)

question = "What is the relationship between David Zinsner and Jen-Hsun Huang?"

gnn = gnn_search(
    embedding_model=SentenceTransformer('all-MiniLM-L6-v2'),
    gnn_model=SimpleMPNN(aggregation='mean', num_layers=5)
)
gnn.ingest_data(graph)
# Example usage
retrieved_paths = gnn.retrieve(question, k_hops=5, top_n_results=5)

for path_data in retrieved_paths:
            print(path_data["path_string_formatted"])


# model = SentenceTransformer('all-MiniLM-L6-v2')
# nlp = spacy.load("en_core_web_sm")


# data, entity2id, relation2id = graph_to_gnn_data(graph, model)


# entities = extract_entities(nlp, question, entity2id)
# print(entities)
# sub_graph, mapping, subset  = extract_k_hop_subgraph_bidirectional([entity2id[ent] for ent in entities], 5, data, len(relation2id))


# question_emb = model.encode(question, convert_to_tensor=True)

# #gnn = GraphGATNet(in_channels=data.x.shape[1], hidden_channels=64, out_channels=data.x.shape[1])
# gnn = SimpleMPNN(aggregation='mean', num_layers=5)
# gnn.eval()

# output = gnn(sub_graph,question_emb)

# retrieved_paths = retrieve_relevant_paths(
#     data=sub_graph,
#     entities=entities,
#     q_emb=question_emb
# )

# for path_data in retrieved_paths:
#             print(path_data["path_string_formatted"])

# top_nodes = get_relevant_nodes(output, question_emb, data, top_k=5)

# for idx, node_info, score in top_nodes:
#     print(f"Node {idx}: {node_info['value']} (type: {node_info['type']}) | similarity: {score:.4f}")