from .SimpleMPNN import SimpleMPNN

from sentence_transformers import SentenceTransformer
import torch
from torch_geometric.data import Data
import torch.nn.functional as F


class CustomWrapper:
    """
    Wrapper for nuclia RAG API access
    """

    def __init__(self):
        self.gnn = SimpleMPNN(aggregation='mean', num_layers=5)
        self.emb_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.current_data = {}

    def ingest_data(self,subgraph):
        self.current_data =  self.graph_to_gnn_data(subgraph) # Encode subgraph

    def input_to_entities(self,question:str):

        if not self.current_data:
            raise ValueError('Please ingest data before doing a query')

        # Preprocessing
        question_emb = self.emb_model.encode(question, convert_to_tensor=True) # Encode question

        # Get relevant nodes according to GNN
        output = self.gnn(self.current_data,question_emb)
        relevant_nodes = self.get_relevant_nodes(output,question_emb,self.current_data,15)

        return relevant_nodes

    def ingest_context(self,context:str):
        raise NotImplemented

    def return_graph(self):
        raise NotImplemented

    def generate_response(self,query:str):
        formatted_prompt = self.qa_prompt.format(query)

    def get_relevant_nodes(self,output,question_emb, graph_data, top_k=3):
        output_norm = F.normalize(output, p=2, dim=1)
        question_emb_norm = F.normalize(question_emb, p=2, dim=0)

        similarities = torch.matmul(output_norm, question_emb_norm)
        top_k_indices = torch.topk(similarities, k=top_k).indices.tolist()

        top_nodes_info = []
        for idx in top_k_indices:
            node_info = graph_data.node_idx_to_info[idx]
            top_nodes_info.append((idx, node_info, similarities[idx].item()))

        return top_nodes_info

    def graph_to_gnn_data(self,graph):
        node_dict = {}
        node_id = 0
        node_texts = []
        node_idx_to_info = {}  # ← añadimos esto

        for edge in graph:
            for node in [edge["from"], edge["to"]]:
                key = (node["value"], node["type"], node["group"])
                if key not in node_dict:
                    node_dict[key] = node_id
                    node_texts.append(node["value"])
                    node_idx_to_info[node_id] = node
                    node_id += 1

        node_embeddings = self.emb_model.encode(node_texts, convert_to_tensor=True)

        edge_index = []
        edge_attr_texts = []
        for edge in graph:
            src_key = (edge["from"]["value"], edge["from"]["type"], edge["from"]["group"])
            dst_key = (edge["to"]["value"], edge["to"]["type"], edge["to"]["group"])
            src = node_dict[src_key]
            dst = node_dict[dst_key]
            edge_index.append([src, dst])
            edge_attr_texts.append(edge["label"])

        edge_attr = self.emb_model.encode(edge_attr_texts, convert_to_tensor=True)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        data = Data(x=node_embeddings, edge_index=edge_index, edge_attr=edge_attr)
        data.node_idx_to_info = node_idx_to_info
        data.edge_attr_texts = edge_attr_texts
        return data


class NucliaWrapper:
    """
    Wrapper for custom exploration method service
    """

    def __init__(self,qa_prompt:str):
        self.qa_prompt = qa_prompt

    def ingest_graph(self,graph):
        raise NotImplemented

    def _process_graph(self):
        raise NotImplemented

    def generate_response(self,query:str):
        formatted_prompt = self.qa_prompt.format(query)