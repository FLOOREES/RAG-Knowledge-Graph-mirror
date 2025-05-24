
# graph = [
#     {
#         "relation": "OTHER",
#         "label": "net sales increased during",
#         "metadata": {
#             "paragraph_id": "122bb3251be909d96d571c8eb26bdb0d/f/122bb3251be909d96d571c8eb26bdb0d/43013-45520",
#             "source_start": 45360,
#             "source_end": 45368,
#             "to_start": 43069,
#             "to_end": 43090,
#             "data_augmentation_task_id": "legal-graph-operation"
#         },
#         "from": {
#             "value": "Americas",
#             "type": "entity",
#             "group": "COURT"
#         },
#         "to": {
#             "value": "third quarter of 2022",
#             "type": "entity",
#             "group": "DATE"
#         }
#     },
#     {
#         "relation": "OTHER",
#         "label": "net sales increased during",
#         "metadata": {
#             "paragraph_id": "122bb3251be909d96d571c8eb26bdb0d/f/122bb3251be909d96d571c8eb26bdb0d/43013-45520",
#             "source_start": 45360,
#             "source_end": 45368,
#             "to_start": 43107,
#             "to_end": 43128,
#             "data_augmentation_task_id": "legal-graph-operation"
#         },
#         "from": {
#             "value": "Americas",
#             "type": "entity",
#             "group": "COURT"
#         },
#         "to": {
#             "value": "third quarter of 2021",
#             "type": "entity",
#             "group": "DATE"
#         }
#     }]

# graph = [
#     {
#         "relation": "DISCOVERY",
#         "label": "discovered the function of",
#         "metadata": {},
#         "from": {"value": "Dr. Alice Carter", "type": "person", "group": "RESEARCHER"},
#         "to": {"value": "protein X", "type": "concept", "group": "BIOLOGY"}
#     },
#     {
#         "relation": "DATE",
#         "label": "in the year",
#         "metadata": {},
#         "from": {"value": "Dr. Alice Carter", "type": "person", "group": "RESEARCHER"},
#         "to": {"value": "2019", "type": "date", "group": "YEAR"}
#     },
#     {
#         "relation": "WORKED_ON",
#         "label": "conducted research at",
#         "metadata": {},
#         "from": {"value": "Dr. Alice Carter", "type": "person", "group": "RESEARCHER"},
#         "to": {"value": "Cambridge Institute of Genetics", "type": "organization", "group": "UNIVERSITY"}
#     },
#     {
#         "relation": "IMPACT",
#         "label": "led to treatment for",
#         "metadata": {},
#         "from": {"value": "protein X", "type": "concept", "group": "BIOLOGY"},
#         "to": {"value": "rare blood disease", "type": "condition", "group": "MEDICINE"}
#     },
#     {
#         "relation": "RELATED_TO",
#         "label": "similar to mechanism of",
#         "metadata": {},
#         "from": {"value": "protein X", "type": "concept", "group": "BIOLOGY"},
#         "to": {"value": "protein Y", "type": "concept", "group": "BIOLOGY"}
#     },
#     {
#         "relation": "DIAGNOSED_WITH",
#         "label": "was diagnosed with",
#         "metadata": {},
#         "from": {"value": "Patient 042", "type": "entity", "group": "PATIENT"},
#         "to": {"value": "rare blood disease", "type": "condition", "group": "MEDICINE"}
#     },
#     {
#         "relation": "DISCOVERY",
#         "label": "discovered mutation in",
#         "metadata": {},
#         "from": {"value": "Dr. Miguel Torres", "type": "person", "group": "RESEARCHER"},
#         "to": {"value": "gene ABC1", "type": "concept", "group": "GENETICS"}
#     },
#     {
#         "relation": "IMPACT",
#         "label": "linked to increased risk of",
#         "metadata": {},
#         "from": {"value": "gene ABC1", "type": "concept", "group": "GENETICS"},
#         "to": {"value": "colon cancer", "type": "condition", "group": "MEDICINE"}
#     },
#     {
#         "relation": "WORKED_ON",
#         "label": "research conducted at",
#         "metadata": {},
#         "from": {"value": "Dr. Miguel Torres", "type": "person", "group": "RESEARCHER"},
#         "to": {"value": "University of Madrid", "type": "organization", "group": "UNIVERSITY"}
#     },
#     {
#         "relation": "DATE",
#         "label": "discovered in",
#         "metadata": {},
#         "from": {"value": "gene ABC1", "type": "concept", "group": "GENETICS"},
#         "to": {"value": "2020", "type": "date", "group": "YEAR"}
#     },
#     {
#         "relation": "RELATED_TO",
#         "label": "has similar mutation as",
#         "metadata": {},
#         "from": {"value": "gene ABC1", "type": "concept", "group": "GENETICS"},
#         "to": {"value": "gene XYZ3", "type": "concept", "group": "GENETICS"}
#     },
#     {
#         "relation": "STUDIED_BY",
#         "label": "was studied by",
#         "metadata": {},
#         "from": {"value": "gene XYZ3", "type": "concept", "group": "GENETICS"},
#         "to": {"value": "Dr. Nina Kapoor", "type": "person", "group": "RESEARCHER"}
#     },
#     {
#         "relation": "WORKED_ON",
#         "label": "conducted clinical trials at",
#         "metadata": {},
#         "from": {"value": "Dr. Nina Kapoor", "type": "person", "group": "RESEARCHER"},
#         "to": {"value": "Stanford Medical Center", "type": "organization", "group": "HOSPITAL"}
#     },
#     {
#         "relation": "TREATED_WITH",
#         "label": "treated with",
#         "metadata": {},
#         "from": {"value": "Patient 077", "type": "entity", "group": "PATIENT"},
#         "to": {"value": "experimental therapy B", "type": "treatment", "group": "MEDICINE"}
#     },
#     {
#         "relation": "TARGETS",
#         "label": "targets",
#         "metadata": {},
#         "from": {"value": "experimental therapy B", "type": "treatment", "group": "MEDICINE"},
#         "to": {"value": "gene XYZ3", "type": "concept", "group": "GENETICS"}
#     }
# ]


import json

from sentence_transformers import SentenceTransformer
import torch
from torch_geometric.data import Data, HeteroData
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

with open('C:/Users/alexn/Desktop/Uni/3r Q2/PIA/RAG-Knowledge-Graph/notebooks/relations.json', 'r') as file:
    graph = json.load(file)

def graph_to_gnn_data(graph, model):
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
                node_idx_to_info[node_id] = node  # ← guardamos la info del nodo
                node_id += 1

    node_embeddings = model.encode(node_texts, convert_to_tensor=True)

    edge_index = []
    edge_attr_texts = []
    for edge in graph:
        src_key = (edge["from"]["value"], edge["from"]["type"], edge["from"]["group"])
        dst_key = (edge["to"]["value"], edge["to"]["type"], edge["to"]["group"])
        src = node_dict[src_key]
        dst = node_dict[dst_key]
        edge_index.append([src, dst])
        edge_attr_texts.append(edge["label"])

    edge_attr = model.encode(edge_attr_texts, convert_to_tensor=True)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    data = HeteroData(x=node_embeddings, edge_index=edge_index, edge_attr=edge_attr)
    data.node_idx_to_info = node_idx_to_info  # ← lo añadimos al objeto Data
    data.edge_attr_texts = edge_attr_texts
    return data


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
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        num_nodes = x.size(0)

        edge_sim = torch.matmul(
            F.normalize(edge_attr, p=2, dim=1),        # [num_edges, dim]
            F.normalize(question_emb, p=2, dim=0)      # [dim]
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


def get_relevant_nodes(output, question_emb, graph_data, top_k=3):
    output_norm = F.normalize(output, p=2, dim=1)
    question_emb_norm = F.normalize(question_emb, p=2, dim=0)

    similarities = torch.matmul(output_norm, question_emb_norm)
    top_k_indices = torch.topk(similarities, k=top_k).indices.tolist()

    top_nodes_info = []
    for idx in top_k_indices:
        node_info = graph_data.node_idx_to_info[idx]
        top_nodes_info.append((idx, node_info, similarities[idx].item()))

    return top_nodes_info



# Ejemplo de uso
model = SentenceTransformer('all-MiniLM-L6-v2')
data = graph_to_gnn_data(graph, model)

question = "What is the net sales increase during the third quarter of 2022?"
question = "What was discovered by Dr. Alice Carter and what was its medical impact?"
question = "Who is Luca Maestri and what is his role in the company?"

question_emb = model.encode(question, convert_to_tensor=True)

#gnn = GraphGATNet(in_channels=data.x.shape[1], hidden_channels=64, out_channels=data.x.shape[1])
gnn = SimpleMPNN(aggregation='mean', num_layers=2)
gnn.eval()

output = gnn(data,question_emb)
top_nodes = get_relevant_nodes(output, question_emb, data, top_k=5)

for idx, node_info, score in top_nodes:
    print(f"Node {idx}: {node_info['value']} (type: {node_info['type']}) | similarity: {score:.4f}")