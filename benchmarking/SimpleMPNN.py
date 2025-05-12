
import json

from sentence_transformers import SentenceTransformer
import torch
from torch_geometric.data import Data
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class SimpleMPNN(nn.Module):
    def __init__(self, aggregation='mean', num_layers=2, alpha=0.5):
        super(SimpleMPNN, self).__init__()
        assert aggregation in ['mean', 'sum'], "Only 'mean' or 'sum' aggregation supported"
        self.aggregation = aggregation
        self.num_layers = num_layers
        self.alpha = alpha  # peso del embedding original

    def forward(self, data: Data, question_emb):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        num_nodes = x.size(0)

        edge_sim = torch.matmul(
            F.normalize(edge_attr, p=2, dim=1),        # [num_edges, dim]
            F.normalize(question_emb, p=2, dim=0)      # [dim]
        )
        for i in range(edge_sim.size(0)):
            print(edge_sim[i],': ',data.edge_attr_texts[i])

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



