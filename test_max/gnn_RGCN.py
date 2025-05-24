import json
import torch
import torch.nn.functional as F
import numpy as np
from collections import deque
from torch_geometric.data import HeteroData
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn import RGCNConv
from model2vec import StaticModel
from utils import extract_k_hop_subgraph_bidirectional

# 1. Carga del modelo de embeddings y definición de la query/semilla
embedding_model = StaticModel.from_pretrained("minishlab/potion-base-8M")
query      = "What is the relation between David Zinsner and VLSI Technology LLC?" #David Zinsner
seed_value = "David Zinsner" #Intel
q_emb      = torch.tensor(embedding_model.encode(query), dtype=torch.float).unsqueeze(0)

# 2. Carga del KG en JSON y construcción de mappings
with open('relations.json', 'r', encoding='utf-8') as f:
    relations = json.load(f)

entity2id = {}
relation2id = {}
edges = []
edge_types = []

for rel in relations:
    src, dst = rel['from']['value'], rel['to']['value']
    label    = rel['label']
    # Mapeo de entidades
    for node in (src, dst):
        if node not in entity2id:
            entity2id[node] = len(entity2id)
    # Mapeo de relaciones
    if label not in relation2id:
        relation2id[label] = len(relation2id)
    # Añadimos la arista
    edges.append((entity2id[src], entity2id[dst]))
    edge_types.append(relation2id[label])

# Inversos
id2entity     = {i: e for e, i in entity2id.items()}
relation_texts = [None] * len(relation2id)
for lab, idx in relation2id.items():
    relation_texts[idx] = lab

print(f"Entities: {len(entity2id)}, Relation types: {len(relation2id)}")

# 3. Generar embeddings de entidades y relaciones
entity_texts = [id2entity[i] for i in range(len(id2entity))]
entity_embeddings = torch.tensor(
    np.array([embedding_model.encode(text) for text in entity_texts]),
    dtype=torch.float
)

rel_embs_full = torch.stack([
    torch.tensor(embedding_model.encode(lbl), dtype=torch.float)
    for lbl in relation_texts
])  # [R_total, D]

# 4. Construcción del HeteroData completo
data = HeteroData()
data['entity'].x = entity_embeddings
data['entity', 'to', 'entity'].edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
data['entity', 'to', 'entity'].edge_type  = torch.tensor(edge_types, dtype=torch.long)

# 5. Extracción del subgrafo K-hop
def extract_k_hop_subgraph(seed_id: int, K: int, data: HeteroData):
    edge_index = data['entity','to','entity'].edge_index
    edge_type  = data['entity','to','entity'].edge_type
    num_nodes  = data['entity'].x.size(0)
    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx      = seed_id,
        num_hops      = K,
        edge_index    = edge_index,
        relabel_nodes = True,
        num_nodes     = num_nodes
    )
    sub_edge_type = edge_type[edge_mask]
    sub_data = HeteroData()
    sub_data['entity'].x = data['entity'].x[subset]
    sub_data['entity','to','entity'].edge_index = sub_edge_index
    sub_data['entity','to','entity'].edge_type  = sub_edge_type
    return sub_data, mapping, subset



# Selección de la semilla
if seed_value not in entity2id:
    raise KeyError(f"Entidad '{seed_value}' no encontrada.")
seed_id = entity2id[seed_value]
sub_data, seed_pos, subset = extract_k_hop_subgraph_bidirectional(seed_id, K=5, data=data,num_rels_total=len(relation2id))
print(f"Subgrafo: {sub_data['entity'].x.size(0)} nodos, "
      f"{sub_data['entity','to','entity'].edge_index.size(1)} aristas")

# 6. Definición y entrenamiento del RGCN
class RGCN(torch.nn.Module):
    def __init__(self, in_c, hid_c, out_c, num_rels):
        super().__init__()
        num_rels = max(1, num_rels)
        self.conv1 = RGCNConv(in_c, hid_c, num_rels)
        self.conv2 = RGCNConv(hid_c, out_c, num_rels)
    def forward(self, x, edge_index, edge_type):
        if edge_type.numel() == 0:
            return x
        x = F.relu(self.conv1(x, edge_index, edge_type))
        return self.conv2(x, edge_index, edge_type)

in_c     = sub_data['entity'].x.size(1)
num_rels = int(sub_data['entity','to','entity'].edge_type.max().item()) + 1 \
           if sub_data['entity','to','entity'].edge_type.numel()>0 else 1

model     = RGCN(in_c, hid_c=128, out_c=256, num_rels=num_rels)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

x          = sub_data['entity'].x
edge_index = sub_data['entity','to','entity'].edge_index
edge_type  = sub_data['entity','to','entity'].edge_type

model.train()
for epoch in range(1, 7):
    optimizer.zero_grad()
    emb = model(x, edge_index, edge_type)
    src, dst = edge_index
    loss = F.mse_loss(emb[src], emb[dst]) if edge_type.numel()>0 else torch.tensor(0.)
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        print(f"Epoch {epoch:02d} — Loss: {loss.item():.4f}")

# 7. Preparar ajacency con r_global
model.eval()
with torch.no_grad():
    node_embs_sub = model(x, edge_index, edge_type)  # [N_sub, D]
node_embs_sub = F.normalize(node_embs_sub, dim=1)
q_emb_norm    = F.normalize(q_emb, dim=1)

N_sub = node_embs_sub.size(0)
adj = [[] for _ in range(N_sub)]
src, dst = edge_index
for e_idx, (u, v) in enumerate(zip(src.tolist(), dst.tolist())):
    r_global = edge_type[e_idx].item()
    adj[u].append((v, r_global))
    adj[v].append((u, r_global))

# 8. Función BFS para extraer relaciones en el camino
def bfs_path_rels(seed, target):
    queue = deque([seed])
    parent = {seed: None}
    while queue:
        u = queue.popleft()
        if u == target: break
        for v, r in adj[u]:
            if v not in parent:
                parent[v] = (u, r)
                queue.append(v)
    if target not in parent:
        return [], []
    rels, nodes = [], []	
    cur = target
    while parent[cur] is not None:
        p, r = parent[cur]
        rels.append(r)
        nodes.append(cur)
        cur = p
    return list(reversed(rels)), list(reversed(nodes))

# 9. Scoring node+relation
scores = []
D = node_embs_sub.size(1)
for local_idx in range(N_sub):
    h_node = node_embs_sub[local_idx]
    rel_ids, nodes = bfs_path_rels(seed_pos, local_idx)
    if rel_ids:
        rel_vecs = rel_embs_full[rel_ids]    # [k, D]
        h_rel    = F.normalize(rel_vecs.mean(dim=0), dim=0)
    else:
        h_rel = torch.zeros(D, device=h_node.device)
    h_comb = F.normalize(0.5*h_node + 0.5*h_rel, dim=0)
    sim    = F.cosine_similarity(q_emb_norm, h_comb.unsqueeze(0)).item()
    scores.append((sim, local_idx, rel_ids, nodes))

# 10. Mostrar top-5 con (nodo, relaciones, sim)
scores.sort(key=lambda x: x[0], reverse=True)
for sim, local_idx, rel_ids, nodes in scores[:5]:
    node_name = id2entity[ subset[local_idx].item() ]
    rel_labels = [relation_texts[r] for r in rel_ids]
    print(f"--------------------")
    for index in range(len(rel_ids)):

        if index == 0:
            print(f"• ({seed_value}) --> {rel_labels[index]} --> ({id2entity[subset[nodes[index]].item()]})")
        else:
            print(f"({id2entity[subset[nodes[index-1]].item()]}) --> {rel_labels[index]} --> ({id2entity[subset[nodes[index]].item()]})")

    print(f"• Path similarity for end node: {node_name} — sim={sim:.4f}")

    print()
