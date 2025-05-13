from torch_geometric.utils import k_hop_subgraph
import torch
from torch_geometric.data import HeteroData

def extract_k_hop_subgraph_bidirectional(seed_id: int,
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

    # 2) Creamos las aristas inversas
    #    Invertimos filas de edge_index y desplazamos el ID de relación
    inv_edge_index = edge_index.flip(0)                       # [2, E]
    inv_edge_type  = edge_type + num_rels_total               # [E]

    # 3) Combinamos originales + inversas
    comb_edge_index = torch.cat([edge_index,    inv_edge_index], dim=1)  # [2, 2E]
    comb_edge_type  = torch.cat([edge_type,     inv_edge_type],  dim=0) # [2E]

    # 4) Usamos k_hop_subgraph sobre este grafo bidireccional temporal
    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx      = seed_id,
        num_hops      = K,
        edge_index    = comb_edge_index,
        relabel_nodes = True,
        num_nodes     = data['entity'].x.size(0)
    )

    # 5) Filtramos los tipos de arista con la máscara
    sub_edge_type = comb_edge_type[edge_mask]

    # 6) Construimos y devolvemos el subgrafo
    sub_data = HeteroData()
    sub_data['entity'].x = data['entity'].x[subset]
    sub_data['entity','to','entity'].edge_index = sub_edge_index
    sub_data['entity','to','entity'].edge_type  = sub_edge_type

    return sub_data, mapping, subset
