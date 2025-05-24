import json
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch_geometric.data import HeteroData
from sentence_transformers import SentenceTransformer 

from torch_geometric.nn import RGCNConv
from typing import List, Dict, Any, Tuple, Optional
import logging
import os
import random

# Assuming StaticModel is in your project's PYTHONPATH or accessible
from model2vec import StaticModel # For initial node/relation features

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# EMBEDDING_MODEL_NAME for initial features:
# - "minishlab/potion-base-8M": Good starting point, relatively small.
# - "sentence-transformers/all-MiniLM-L6-v2": Very popular, good quality for its size.
# - "sentence-transformers/all-mpnet-base-v2": Larger, generally better performance than MiniLM.
# - For very specific domains, you might fine-tune one of these or use a domain-specific model.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # UPDATED for better general text features
KG_DATA_PATH = "./test_max/relations.json" # REPLACE with your actual relations data path
OUTPUT_EMBEDDINGS_PATH = "pretrained_entity_embeddings_distmult.pt"
OUTPUT_RELATION_EMBEDDINGS_PATH = "pretrained_relation_embeddings_distmult.pt" # NEW: if learning relation embeddings
OUTPUT_METADATA_PATH = "kg_metadata_distmult.json"

# RGCN Hyperparameters & Training
INITIAL_EMBEDDING_DIM_TEXT = 384 # For all-MiniLM-L6-v2. If using Potion, it's 768. Auto-detected later.
# GNN_EMBEDDING_DIM will be the dimension of the output entity AND relation embeddings from the GNN.
# It's often good practice to have entity and relation embeddings in the same space for scoring functions like DistMult.
GNN_EMBEDDING_DIM = 384 # Common choice for KG embeddings, can be tuned (e.g., 100, 200, 500)
GNN_HIDDEN_CHANNELS_RGCN = 256 # Hidden channels specifically for RGCN layers if RGCN output is not final GNN_EMBEDDING_DIM
NUM_GNN_LAYERS = 2
NUM_EPOCHS = 10 # Increased epochs for link prediction
LEARNING_RATE = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NEGATIVE_SAMPLES_RATIO = 5 # Number of negative samples per positive sample
BATCH_SIZE = 1024 # For link prediction training
MARGIN = 1.0 # For margin-based loss (like TransE, not directly used by DistMult with BCE)
LOSS_TYPE = "BCE" # "BCE" (Binary Cross-Entropy) or "Margin" (MarginRankingLoss)

# Whether the GNN itself learns separate relation embeddings or uses pre-defined ones.
# For DistMult, we need distinct relation embeddings. RGCNConv doesn't learn them directly in its basic form.
# We will define relation embeddings as nn.Embedding and learn them.
LEARN_RELATION_EMBEDDINGS = True


class KGEModel(torch.nn.Module):
    def __init__(self, num_entities: int, num_relations: int, embedding_dim: int,
                 initial_entity_features: Optional[torch.Tensor] = None,
                 text_feature_dim: Optional[int] = None,
                 num_rgcn_layers: int = 2, rgcn_hidden_channels: int = 128):
        super().__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations # This is 2 * num_original_relations
        self.embedding_dim = embedding_dim # Target GNN embedding dimension

        # Entity Embeddings: Initialize from text features if provided, then make them learnable
        if initial_entity_features is not None:
            assert text_feature_dim is not None, "text_feature_dim must be provided with initial_entity_features"
            self.entity_feature_projection = torch.nn.Linear(text_feature_dim, embedding_dim)
            # We'll use RGCN to refine these projections
            self.initial_entity_features_projected = self.entity_feature_projection(initial_entity_features)
            # The RGCN will operate on these projected features
            self.rgcn = FullGraphRGCN(embedding_dim, rgcn_hidden_channels, embedding_dim,
                                      num_relations, num_rgcn_layers)
        else: # If no initial text features, learn entity embeddings from scratch + RGCN
            # This path is less common if you have good text features, but possible
            self.entity_embeds_raw = torch.nn.Embedding(num_entities, embedding_dim)
            torch.nn.init.xavier_uniform_(self.entity_embeds_raw.weight.data)
            self.rgcn = FullGraphRGCN(embedding_dim, rgcn_hidden_channels, embedding_dim,
                                      num_relations, num_rgcn_layers)


        # Relation Embeddings (learned directly)
        if LEARN_RELATION_EMBEDDINGS:
            self.relation_embeds = torch.nn.Embedding(num_relations, embedding_dim) # Relations in the same dim space
            torch.nn.init.xavier_uniform_(self.relation_embeds.weight.data)
        else:
            # If not learning, you'd need to provide pre-computed relation embeddings (e.g., from StaticModel)
            # and potentially project them. This is less common for SOTA KGE models.
            pass

    def get_entity_embeddings(self, x_initial_text_features: Optional[torch.Tensor],
                              edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'entity_feature_projection'):
            # Use RGCN to refine projected text features
            projected_features = self.entity_feature_projection(x_initial_text_features) # type: ignore
            entity_embeddings = self.rgcn(projected_features, edge_index, edge_type)
        else:
            # Use RGCN to refine randomly initialized embeddings
            entity_embeddings = self.rgcn(self.entity_embeds_raw.weight, edge_index, edge_type)
        return entity_embeddings

    def forward(self, h_indices: torch.Tensor, r_indices: torch.Tensor, t_indices: torch.Tensor,
                all_entity_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculates scores for (h, r, t) triples using DistMult.
        all_entity_embeddings are the GNN-processed entity embeddings.
        """
        h_emb = all_entity_embeddings[h_indices] # [batch_size, emb_dim]
        t_emb = all_entity_embeddings[t_indices] # [batch_size, emb_dim]
        
        if LEARN_RELATION_EMBEDDINGS:
            r_emb = self.relation_embeds(r_indices) # [batch_size, emb_dim]
        else:
            # Handle case where relation embeddings are not learned by this module
            # For now, this path would error if not LEARN_RELATION_EMBEDDINGS
            raise NotImplementedError("Static relation embeddings not fully implemented for scoring here.")

        # DistMult scoring function: <h, R, t> = sum(h_i * r_i * t_i)
        score = torch.sum(h_emb * r_emb * t_emb, dim=-1) # [batch_size]
        return score

# RGCN component (can be kept similar)
class FullGraphRGCN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_relations: int, num_layers: int = 2):
        super().__init__()
        self.num_relations = num_relations
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList() # Batch Norm for GNNs can help

        if num_layers == 1:
            self.convs.append(RGCNConv(in_channels, out_channels, num_relations))
        else:
            self.convs.append(RGCNConv(in_channels, hidden_channels, num_relations))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(RGCNConv(hidden_channels, hidden_channels, num_relations))
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            self.convs.append(RGCNConv(hidden_channels, out_channels, num_relations))
        
        logger.info(f"Initialized RGCN with {num_layers} layers. In: {in_channels}, Hidden: {hidden_channels}, Out: {out_channels}, Relations: {num_relations}")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        if x is None or x.numel() == 0: # Should not happen if KGEModel handles init
            logger.warning("RGCN: Input node features 'x' are empty.")
            num_nodes_inferred = int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
            return torch.empty(num_nodes_inferred, self.convs[-1].out_channels, device=DEVICE)

        if edge_index is None or edge_index.numel() == 0 or edge_type is None or edge_type.numel() == 0:
            logger.warning("RGCN: Edge index or edge types are empty. If input dim matches output, passing through.")
            return x if x.size(1) == self.convs[-1].out_channels else torch.zeros(x.size(0), self.convs[-1].out_channels, device=x.device)

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < self.num_layers - 1:
                if self.bns: x = self.bns[i](x) # Apply Batch Norm
                x = F.relu(x)
                # x = F.dropout(x, p=0.5, training=self.training) # Optional Dropout
        return x


def load_and_prepare_data_for_kge(relations_json_path: str, embedding_model_name: str) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, # edges (h,r,t), initial_entity_features
    Dict[str, int], Dict[str, int], # entity2id, relation2id (original)
    int, int, int # num_entities, num_original_relations, text_feature_dim
]:
    logger.info(f"Loading relations from: {relations_json_path}")
    try:
        with open(relations_json_path, 'r', encoding='utf-8') as f:
            relations_data = json.load(f)
    except Exception as e:
        logger.error(f"Error loading/parsing {relations_json_path}: {e}")
        raise

    entity2id: Dict[str, int] = {}
    relation2id_original: Dict[str, int] = {} # Original relation labels to ID (0 to N-1)
    
    triples_original: List[Tuple[int, int, int]] = [] # (h_id, r_original_id, t_id)

    for rel in relations_data:
        src_val, dst_val, label_val = rel['from']['value'], rel['to']['value'], rel['label']
        for node in (src_val, dst_val):
            if node not in entity2id:
                entity2id[node] = len(entity2id)
        if label_val not in relation2id_original:
            relation2id_original[label_val] = len(relation2id_original)
        
        triples_original.append((entity2id[src_val], relation2id_original[label_val], entity2id[dst_val]))

    num_entities = len(entity2id)
    num_original_relations = len(relation2id_original)
    id2entity = {i: e for e, i in entity2id.items()}
    
    logger.info(f"KG loaded: {num_entities} entities, {num_original_relations} original relation types.")

    # --- Create full set of triples (h, r_full, t) for GNN and Link Prediction ---
    # r_full: 0 to N-1 are original, N to 2N-1 are inverse
    # These triples are used for GNN message passing (edge_index, edge_type) and as positive samples for training
    all_triples_for_gnn_and_lp: List[Tuple[int, int, int]] = [] # (h_id, r_full_id, t_id)
    
    # For GNN structure (edge_index and edge_type)
    edge_index_list_for_gnn: List[Tuple[int, int]] = []
    edge_type_list_for_gnn: List[int] = []


    for h, r_orig, t in triples_original:
        # Original direction for GNN and LP
        all_triples_for_gnn_and_lp.append((h, r_orig, t))
        edge_index_list_for_gnn.append((h,t))
        edge_type_list_for_gnn.append(r_orig)

        # Inverse direction for GNN and LP
        r_inv = r_orig + num_original_relations
        all_triples_for_gnn_and_lp.append((t, r_inv, h))
        edge_index_list_for_gnn.append((t,h))
        edge_type_list_for_gnn.append(r_inv)
        
    # Convert to tensors
    # `train_triples` will be used for batching in link prediction
    train_triples = torch.tensor(all_triples_for_gnn_and_lp, dtype=torch.long)
    
    # `gnn_edge_index` and `gnn_edge_type` for the RGCN component
    gnn_edge_index = torch.tensor(edge_index_list_for_gnn, dtype=torch.long).t().contiguous()
    gnn_edge_type = torch.tensor(edge_type_list_for_gnn, dtype=torch.long)


    # --- Initial Node Features (from text embedding model) ---
    text_feature_dim_actual = INITIAL_EMBEDDING_DIM_TEXT # Default
    if num_entities > 0:
        sbert_model = SentenceTransformer(embedding_model_name, device=DEVICE)
        text_feature_dim_actual = sbert_model.get_sentence_embedding_dimension()
        entity_texts = [id2entity[i] for i in range(num_entities)]
        raw_embeddings_np = sbert_model.encode(entity_texts, show_progress_bar=True, convert_to_numpy=True)
        initial_entity_features = torch.tensor(raw_embeddings_np, dtype=torch.float)
        processed_embeddings = []
        for emb_np in raw_embeddings_np:
            if emb_np.ndim == 2 and emb_np.shape[0] == 1: processed_embeddings.append(emb_np.squeeze(0))
            elif emb_np.ndim == 1: processed_embeddings.append(emb_np)
            else: processed_embeddings.append(np.zeros(text_feature_dim_actual))
        
        if processed_embeddings:
            initial_entity_features_np = np.array(processed_embeddings)
            initial_entity_features = torch.tensor(initial_entity_features_np, dtype=torch.float)
            text_feature_dim_actual = initial_entity_features.size(1)
            logger.info(f"Initial entity features from text model shape: {initial_entity_features.shape}")
        else:
            initial_entity_features = torch.randn(num_entities, text_feature_dim_actual)
            logger.warning(f"Using random initial entity features of shape: {initial_entity_features.shape}")
    else:
        initial_entity_features = torch.empty(0, text_feature_dim_actual, dtype=torch.float)

    return train_triples, gnn_edge_index, gnn_edge_type, initial_entity_features, \
           entity2id, relation2id_original, \
           num_entities, num_original_relations, text_feature_dim_actual


def generate_negative_samples(positive_triples_batch: torch.Tensor, num_entities: int,
                              negative_ratio: int) -> torch.Tensor:
    batch_size = positive_triples_batch.size(0)
    num_negative_samples_total = batch_size * negative_ratio
    
    negative_h = torch.zeros(num_negative_samples_total, dtype=torch.long)
    negative_r = torch.zeros(num_negative_samples_total, dtype=torch.long)
    negative_t = torch.zeros(num_negative_samples_total, dtype=torch.long)

    current_idx = 0
    for i in range(batch_size):
        h, r, t = positive_triples_batch[i]
        for _ in range(negative_ratio):
            if random.random() < 0.5: # Corrupt head
                neg_h_candidate = random.randint(0, num_entities - 1)
                while neg_h_candidate == h: # Ensure it's different
                    neg_h_candidate = random.randint(0, num_entities - 1)
                negative_h[current_idx] = neg_h_candidate
                negative_r[current_idx] = r
                negative_t[current_idx] = t
            else: # Corrupt tail
                neg_t_candidate = random.randint(0, num_entities - 1)
                while neg_t_candidate == t:
                    neg_t_candidate = random.randint(0, num_entities - 1)
                negative_h[current_idx] = h
                negative_r[current_idx] = r
                negative_t[current_idx] = neg_t_candidate
            current_idx += 1
            
    # Combine into a single tensor [num_negative_samples_total, 3]
    return torch.stack([negative_h, negative_r, negative_t], dim=1)


def train_link_prediction(model: KGEModel, optimizer: optim.Optimizer,
                          train_triples_all: torch.Tensor,
                          initial_text_features_all_entities: torch.Tensor, # For RGCN input
                          gnn_edge_index: torch.Tensor, gnn_edge_type: torch.Tensor, # For RGCN
                          num_entities: int, batch_size: int, negative_ratio: int,
                          loss_type: str, margin: float, device: torch.device) -> float:
    model.train()
    total_loss = 0
    num_batches = (train_triples_all.size(0) + batch_size - 1) // batch_size
    
    # Shuffle training triples each epoch
    perm = torch.randperm(train_triples_all.size(0))
    train_triples_shuffled = train_triples_all[perm]

    # Get all entity embeddings from GNN once per epoch (or more frequently if GNN is very deep/complex)
    # This assumes the GNN structure (edge_index, edge_type) doesn't change during training.
    with torch.no_grad() if not any(p.requires_grad for p in model.rgcn.parameters()) else torch.enable_grad(): # type: ignore
        # if RGCN params are frozen, no_grad. Otherwise, they are part of the KGEModel's parameters.
        all_entity_embeddings_gnn = model.get_entity_embeddings(
            initial_text_features_all_entities.to(device) if initial_text_features_all_entities.numel() > 0 else None,
            gnn_edge_index.to(device),
            gnn_edge_type.to(device)
        )


    for i in range(num_batches):
        optimizer.zero_grad()
        
        positive_batch = train_triples_shuffled[i * batch_size : (i + 1) * batch_size].to(device)
        if positive_batch.size(0) == 0: continue

        h_pos, r_pos, t_pos = positive_batch[:, 0], positive_batch[:, 1], positive_batch[:, 2]
        
        negative_batch = generate_negative_samples(positive_batch, num_entities, negative_ratio).to(device)
        h_neg, r_neg, t_neg = negative_batch[:, 0], negative_batch[:, 1], negative_batch[:, 2]

        # Scores for positive and negative samples
        # Pass the pre-computed GNN entity embeddings to the scoring function
        pos_scores = model(h_pos, r_pos, t_pos, all_entity_embeddings_gnn)
        neg_scores = model(h_neg, r_neg, t_neg, all_entity_embeddings_gnn)

        if loss_type == "BCE":
            pos_labels = torch.ones_like(pos_scores)
            neg_labels = torch.zeros_like(neg_scores)
            scores = torch.cat([pos_scores, neg_scores], dim=0)
            labels = torch.cat([pos_labels, neg_labels], dim=0)
            loss = F.binary_cross_entropy_with_logits(scores, labels) # Assumes scores are logits
        elif loss_type == "Margin":
            # This requires scores to be similarity scores (higher is better)
            # DistMult naturally produces scores that can be used this way (though often used with BCE)
            # Ensure neg_scores are reshaped correctly if negative_ratio > 1
            # For simplicity, let's assume negative_ratio=1 for margin loss or average neg_scores per positive.
            # A common margin loss: max(0, margin - pos_score + neg_score)
            if neg_scores.size(0) != pos_scores.size(0): # Reshape for margin loss
                 neg_scores = neg_scores.view(pos_scores.size(0), -1).mean(dim=1) # Average negative scores
            loss = F.margin_ranking_loss(pos_scores, neg_scores, torch.ones_like(pos_scores), margin=margin)
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    return total_loss / num_batches


def main_kge_training():
    logger.info(f"Using device: {DEVICE}")

    if not os.path.exists(KG_DATA_PATH):
        logger.error(f"KG data file not found: {KG_DATA_PATH}. Please provide the correct path.")
        return

    train_triples, gnn_edge_index, gnn_edge_type, initial_entity_text_features, \
    entity2id, relation2id_original, \
    num_entities, num_original_relations, text_feature_dim_actual = \
        load_and_prepare_data_for_kge(KG_DATA_PATH, EMBEDDING_MODEL_NAME)

    if num_entities == 0:
        logger.error("No entities found in the graph. Cannot train KGE model.")
        return
    
    num_total_relations_for_gnn_and_kge = 2 * num_original_relations
    if num_total_relations_for_gnn_and_kge == 0 and num_entities > 0: # No relations but entities exist
        num_total_relations_for_gnn_and_kge = 1 # For Embedding layers
        logger.warning("No relations in graph. num_total_relations set to 1.")


    logger.info(f"Number of entities: {num_entities}")
    logger.info(f"Number of original relations: {num_original_relations}")
    logger.info(f"Number of total relation types (orig+inv) for GNN/KGE: {num_total_relations_for_gnn_and_kge}")
    logger.info(f"Initial text feature dimension: {text_feature_dim_actual}")
    logger.info(f"Target GNN/KGE embedding dimension: {GNN_EMBEDDING_DIM}")

    kge_model = KGEModel(
        num_entities=num_entities,
        num_relations=num_total_relations_for_gnn_and_kge, # For relation embeddings
        embedding_dim=GNN_EMBEDDING_DIM,
        initial_entity_features=initial_entity_text_features.to(DEVICE) if initial_entity_text_features.numel() > 0 else None,
        text_feature_dim=text_feature_dim_actual if initial_entity_text_features.numel() > 0 else None,
        num_rgcn_layers=NUM_GNN_LAYERS,
        rgcn_hidden_channels=GNN_HIDDEN_CHANNELS_RGCN
    ).to(DEVICE)

    optimizer = optim.Adam(kge_model.parameters(), lr=LEARNING_RATE)

    logger.info("Starting KGE model training (Link Prediction with DistMult)...")
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_loss = train_link_prediction(
            kge_model, optimizer, train_triples,
            initial_entity_text_features, # Pass CPU tensor, will be moved to device in train_link_prediction
            gnn_edge_index, gnn_edge_type, # Pass CPU tensors
            num_entities, BATCH_SIZE, NEGATIVE_SAMPLES_RATIO,
            LOSS_TYPE, MARGIN, DEVICE
        )
        if epoch % 10 == 0 or epoch == 1 or epoch == NUM_EPOCHS :
            logger.info(f"Epoch {epoch:03d}/{NUM_EPOCHS:03d} | Avg Loss: {epoch_loss:.4f}")

    logger.info("KGE Training finished.")

    # Get final entity and relation embeddings
    kge_model.eval()
    with torch.no_grad():
        # Final entity embeddings from GNN part of KGEModel
        final_entity_embeddings = kge_model.get_entity_embeddings(
            initial_entity_text_features.to(DEVICE) if initial_entity_text_features.numel() > 0 else None,
            gnn_edge_index.to(DEVICE),
            gnn_edge_type.to(DEVICE)
        ).cpu()

        final_relation_embeddings = None
        if LEARN_RELATION_EMBEDDINGS:
            final_relation_embeddings = kge_model.relation_embeds.weight.cpu()

    logger.info(f"Final entity embeddings shape: {final_entity_embeddings.shape}")
    torch.save(final_entity_embeddings, OUTPUT_EMBEDDINGS_PATH)
    logger.info(f"Pre-trained GNN entity embeddings saved to {OUTPUT_EMBEDDINGS_PATH}")

    if final_relation_embeddings is not None:
        logger.info(f"Final relation embeddings shape: {final_relation_embeddings.shape}")
        torch.save(final_relation_embeddings, OUTPUT_RELATION_EMBEDDINGS_PATH)
        logger.info(f"Pre-trained relation embeddings saved to {OUTPUT_RELATION_EMBEDDINGS_PATH}")

    metadata = {
        'entity2id': entity2id,
        'relation2id_original': relation2id_original, # Save original relation mapping
        'embedding_dim': GNN_EMBEDDING_DIM, # This is the KGE embedding dimension
        'text_feature_dim_initial': text_feature_dim_actual,
        'gnn_architecture': {
            'type': 'RGCN_enhanced_KGE_DistMult',
            'rgcn_layers': NUM_GNN_LAYERS,
            'rgcn_hidden': GNN_HIDDEN_CHANNELS_RGCN,
            'kge_embedding_dim': GNN_EMBEDDING_DIM,
            'num_relations_trained_on': num_total_relations_for_gnn_and_kge
        },
        'training_params': {
            'epochs': NUM_EPOCHS, 'lr': LEARNING_RATE, 'batch_size': BATCH_SIZE,
            'negative_ratio': NEGATIVE_SAMPLES_RATIO, 'loss': LOSS_TYPE, 'margin': MARGIN if LOSS_TYPE=="Margin" else None
        }
    }
    with open(OUTPUT_METADATA_PATH, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"KG metadata and KGE config saved to {OUTPUT_METADATA_PATH}")


if __name__ == '__main__':
    if KG_DATA_PATH == "relations.json":
        logger.error("Please update KG_DATA_PATH in the script before running.")
    else:
        main_kge_training()