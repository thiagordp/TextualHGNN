import torch
import torch.nn as nn
import random
import logging
import sys
import os
import re
import numpy as np
from pathlib import Path
from typing import List, Any, Optional

import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data, Batch

from src.data.text_graph_dataset_ondisk import TextGraphDatasetOnDisk
from src.models.graph_classification.gnn import DiffPoolMinCut
from src.models.graph_classification.train_and_evaluate import load_datasets
from src.utils.general_utils import load_config
from src.models.graph_explainability.subgraphx import SubgraphX
from src.models.graph_explainability.explainability_metrics import (
    calculate_fidelity,
    calculate_stability,
    GraphFidelityEvaluator, calculate_stability_adapted
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# UPDATE THIS PATH to your actual model
MODEL_PATH = "models/IMDB/best_model_DiffPoolMinCut_2025.11.24_23.28.59_final_lr0.0009615826222685446_hd_64_bs32_dec0.0624919749252487_lk0.01_en0.001_rc0.0_ct0.1_bl0.01_rp1.0_l20.009909917114455734_rep00.pth"
CONFIG_PATH = "src/utils/config.json"
DATASET = "IMDB"
LANG = "english"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 2. Model Wrapper (Sparse -> Dense Adapter)
# TODO: Por enquanto deixa assim, mas depois preciso alterar o SubgraphX para aceitar adj_dense
# ==========================================
class DiffPoolSparseWrapper(nn.Module):
    """
    Adapts the Dense DiffPool model to work with SubgraphX (which uses Sparse inputs).
    SubgraphX passes a PyG 'Data' object. DiffPool needs (x, adj, mask).
    """

    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device

    def forward(self, data):
        # 1. Extract Sparse Features
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        # 2. Convert to Dense (Batch Size=1 for SubgraphX logic)
        # If batch is None, create a dummy batch vector
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=self.device)

        # Convert sparse edge_index to dense adjacency [B, N, N]
        adj_dense = to_dense_adj(edge_index, batch=batch)

        # Create Mask [B, N]
        # For single graph processing, all nodes are valid
        mask = torch.ones((1, x.size(0)), dtype=torch.bool, device=self.device)

        # Reshape x to [B, N, F] (Assuming B=1)
        x_dense = x.unsqueeze(0)

        # 3. Forward Pass
        # Only return the logits (first element of tuple)
        out = self.model(x_dense, adj_dense, mask)
        if isinstance(out, tuple):
            return out[0]
        return out


# ==========================================
# 3. Hyperparameter Parsing
# ==========================================

def parse_hyperparameters(model_path, config):
    filename = Path(model_path).name
    params = {}

    patterns = {
        "inner_channels": r"_hd_(\d+)_",
        "decrease_proportion": r"_dec([\d\.]+)_",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, filename)

        if match:
            val = float(match.group(1))
            params[key] = int(val) if val.is_integer() else val

    defaults = {
        "max_num_nodes": config.get("NUM_NODES", 1000),
        "in_channels": config.get("NODE_FEATURE_DIM", 100),
        "out_channels": 2,
        "hidden_channels": config.get("HIDDEN_DIM", 100),
        "softmax_assign": True
    }

    return {**defaults, **params}


# ==========================================
# 4. SubgraphX Wrapper Function
# ==========================================

def run_subgraphx(model, x, adj, edge_index, target_class, **kwargs):
    """
    Instantiates and runs SubgraphX.
    Adapts input shapes and output formats.
    Args:
        model:
        x:
        adj:
        edge_index:
        target_class:
        **kwargs:

    Returns:

    """

    x_sparse = x.squeeze(0)
    if edge_index is None:
        from torch_geometric.utils import dense_to_sparse
        edge_index_sparse, _ = dense_to_sparse(adj.squeeze(0))
    else:
        edge_index_sparse = edge_index

    graph_data = Data(x=x_sparse, edge_index=edge_index_sparse).to(x.device)

    adapter = DiffPoolSparseWrapper(model, x.device)

    explainer = SubgraphX(
        model=adapter,
        num_hops=kwargs.get("num_hops", 3),
        coef=kwargs.get("coef", 10.0),
        high2low=kwargs.get("high2low", True),
        num_child=kwargs.get("num_child", 12),
        num_rollouts=kwargs.get("num_rollouts", 20),
        node_min=kwargs.get("node_min", 3),
        shapley_steps=kwargs.get("shapley_steps", 50),
        log=True,
        device=x.device
    )

    explanation_dict = explainer.explain_graph(
        graph=graph_data,
        feat=x_sparse,
        target_class=target_class,
    )

    if 'nodes' in explanation_dict:
        return explanation_dict["nodes"].tolist()

    return []


# ==========================================
# 5. Stability Metric (Local Fix)
# ==========================================
def jaccard_distance(list_a, list_b):
    s1 = set(list_a)
    s2 = set(list_b)
    if not s1 and not s2: return 0.0
    return 1.0 - len(s1.intersection(s2)) / len(s1.union(s2))


# ==========================================
# 6. Main Execution
# ==========================================
def main():
    # --- A. Setup ---
    config = load_config(LANG, CONFIG_PATH)
    params = parse_hyperparameters(MODEL_PATH, config)
    logging.info(f"Loaded Params: {params}")

    # --- B. Model ---
    model = DiffPoolMinCut(
        max_num_nodes=params['max_num_nodes'],
        in_channels=params['in_channels'],
        hidden_channels=params['hidden_channels'],
        out_channels=params['out_channels'],
        inner_channels=params['inner_channels'],
        softmax_assign=params['softmax_assign'],
        decrease_proportion=params['decrease_proportion'],
    ).to(DEVICE)

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # --- C. Data ---
    logging.info("Loading Test Data...")
    # We use ToDense to get ADJ, but we also need to preserve edge_index for SubgraphX
    # TextGraphDatasetOnDisk usually handles this if transform is ToDense
    transform = T.Compose([T.ToDense(num_nodes=params['max_num_nodes'])])

    # We load all splits but only use test
    _, _, tgd_test = load_datasets(
        root=config["ROOT"],
        max_num_nodes=params['max_num_nodes'],
        node_feature_size=params['in_channels'],
        lang=LANG,
        preprocessing_fn=None,  # Not needed for loading existing ondisk
        graph_builder_type="phrase_subgraphs",
        use_pmi=False
    )

    # Apply transform manually if load_datasets didn't (depends on implementation)
    # Assuming load_datasets applies it via the dataset class internally if passed
    # If not, we apply to the sample.

    # --- D. Sample Selection ---
    idx = random.randint(0, len(tgd_test) - 1)
    data = tgd_test[idx]

    print("ID", int.from_bytes(data.doc_id, byteorder='little'))

    # Manually apply ToDense if the dataset didn't give us adj
    if not hasattr(data, "adj") or data.adj is None:
        data = transform(data)

    data = data.to(DEVICE)

    # Batch dims
    x = data.x.unsqueeze(0)
    adj = data.adj.unsqueeze(0)
    mask = data.mask.unsqueeze(0)
    edge_index = data.edge_index

    with torch.no_grad():
        logits = model(x, adj, mask)[0]
        pred = torch.argmax(logits, dim=1).item()

    logging.info(f"Sample {idx} | True: {data.y[0]} | Pred: {pred}")

    # --- E. Evaluation Loop ---
    # 1. Run SubgraphX
    logging.info("Running SubgraphX (this may take time)...")

    # SubgraphX settings
    sx_args = {
        "coef":10,
        "num_rollouts": 20,
        "num_child":12,
        "shapley_steps": 100,  # Low for speed
        "node_min": 5
    }

    explanation = run_subgraphx(model, x, adj, edge_index, pred, **sx_args)
    logging.info(f"Explanation Nodes: {explanation}")

    # 2. Fidelity
    fid_plus, fid_minus = calculate_fidelity(
        model=model, x=x, explanation=explanation, level=0,
        adj=adj, mask=mask
    )

    # 3. Stability
    stab = calculate_stability_adapted(
        model=model, explainer_func=run_subgraphx,
        x=x, adj=adj, mask=mask, edge_index=edge_index, target_class=pred,
        n_samples=10, noise_std=0.01, **sx_args
    )

    print("\n" + "=" * 30)
    print(f"RESULTS FOR SAMPLE {idx}")
    print(f"Fidelity+ (Necessity):   {fid_plus:.4f}")
    print(f"Fidelity- (Sufficiency): {fid_minus:.4f}")
    print(f"Stability (Jaccard):     {stab:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    main()