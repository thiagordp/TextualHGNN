"""
Pyvis

30/09/2025
"""

import glob
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

import networkx as nx
import torch
import tqdm
from pyvis.network import Network
import torch_geometric.transforms as T

# Assuming all imports from the original file are available in the environment
from src.data.text_graph_dataset_ondisk import TextGraphDatasetOnDisk
from src.models.graph_classification.gnn import DiffPool
from src.models.graph_explainability.concept_grounding import ConceptGrounding
from src.models.graph_explainability.embeddings_oracle import EmbeddingOracle
from src.models.graph_explainability.llm_oracle import LLMOracle
from src.pipelines.concept_grounding_pipeline import retrieve_graph
from src.utils.general_utils import setup_logging, load_config
from src.utils.models_utils import get_timestamp


@dataclass
class HierarchicalVisualizationData:
    """Structure container for all data required for visualization"""

    adj_l0: torch.Tensor
    adj_l1: torch.Tensor
    adj_l2: torch.Tensor
    s01: torch.Tensor
    s12: torch.Tensor
    l0_names: Dict[int, str]
    l1_names: Dict[int, str]
    l2_names: Dict[int, str]


class HierarchicalGraphVisualizer:
    """
    Generates an interactive, multi-layered graph that faithfully implements
    the visual encoding scheme from the research proposal.
    """
    # Configuration constants aligned with the research proposal
    LEVEL_Y_COORDS = {0: 800, 1: 400, 2: 0}
    NODE_SHAPES = {0: 'dot', 1: 'square', 2: 'star'}
    NODE_SIZES = {0: 15, 1: 30, 2: 45}

    PALETTE = ['#FF6347', '#4682B4', '#32CD32', '#FFD700', '#6A5ACD',
               '#DA70D6', '#40E0D0', '#FF8C00', '#9932CC', '#00FA9A']

    def __init__(self, data: HierarchicalVisualizationData):
        self.data = data
        self.net = Network(
            height="900px",
            width="100%",
            notebook=False,
            bgcolor="#222222",
            font_color="white",
            select_menu=True,
            filter_menu=True,
            directed=True
        )

        self.net.set_options(
            """
            var options = {
              "physics": {
                "barnesHut": { "gravitationalConstant": -30000, "springLength": 200 },
                "stabilization": { "iterations": 1000 }
              }
            }
            """.strip()
        )

    def _generate_tooltip_html(self, level: int, node_id: int) -> str:
        """Generates HTML for tooltips"""
        if level == 0:
            name = self.data.l0_names.get(node_id, f"Node {node_id}")
            return f"Word (L0)\nID: {node_id}\nName: {name}"
        elif level == 1:
            s01_col = self.data.s01[:, node_id]
            top_scores, top_indices = torch.topk(s01_col, k=min(5, len(s01_col)))

            constituents = "\n".join([
                f"{self.data.l0_names.get(idx.item(), 'N/A')} ({score:.2f})"
                for idx, score in zip(top_indices, top_scores) if score > 1e-3
            ])
            rep_word_idx = torch.argmax(s01_col).item()
            rep_word = self.data.l0_names.get(rep_word_idx, "N/A")
            return f"Cluster (L1)\nID: C1_{node_id}\nRep: {rep_word} \n {constituents}"
        elif level == 2:
            # Find top constituent L1 clusters
            s12_col = self.data.s12[:, node_id]
            top_scores, top_indices = torch.topk(s12_col, k=min(5, len(s12_col)))

            constituents = "<br>".join([
                f"C1_{idx.item()} ({score:.2f})"
                for idx, score in zip(top_indices, top_scores) if score > 0
            ])
            return f"Super-Cluster (L2)\nID: C2_{node_id} -- {constituents}"
        return ""

    def _add_level_0_nodes_and_edges(self):
        """Renders the word-level graph (L0)."""
        l0_to_l1_assignments = self.data.s01.argmax(dim=1)
        num_nodes = self.data.adj_l0.shape[0]

        for i in range(num_nodes):
            cluster_id = l0_to_l1_assignments[i].item()
            color = self.PALETTE[cluster_id % len(self.PALETTE)]
            self.net.add_node(
                f"L0_{i}",
                label=self.data.l0_names.get(i, str(i)),
                shape=self.NODE_SHAPES[0],
                size=self.NODE_SIZES[0],
                color=color,
                title=self._generate_tooltip_html(0, i),
                x=random.uniform(-500, 500),  # Spread nodes horizontally
                y=self.LEVEL_Y_COORDS[0],
                physics=True
            )

        # Add L0 edges (syntactic dependencies)
        rows, cols = self.data.adj_l0.nonzero(as_tuple=True)
        for u, v in zip(rows, cols):
            if u < v:  # Avoid duplicate edges
                self.net.add_edge(f"L0_{u}", f"L0_{v}", width=1, color='rgba(200, 200, 200, 0.3)')

    def _add_higher_level_nodes_and_edges(self, level: int):
        """Generic renderer for L1 and L2 graphs"""
        if level == 1:
            adj = self.data.adj_l1
            s_matrix = self.data.s12
            num_nodes = self.data.adj_l1.shape[0]
            prefix = "L1"
            name_map = self.data.l1_names
        elif level == 2:
            adj = self.data.adj_l2
            s_matrix = None
            num_nodes = adj.shape[0]
            prefix = "L2"
            name_map = self.data.l2_names
        else:
            return

        assignments = s_matrix.argmax(dim=1) if s_matrix is not None else torch.zeros(num_nodes, dtype=torch.long)

        for i in range(num_nodes):
            super_cluster_id = assignments[i].item()
            border_color = self.PALETTE[super_cluster_id % len(self.PALETTE)] if s_matrix is not None else '#FFFFFF'
            bg_color = "#555555"

            self.net.add_node(
                f"{prefix}_{i}",
                label=name_map.get(i, f"C{level}_{i}"),
                shape=self.NODE_SHAPES[level],
                size=self.NODE_SIZES[level],
                color={"background": bg_color, "border": border_color},
                borderWidth=4,
                title=self._generate_tooltip_html(level, i),
                x=random.uniform(-400, 400),
                y=self.LEVEL_Y_COORDS[level],
                physics=True
            )

        rows, cols = adj.nonzero(as_tuple=True)

        for u, v in zip(rows, cols):
            if u < v:
                weight = adj[u, v].item()
                self.net.add_edge(
                    f"{prefix}_{u}", f"{prefix}_{v}",
                    value=weight * 5,
                    width=0.5 + weight * 4,
                    color="rgba(255, 107, 71, 0.7)"
                )

    def generate_html(self, filename: str):

        """Generates HTML for visualization"""

        self._add_level_0_nodes_and_edges()
        self._add_higher_level_nodes_and_edges(level=1)
        self._add_higher_level_nodes_and_edges(level=2)

        for l0_idx in range(self.data.s01.shape[0]):
            l1_idx = self.data.s01[l0_idx].argmax().item()
            self.net.add_edge(f"L0_{l0_idx}", f"L1_{l1_idx}", dashes=True, color='rgba(200, 200, 200, 0.2)', width=0.5)

        self.net.show(filename, notebook=False)
        logging.info(f"Saved visualization to {filename}")


# Step 1: Isolate Data Preparation into a cleaner main function
def main():
    """Main function to drive the visualization pipeline."""
    # --- Constants and Setup ---
    MODEL_PATH = f"models/grid_search/STF_HC_Voto_Relatorio/best_model_DiffPool_20250903_135039_lr0.0001_hd_32_bs4_dec0.1_lk10.0_en0.01_rc0.1_ct0.1_bl0.1_rp0.1_l20.01_rep02.pth"
    DATASET = "STF_HC_Voto_Relatorio"
    LANG = "portuguese_voto"
    CONFIG = load_config(LANG, "src/utils/config.json")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ROOT = f"data/datasets/{DATASET}"

    timestamp = get_timestamp()
    setup_logging(log_file=f"cg_visualization_{DATASET}_{timestamp}.log")
    logging.info(f"============  STARTING CG VISUALIZATION: '{DATASET}'  ============")

    # --- Load Model and Data ---
    logging.info("-- Loading model and test dataset --")
    weights = torch.load(MODEL_PATH, map_location=DEVICE)
    diffpool_model = DiffPool(
        max_num_nodes=3000, in_channels=100, hidden_channels=100,
        out_channels=2, inner_channels=32, softmax_assign=True, decrease_proportion=0.1
    )
    diffpool_model.load_state_dict(weights)
    diffpool_model = diffpool_model.to(device=DEVICE)

    tgd_test = TextGraphDatasetOnDisk(
        root=ROOT, split="test", batch_size=1, node_feature_size=100,
        transform=T.ToDense(num_nodes=3000), max_num_nodes=3000, lang=LANG
    )

    # --- Setup Oracles for Concept Grounding ---
    oracle = EmbeddingOracle(
        model_path='data/external/embeddings/glove_legal_100.bin',
        db_path=f"data/oracle/embeddings_{DATASET}.db",
        batch_size=500000
    )
    api_key = os.getenv("TOGETHER_API_KEY")
    llm_oracle = LLMOracle(
        system_prompt="data/prompts/llm_oracle.txt", language=LANG, api_key=api_key
    )

    # --- Run Concept Grounding to get the data ---
    doc_index = 1
    data_element = tgd_test[doc_index * -1].to(device=DEVICE)
    cg = ConceptGrounding(
        graph_model=diffpool_model, embedding_oracle=oracle, llm_oracle=llm_oracle,
        graph=data_element, original_raw_file_path=f"{ROOT}/test/raw/", language=LANG
    )
    logging.info("Running concept grounding...")
    cg.concept_grounding()

    # --- Data Extraction and Structuring ---
    logging.info("Extracting hierarchical data for visualization...")
    cutoff_idx = (cg.x_l0.abs().sum(dim=1) > 1e-5).nonzero().max().item() + 1

    # Prune adjacency matrices for efficiency in visualization
    adj_l1_pruned = cg.adj_l1.clone()
    K = int(adj_l1_pruned.shape[0] * 2.5)  # Keep top 2.5 edges per node on average
    values, indices = torch.topk(adj_l1_pruned.abs().flatten(), K)
    mask = torch.zeros_like(adj_l1_pruned.flatten(), dtype=torch.bool)
    mask[indices] = True
    adj_l1_pruned = torch.where(mask.view_as(adj_l1_pruned), adj_l1_pruned, 0.0)

    viz_data = HierarchicalVisualizationData(
        adj_l0=cg.adj_l0[:cutoff_idx, :cutoff_idx],
        adj_l1=adj_l1_pruned,
        adj_l2=cg.adj_l2,
        s01=cg.s01[:cutoff_idx],
        s12=cg.s12,
        l0_names={i: name for i, name in enumerate(cg.l0_names[:cutoff_idx])},
        l1_names={i: name for i, name in enumerate(cg.l1_names)},
        l2_names={i: f"C2_{i}" for i in range(cg.adj_l2.shape[0])}  # L2 names are not grounded
    )

    # --- Generate Visualization ---
    logging.info("Generating interactive HTML visualization...")
    visualizer = HierarchicalGraphVisualizer(data=viz_data)
    visualizer.generate_html(
        filename="test_visualization.html"
    )


if __name__ == "__main__":
    main()
