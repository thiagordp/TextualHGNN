"""
Pyvis for Hierarchical Graph Visualization (Improved Version)

30/09/2025
"""

import glob
import json
import logging
import os
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict

import torch
from pyvis.network import Network
import torch_geometric.transforms as T

# Assuming all imports from the original file are available in the environment
from src.data.text_graph_dataset_ondisk import TextGraphDatasetOnDisk
from src.models.graph_classification.gnn import DiffPool
from src.models.graph_explainability.concept_grounding import ConceptGrounding
from src.models.graph_explainability.embeddings_oracle import EmbeddingOracle
from src.models.graph_explainability.llm_oracle import LLMOracle
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
    the visual encoding scheme from the research proposal, with new features.
    """
    # Configuration constants aligned with the research proposal
    LEVEL_Y_COORDS = {0: 800, 1: 400, 2: 0}
    # NEW: Added 'triangle' for unassigned nodes
    NODE_SHAPES = {0: 'dot', 1: 'square', 2: 'star', 'unassigned': 'triangle'}
    NODE_SIZES = {0: 15, 1: 30, 2: 45}

    PALETTE = ['#FF6347', '#4682B4', '#32CD32', '#FFD700', '#6A5ACD',
               '#DA70D6', '#40E0D0', '#FF8C00', '#9932CC', '#00FA9A']

    def __init__(self,
                 data: HierarchicalVisualizationData,
                 l0_assignment_threshold: float = 0.1,
                 l1_assignment_threshold: float = 0.1):

        self.data = data
        # Threshold for styling L0 nodes
        self.l0_assignment_threshold = l0_assignment_threshold
        # Threshold for L1->L2 assignments
        self.l1_assignment_threshold = l1_assignment_threshold
        self.included_l1_nodes: Set[int] = set()

        self.net = Network(
            height="900px", width="100%", notebook=False, bgcolor="#222222",
            font_color="white", select_menu=False, filter_menu=False, directed=False
        )

        self.net.set_options(
            """
            var options = {
              "physics": {
                "barnesHut": { "gravitationalConstant": -40000, "springLength": 350, "centralGravity": 0.4 },
                "stabilization": { "iterations": 1200 }
              }
            }
            """.strip()
        )

    def _generate_tooltip_html(self, level: int, node_id: int) -> str:
        """Generates HTML for tooltips based on user's preferred format."""
        if level == 0:
            name = self.data.l0_names.get(node_id, f"Node {node_id}")
            max_assignment = self.data.s01[node_id].max().item()
            return f"Word (L0)\nID: {node_id}\nName: {name}\nMax Assign: {max_assignment:.3f}"
        elif level == 1:
            s01_col = self.data.s01[:, node_id]
            top_scores, top_indices = torch.topk(s01_col, k=min(3, len(s01_col)))
            constituents = "\n".join([
                f"{self.data.l0_names.get(idx.item(), 'N/A')} ({score:.2f})"
                for idx, score in zip(top_indices, top_scores) if score > 1e-3
            ])
            rep_word_idx = torch.argmax(s01_col).item()
            rep_word = self.data.l0_names.get(rep_word_idx, "N/A")
            return f"Cluster (L1)\nID: C1_{node_id}\nName: {rep_word}\nConstituents:\n{constituents}"
        elif level == 2:
            s12_col = self.data.s12[:, node_id]
            name = self.data.l2_names.get(node_id, f"Node {node_id}")
            top_scores, top_indices = torch.topk(s12_col, k=min(3, len(s12_col)))
            constituents = "\n".join([
                f"C1_{idx.item()} ({score:.2f})"
                for idx, score in zip(top_indices, top_scores) if score > 0
            ])
            return f"Cluster (L2)\nID: C2_{node_id}\nName: {name}\nConstituents:\n{constituents}"
        return ""

    def _add_level_0_nodes_and_edges(self):
        """Renders the word-level graph (L0), styling nodes based on assignment strength."""
        num_nodes = self.data.adj_l0.shape[0]

        for i in range(num_nodes):
            # REQUIREMENT 3: Check assignment strength
            max_assignment_score = self.data.s01[i].max().item()

            if max_assignment_score < self.l0_assignment_threshold:
                # This node is weakly assigned or unassigned
                shape = self.NODE_SHAPES['unassigned']
                color = '#808080'  # Grey color for unassigned
            else:
                # This node is assigned
                shape = self.NODE_SHAPES[0]
                cluster_id = self.data.s01[i].argmax().item()
                color = self.PALETTE[cluster_id % len(self.PALETTE)]

            self.net.add_node(
                f"L0_{i}",
                label=self.data.l0_names.get(i, str(i)),
                shape=shape,
                size=self.NODE_SIZES[0],
                color=color,
                title=self._generate_tooltip_html(0, i),
                x=random.uniform(-500, 500),
                y=self.LEVEL_Y_COORDS[0] + random.uniform(-50, 50),  # Add jitter
                physics=True
            )

        # Add L0 edges (syntactic dependencies) - No changes here
        rows, cols = self.data.adj_l0.nonzero(as_tuple=True)
        for u, v in zip(rows, cols):
            if u.item() < v.item():
                self.net.add_edge(f"L0_{u}", f"L0_{v}", width=1, color='rgba(200, 200, 200, 0.3)')

    def _add_higher_level_nodes_and_edges(self, level: int):
        if level == 1:
            adj = self.data.adj_l1
            s_matrix_next = self.data.s12
            num_nodes = adj.shape[0]
            for i in range(num_nodes):
                l0_connection_strength = self.data.s01[:, i].max().item()
                l2_connection_strength = s_matrix_next[i].max().item()
                if (l0_connection_strength >= self.l0_assignment_threshold or
                        l2_connection_strength >= self.l1_assignment_threshold):
                    self.included_l1_nodes.add(i)
                    assignments_to_next = s_matrix_next.argmax(dim=1)
                    super_cluster_id = assignments_to_next[i].item()
                    border_color = self.PALETTE[super_cluster_id % len(self.PALETTE)]
                    self.net.add_node(
                        f"L1_{i}",
                        label=self.data.l1_names.get(i, f"C1_{i}"),
                        shape=self.NODE_SHAPES[1], size=self.NODE_SIZES[1],
                        color={"background": "#555555", "border": border_color},
                        borderWidth=4, title=self._generate_tooltip_html(1, i),
                        x=random.uniform(-400, 400), y=self.LEVEL_Y_COORDS[1],
                        physics=True
                    )
            rows, cols = adj.nonzero(as_tuple=True)
            for u_item, v_item in zip(rows.tolist(), cols.tolist()):
                if u_item < v_item and u_item in self.included_l1_nodes and v_item in self.included_l1_nodes:
                    weight = adj[u_item, v_item].item()
                    self.net.add_edge(
                        f"L1_{u_item}", f"L1_{v_item}", value=weight,
                        width=0.5 + weight * 4, color="rgba(255, 107, 71, 0.7)",
                        title=f'Adj: {weight:.3f}'
                    )
        elif level == 2:
            adj = self.data.adj_l2
            num_nodes = adj.shape[0]
            for i in range(num_nodes):
                self.net.add_node(
                    f"L2_{i}",
                    label=self.data.l2_names.get(i, f"C2_{i}"),
                    shape=self.NODE_SHAPES[2], size=self.NODE_SIZES[2],
                    color={"background": "#CCCCCC", "border": "#FFFFFF"}, borderWidth=1,
                    title=self._generate_tooltip_html(2, i), x=random.uniform(-400, 400),
                    y=self.LEVEL_Y_COORDS[2], physics=True
                )
            rows, cols = adj.nonzero(as_tuple=True)
            for u_item, v_item in zip(rows.tolist(), cols.tolist()):
                if u_item < v_item:
                    weight = adj[u_item, v_item].item()
                    self.net.add_edge(
                        f"L2_{u_item}", f"L2_{v_item}", value=weight,
                        width=0.5 + weight * 4, color="rgba(70, 130, 180, 0.9)",
                        title=f'Adj: {weight:.3f}'
                    )

    def _add_inter_level_edges(self):
        """Adds edges representing assignments between hierarchical levels."""
        # L0 -> L1 edges (only draw if the target L1 node was included)
        for l0_idx in range(self.data.s01.shape[0]):
            max_assignment_score = self.data.s01[l0_idx].max().item()
            if max_assignment_score >= self.l0_assignment_threshold:
                l1_idx = self.data.s01[l0_idx].argmax().item()
                self.net.add_edge(
                    f"L0_{l0_idx}", f"L1_{l1_idx}",
                    dashes=True, color='rgba(200, 200, 200, 0.2)', width=0.5,
                    title=f'Assign (L0-L1): {max_assignment_score:.2f}'
                )

        # L1 -> L2 edges (only draw if the source L1 node was included)
        for l1_idx in range(self.data.s12.shape[0]):
            max_assignment_score = self.data.s12[l1_idx].max().item()
            # You can add a threshold here as well if needed
            if max_assignment_score >= self.l1_assignment_threshold:
                l2_idx = self.data.s12[l1_idx].argmax().item()
                self.net.add_edge(
                    f"L1_{l1_idx}", f"L2_{l2_idx}", dashes=[5, 5],
                    color='rgba(70, 130, 180, 0.5)', width=1.5,
                    title=f'Assign (L1-L2): {max_assignment_score:.2f}'
                )

    def generate_html(self, filename: str):
        """Generates HTML for visualization"""
        self._add_level_0_nodes_and_edges()
        # The filtering happens inside this method for L1
        self._add_higher_level_nodes_and_edges(level=1)
        self._add_higher_level_nodes_and_edges(level=2)
        # Inter-level edges are now drawn considering the filtered nodes
        self._add_inter_level_edges()

        self.net.show(filename, notebook=False)
        logging.info(f"Saved visualization to {filename}")


# Main function remains the same, but we pass the new threshold to the visualizer
def main():
    """Main function to drive the visualization pipeline."""

    # --- Constants and Setup ---
    MODEL_PATH = f"models/grid_search/STF_HC_Voto_Relatorio/best_model_DiffPool_20250903_135039_lr0.0001_hd_32_bs4_dec0.1_lk10.0_en0.01_rc0.1_ct0.1_bl0.1_rp0.1_l20.01_rep02.pth"
    DATASET = "STF_HC_Voto_Relatorio"
    LANG = "portuguese_voto"
    CONFIG = load_config(LANG, "src/utils/config.json")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ROOT = f"data/datasets/{DATASET}"
    SPLIT_USED = "test"

    timestamp = get_timestamp()
    setup_logging(log_file=f"cg_visualization_{DATASET}_{timestamp}.log")
    logging.info(f"============  STARTING CG VISUALIZATION: '{DATASET}'  ============")

    # --- Load Model and Data ---
    weights = torch.load(MODEL_PATH, map_location=DEVICE)
    diffpool_model = DiffPool(
        max_num_nodes=3000, in_channels=100, hidden_channels=100,
        out_channels=2, inner_channels=32, softmax_assign=True, decrease_proportion=0.1
    )
    diffpool_model.load_state_dict(weights)
    diffpool_model = diffpool_model.to(device=DEVICE)

    tgd_test = TextGraphDatasetOnDisk(
        root=ROOT, split=SPLIT_USED, batch_size=1, node_feature_size=100,
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

    adj_l1_pruned = cg.adj_l1.clone()
    K = int(adj_l1_pruned.shape[0] * 2.5)
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
        l2_names={i: f"C2_{i}" for i in range(cg.adj_l2.shape[0])}
    )

    # --- Generate Visualization ---
    logging.info("Generating interactive HTML visualization...")
    # Pass the threshold to the visualizer
    visualizer = HierarchicalGraphVisualizer(data=viz_data,
                                             l0_assignment_threshold=0.01,
                                             l1_assignment_threshold=0.01)


    path_to_visualization = Path(f"data/visualization/{DATASET}/{SPLIT_USED}/")
    os.makedirs(path_to_visualization, exist_ok=True)

    visualization_output = path_to_visualization / f"visualization_{cg.data_sample_id:05d}_{timestamp}.html"
    visualizer.generate_html(
        filename=str(visualization_output)
    )


if __name__ == "__main__":
    main()
