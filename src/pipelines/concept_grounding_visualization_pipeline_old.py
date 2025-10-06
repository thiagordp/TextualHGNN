"""
Visualizing CG results in a Pyvis graph
"""
import glob
import json
import logging
import os
import time
from pathlib import Path

import torch
import tqdm
from matplotlib import pyplot as plt

from src.models.graph_classification.gnn import DiffPool
from src.models.graph_explainability.concept_grounding import ConceptGrounding
from src.models.graph_explainability.embeddings_oracle import EmbeddingOracle
from src.models.graph_explainability.llm_oracle import LLMOracle
from src.pipelines.concept_grounding_pipeline import retrieve_graph
from src.utils.general_utils import setup_logging, load_config

from src.utils.models_utils import get_timestamp
import torch_geometric.transforms as T
from src.data.text_graph_dataset_ondisk import TextGraphDatasetOnDisk


def main():
    """Main function"""

    #
    # Constants
    #
    MODEL_PATH = f"models/grid_search/STF_HC_Voto_Relatorio/best_model_DiffPool_20250903_135039_lr0.0001_hd_32_bs4_dec0.1_lk10.0_en0.01_rc0.1_ct0.1_bl0.1_rp0.1_l20.01_rep02.pth"
    DATASET = "STF_HC_Voto_Relatorio"
    max_num_nodes = 3000
    EMBEDDING_PATH = 'data/external/embeddings/glove_legal_100.bin'
    LANG = "portuguese_voto"
    CONFIG = load_config(LANG, "src/utils/config.json")
    EMBEDDINGS_DATABASE_PATH = f"data/oracle/embeddings_{DATASET}.db"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ROOT = f"data/datasets/{DATASET}"
    DOCUMENTS_TO_EXPLAIN = 1

    timestamp = get_timestamp()
    setup_logging(log_file=f"cg_dataset_{DATASET}_Diffpool_pipeline_{timestamp}.log")

    #
    # Load graph
    #
    logging.info(f"============  STARTING CG VISUALIZATION: '{DATASET}'  ============")
    logging.info(f"CONFIG: \n{json.dumps(CONFIG, indent=3)}\n")
    logging.info(f"MODEL CHECKPOINT: {MODEL_PATH}")

    #
    # Load Model
    #
    logging.info("-- Loading weights --")
    weights = torch.load(MODEL_PATH)
    diffpool_model = DiffPool(
        max_num_nodes=3000,
        in_channels=100,
        hidden_channels=100,
        out_channels=2,
        inner_channels=32,
        softmax_assign=True,
        decrease_proportion=0.1,
    )

    diffpool_model.load_state_dict(weights)
    diffpool_model = diffpool_model.to(device=DEVICE)

    #
    # Load dataset
    #
    logging.info("-- Loading test dataset ---")
    tgd_test = TextGraphDatasetOnDisk(
        root=ROOT,
        split="test",
        batch_size=1,
        node_feature_size=100,
        transform=T.ToDense(num_nodes=3000),
        max_num_nodes=3000,
        lang=LANG
    )
    logging.info(f"Test set size: {len(tgd_test)}")

    #
    # Setup CG
    #
    oracle = EmbeddingOracle(
        model_path=EMBEDDING_PATH,
        db_path=EMBEDDINGS_DATABASE_PATH,
        batch_size=500000
    )
    PROMPT_LLM_ORACLE_PATH = "data/prompts/llm_oracle.txt"

    # Retrieve the API key from environment variables for security
    api_key = os.getenv("TOGETHER_API_KEY")
    llm_oracle = LLMOracle(
        system_prompt=PROMPT_LLM_ORACLE_PATH,
        language=LANG,
        api_key=api_key,
    )

    nx_graphs = Path(ROOT) / "test" / "interim"
    nx_graphs = str(nx_graphs) + "/*.pt"
    stored_graph_paths = glob.glob(nx_graphs)
    stored_graph_paths = list(stored_graph_paths)
    print(f"NX Graphs at '{nx_graphs}' with: {len(stored_graph_paths)} NX graphs.")

    #
    # Run CG over all L0/L1
    #
    docs_explained = 0
    doc_index = 1
    while docs_explained < DOCUMENTS_TO_EXPLAIN:
        docs_explained = 1
        max_per_label = max(int(DOCUMENTS_TO_EXPLAIN / 2), 1)
        predicted_per_label = {}

        model_checkpoint_file = MODEL_PATH.split('/')[-1].replace(".pth", "")
        output_path = f"data/explanations/{DATASET}/{model_checkpoint_file}"

        data_element = tgd_test[doc_index * -1].to(device=DEVICE)
        cg = ConceptGrounding(
            graph_model=diffpool_model,
            embedding_oracle=oracle,
            llm_oracle=llm_oracle,
            graph=data_element,
            hyper_nodes_to_explain=None,
            nodes_per_hyper_node=None,
            original_raw_file_path=f"{ROOT}/test/raw/",
            language=LANG
        )

        graph = retrieve_graph(stored_graph_paths, str(cg.data_sample_id), output_path=output_path)

        cg.concept_grounding()
        os.makedirs(output_path, exist_ok=True)
        # cg.save_explanation(Path(output_path) / f"{cg.data_sample_id}.json")

        #
        # Retrieve G0 and G1 and assignments.
        #

        x_l0 = cg.x_l0
        cutoff_idx = (x_l0.abs().sum(dim=1) > 1e-5).nonzero().max().item() + 1
        x_l0 = x_l0[:cutoff_idx]

        s01 = cg.s01[:cutoff_idx]
        s12 = cg.s12

        x_l1 = cg.x_l1
        adj_l0 = cg.adj_l0[:cutoff_idx, :cutoff_idx]
        adj_l1 = cg.adj_l1

        K = int(x_l1.shape[0] * 2)
        values, indices = torch.topk(adj_l1.abs().flatten(), K)  # top-K by magnitude
        mask = torch.zeros_like(adj_l1, dtype=torch.bool).flatten()
        mask[indices] = True
        mask = mask.view_as(adj_l1)
        adj_l1 = torch.where(mask, adj_l1, torch.tensor(0., device=adj_l1.device, dtype=adj_l1.dtype))

        l0_names = cg.l0_names[:cutoff_idx]
        l1_names = cg.l1_names

        logging.info(f"------- L0 names --------")
        logging.info(l0_names)
        logging.info(f"------- L1 names --------")
        logging.info(l1_names)
        logging.info("*" * 77)

        #
        # Plot the graph in PyVis.
        #

        g0 = nx.from_numpy_array(adj_l0.cpu().numpy())
        g1 = nx.from_numpy_array(adj_l1.cpu().numpy())

        assignments_data = []
        for l0_index in tqdm.tqdm(range(len(l0_names))):
            for l1_index in range(len(l1_names)):
                assignment = cg.s01[l0_index, l1_index].item()  # Use .item() to get float
                if assignment > 0.1:
                    assignments_data.append(
                        (
                            l0_index,
                            l1_index,
                            assignment,
                        )
                    )

        visualizer = HierarchicalGraphVisualizer(
            g0=g0,
            g1=g1,
            g0_names={i: name for i, name in enumerate(cg.l0_names)},
            g1_names={i: name for i, name in enumerate(cg.l1_names)},
            assignments=assignments_data
        )

        visualizer.generate_html("visualizing_cg.html")


import random
import networkx as nx
from pyvis.network import Network
from typing import List, Tuple, Any, Dict


class HierarchicalGraphVisualizer:
    """
    Generates an interactive, multi-layered network graph from NetworkX graphs.

    This class creates a visualization with distinct horizontal "zones" for each level,
    where nodes can spread out freely. It's designed to show how lower-level
    nodes (e.g., tokens) are grouped into higher-level abstract concepts.
    """

    # --- Configuration Constants ---
    LEVEL_Y_COORDS = {0: 600, 1: 350, 2: 100, 3: -50}
    NODE_COLORS = {
        0: '#D3D3D3',
        1: '#a3c4f3',
        2: '#b39ddb',
        'prediction': '#ff6f61'
    }
    PHYSICS_OPTIONS = """
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -2000,
          "centralGravity": 0.3,
          "springLength": 120,
          "avoidOverlap": 0.5
        },
        "stabilization": {
          "enabled": true,
          "iterations": 500,
          "updateInterval": 25
        },
        "minVelocity": 0.75
      }
    }
    """.strip()

    def __init__(self,
                 g0: nx.Graph,
                 g1: nx.Graph,
                 assignments: List[Tuple[Any, Any, float]],
                 g0_names: Dict[Any, str] = None,
                 g1_names: Dict[Any, str] = None):
        """
        Initializes the visualizer with graph data.

        Args:
            g0 (nx.Graph): A NetworkX graph for Level 0.
            g1 (nx.Graph): A NetworkX graph for Level 1.
            assignments (List[Tuple[Any, Any, float]]): A list of tuples, where each
                tuple represents an assignment `(node_idx_from_g0, node_idx_from_g1, weight)`.
            g0_names (Dict[Any, str], optional): A dictionary mapping node IDs in g0
                to their display names. If None, the node IDs themselves are used.
            g1_names (Dict[Any, str], optional): A dictionary mapping node IDs in g1
                to their display names. If None, the node IDs themselves are used.
        """
        self.g0 = g0
        self.g1 = g1
        self.assignments = assignments
        self.g0_names = g0_names or {n: str(n) for n in g0.nodes()}
        self.g1_names = g1_names or {n: str(n) for n in g1.nodes()}
        self.net = Network(
            height="100vh",
            width="100%",
            notebook=False,
            directed=True,
            select_menu=True,
            filter_menu=True
        )

    def _add_nodes(self):
        """Adds and styles all nodes for all levels to the PyVis graph."""
        # --- MODIFICATION: Add nodes with prefixed IDs to avoid collision ---
        # Level 0 Nodes
        for node in self.g0.nodes():
            node_id = f"g0_{node}"
            self.net.add_node(
                node_id,
                label=self.g0_names.get(node, str(node)),
                level=0,
                y=int(self.LEVEL_Y_COORDS[0] * (1 + 0.3 * random.random())),
                fixed={'y': True},
                shape='ellipse',
                color=self.NODE_COLORS[0],
                font={'color': 'black'}
            )

        # Level 1 Nodes
        for node in self.g1.nodes():
            node_id = f"g1_{node}"
            self.net.add_node(
                node_id,
                label=self.g1_names.get(node, str(node)),
                level=1,
                y=int(self.LEVEL_Y_COORDS[1] * (1 + 0.5 * random.random())),
                fixed={'y': True},
                shape='ellipse',
                color=self.NODE_COLORS[1]
            )
        # --- END MODIFICATION ---

        # Higher-level and Prediction Nodes (IDs are already unique)
        g_node_id = "L2_Aggregate"
        self.net.add_node(
            g_node_id,
            label="L1 Concepts",
            level=2,
            y=self.LEVEL_Y_COORDS[2],
            fixed={'y': True},
            shape='ellipse',
            color=self.NODE_COLORS[2]
        )

        pred_node_id = "Prediction"
        self.net.add_node(
            pred_node_id,
            label="Prediction",
            level=3,
            x=0,
            y=self.LEVEL_Y_COORDS[3],
            fixed=True,
            shape='box',
            color=self.NODE_COLORS['prediction']
        )

    def _add_edges(self):
        """Adds and styles all edges (intra-level and inter-level)."""
        smooth_options = {'type': 'curvedCW', 'roundness': 0.15}

        # --- MODIFICATION: Add edges using prefixed IDs ---
        # Intra-level edges for G0
        for u, v in self.g0.edges():
            self.net.add_edge(f"g0_{u}", f"g0_{v}", dashes=False, color='gray', smooth=smooth_options)

        # Intra-level edges for G1
        for u, v in self.g1.edges():
            self.net.add_edge(f"g1_{u}", f"g1_{v}", dashes=False, color='black', smooth=smooth_options)

        # Inter-level assignment edges (L0 -> L1) now reflect their weight.
        for u, v, w in self.assignments:
            alpha = max(0.15, w)
            color = f'rgba(180, 180, 180, {alpha})'
            width = 0.5 + (w * 2)
            dashes = [7, 4]

            self.net.add_edge(
                f"g0_{u}", f"g1_{v}",  # Use prefixed IDs
                dashes=dashes,
                color=color,
                width=width,
                arrows='',
                title=f"Weight: {w:.2f}"
            )

        # Edges from L1 to L2 aggregate node
        g_node_id = "L2_Aggregate"
        for node in self.g1.nodes():
            self.net.add_edge(f"g1_{node}", g_node_id, dashes=True, color='rgba(179, 157, 219, 0.5)', arrows='')
        # --- END MODIFICATION ---

        # Final edge to prediction
        self.net.add_edge(g_node_id, "Prediction", dashes=True, color='rgba(255, 111, 97, 0.7)')

    def generate_html(self, filename: str = "hierarchical_graph.html"):
        """
        Generates and saves the interactive HTML graph file.

        Args:
            filename (str): The name of the output HTML file.
        """
        self._add_nodes()
        self._add_edges()
        # self.net.set_options(self.PHYSICS_OPTIONS)
        self.net.toggle_physics(True)
        time.sleep(1)
        self.net.toggle_physics(False)
        self.net.show(filename, notebook=False)
        print(f"Graph saved to {filename}")


if __name__ == "__main__":
    main()
