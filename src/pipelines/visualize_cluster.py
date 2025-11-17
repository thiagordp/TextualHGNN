# src/pipelines/visualize_clusters.py
import logging

import torch
import pandas as pd
from collections import Counter, defaultdict
import os
import json
import tqdm

# Assuming the script is in src/pipelines, we adjust the path
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.general_utils import load_config
from src.models.graph_classification.gnn import DiffPool
from src.models.graph_classification.train_and_evaluate import load_datasets, create_loaders
from src.data.text_graph_dataset_ondisk import TextGraphDatasetOnDisk


def load_trained_model(model_path, config, device):
    """Loads a pre-trained DiffPool model from a state dictionary."""
    # Initialize a new model with the same architecture
    model = DiffPool(
        max_num_nodes=config["NUM_NODES"],
        in_channels=config["NODE_FEATURE_DIM"],
        hidden_channels=config["HIDDEN_DIM"],
        out_channels=2,
        inner_channels=config["INNER_DIM"],
        softmax_assign=config["SOFTMAX_ASSIGN"],
        decrease_proportion=config["DECREASE_PROPORTION"]
    ).to(device)

    # Load the saved weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logging.info(f"Model loaded successfully from {model_path}")
    return model


def get_token_mapping(dataset: TextGraphDatasetOnDisk):
    """
    Retrieves the mapping from token indices to token strings.
    This assumes the dataset object has an 'inv_vocab' attribute or similar.
    """
    # Based on your dataset implementation, the token mapping seems to be
    # stored in an inverted vocabulary. We will try to access it.
    if hasattr(dataset, 'inv_vocab'):
        # Creates a simple dictionary from index to token
        return {i: token for i, token in enumerate(dataset.inv_vocab)}
    else:
        raise AttributeError("The dataset does not have an 'inv_vocab' attribute to map token IDs to strings.")


def analyze_cluster_assignments(model, loader, token_map, device):
    """
    Runs the model on the data and aggregates the tokens assigned to each cluster.
    """
    # {cluster_id: [list of all tokens assigned to it]}
    cluster_token_aggregator = defaultdict(list)

    logging.info("Analyzing cluster assignments across the dataset...")
    with torch.no_grad():
        for data in tqdm.tqdm(loader, desc="Processing Batches"):
            data = data.to(device)
            # Run model in debug mode to get the assignment matrix s01
            _, _, _, _, _, (s01, _) = model(data.x, data.adj, data.mask, debug=True)

            # Get hard assignments for the first pooling layer
            concept_ids_batch = s01.argmax(dim=-1)  # Shape: [batch_size, num_nodes]

            num_graphs_in_batch = data.x.size(0)
            for i in range(num_graphs_in_batch):
                num_nodes = int(data.mask[i].sum())
                if num_nodes == 0: continue

                # Node indices in the original vocabulary are simply their position
                node_indices = data.node_indices[i, :num_nodes].cpu().tolist()

                # Get the concepts for this specific graph
                single_graph_concepts = concept_ids_batch[i, :num_nodes].cpu().tolist()

                for node_idx, concept_id in zip(node_indices, single_graph_concepts):
                    token_string = token_map.get(node_idx, "UNK")
                    cluster_token_aggregator[concept_id].append(token_string)

    return cluster_token_aggregator


def get_top_k_tokens_for_clusters(cluster_tokens, k=2):
    """
    Finds the top k most frequent tokens for each cluster.
    """
    logging.info("\nCalculating top tokens for each cluster...")
    cluster_labels = {}

    # Sort clusters by ID for consistent output
    sorted_cluster_ids = sorted(cluster_tokens.keys())

    for cluster_id in sorted_cluster_ids:
        tokens = cluster_tokens[cluster_id]
        if not tokens:
            cluster_labels[cluster_id] = "Empty"
            continue

        # Count token frequencies and get the top k
        token_counts = Counter(tokens)
        top_k = token_counts.most_common(k)

        # Format the label string
        label = ", ".join([f"'{token}' ({count})" for token, count in top_k])
        cluster_labels[cluster_id] = label

    return cluster_labels


if __name__ == '__main__':
    # --- Configuration ---
    LANG = "portuguese"
    CONFIG = load_config(LANG, "src/utils/config.json")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- 1. EDIT THIS LINE: Point to your best trained model file ---
    BEST_MODEL_PATH = "models/grid_search/STF_HC/best_model_20250822_022641_lk1000.0_en0.1_rc0.1_ct1.0_bl1.0_rp10.0_l210.0.pth"  # <-- EDIT HERE

    if not os.path.exists(BEST_MODEL_PATH):
        logging.info(f"Error: Model file not found at '{BEST_MODEL_PATH}'")
        logging.info("Please update the BEST_MODEL_PATH variable with the correct path to your trained model.")
    else:
        # --- 2. Load Data and Token Mapping ---
        # We only need the test set for this analysis
        _, _, tgd_test = load_datasets(
            root=CONFIG["ROOT"],
            max_num_nodes=CONFIG["NUM_NODES"],
            node_feature_size=CONFIG["NODE_FEATURE_DIM"],
            lang=LANG
        )
        # It's important to use the same batch size you trained with, or a smaller one
        _, _, test_loader = create_loaders(tgd_test, tgd_test, tgd_test, batch_size=1)


        for data in test_loader:
            logging.info(data.keys())
            doc_id = data_sample_id = int.from_bytes(data.doc_id, byteorder='little')
            # e.g. ['x', 'edge_index', 'edge_attr', 'y', ...]

            for key, value in data.items():
                logging.info(f"{key}: {value}")

        try:
            token_map = get_token_mapping(tgd_test)

            # --- 3. Load Model ---
            # We need to add n_classes to the config for model initialization
            CONFIG['N_CLASSES'] = tgd_test.num_classes
            model = load_trained_model(BEST_MODEL_PATH, CONFIG, DEVICE)

            # --- 4. Analyze Assignments and Get Labels ---
            cluster_tokens = analyze_cluster_assignments(model, test_loader, token_map, DEVICE)
            cluster_labels = get_top_k_tokens_for_clusters(cluster_tokens, k=2)

            # --- 5. Print the Results ---
            logging.info("\n--- Cluster Visualization Results (L1) ---")
            logging.info("Each cluster is labeled with its top 2 most frequent tokens.\n")

            results_df = pd.DataFrame(list(cluster_labels.items()), columns=['Cluster ID', 'Top Tokens'])
            logging.info(results_df.to_string(index=False))

        except AttributeError as e:
            logging.info(f"\nAn error occurred: {e}")
            logging.info("Could not find the token vocabulary in the dataset object.")
            logging.info("Please check the implementation of 'TextGraphDatasetOnDisk' to ensure 'inv_vocab' is available.")
        except Exception as e:
            logging.info(f"\nAn unexpected error occurred: {e}")