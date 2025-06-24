# main.py
import logging
import os
from datetime import datetime
import random
from pathlib import Path
import torch
import tqdm
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Local imports from our modularized project structure
from config import ModelConfig, TrainingConfig, DataConfig, VisualizationConfig
from data_preprocessing import preprocess_text
from engine import train, test
from graph_builder import DocumentGraphBuilder
from model import ExplainableHierarchicalGNN
from src.multi_graph.dataset import MultiLevelGraphOnDiskDataset
from visualization import visualize_learned_attentions


def setup_logging():
    """Configures logging to file and console."""
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
    log_filename = f"logs/run_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    logging.info("Logging setup complete.")


def run():
    """Main function to run the entire pipeline using OnDiskDataset."""
    setup_logging()
    model_cfg, train_cfg, data_cfg, viz_cfg = ModelConfig(), TrainingConfig(), DataConfig(), VisualizationConfig()

    # 1. Instantiate the builder once; it's a dependency for the dataset class
    builder = DocumentGraphBuilder(
        spacy_model=data_cfg.SPACY_MODEL,
        embedding_model=data_cfg.EMBEDDING_MODEL,
        similarity_threshold=data_cfg.SIMILARITY_THRESHOLD
    )

    # 2. Instantiate On-Disk Datasets for each split
    # NOTE: This assumes you have structured your raw data into train/validation/test folders
    # e.g., data/datasets/IMDB/raw/train/, data/datasets/IMDB/raw/validation/, etc.
    logging.info("--- Initializing On-Disk Datasets (processing will run once if needed) ---")
    dataset_root = f"data/datasets_multilevel/{data_cfg.DATASET_NAME}"

    train_dataset = MultiLevelGraphOnDiskDataset(root=dataset_root, split="train", builder=builder)
    val_dataset = MultiLevelGraphOnDiskDataset(root=dataset_root, split="validation", builder=builder)
    test_dataset = MultiLevelGraphOnDiskDataset(root=dataset_root, split="test", builder=builder)

    # 3. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=train_cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=train_cfg.BATCH_SIZE)

    logging.info(f"Datasets ready: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # 4. Initialize Model and Training Components
    class_names = sorted(train_dataset.class_map.keys())
    model_cfg.OUT_CHANNELS = len(class_names)
    model = ExplainableHierarchicalGNN(builder.embedding_dim, model_cfg.HIDDEN_CHANNELS, model_cfg.OUT_CHANNELS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    # 5. Training Loop
    # ... (Training loop logic is unchanged) ...

    # 6. Final Evaluation and Explainability (UPDATED WORKFLOW)
    logging.info("\n--- Final Evaluation on Test Set ---")
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))

    # For explainability, we pass the smaller test_dataset into memory.
    # For very large test sets, the 'test' function would also need to be adapted.
    test_graphs_in_memory = [g for g in test_dataset]
    test_metrics, _, _, final_explanations = test(model, test_loader, test_graphs_in_memory)
    logging.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}, F1-Score: {test_metrics['f1']:.4f}")

    # --- CHANGE: Loop through ALL explanations and generate a visualization for each ---
    if final_explanations:
        logging.info(f"\n--- Generating {len(final_explanations)} Explanation Visualizations ---")
        output_viz_dir = Path(viz_cfg.VISUALIZATION_FOLDER) / data_cfg.DATASET_NAME
        output_viz_dir.mkdir(parents=True, exist_ok=True)

        # Create the map from filename to its original nx_graph object once
        nx_graph_map_for_viz = {g.graph['filename']: g for g in test_graphs_in_memory}

        # Use tqdm for a progress bar
        for explanation in tqdm.tqdm(final_explanations, desc="Generating all explanation visualizations"):
            doc_filename = explanation['doc_id']
            original_nx_graph = nx_graph_map_for_viz.get(doc_filename)

            if original_nx_graph:
                output_html_path = output_viz_dir / f"{Path(doc_filename).stem}_learned_attention.html"

                # Call the visualization function for the current explanation
                visualize_learned_attentions(
                    nx_graph=original_nx_graph,
                    explanation=explanation,
                    dep_label_map=builder.dep_label_map,
                    output_filename=str(output_html_path)
                )
            else:
                logging.warning(f"Could not find original NetworkX graph for doc_id: {doc_filename} to visualize.")
    else:
        logging.warning("No explanations were generated, skipping visualization.")


if __name__ == '__main__':
    run()
