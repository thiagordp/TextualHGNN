"""
DiffPool Training Pipeline for Graph Classification
===================================================

This script implements the training and evaluation pipeline for a Graph Neural Network (GNN)
with Differentiable Pooling (DiffPool) on text-based graph datasets using PyTorch Geometric.

Key Features:
-------------
1. Model Architecture: DiffPool with GraphSAGE layers for embedding and pooling.
2. Automatic Class Weighting: Computes class weights dynamically using `sklearn.utils.class_weight`.
3. Embedding Interpretation: Saves node embeddings from L1 and L2 pooling layers for concept grounding.
4. Comprehensive Logging: Logs to terminal and `logs/{DATASET}_diffpool_training.log`.
5. Visualization: Supports embedding visualization using t-SNE.
6. Early Stopping: Monitors validation performance to avoid overfitting.


Date: 2024-04-12
"""

import json
import logging
import os
import time
from typing import List

import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from torch import nn

from src.models.graph_classification.train_and_evaluate import initialize_model, calculate_class_weights, \
    train_and_validate, evaluate_model, load_datasets, create_loaders
from src.utils.general_utils import load_config, setup_logging
from src.utils.models_utils import get_timestamp

# from utils.general_utils import format_time_elapsed

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LANG = "italian"
LANG = "english"

CONFIG = load_config(LANG, "src/utils/config.json")


# Setup Logging


def format_time_elapsed(start_time: float) -> str:
    """
    Format elapsed time into 'HH:MM:SS.mmm' format.

    Args:
        start_time (float): The start time as returned by `time.time()`.

    Returns:
        str: The formatted time elapsed.
    """
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours):02}:{int(minutes):02}:{seconds:06.3f}"


setup_logging(log_file=f"training_model_experiment_{CONFIG['DATASET']}.log")
logging.info(f"============  STARTING EXPERIMENT {CONFIG['DATASET']}  ============")
logging.info(f"CONFIG: \n{json.dumps(CONFIG, indent=3)}")


# Visualization of Embeddings
def visualize_embeddings(embeddings: torch.Tensor, labels: List[int], title: str = "Node Embeddings") -> None:
    tsne = TSNE(n_components=2)
    reduced_embeddings = tsne.fit_transform(embeddings.cpu().numpy())
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', s=10)
    plt.colorbar()
    plt.title(title)
    plt.show()


def main():
    """
    Main function to orchestrate the training, validation, and testing of the GNN model.
    """
    logging.info("Current directory: " + os.getcwd())
    timestamp = get_timestamp()

    # Load datasets
    start_time = time.time()
    tgd_train, tgd_val, tgd_test = load_datasets(
        root=CONFIG["ROOT"],
        max_num_nodes=CONFIG["NUM_NODES"],
        node_feature_size=CONFIG["NODE_FEATURE_DIM"],
        lang=LANG
    )
    logging.info(f"Loaded datasets finished after {format_time_elapsed(start_time)}")

    # Create data loaders
    start_time = time.time()
    train_loader, val_loader, test_loader = create_loaders(
        tgd_train,
        tgd_val,
        tgd_test,
        batch_size=CONFIG["BATCH_SIZE"]
    )
    logging.info(f"Data loaders creation finished after {format_time_elapsed(start_time)}")

    start_time = time.time()
    # Initialize model and optimizer
    model, optimizer = initialize_model(
        in_channels=tgd_train.num_node_attributes,
        out_channels=tgd_train.num_classes,
        max_num_nodes=CONFIG["NUM_NODES"],
        lr=0.0001,
        hidden_dim=CONFIG["HIDDEN_DIM"],
        inner_dim=64,
        softmax_assign=True,
        decrease_proportion=0.1,
        device=DEVICE
    )
    logging.info(f"Model Init finished after {format_time_elapsed(start_time)}")

    # Define loss function
    start_time = time.time()
    class_weights = calculate_class_weights(tgd_train, device=DEVICE)
    # loss_fn = nn.NLLLoss(weight=class_weights)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # # Train and validate the model
    best_model_path = train_and_validate(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        patience=CONFIG["PATIENCE"],
        epochs=CONFIG["EPOCHS"],
        lr=CONFIG["LR"],
        hidden_dim=CONFIG["HIDDEN_DIM"],
        batch_size=CONFIG["BATCH_SIZE"],
        use_softmax=CONFIG["SOFTMAX_ASSIGN"],
        decrease_prop=CONFIG["DECREASE_PROPORTION"],
        dataset_name = CONFIG["DATASET"],
        grid_search=False,
        verbose=False,
        device=DEVICE,
        timestamp=timestamp
    )
    best_model_path = "models/grid_search/IMDB/IMDB_DiffPool_20250425_142400_lr0.005_hd100_bs64_softmaxTrue_decrease_prop0.1_valmacrof1score0.7986_epoch014.pth"
    logging.info(f"Train and Validation model finished after {format_time_elapsed(start_time)}")

    # Load the best model and evaluate on train, validation, and test datasets

    start_time = time.time()
    if best_model_path:
        model.load_state_dict(torch.load(best_model_path))
        #evaluate_model(model, train_loader, loss_fn, "Train", device=DEVICE)
        #evaluate_model(model, val_loader, loss_fn, "Validation", device=DEVICE)
        evaluate_model(model, test_loader, loss_fn, "Test", device=DEVICE)

        logging.info(f"Best model evaluation finished after {format_time_elapsed(start_time)}")
    else:
        logging.info("No best model was saved.")




if __name__ == "__main__":
    main()
