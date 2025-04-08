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

Author:
-------
Thiago Raulino Dal Pont
Date: 2024-04-12
"""

import os
import time
import logging
from datetime import datetime
from typing import List, Tuple

import json
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.loader import DenseDataLoader
import torch_geometric.transforms as T

from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.data.text_graph_dataset_ondisk import TextGraphDatasetOnDisk
from src.models.graph_classification.gnn import DiffPool
from torch.utils.tensorboard import SummaryWriter

from src.utils.models_utils import get_timestamp

# from utils.general_utils import format_time_elapsed

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Configuration
CONFIG = {
    "EPOCHS": 100,
    "PATIENCE": 20,
    "LR": 1e-5,
    "LR_DECAY": 0,
    "NODE_FEATURE_DIM": 100,
    "HIDDEN_DIM": 100,
    "NUM_NODES": 1000,
    "BATCH_SIZE": 16,
    "DATASET": "Imprisonment-IT",  # or "Imprisonment-IT"
    "LANG": "italian",  # or "italian"
    "ROOT": "data/datasets/Imprisonment-IT",
}


# Setup Logging
def setup_logging(log_folder='logs', log_file='diffpool_training.log'):
    os.makedirs(log_folder, exist_ok=True)
    log_path = os.path.join(log_folder, log_file)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )


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


def calculate_class_weights(dataset: TextGraphDatasetOnDisk) -> torch.Tensor:
    """
    Calculate class weights automatically for imbalanced datasets.

    This function computes balanced class weights using the `compute_class_weight` method
    from `sklearn.utils.class_weight`, making it suitable for handling class imbalance during
    training. The computed weights are returned as a `torch.Tensor` compatible with PyTorch
    loss functions like `nn.NLLLoss` or `nn.CrossEntropyLoss`.

    Args:
        dataset (TextGraphDatasetOnDisk): A PyTorch Geometric dataset where each `data` object
                                          must have a `y` attribute representing the class label.

    Returns:
        torch.Tensor: A tensor containing the computed class weights.
    """
    labels = []
    for i, data in enumerate(dataset):
        if isinstance(data.y, torch.Tensor):
            labels.append(data.y.item())  # Safe extraction if y is a tensor

        elif isinstance(data.y, (int, float)):
            labels.append(data.y)  # Directly use numeric types

        elif isinstance(data.y, bytes):
            try:
                # Attempt to convert from byte data to an integer
                decoded_value = int.from_bytes(data.y, byteorder='little', signed=False)
                labels.append(decoded_value)
                logging.debug(f"Decoded byte label at index {i}: {decoded_value}")

            except ValueError as e:
                logging.warning(f"Failed to decode byte label at index {i}: {data.y}. Skipping. Error: {e}")

        else:
            logging.warning(f"Unsupported label type at index {i}: {type(data.y)}. Skipping this data point.")

    if not labels:
        logging.warning("Empty dataset or unsupported label types. Returning default class weights.")
        return torch.ones(1, device=DEVICE, dtype=torch.float)

    # Compute balanced class weights using sklearn
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=torch.unique(torch.tensor(labels)).numpy(),
        y=labels
    )

    logging.info(f"Computed Class Weights: {class_weights}")
    return torch.tensor(class_weights, device=DEVICE, dtype=torch.float)


# Visualization of Embeddings
def visualize_embeddings(embeddings: torch.Tensor, labels: List[int], title: str = "Node Embeddings") -> None:
    tsne = TSNE(n_components=2)
    reduced_embeddings = tsne.fit_transform(embeddings.cpu().numpy())
    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=labels, cmap='viridis', s=10)
    plt.colorbar()
    plt.title(title)
    plt.show()


# Load Datasets with Transformations
def load_datasets(root: str, num_nodes: int, node_feature_size: int) -> Tuple[TextGraphDatasetOnDisk, ...]:
    """
    Load datasets and apply transformations.

    Args:
        root (str): The root directory for the dataset.
        num_nodes (int): Maximum number of nodes in the graphs.
        node_feature_size (int): Size of the node feature vectors.

    Returns:
        tuple: Train, validation, and test datasets.
    """

    splits = ['train', 'validation', 'test']
    datasets = []

    for split in splits:
        datasets.append(TextGraphDatasetOnDisk(
            root=root,
            split=split,
            batch_size=1,
            transform=T.ToDense(num_nodes=num_nodes),
            max_num_nodes=CONFIG["NUM_NODES"],
            node_feature_size=node_feature_size,
            lang=CONFIG['LANG']
        ))
    return tuple(datasets)


def create_loaders(train_dataset, val_dataset, test_dataset, batch_size):
    """
    Create data loaders for training, validation, and testing.

    Args:
        train_dataset: Training dataset.
        val_dataset: Validation dataset.
        test_dataset: Testing dataset.
        batch_size (int): Batch size for the loaders.

    Returns:
        tuple: Train, validation, and test data loaders.
    """
    return (
        DenseDataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        DenseDataLoader(val_dataset, batch_size=batch_size, shuffle=False),
        DenseDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    )


def initialize_model(in_channels, out_channels, max_num_nodes, lr, hidden_dim):
    """
    Initialize the GNN model and optimizer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output classes.
        max_num_nodes (int): Maximum number of nodes in the graphs.
        lr (float): Learning rate for the optimizer.
        hidden_dim(int): Hidden embedding dim.

    Returns:
        tuple: The initialized model and optimizer.

    """
    model = DiffPool(
        max_num_nodes=max_num_nodes,
        in_channels=in_channels,
        hidden_channels=hidden_dim,
        out_channels=out_channels
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    logging.info(f"Model Initialized with {model.num_parameters} parameters.")
    return model, optimizer


def train(model, loader, optimizer, loss_fn, return_acc=False):
    """
    Trains the GNN model for one epoch.

    Args:
        model (torch.nn.Module): The GNN model to be trained.
        loader (torch_geometric.data.DataLoader): DataLoader providing the training data.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        loss_fn (callable): Loss function to be minimized.
        return_acc (bool, optional): If True, returns the training accuracy along with the loss. Defaults to False.

    Returns:
        float: Average training loss over the epoch.
        float, optional: Training accuracy if `return_acc` is True.

    Notes:
        - The function trains the model for one full pass over the dataset provided by the DataLoader.
        - The model is set to training mode (`model.train()`) to ensure layers like dropout and batch normalization
          behave correctly during training.
        - The loss is calculated for each batch and accumulated to compute the average loss.
        - If `return_acc` is True, the function also computes and returns the accuracy.
    """
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for data in tqdm(loader, desc="Processing Training Batches"):
        data = data.to(DEVICE)

        optimizer.zero_grad()

        # Forward pass
        output, l, e = model(data.x, data.adj, data.mask)

        # Ensure data.y is already on the device and properly processed
        y = torch.tensor(data.y).to(DEVICE)

        # If data.y needs to be summed along a dimension, do it directly
        y = torch.sum(y, dim=1)

        # Compute loss
        loss = loss_fn(output, y)
        loss.backward()
        optimizer.step()

        # Accumulate loss
        total_loss += loss.item() * y.size(0)

        # Calculate accuracy
        if return_acc:
            _, predicted = torch.max(output, dim=1)  # Assuming output is logits
            correct_predictions += (predicted == y).sum().item()
            total_samples += y.size(0)

    average_loss = total_loss / len(loader.dataset)

    if return_acc:
        accuracy = correct_predictions / total_samples
        return average_loss, accuracy

    return average_loss


@torch.no_grad()
def test(model, loader, loss_fn, show_pred=False):
    """
    Evaluate the performance of the GNN model on the given dataset.

    Args:
        model (torch.nn.Module): The GNN model to be evaluated.
        loader (torch_geometric.loader.DataLoader): DataLoader providing the dataset for evaluation.
        loss_fn (callable): The loss function used to calculate the test loss.
        show_pred (bool, optional): If True, prints the predicted and true labels for each batch. Defaults to False.

    Returns:
        float: Accuracy of the model on the dataset.
        float: Macro F1 score of the model on the dataset.
        list: True labels of the dataset.
        list: Predicted labels of the dataset.
        float: Average test loss over the dataset.

    Notes:
        - This function evaluates the model's performance by computing the accuracy, macro F1 score, and test loss.
        - The model is set to evaluation mode (`model.eval()`) to disable dropout and batch normalization layers.
        - The predictions and true labels can be printed for inspection by setting `show_pred=True`.
    """
    model.eval()  # Set the model to evaluation mode
    total_correct = 0
    total_loss = 0.0
    true_labels = []
    pred_labels = []

    for data in tqdm(loader, desc="Processing Batches"):
        data = data.to(DEVICE)

        # Forward pass: Get model predictions
        output, _, _ = model(data.x, data.adj, data.mask)

        # Prepare target labels
        y = torch.tensor(data.y).to(DEVICE)
        y = torch.sum(y, dim=1)  # Summing along the appropriate dimension

        # Compute the loss
        loss = loss_fn(output, y)
        total_loss += loss.item() * y.size(0)

        # Get the predicted labels
        _, predicted = torch.max(output, dim=1)

        # Accumulate true and predicted labels for evaluation
        true_labels.extend(y.tolist())
        pred_labels.extend(predicted.tolist())

        # Count correct predictions
        total_correct += (predicted == y).sum().item()

        # Optionally print predictions
        if show_pred:
            print("*" * 50)
            print(f"y_pred: {predicted.tolist()}")
            print(f"y_true: {y.tolist()}")

    # Calculate evaluation metrics
    accuracy = total_correct / len(loader.dataset)
    macro_f1 = f1_score(true_labels, pred_labels, average='macro')
    average_loss = total_loss / len(loader.dataset)

    cr = classification_report(true_labels, pred_labels, output_dict=True)

    return accuracy, macro_f1, true_labels, pred_labels, average_loss, cr


# Training and Validation Loop
def train_and_validate(model, train_loader, val_loader, optimizer, loss_fn, patience, epochs):
    """
    Train and validate the model.

    Args:
        model: The GNN model to train.
        train_loader: DataLoader for the training dataset.
        val_loader: DataLoader for the validation dataset.
        optimizer: Optimizer for training the model.
        loss_fn: Loss function to use for training.
        patience (int): Number of epochs to wait for improvement before early stopping.
        epochs (int): Total number of epochs to train.

    Returns:
        str: Path to the best model saved during training.
    """
    best_val_f1 = 0
    best_epoch = 0
    best_model_path = None
    times = []

    timestamp = get_timestamp()
    writer = SummaryWriter(f"runs/diffpool_{CONFIG['DATASET']}_{timestamp}")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train(model, train_loader, optimizer, loss_fn, return_acc=True)
        val_acc, val_macro_f1, _, _, val_loss, cr_val = test(model, val_loader, loss_fn)
        train_acc, train_macro_f1, _, _, _, cr_train = test(model, train_loader, loss_fn)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_epoch = epoch

            model_name = type(model).__name__
            lr = optimizer.param_groups[0]['lr']

            best_model_path = f'models/{CONFIG["DATASET"]}_{model_name}_{timestamp}_lr{lr}_valmacrof1score{best_val_f1:.4f}_epoch{best_epoch:03d}.pth'
            torch.save(model.state_dict(), best_model_path)

            logging.info("Classification Report for 'train' set")
            logging.info(f"\n{json.dumps(cr_train, indent=3)}")

            logging.info("Classification Report for 'validation' set")
            logging.info(f"\n{json.dumps(cr_val, indent=3)}")

        logging.info(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
                     f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, '
                     f'Val Macro F1: {val_macro_f1:.4f}')

        times.append(time.time() - start_time)

        if (epoch - best_epoch) >= patience:
            logging.info(
                f"'Early stopping' triggered at epoch {epoch} after {patience} epochs without improvement on Macro F1 score")
            break

    logging.info(f"Median time per epoch: {torch.tensor(times).median():.3f}s")
    writer.close()

    return best_model_path


def evaluate_model(model, loader, loss_fn, dataset_name):
    """
    Evaluate the model on a given dataset.

    Args:
        model: The trained model.
        loader: DataLoader for the dataset to evaluate.
        dataset_name (str): Name of the dataset (e.g., 'Train', 'Validation', 'Test').

    Returns:
        None
    """
    logging.info("-" * 50)
    logging.info(dataset_name)
    acc, macro_f1, true_labels, pred_labels, loss, cr = test(model, loader, loss_fn, show_pred=True)

    logging.info(f"\n{dataset_name} Accuracy: {acc:.4f}")
    logging.info(f"{dataset_name} Macro F1: {macro_f1:.4f}")
    logging.info(f"{dataset_name} Loss: {loss:.4f}")

    report = classification_report(
        true_labels,
        pred_labels,
        target_names=[str(i) for i in range(len(set(true_labels)))],
        digits=4,
        output_dict=True
    )
    logging.info(f"\n{dataset_name} Classification Report:\n{json.dumps(report, indent=3)}")

    cm = confusion_matrix(true_labels, pred_labels)
    logging.info(f"\n{dataset_name} Confusion Matrix:\n{cm}")


def main():
    """
    Main function to orchestrate the training, validation, and testing of the GNN model.
    """
    logging.info("Current directory: " + os.getcwd())

    # Load datasets
    start_time = time.time()
    tgd_train, tgd_val, tgd_test = load_datasets(
        CONFIG["ROOT"],
        CONFIG["NUM_NODES"],
        CONFIG["NODE_FEATURE_DIM"]
    )
    logging.info(f"Loaded datasets finished after {format_time_elapsed(start_time)}")

    # Create data loaders
    start_time = time.time()
    train_loader, val_loader, test_loader = create_loaders(
        tgd_train,
        tgd_val,
        tgd_test,
        CONFIG["BATCH_SIZE"]
    )
    logging.info(f"Data loaders creation finished after {format_time_elapsed(start_time)}")

    start_time = time.time()
    # Initialize model and optimizer
    model, optimizer = initialize_model(
        in_channels=tgd_train.num_node_attributes,
        out_channels=tgd_train.num_classes,
        max_num_nodes=CONFIG["NUM_NODES"],
        hidden_dim=CONFIG["HIDDEN_DIM"],
        lr=CONFIG["LR"]
    )
    logging.info(f"Model Init finished after {format_time_elapsed(start_time)}")

    # Define loss function
    start_time = time.time()
    class_weights = calculate_class_weights(tgd_train)
    loss_fn = nn.NLLLoss(weight=class_weights)

    # Train and validate the model
    best_model_path = train_and_validate(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        CONFIG["PATIENCE"],
        CONFIG["EPOCHS"]
    )
    logging.info(f"Train and Validation model finished after {format_time_elapsed(start_time)}")

    # Load the best model and evaluate on train, validation, and test datasets

    start_time = time.time()
    if best_model_path:
        model.load_state_dict(torch.load(best_model_path))
        evaluate_model(model, train_loader, loss_fn, "Train")
        evaluate_model(model, val_loader, loss_fn, "Validation")
        evaluate_model(model, test_loader, loss_fn, "Test")

        logging.info(f"Best model evaluation finished after {format_time_elapsed(start_time)}")
    else:
        logging.info("No best model was saved.")


if __name__ == "__main__":
    main()
