import json
import logging
import os
import time
from collections import deque
from typing import Tuple

import torch
import torch_geometric.transforms as T
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DenseDataLoader
from tqdm import tqdm

from src.data.text_graph_dataset_ondisk import TextGraphDatasetOnDisk
from src.models.graph_classification.gnn import DiffPool
from src.models.graph_explainability.cg_evaluation_metrics import ConceptCompletenessCalculatorV2
from src.utils.models_utils import get_timestamp


def calculate_class_weights(dataset: TextGraphDatasetOnDisk, device: torch.device) -> torch.Tensor:
    """
    Calculate class weights automatically for imbalanced datasets.

    This function computes balanced class weights using the `compute_class_weight` method
    from `sklearn.utils.class_weight`, making it suitable for handling class imbalance during
    training. The computed weights are returned as a `torch.Tensor` compatible with PyTorch
    loss functions like `nn.NLLLoss` or `nn.CrossEntropyLoss`.

    Args:
        dataset (TextGraphDatasetOnDisk): A PyTorch Geometric dataset where each `data` object
                                          must have a `y` attribute representing the class label.
        device (torch.device): A PyTorch device on which to run the computation.
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
        return torch.ones(1, device=device, dtype=torch.float)

    # Compute balanced class weights using sklearn
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=torch.unique(torch.tensor(labels)).numpy(),
        y=labels
    )

    # Square each weight
    # class_weights = np.square(class_weights)
    # class_weights = np.array([1.0,2.0])

    logging.info(f"Computed Class Weights: {class_weights}")
    return torch.tensor(class_weights, device=device, dtype=torch.float)


# Load Datasets with Transformations
def load_datasets(root: str, max_num_nodes: int, node_feature_size: int, lang: str) -> Tuple[
    TextGraphDatasetOnDisk, ...]:
    """
    Load datasets and apply transformations.

    Args:
        root (str): The root directory for the dataset.
        max_num_nodes (int): Maximum number of nodes in the graphs.
        node_feature_size (int): Size of the node feature vectors.
        lang (str)

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
            transform=T.ToDense(num_nodes=max_num_nodes),
            max_num_nodes=max_num_nodes,
            node_feature_size=node_feature_size,
            lang=lang
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


def initialize_model(in_channels, out_channels, max_num_nodes, lr, hidden_dim,
                     inner_dim, softmax_assign: bool, decrease_proportion, device: str | torch.device):
    """
    Initialize the GNN model and optimizer.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output classes.
        max_num_nodes (int): Maximum number of nodes in the graphs.
        lr (float): Learning rate for the optimizer.
        hidden_dim(int): Hidden embedding dim.
        inner_dim (int): Inner embedding dim.
        softmax_assign (bool): Assign softmax weights.
        decrease_proportion (float): Decrease proportion.
        device (str): Device to use.

    Returns:
        tuple: The initialized model and optimizer.

    """
    model = DiffPool(
        max_num_nodes=max_num_nodes,
        in_channels=in_channels,
        hidden_channels=hidden_dim,
        out_channels=out_channels,
        inner_channels=inner_dim,
        softmax_assign=softmax_assign,
        decrease_proportion=decrease_proportion
    ).to(device)

    logging.info(f"Model structure:")
    # logging.info(summary(model))
    logging.info(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    logging.info(f"Model Initialized with {model.num_parameters} parameters.")

    return model, optimizer


def train(model, loader, optimizer, loss_fn, device,
          alpha_link=0.7, alpha_entropy=0.3, return_acc=False, verbose=True):
    """
    Trains the GNN model for one epoch.

    Args:
        model (torch.nn.Module): The GNN model to be trained.
        loader (torch_geometric.data.DataLoader): DataLoader providing the training data.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        device
        alpha_link
        alpha_entropy
        verbose
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

    batch = 0

    window_size = 10
    rolling_cls_loss = deque(maxlen=window_size)
    rolling_link_loss = deque(maxlen=window_size)
    rolling_entropy = deque(maxlen=window_size)

    # ✅ Create tqdm instance as a variable
    pbar = tqdm(loader, desc="Processing Training Batches")

    cls_loss_timeline = []
    link_loss_timeline = []
    entropy_loss_timeline = []

    scaler = GradScaler()

    for data in pbar:  # iterate over pbar, not tqdm(loader)
        batch += 1
        data = data.to(device)
        optimizer.zero_grad()

        with autocast():  # AMP context
            output, l, e = model(data.x, data.adj, data.mask)
            y = torch.sum(torch.tensor(data.y).to(device), dim=1)
            cls_loss = loss_fn(output, y)
            loss = cls_loss + alpha_link * l.mean() + alpha_entropy * e.mean()


        # Compute and store metrics
        cls_loss_val = cls_loss.item()
        link_loss_val = l.mean().item()
        entropy_val = e.mean().item()

        # Append to smoothing buffers
        rolling_cls_loss.append(cls_loss_val)
        rolling_link_loss.append(link_loss_val)
        rolling_entropy.append(entropy_val)

        # Compute smoothed averages
        smoothed_cls = sum(rolling_cls_loss) / len(rolling_cls_loss)
        smoothed_link = sum(rolling_link_loss) / len(rolling_link_loss)
        smoothed_entropy = sum(rolling_entropy) / len(rolling_entropy)

        cls_loss_timeline.append(cls_loss_val)
        link_loss_timeline.append(link_loss_val)
        entropy_loss_timeline.append(entropy_val)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update tqdm bar
        pbar.set_postfix({
            "ClsLoss": f"{smoothed_cls:.4f}",
            "Link": f"{alpha_link * smoothed_link:.4f}",
            "Entropy": f"{alpha_entropy * smoothed_entropy:.4f}"
        })

        # Accumulate loss
        total_loss += (loss.item() * y.size(0))

        # Calculate accuracy
        if return_acc:
            _, predicted = torch.max(output, dim=1)  # Assuming output is logits
            correct_predictions += (predicted == y).sum().item()
            total_samples += y.size(0)

    average_loss = total_loss / len(loader.dataset)

    if return_acc:
        accuracy = correct_predictions / total_samples
        return average_loss, accuracy

    plt.figure(figsize=(12, 8), dpi=300)
    plt.plot(rolling_cls_loss)
    plt.plot(rolling_link_loss)
    plt.plot(rolling_entropy)
    plt.show()

    return average_loss


@torch.no_grad()
def test(model, loader, loss_fn, device, verbose=True):
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

    for data in loader:
        data = data.to(device)

        # Forward pass: Get model predictions
        output, _, _ = model(data.x, data.adj, data.mask)

        # Prepare target labels
        y = torch.tensor(data.y).to(device)
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
        if verbose:
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
def train_and_validate(model, train_loader, val_loader,
                       optimizer, loss_fn, patience, epochs, lr, hidden_dim, batch_size,
                       use_softmax, decrease_prop, dataset_name, device, timestamp, grid_search=False, verbose=True):
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
        lr (float): Initial learning rate.
        hidden_dim (int): Number of hidden dimensions.
        batch_size (int): Batch size.
        use_softmax (bool): If True, use softmax classification.
        decrease_prop (float): Decrease learning rate by this amount.
        verbose (bool): verbosity
        grid_search (bool): Whether to use a grid search.
        dataset_name (string): Name of the dataset.

    Returns:
        str: Path to the best model saved during training.

    """
    best_val_f1 = 0
    best_epoch = 0
    best_model_path = None
    times = []

    # timestamp = get_timestamp()

    writer = SummaryWriter(
        f"runs/diffpool_{dataset_name}_{timestamp}_pat{patience}_ep{epochs}_lr{lr}_hd{hidden_dim}_bs{batch_size}_sm{use_softmax}_dp{decrease_prop}_gs{grid_search}"
    )

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train(
            model=model, loader=train_loader, optimizer=optimizer, loss_fn=loss_fn,
            return_acc=True, verbose=verbose, device=device, alpha_link=1000, alpha_entropy=0.1)
        val_acc, val_macro_f1, val_y_true, val_y_pred, val_loss, cr_val = test(model, val_loader, loss_fn,
                                                                               device=device, verbose=verbose)
        # train_acc, train_macro_f1, train_y_true, train_y_pred, train_loss, cr_train = test(model, train_loader, loss_fn)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        logging.info(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
                     f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, '
                     f'Val Macro F1: {val_macro_f1:.4f}')

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_epoch = epoch

            model_name = type(model).__name__

            if grid_search:
                best_model_path = f"models/grid_search/{dataset_name}/"
            else:
                best_model_path = f"models/{dataset_name}/"

            os.makedirs(best_model_path, exist_ok=True)

            best_model_path = best_model_path + (f'{dataset_name}_{model_name}_{timestamp}_'
                                                 f'lr{lr}_hd{hidden_dim}_bs{batch_size}_'
                                                 f'softmax{use_softmax}_decrease_prop{decrease_prop}_'
                                                 f'valmacrof1score{best_val_f1:.4f}_epoch{best_epoch:03d}.pth')
            torch.save(model.state_dict(), best_model_path)

            # logging.info("Classification Report for 'train' set")
            # logging.info(f"\n{json.dumps(cr_train, indent=3)}")
            #
            # logging.info(f"Confusion matrix for 'train' set")
            # cm = confusion_matrix(train_y_true, train_y_pred)
            # logging.info(f"\n{cm}")

            if verbose:
                logging.info("Classification Report for 'validation' set")
                logging.info(f"\n{json.dumps(cr_val, indent=3)}")

                logging.info(f"Confusion matrix for 'validation' set")
                cm = confusion_matrix(val_y_true, val_y_pred)
                logging.info(f"\n{cm}")

        completeness_calculator = ConceptCompletenessCalculatorV2(model, train_loader, val_loader, device)
        completeness_scores = completeness_calculator.calculate_concept_completeness()

        logging.info("\nFinal Concept Completeness Scores (per layer):")
        for layer, (avg_score, std_score) in enumerate(list(completeness_scores), start=1):
            logging.info(f"Layer {layer}: {avg_score:.4f} ± {std_score:.4f}")

        times.append(time.time() - start_time)

        if (epoch - best_epoch) >= patience:
            logging.info(
                f"'Early stopping' triggered at epoch {epoch} after {patience} epochs without improvement on Macro F1 score")
            break

    if verbose:
        logging.info(f"Median time per epoch: {torch.tensor(times).median():.3f}s")
    writer.close()

    return best_model_path


def evaluate_model(model, loader, loss_fn, dataset_name, device):
    """
    Evaluate the model on a given dataset.

    Args:
        model: The trained model.
        loader: DataLoader for the dataset to evaluate.
        loss_fn:
        dataset_name (str): Name of the dataset (e.g., 'Train', 'Validation', 'Test').
        device

    Returns:
        None
    """
    logging.info("-" * 50)
    logging.info(dataset_name)
    acc, macro_f1, true_labels, pred_labels, loss, cr = test(model, loader, loss_fn, verbose=True, device=device)

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
