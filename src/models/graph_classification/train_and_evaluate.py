import json
import logging
import os
import time
from collections import deque
from typing import Tuple, Optional

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.loader import DenseDataLoader
from tqdm import tqdm

from src.data.text_graph_dataset_ondisk import TextGraphDatasetOnDisk
from src.models.graph_classification.gnn import DiffPool
from src.models.graph_classification.utils import loss_config_to_tag
from src.models.graph_explainability.cg_evaluation_metrics import ConceptCompletenessCalculatorV2


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
    logging.info(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    logging.info(f"Model Initialized with {model.num_parameters} parameters.")

    return model, optimizer



def compute_aux_losses(model_outputs, data, loss_config):
    """
    Compute auxiliary losses using the detailed DiffPool model output.

    Args:
        model_outputs: Output tuple from model(x, adj, mask, debug=True)
        data: Input batch containing .x
        loss_config: Dict with keys like 'link', 'entropy', 'reconstruction'

    Returns:
        torch.Tensor (scalar loss)
    """
    logits, link_loss, entropy_loss, (emb_l1, _), (emb_l2, _), (s01, s12) = model_outputs
    x = data.x  # [B, N, d]

    aux_loss = 0.0

    # Link prediction loss
    if loss_config.get("link", 0.0) > 0:
        aux_loss += loss_config["link"] * link_loss.mean()

    # Entropy regularization
    if loss_config.get("entropy", 0.0) > 0:
        aux_loss += loss_config["entropy"] * entropy_loss.mean()

    # Reconstruction loss (batched)
    if loss_config.get("reconstruction", 0.0) > 0:
        # x: [B, N, d], s01: [B, N, C], emb_l1: [B, C, d]
        B, N, d = x.shape
        _, _, C = s01.shape

        s_sum = s01.sum(dim=1, keepdim=True).clamp(min=1e-8)  # [B, 1, C]
        s_norm = s01 / s_sum  # [B, N, C]
        x_bar = torch.einsum("bnc,bnd->bcd", s_norm, x)  # [B, C, d]

        recon_loss = F.mse_loss(emb_l1, x_bar)
        aux_loss += loss_config["reconstruction"] * recon_loss

    return aux_loss


def train(
        model: torch.nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        device: torch.device,
        loss_config: dict,
        accumulation_steps: int = 5,  # 👈  New
        scaler: Optional[GradScaler] = None,
        return_acc: bool = False,
        verbose: bool = True,
) -> Tuple[float, Optional[float]]:
    """
    Trains the GNN model for one epoch.

    Args:
        model (torch.nn.Module): The GNN model to be trained.
        loader (torch_geometric.data.DataLoader): DataLoader providing the training data.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        device
        loss_config (dict): Loss configuration dictionary.
        verbose
        loss_fn (callable): Loss function to be minimized.
        accumulation_steps
        scaler
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
    assert accumulation_steps >= 1, "`accumulation_steps` must be ≥ 1"

    model.train()

    scaler = scaler or GradScaler(enabled=torch.cuda.is_available())

    total_loss, seen_samples = 0.0, 0
    correct_preds = 0

    # Smoothing helpers
    window_size = 10
    roll_cls, roll_link, roll_ent = (
        deque(maxlen=window_size) for _ in range(3)
    )

    pbar = tqdm(loader, disable=False, desc="Train")
    optimizer.zero_grad(set_to_none=True)  # more efficient

    for step, data in enumerate(pbar, start=1):

        data = data.to(device)
        optimizer.zero_grad()

        with autocast(enabled=torch.cuda.is_available()):
            model_outputs =model(data.x, data.adj, data.mask, debug=True)
            logits, obj1, obj2, g_layer1, g_layer2, s = model_outputs

            y = torch.as_tensor(data.y, device=device).squeeze()
            if y.dim() > 1:
                y = y.sum(dim=1)  # keep your original intention

            cls_loss = loss_fn(logits, y)
            aux_loss = compute_aux_losses(model_outputs, data, loss_config or {})
            loss = (cls_loss + aux_loss) / accumulation_steps

        scaler.scale(loss).backward()

        # Update after every `accumulation_steps` mini-batches
        if step % accumulation_steps == 0 or step == len(loader):
            # Optional gradient clipping BEFORE scaler.step
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # # --- Metrics & logging ------------------------------------------------
        #
        # cls_val, link_val, ent_val = (
        #     cls_loss.item(),
        #     link_loss.mean().item(),
        #     ent_loss.mean().item(),
        # )
        # roll_cls.append(cls_val)
        # roll_link.append(link_val)
        # roll_ent.append(ent_val)
        #
        # pbar.set_postfix(
        #     cls=f"{sum(roll_cls) / len(roll_cls):.4f}",
        #     link=f"{alpha_link * sum(roll_link) / len(roll_link):.4f}",
        #     ent=f"{alpha_entropy * sum(roll_ent) / len(roll_ent):.4f}",
        # )
        # pbar.update(0)  # 👈 Force redraw

        batch_size = y.size(0)
        total_loss += loss.item() * accumulation_steps * batch_size  # undo /acc_steps
        seen_samples += batch_size

        if return_acc:
            preds = logits.argmax(dim=1)
            correct_preds += (preds == y).sum().item()

    # logging.info(
    #     f"Loss information:\tcls={sum(roll_cls) / len(roll_cls):.4f},\tlink={alpha_link * sum(roll_link) / len(roll_link):.4f},\tent={alpha_entropy * sum(roll_ent) / len(roll_ent):.4f}")
    avg_loss = total_loss / seen_samples
    avg_acc = (correct_preds / seen_samples) if return_acc else None
    return (avg_loss, avg_acc) if return_acc else avg_loss


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
            # print("*" * 50)
            # print(f"y_pred: {predicted.tolist()}")
            # print(f"y_true: {y.tolist()}")
            pass

    # Calculate evaluation metrics
    accuracy = total_correct / len(loader.dataset)
    macro_f1 = f1_score(true_labels, pred_labels, average='macro')
    average_loss = total_loss / len(loader.dataset)

    cr = classification_report(true_labels, pred_labels, output_dict=True)

    return accuracy, macro_f1, true_labels, pred_labels, average_loss, cr


# Training and Validation Loop
def train_and_validate(model, train_loader, val_loader,
                       optimizer, loss_fn, patience, epochs, lr, hidden_dim, batch_size,
                       use_softmax, decrease_prop, dataset_name, device, timestamp, grid_search=False, verbose=True,
                       loss_config={}):
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
    best_hybrid = 0
    best_epoch = 0
    best_model_path = None
    best_completeness = 0
    times = []

    # timestamp = get_timestamp()
    loss_tag = loss_config_to_tag(loss_config)
    writer = SummaryWriter(
        f"runs/diffpool_{dataset_name}_{timestamp}_pat{patience}_ep{epochs}_lr{lr}_hd{hidden_dim}_bs{batch_size}_sm{use_softmax}_dp{decrease_prop}_{loss_tag}_gs{grid_search}"
    )

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            return_acc=True,
            verbose=verbose,
            device=device,
            loss_config=loss_config
        )

        val_acc, val_macro_f1, val_y_true, val_y_pred, val_loss, cr_val = test(
            model, val_loader, loss_fn, device=device, verbose=verbose
        )

        completeness_calculator = ConceptCompletenessCalculatorV2(model, train_loader, val_loader, device)
        completeness_scores = completeness_calculator.calculate_concept_completeness()
        avg_completeness = sum(score for score, _ in completeness_scores) / len(completeness_scores)

        hybrid_score = 0.5 * val_macro_f1 + 0.5 * avg_completeness

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('ConceptCompleteness/val', avg_completeness, epoch)
        writer.add_scalar('HybridScore/val', hybrid_score, epoch)

        logging.info(f'Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | '
                     f'Val Acc: {val_acc:.4f} | Val F1: {val_macro_f1:.4f} | '
                     f'Completeness: {avg_completeness:.4f} | Hybrid: {hybrid_score:.4f}')

        if hybrid_score > best_hybrid:
            best_val_f1 = val_macro_f1
            best_completeness = avg_completeness
            best_hybrid = hybrid_score
            best_epoch = epoch

            model_name = type(model).__name__
            path_prefix = "models/grid_search" if grid_search else "models"
            model_dir = f"{path_prefix}/{dataset_name}/"
            os.makedirs(model_dir, exist_ok=True)

            best_model_path = os.path.join(
                model_dir,
                f'{dataset_name}_{model_name}_{timestamp}_'
                f'lr{lr}_hd{hidden_dim}_bs{batch_size}_'
                f'softmax{use_softmax}_dp{decrease_prop}_'
                f'{loss_tag}_f1{val_macro_f1:.4f}_comp{avg_completeness:.4f}_hyb{hybrid_score:.4f}_ep{best_epoch:03d}.pth'
            )
            torch.save(model.state_dict(), best_model_path)

            if verbose:
                logging.info("Classification Report for 'validation' set")
                logging.info(f"\n{json.dumps(cr_val, indent=3)}")

                cm = confusion_matrix(val_y_true, val_y_pred)
                logging.info(f"\n{cm}")

        times.append(time.time() - start_time)

        if (epoch - best_epoch) >= patience:
            logging.info("Early stopping triggered.")
            break

    if verbose:
        logging.info(f"Median time per epoch: {torch.tensor(times).median():.3f}s")

    writer.close()

    return best_model_path, best_val_f1, best_completeness, best_hybrid


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
