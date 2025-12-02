import json
import logging
import os
import time
from collections import deque, defaultdict
from typing import Tuple, Optional, Callable, Dict

import numpy as np
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
from sklearn.metrics import classification_report, f1_score, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.loader import DenseDataLoader
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm
from torch_geometric.data import Data

from src.data.text_graph_dataset_ondisk import TextGraphDatasetOnDisk
from src.models.graph_classification.gnn import DiffPool, DiffPoolMinCut
from src.models.graph_classification.utils import loss_config_to_tag
from src.models.graph_explainability.cg_evaluation_metrics import \
    ConceptConformityCalculator, ModularityCalculator, SilhouetteScoreCalculator, ConceptCompletenessCalculator


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
def load_datasets(root: str, max_num_nodes: int, node_feature_size: int, lang: str, preprocessing_fn=None,
                  graph_builder_type: str = "graph_of_words", use_pmi:bool=False) -> Tuple[
    TextGraphDatasetOnDisk, ...]:
    """
    Load datasets and apply transformations.

    Args:
        root (str): The root directory for the dataset.
        max_num_nodes (int): Maximum number of nodes in the graphs.
        node_feature_size (int): Size of the node feature vectors.
        lang (str)
        preprocessing_fn (Callable, optional): Preprocessing function.
        graph_builder_type (str, optional): The name of the builder to use.

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
            lang=lang,
            preprocessing_fn=preprocessing_fn,
            graph_builder_type=graph_builder_type,
            use_pmi=use_pmi
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
                     inner_dim, softmax_assign: bool, decrease_proportion, device: str | torch.device, l2=0):
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
        l2

    Returns:
        tuple: The initialized model and optimizer.

    """
    # model = DiffPool(
    #     max_num_nodes=max_num_nodes,
    #     in_channels=in_channels,
    #     hidden_channels=hidden_dim,
    #     out_channels=out_channels,
    #     inner_channels=inner_dim,
    #     softmax_assign=softmax_assign,
    #     decrease_proportion=decrease_proportion
    # ).to(device)
    model = DiffPoolMinCut(
        max_num_nodes=max_num_nodes,
        in_channels=in_channels,
        hidden_channels=hidden_dim,
        out_channels=out_channels,
        inner_channels=inner_dim,
        softmax_assign=softmax_assign,
        decrease_proportion=decrease_proportion
    ).to(device)

    logging.info(f"Model structure: {model.__class__.__name__}")
    logging.info(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=l2)
    logging.info(f"Model Initialized with {model.num_parameters} parameters.")

    return model, optimizer


def _compute_layer_losses(S: torch.Tensor, Z: torch.Tensor, X: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Helper function to compute regularization losses for a *single* pooling layer.

    Args:
        S: Assignment matrix [B, N_in, N_out]
        Z: Cluster embeddings [B, N_out, D]
        X: Input node embeddings [B, N_in, D]

    Returns:
        A dictionary of scalar loss tensors.
    """
    if Z is None or S is None:
        return {
            "recon": torch.tensor(0.0, device=X.device),
            "contrastive": torch.tensor(0.0, device=X.device),
            "balance": torch.tensor(0.0, device=X.device),
            "repel": torch.tensor(0.0, device=X.device)
        }

    # --- Reconstruction Loss (from paper) ---
    # L_recon = || Z - S^T_norm @ X ||
    # This ensures the cluster embedding Z is close to the mean of its members X.
    s_sum = S.sum(dim=1, keepdim=True).clamp(min=1e-8)  # [B, 1, N_out]
    s_norm = S / s_sum  # [B, N_in, N_out]

    S_transpose_norm = s_norm.transpose(1, 2)  # [B, N_out, N_in]
    X_bar = S_transpose_norm @ X  # [B, N_out, D] (mean of members)
    recon_loss = F.mse_loss(Z, X_bar)

    # --- Contrastive Loss ---
    Z_norm = F.normalize(Z, dim=-1)  # [B, N_out, D]
    X_bar_norm = F.normalize(X_bar, dim=-1)  # [B, N_out, D]

    sim_matrix = torch.einsum("bcd,bkd->bck", Z_norm, X_bar_norm) / 0.07  # temp=0.07

    B, N_out, _ = Z.shape
    device = Z.device
    labels = torch.arange(N_out, device=device).unsqueeze(0).expand(B, -1)  # [B, N_out]

    contrastive_loss = F.cross_entropy(
        sim_matrix.reshape(-1, N_out),
        labels.reshape(-1)
    )

    # --- Balance Loss ---
    # Encourages clusters to have equal size by penalizing deviation from uniform.
    S_sum_per_cluster = S.sum(dim=1)  # [B, N_out]
    S_sum_total = S_sum_per_cluster.sum(dim=1, keepdim=True).clamp(min=1e-8)  # [B, 1]
    S_dist = S_sum_per_cluster / S_sum_total  # [B, N_out]

    target_dist = torch.full_like(S_dist, 1.0 / N_out)
    balance_loss = F.kl_div(
        S_dist.log().clamp(min=-100),
        target_dist,
        reduction='batchmean'
    )

    # --- Repel Loss ---
    # Encourage cluster embeddings Z to be far apart
    Z_norm = F.normalize(Z, dim=-1)  # [B, N_out, D]
    Z_sim = Z_norm @ Z_norm.transpose(1, 2)
    identity_mask = torch.eye(N_out, device=device).unsqueeze(0)  # [1, N_out, N_out]
    repel_loss = F.relu(Z_sim * (1.0 - identity_mask)).mean()

    return {
        "recon": recon_loss,
        "contrastive": contrastive_loss,
        "balance": balance_loss,
        "repel": repel_loss
    }


def compute_aux_losses(
        model_outputs: tuple,
        data: Data,
        loss_config: dict
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    [REFACTORED]
    Compute auxiliary losses for *all* hierarchical layers (L1 and L2).
    """

    # [FIX] Unpack the new 7-item tuple from our 2-level model
    logits, link_loss_sum, entropy_loss_sum, (emb_l1, adj_l1), (emb_l2, adj_l2), cluster_logits, (s01,
                                                                                                  s12) = model_outputs

    X_l0 = data.x  # Initial node features [B, N, D]

    aux_loss = torch.tensor(0.0, device=X_l0.device)
    loss_report = {}

    # --- 1. DiffPool/MinCut Structural Losses (already summed) ---
    if loss_config.get("link", 0.0) > 0:
        aux_loss += loss_config["link"] * link_loss_sum
        loss_report["link_loss"] = link_loss_sum.item()

    if loss_config.get("entropy", 0.0) > 0:
        aux_loss += loss_config["entropy"] * entropy_loss_sum
        loss_report["entropy_loss"] = entropy_loss_sum.item()

    # --- 2. Per-Layer Regularization Losses ---

    # --- Level 1 Losses (Tokens -> Concepts) ---
    l1_losses = _compute_layer_losses(s01, emb_l1, X_l0)

    # --- Level 2 Losses (Concepts -> Arguments) ---
    # L2 losses are computed using L1's output (emb_l1) as input
    l2_losses = _compute_layer_losses(s12, emb_l2, emb_l1)

    # --- Sum L1 and L2 losses and add to total aux_loss ---
    if loss_config.get("reconstruction", 0.0) > 0:
        recon_total = l1_losses["recon"] + l2_losses["recon"]
        aux_loss += loss_config["reconstruction"] * recon_total
        loss_report["recon_loss"] = recon_total.item()

    if loss_config.get("contrastive", 0.0) > 0:
        contrastive_total = l1_losses["contrastive"] + l2_losses["contrastive"]
        aux_loss += loss_config["contrastive"] * contrastive_total
        loss_report["contrastive_loss"] = contrastive_total.item()

    if loss_config.get("balance", 0.0) > 0:
        balance_total = l1_losses["balance"] + l2_losses["balance"]
        aux_loss += loss_config["balance"] * balance_total
        loss_report["balance_loss"] = balance_total.item()

    if loss_config.get("repel", 0.0) > 0:
        repel_total = l1_losses["repel"] + l2_losses["repel"]
        aux_loss += loss_config["repel"] * repel_total
        loss_report["repel_loss"] = repel_total.item()

    return aux_loss, loss_report


def train(
        model: torch.nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn,
        device: torch.device,
        loss_config: dict,
        accumulation_steps: int = 1,  # 👈  New
        scaler: Optional[GradScaler] = None,
        return_acc: bool = False,
        verbose: bool = True,
) -> Tuple[float, Optional[float], Dict[str, float]]:
    """
    Trains the GNN model for one epoch.
    Returns dictionary of all computed losses.
    """
    model.train()
    scaler = scaler or GradScaler(enabled=torch.cuda.is_available())

    total_cls_loss, total_aux_loss, seen_samples = 0.0, 0.0, 0
    correct_preds = 0

    # Create an accumulator for the loss report
    epoch_loss_report = defaultdict(float)

    pbar = tqdm(loader, disable=False, desc="Train")
    optimizer.zero_grad(set_to_none=True)  # more efficient

    for step, data in enumerate(pbar, start=1):
        data = data.to(device)

        # Note: data.adj is the dense, weighted adj matrix from ToDense
        # data.x is the node features
        with autocast(enabled=torch.cuda.is_available(), device_type=str(device)):
            # Pass all dense data to the model
            model_outputs = model(data.x, data.adj, data.mask, debug=True)
            logits, _, _, _, _, _, _ = model_outputs

            y = torch.tensor([label[0] for label in data.y], dtype=torch.long, device=device)

            # 1. Compute Classification Loss
            cls_loss = loss_fn(logits, y)
            # 2. Compute All Auxiliary Losses
            aux_loss, batch_loss_report = compute_aux_losses(
                model_outputs, data, loss_config or {}
            )

            loss = cls_loss + aux_loss
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        # --- Metrics & logging ---
        batch_size = y.size(0)
        total_cls_loss += cls_loss.item() * batch_size
        total_aux_loss += aux_loss.item() * batch_size
        seen_samples += batch_size

        # Accumulate metrics for epoch report
        for k, v in batch_loss_report.items():
            epoch_loss_report[k] += v * batch_size

        if return_acc:
            preds = logits.argmax(dim=1)
            correct_preds += (preds == y).sum().item()

    avg_cls_loss = total_cls_loss / seen_samples
    avg_aux_loss = total_aux_loss / seen_samples
    avg_acc = (correct_preds / seen_samples) if return_acc else None

    # Finalize epoch loss report
    final_loss_report = {k: v / seen_samples for k, v in epoch_loss_report.items()}
    final_loss_report["cls_loss"] = avg_cls_loss
    final_loss_report["aux_loss"] = avg_aux_loss
    final_loss_report["total_loss"] = avg_cls_loss + avg_aux_loss

    return avg_cls_loss, avg_acc, final_loss_report


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
    total_samples = 0

    for data in loader:
        data = data.to(device)
        batch_size = len(data.y)  # [FIX] Get batch size from list

        # Forward pass: Get model predictions
        output, _, _ = model(data.x, data.adj, data.mask)

        # Prepare target labels
        y = torch.tensor([label[0] for label in data.y], dtype=torch.long, device=device)

        # Compute the loss
        loss = loss_fn(output, y)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        # Get the predicted labels
        _, predicted = torch.max(output, dim=1)

        # Accumulate true and predicted labels for evaluation
        true_labels.extend(y.tolist())
        pred_labels.extend(predicted.tolist())

    accuracy = accuracy_score(true_labels, pred_labels)
    macro_f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0.0)
    average_loss = total_loss / total_samples

    cr = classification_report(true_labels, pred_labels, output_dict=True, zero_division=0.0)
    return accuracy, macro_f1, true_labels, pred_labels, average_loss, cr


def train_and_validate(model, train_loader, val_loader,
                       optimizer, loss_fn, patience, epochs, lr, hidden_dim, batch_size,
                       use_softmax, decrease_prop, dataset_name, device, timestamp, grid_search=False, verbose=True,
                       loss_config={}, repetition=None):
    """
    Train and validate the model, calculating a full suite of interpretability metrics.
    """
    best_e_score = -1.0
    best_model_path = None
    best_epoch = 0
    best_epoch_results = {}
    times = []

    loss_tag = loss_config_to_tag(loss_config)

    model_name_str = model.__class__.__name__
    run_name = f"runs/{model_name_str}_{dataset_name}_{timestamp}_rep{repetition}_{loss_tag}"
    writer = SummaryWriter(run_name)
    logging.info(f"TensorBoard Run: {run_name}")

    for epoch in tqdm(range(1, epochs + 1), desc="Training epochs:"):

        start_time = time.time()

        # --- TRAINING ---
        train_cls_loss, train_acc, train_loss_report = train(
            model=model, loader=train_loader, optimizer=optimizer, loss_fn=loss_fn,
            return_acc=True, verbose=verbose, device=device, loss_config=loss_config
        )

        for k, v in train_loss_report.items():
            writer.add_scalar(f'Loss/train_{k}', v, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)

        # --- VALIDATION & METRICS ---
        model.eval()
        all_preds, all_labels = [], []

        # Initialize metric calculators for the epoch
        completeness_calc = ConceptCompletenessCalculator()
        conformity_calc = ConceptConformityCalculator()
        modularity_calc = ModularityCalculator()
        silhouette_calc = SilhouetteScoreCalculator()

        val_cls_loss, val_aux_loss, val_samples = 0.0, 0.0, 0
        val_pbar = tqdm(val_loader, disable=False, desc="Validate")

        with torch.no_grad():
            for data in val_pbar:

                data = data.to(device)
                B = len(data.y)  # Get batch size from list

                model_outputs = model(data.x, data.adj, data.mask, debug=True)
                logits, loss_lp, loss_entropy, (emb_l1, adj_l1), (emb_l2, adj_l2), cluster_logits, (s01, s12) = model_outputs

                y = torch.tensor([label[0] for label in data.y], dtype=torch.long, device=device)

                # --- Calculate Validation Loss ---
                cls_loss = loss_fn(logits, y)
                aux_loss, _ = compute_aux_losses(model_outputs, data, loss_config)
                val_cls_loss += cls_loss.item() * B
                val_aux_loss += aux_loss.item() * B
                val_samples += B

                # --- Prepare for Metrics ---

                pred = logits.argmax(dim=1)
                all_preds.append(pred.cpu())
                y_indices = torch.tensor([x[0] for x in data.y], dtype=torch.long).to(device)
                all_labels.append(y_indices)

                # Get concept assignments (we focus on the first layer for interpretability)
                concept_ids_batch = s01.argmax(dim=-1)

                # The 'batch' attribute maps each node to its graph in the batch
                batch_vector = data.batch if hasattr(data, 'batch') else torch.zeros(s01.size(1), dtype=torch.long,
                                                                                     device=device)
                num_graphs_in_batch = data.x.size(0)

                for i in range(num_graphs_in_batch):
                    num_nodes = int(data.mask[i].sum())
                    if num_nodes == 0: continue

                    x_i = data.x[i, :num_nodes]
                    adj_i = data.adj[i, :num_nodes, :num_nodes]

                    # Ensure y_i is a tensor for the Data object
                    y_val = data.y[i]

                    if isinstance(y_val, list):
                        # Handle case where y is a list of lists/tensors
                        y_i = torch.tensor(y_val, device=device, dtype=torch.long)
                    elif isinstance(y_val, (int, float)):
                        # Handle case where y is a flat list of numbers
                        y_i = torch.tensor([y_val], device=device, dtype=torch.long)
                    else:  # It's already a tensor
                        y_i = y_val

                    y_i = y_i.unsqueeze(dim=0)
                    if y_i.ndim > 1 and y_i.shape[1] > 1:
                        y_i = y_i.sum(dim=1)
                    else:
                        y_i = y_i.long()  # Ensure it's integer type

                    edge_index_i = dense_to_sparse(adj_i)[0]
                    single_graph_data = Data(x=x_i.clone(), edge_index=edge_index_i.clone(), y=y_i.clone())

                    single_graph_concepts = concept_ids_batch[i, :num_nodes]

                    completeness_calc.add_item(y_i, single_graph_concepts)
                    conformity_calc.add_item(single_graph_data, single_graph_concepts)
                    modularity_calc.add_item(single_graph_data, single_graph_concepts)
                    silhouette_calc.add_item(single_graph_data, single_graph_concepts)

        # --- Finalize Epoch Metrics ---
        preds, labels = torch.cat(all_preds).cpu().numpy(), torch.cat(all_labels).cpu().numpy()

        avg_val_cls_loss = val_cls_loss / val_samples
        avg_val_aux_loss = val_aux_loss / val_samples
        avg_val_total_loss = avg_val_cls_loss + avg_val_aux_loss

        with torch.no_grad():
            total_params = 0
            sum_abs_params = 0.0
            for param in model.parameters():
                if param.requires_grad:
                    total_params += param.numel()
                    sum_abs_params += torch.sum(torch.abs(param.data)).item()

        avg_abs_params = sum_abs_params / total_params if total_params > 0 else 0

        macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
        acc = accuracy_score(labels, preds)
        completeness = completeness_calc.calculate()
        conformity = conformity_calc.calculate()
        modularity = modularity_calc.calculate()
        silhouette = silhouette_calc.calculate()

        # Calculate HI-Score and E-Score
        hi_score = (completeness + conformity) / 2
        e_score = (2 * macro_f1 * hi_score) / (macro_f1 + hi_score) if (macro_f1 + hi_score) > 0 else 0

        # --- LOGGING ---
        writer.add_scalar('Loss/val_cls', avg_val_cls_loss, epoch)
        writer.add_scalar('Loss/val_aux', avg_val_aux_loss, epoch)
        writer.add_scalar('Loss/val_total', avg_val_total_loss, epoch)
        writer.add_scalar('Accuracy/val', acc, epoch)
        writer.add_scalar('F1/val_macro', macro_f1, epoch)
        writer.add_scalar('Metrics/Completeness', completeness, epoch)
        writer.add_scalar('Metrics/Conformity', conformity, epoch)
        writer.add_scalar('Metrics/Modularity', modularity, epoch)
        writer.add_scalar('Metrics/Silhouette', silhouette, epoch)
        writer.add_scalar('Overall/HI_Score', hi_score, epoch)
        writer.add_scalar('Overall/E_Score', e_score, epoch)

        # [FIX] Updated logging to use the new loss report

        loss_log = " | ".join(
            [f"{k.replace('_loss', '')}: {v:.3f}" for k, v in train_loss_report.items() if k != 'total_loss'])

        logging.info(
            f'\nEpoch {epoch:03d} | Train Loss: {train_loss_report["total_loss"]:.4f} | Val Loss: {avg_val_total_loss:.4f} | E-Score: {e_score:.4f} (F1: {macro_f1:.4f}, HI: {hi_score:.4f})'
        )
        if loss_log:  # Only print if there are aux losses
            logging.info(f'           | Train Losses: {loss_log}')

        # --- MODEL CHECKPOINTING ---
        if e_score > best_e_score:
            best_e_score = e_score
        best_epoch = epoch
        best_epoch_results = {
            "loss_config": loss_config, "best_epoch": epoch, "macro_f1": macro_f1,
            "completeness": completeness, "conformity": conformity,
            "modularity": modularity, "silhouette": silhouette,
            "hi_score": hi_score, "e_score": e_score, 'accuracy': acc,
            "val_loss": avg_val_total_loss,
            "train_loss": train_loss_report["total_loss"],
            'sum_abs_params': sum_abs_params,
            'avg_abs_params': avg_abs_params,

        }

        model_name = type(model).__name__
        path_prefix = "models/grid_search" if grid_search else "models"
        model_dir = os.path.join(path_prefix, dataset_name)
        os.makedirs(model_dir, exist_ok=True)

        hyperparams = f"lr{lr}_hd_{hidden_dim}_bs{batch_size}_dec{decrease_prop}"

        model_file = f'best_model_{model_name}_{timestamp}_{hyperparams}_{loss_tag}_rep{repetition:02d}.pth' if repetition is not None else f'best_model_{model_name}_{timestamp}_{hyperparams}_{loss_tag}.pth'
        best_model_path = os.path.join(model_dir, model_file)

        torch.save(model.state_dict(), best_model_path)
        logging.info(f"✅ New best model saved with E-Score: {best_e_score:.4f} into '{best_model_path}'")
        times.append(time.time() - start_time)

        if (epoch - best_epoch) >= patience:
            logging.info(f"Early stopping triggered after {epoch} epochs.")

    logging.info(f"Median time per epoch: {torch.tensor(times).median():.3f}s")
    writer.close()

    # Return the results from the best epoch
    return best_model_path, best_epoch_results


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
        output_dict=True,
        zero_division=0.0
    )
    logging.info(f"\n{dataset_name} Classification Report:\n{json.dumps(report, indent=3)}")

    cm = confusion_matrix(true_labels, pred_labels)
    logging.info(f"\n{dataset_name} Confusion Matrix:\n{cm}")
