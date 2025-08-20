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
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataLoader
from torch_geometric.loader import DenseDataLoader
from torch_geometric.utils import dense_to_sparse
from tqdm import tqdm
from torch_geometric.data import Data


from src.data.text_graph_dataset_ondisk import TextGraphDatasetOnDisk
from src.models.graph_classification.gnn import DiffPool
from src.models.graph_classification.utils import loss_config_to_tag
from src.models.graph_explainability.cg_evaluation_metrics import  \
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
    Compute auxiliary losses based on interpretable concept structure.

    Input:
        model_outputs:
            Tuple from model(..., debug=True):
                logits: [B, num_classes]
                link_loss: scalar per-graph structural loss
                entropy_loss: scalar per-graph entropy regularization
                emb_l1: [B, C, d]     # Level-1 concept embeddings
                emb_l2: [B, C2, d]    # Level-2 concept embeddings (unused here)
                s01: [B, N, C]        # Soft assignment of N tokens to C clusters
                s12: [B, C, C2]       # (unused)

        data:
            Batch object with:
                x: [B, N, d] — token embeddings

        loss_config:
            Dictionary specifying the weight for each loss component.

    Returns:
        aux_loss: scalar tensor to be added to total loss
    """

    logits, link_loss, entropy_loss, (emb_l1, _), (emb_l2, _), (s01, s12) = model_outputs
    x = data.x  # [B, N, d]
    B, N, d = x.shape
    _, _, C = s01.shape  # C: number of concepts

    aux_loss = 0.0

    # ─────────────────────────────────────────────────────────────────────────────
    # (1) Link Prediction Loss
    # Encourages cluster assignments to preserve adjacency structure:
    #
    #   \mathcal{L}_{link} = \lVert A - SS^\top \rVert_F^2
    #
    # (computed inside DiffPool)
    # ─────────────────────────────────────────────────────────────────────────────
    if loss_config.get("link", 0.0) > 0:
        aux_loss += loss_config["link"] * link_loss.mean()

    # ─────────────────────────────────────────────────────────────────────────────
    # (2) Entropy Loss
    # Encourages sharper (low-entropy) cluster assignments:
    #
    #   \mathcal{L}_{entropy} = \sum_{i,c} S_{ic} \log S_{ic}
    #
    # (also from DiffPool)
    # ─────────────────────────────────────────────────────────────────────────────
    if loss_config.get("entropy", 0.0) > 0:
        aux_loss += loss_config["entropy"] * entropy_loss.mean()

    # ─────────────────────────────────────────────────────────────────────────────
    # (3) Reconstruction Loss
    # Encourage each concept embedding to be close to the mean of its tokens:
    #
    #   \bar{x}_c = \frac{1}{\sum_i S_{ic}} \sum_i S_{ic} x_i
    #   \mathcal{L}_{recon} = \sum_c \lVert z_c - \bar{x}_c \rVert^2
    #
    # where z_c = emb_l1[c]
    # ─────────────────────────────────────────────────────────────────────────────
    s_sum = s01.sum(dim=1, keepdim=True).clamp(min=1e-8)  # [B, 1, C]
    s_norm = s01 / s_sum  # [B, N, C]
    x_bar = torch.einsum("bnc,bnd->bcd", s_norm, x)  # [B, C, d]

    if loss_config.get("reconstruction", 0.0) > 0:
        recon_loss = F.mse_loss(emb_l1, x_bar)
        aux_loss += loss_config["reconstruction"] * recon_loss

    # ─────────────────────────────────────────────────────────────────────────────
    # (4) Contrastive Loss (InfoNCE)
    # Pull each cluster embedding toward its x_bar; push away from others:
    #
    #   \mathcal{L}_{con} = -\sum_c \log \frac{ \exp(\cos(z_c, \bar{x}_c)/\tau) }
    #                                      { \sum_{c'} \exp(\cos(z_c, \bar{x}_{c'})/\tau) }
    #
    # with temperature \tau = 0.07
    # ─────────────────────────────────────────────────────────────────────────────
    if loss_config.get("contrastive", 0.0) > 0:
        z_c = F.normalize(emb_l1, dim=-1)  # [B, C, d]
        x_c = F.normalize(x_bar, dim=-1)  # [B, C, d]
        sim = torch.einsum("bcd,bkd->bck", z_c, x_c) / 0.07  # [B, C, C]
        log_prob = F.log_softmax(sim, dim=-1)
        contrastive_loss = -log_prob.diagonal(dim1=1, dim2=2).mean()
        aux_loss += loss_config["contrastive"] * contrastive_loss

    # ─────────────────────────────────────────────────────────────────────────────
    # (5) Balance Loss
    # Encourage clusters to receive equal total assignment:
    #
    #   \mathcal{L}_{balance} = \sum_j \left( \sum_i S_{ij} \right)^2
    #
    # ─────────────────────────────────────────────────────────────────────────────
    if loss_config.get("balance", 0.0) > 0:
        size = s01.sum(dim=1)  # [B, C]
        balance_loss = (size ** 2).mean()
        aux_loss += loss_config["balance"] * balance_loss

    # ─────────────────────────────────────────────────────────────────────────────
    # (6) Repel Loss
    # Encourage concept embeddings to be diverse (repel each other):
    #
    #   \mathcal{L}_{repel} = \sum_{j \ne k} \max(0, \delta - \lVert z_j - z_k \rVert^2)
    #
    # ─────────────────────────────────────────────────────────────────────────────
    if loss_config.get("repel", 0.0) > 0:
        z = emb_l1  # [B, C, d]
        zz = z.unsqueeze(2) - z.unsqueeze(1)  # [B, C, C, d]
        dist = (zz ** 2).sum(dim=-1)  # [B, C, C]
        mask = ~torch.eye(C, device=z.device).bool()  # [C, C]
        repel_loss = F.relu(1.0 - dist.masked_select(mask[None])).mean()
        aux_loss += loss_config["repel"] * repel_loss

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

        with autocast(enabled=torch.cuda.is_available(), device_type=str(device)):
            model_outputs = model(data.x, data.adj, data.mask, debug=True)
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
    macro_f1 = f1_score(true_labels, pred_labels, average='macro', zero_division=0.0)
    average_loss = total_loss / len(loader.dataset)

    cr = classification_report(true_labels, pred_labels, output_dict=True)
    return accuracy, macro_f1, true_labels, pred_labels, average_loss, cr


# Replace the existing train_and_validate function with this one:
def train_and_validate(model, train_loader, val_loader,
                       optimizer, loss_fn, patience, epochs, lr, hidden_dim, batch_size,
                       use_softmax, decrease_prop, dataset_name, device, timestamp, grid_search=False, verbose=True,
                       loss_config={}):
    """
    Train and validate the model, calculating a full suite of interpretability metrics.
    """
    best_e_score = -1.0
    best_model_path = None
    best_epoch = 0
    best_epoch_results = {}
    times = []

    loss_tag = loss_config_to_tag(loss_config)
    writer = SummaryWriter(
        f"runs/diffpool_{dataset_name}_{timestamp}_pat{patience}_ep{epochs}_lr{lr}_hd{hidden_dim}_bs{batch_size}_sm{use_softmax}_dp{decrease_prop}_{loss_tag}_gs{grid_search}"
    )

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # --- TRAINING ---
        train_loss, train_acc = train(
            model=model, loader=train_loader, optimizer=optimizer, loss_fn=loss_fn,
            return_acc=True, verbose=False, device=device, loss_config=loss_config
        )

        # --- VALIDATION & METRICS ---
        model.eval()
        all_preds, all_labels = [], []

        # Initialize metric calculators for the epoch
        completeness_calc = ConceptCompletenessCalculator()
        conformity_calc = ConceptConformityCalculator()
        modularity_calc = ModularityCalculator()
        silhouette_calc = SilhouetteScoreCalculator()

        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                model_outputs = model(data.x, data.adj, data.mask, debug=True)
                logits, _, _, _, _, (s01, s12) = model_outputs
                pred = logits.max(dim=1)[1]

                all_preds.append(pred.cpu())

                y_tensor = data.y
                if isinstance(y_tensor, list):
                    y_tensor = torch.tensor(y_tensor, device=device, dtype=torch.float)

                # Convert one-hot labels to class indices
                if y_tensor.ndim > 1 and y_tensor.shape[1] > 1:
                    y_indices = y_tensor.argmax(dim=1)
                else:
                    y_indices = y_tensor.long()  # Ensure it's integer type

                all_labels.append(y_indices.view(-1).cpu())

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

                    edge_index_i = dense_to_sparse(adj_i)[0]
                    single_graph_data = Data(x=x_i.clone(), edge_index=edge_index_i.clone(), y=y_i.clone())

                    single_graph_concepts = concept_ids_batch[i, :num_nodes]

                    completeness_calc.add_item(single_graph_data.y, single_graph_concepts)
                    conformity_calc.add_item(single_graph_data, single_graph_concepts)
                    modularity_calc.add_item(single_graph_data, single_graph_concepts)
                    silhouette_calc.add_item(single_graph_data, single_graph_concepts)

        # Calculate final metrics for the epoch
        preds, labels = torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()

        macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
        completeness = completeness_calc.calculate()
        conformity = conformity_calc.calculate()
        modularity = modularity_calc.calculate()
        silhouette = silhouette_calc.calculate()

        # Calculate HI-Score and E-Score
        hi_score = (completeness * conformity * modularity * silhouette) ** 0.25
        e_score = (2 * macro_f1 * hi_score) / (macro_f1 + hi_score) if (macro_f1 + hi_score) > 0 else 0

        # --- LOGGING ---
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('F1/val', macro_f1, epoch)
        writer.add_scalar('Metrics/Completeness', completeness, epoch)
        writer.add_scalar('Metrics/Conformity', conformity, epoch)
        writer.add_scalar('Metrics/Modularity', modularity, epoch)
        writer.add_scalar('Metrics/Silhouette', silhouette, epoch)
        writer.add_scalar('Overall/HI_Score', hi_score, epoch)
        writer.add_scalar('Overall/E_Score', e_score, epoch)

        logging.info(
            f'Epoch {epoch:03d} | F1: {macro_f1:.4f} | Comp: {completeness:.4f} | Conf: {conformity:.4f} | Mod: {modularity:.4f} | Sil: {silhouette:.4f} | E-Score: {e_score:.4f}')

        # --- MODEL CHECKPOINTING ---
        if e_score > best_e_score:
            best_e_score = e_score
            best_epoch_results = {
                "loss_config": loss_config, "best_epoch": epoch, "macro_f1": macro_f1,
                "completeness": completeness, "conformity": conformity,
                "modularity": modularity, "silhouette": silhouette,
                "hi_score": hi_score, "e_score": e_score
            }

            model_name = type(model).__name__
            path_prefix = "models/grid_search" if grid_search else "models"
            model_dir = os.path.join(path_prefix, dataset_name)
            os.makedirs(model_dir, exist_ok=True)

            best_model_path = os.path.join(model_dir, f'best_model_{timestamp}_{loss_tag}.pth')
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"✅ New best model saved with E-Score: {best_e_score:.4f}")

        times.append(time.time() - start_time)

        if (epoch - best_epoch) >= patience:
            logging.info("Early stopping triggered.")
            break

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
