import logging
import os
import time
from datetime import datetime

import torch
import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DenseDataLoader, DataLoader

from src.data.text_graph_dataset_ondisk import TextGraphDatasetOnDisk
from src.models.graph_classification.gnn import GCN
from src.utils.general_utils import setup_logging

# Determine the device to run the model on
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

EPOCHS = 100
PATIENCE = 20
TRAIN_SIZE = 0.7
CLASS_WEIGHTS = [0.7109853029251099, 1.6849168539047241]
LR = 1e-4
NODE_FEATURE_DIM = HIDDEN_DIM = 100
DATASET = "STF_HC_Voto_Relatorio"
ROOT = f"data/datasets/{DATASET}"
NUM_NODES = 3000
BATCH_SIZE = 4

setup_logging(log_file=f"training_model_experiment_gcn_{DATASET}.log")
logging.info(f"============  STARTING EXPERIMENT {DATASET}  ============")


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

    for data in tqdm.tqdm(loader, desc="Training"):
        data = data.to(DEVICE)

        optimizer.zero_grad()

        # Forward pass
        output = model(data.x, data.edge_index, data.batch)

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

    for data in tqdm.tqdm(loader, desc="Evaluating"):
        data = data.to(DEVICE)

        # Forward pass: Get model predictions
        output = model(data.x, data.edge_index, data.batch)

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
            pass
            # logging.info("*" * 50)
            # logging.info(f"y_pred: {predicted.tolist()}")
            # logging.info(f"y_true: {y.tolist()}")

    # Calculate evaluation metrics
    accuracy = total_correct / len(loader.dataset)
    macro_f1 = f1_score(true_labels, pred_labels, average='macro')
    average_loss = total_loss / len(loader.dataset)

    return accuracy, macro_f1, true_labels, pred_labels, average_loss


def save_model(model, optimizer, path, epoch, loss, val_acc):
    """
    Save the model checkpoint along with optimizer state and training metadata.

    Args:
        model (torch.nn.Module): The model to be saved.
        optimizer (torch.optim.Optimizer): The optimizer associated with the model.
        path (str): The file path where the checkpoint will be saved.
        epoch (int): The current epoch number.
        loss (float): The loss value at the time of saving.
        val_acc (float): The validation accuracy at the time of saving.

    Returns:
        None

    Notes:
        - The function saves a dictionary containing the model's state dictionary,
          optimizer's state dictionary, and additional metadata like the epoch, loss, and validation accuracy.
        - This checkpoint can be used to resume training or evaluate the model later.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'val_acc': val_acc,
    }

    torch.save(checkpoint, path)


def get_timestamp():
    """
    Generate a timestamp string in the format 'YYYYMMDD_HHMMSS'.

    Returns:
        str: The current timestamp as a string in the format 'YYYYMMDD_HHMMSS'.

    Example:
        get_timestamp()
        '20240814_103045'
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def load_datasets(root, num_nodes, node_feature_size):
    """
    Load datasets and apply transformations.

    Args:
        root (str): The root directory for the dataset.
        num_nodes (int): Maximum number of nodes in the graphs.
        node_feature_size (int): Size of the node feature vectors.

    Returns:
        tuple: Train, validation, and test datasets.
    """
    tgd_train = TextGraphDatasetOnDisk(
        root=root,
        split="train",
        batch_size=1,
        # transform=T.ToDense(num_nodes=num_nodes),
        node_feature_size=node_feature_size
    )

    tgd_val = TextGraphDatasetOnDisk(
        root=root,
        split="validation",
        batch_size=1,
        # transform=T.ToDense(num_nodes=num_nodes),
        node_feature_size=node_feature_size
    )

    tgd_test = TextGraphDatasetOnDisk(
        root=root,
        split="test",
        batch_size=1,
        # transform=T.ToDense(num_nodes=num_nodes),
        node_feature_size=node_feature_size
    )

    return tgd_train, tgd_val, tgd_test


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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


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
    model = GCN(
        num_node_features=in_channels,
        hidden_dim=hidden_dim,
        num_classes=out_channels
    ).to(DEVICE)

    logging.info(f"Number of parameters: {model.num_parameters}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1)

    return model, optimizer


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
    writer = SummaryWriter(f"runs/gcn_glove_{timestamp}")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        train_loss, train_acc = train(model, train_loader, optimizer, loss_fn, return_acc=True)
        val_acc, val_macro_f1, _, _, val_loss = test(model, val_loader, loss_fn, )
        train_acc, train_macro_f1, _, _, _ = test(model, train_loader, loss_fn, )

        with torch.no_grad():
            total_params = 0
            sum_abs_params = 0.0
            for param in model.parameters():
                if param.requires_grad:
                    total_params += param.numel()
                    sum_abs_params += torch.sum(torch.abs(param.data)).item()
        avg_abs_param = sum_abs_params / total_params if total_params > 0 else 0

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Metrics/Avg_Abs_Param', avg_abs_param, epoch)

        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_epoch = epoch

            model_name = type(model).__name__
            lr = optimizer.param_groups[0]['lr']

            best_model_path = f'models/{model_name}_{timestamp}_lr{lr}_valmacrof1score{best_val_f1:.4f}_epoch{best_epoch:03d}.pth'
            torch.save(model.state_dict(), best_model_path)

        logging.info(
            f'Epoch: {epoch:03d} | Train Loss: {train_loss:.4f} | Val Macro F1: {val_macro_f1:.4f} | Avg Abs Param: {avg_abs_param:.4f}')

        times.append(time.time() - start_time)

        if (epoch - best_epoch) >= patience:
            logging.info("Early stopping triggered")
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
    acc, macro_f1, true_labels, pred_labels, loss = test(model, loader, loss_fn, show_pred=True)

    logging.info(f"\n{dataset_name} Accuracy: {acc:.4f}")
    logging.info(f"{dataset_name} Macro F1: {macro_f1:.4f}")
    logging.info(f"{dataset_name} Loss: {loss:.4f}")

    report = classification_report(
        true_labels,
        pred_labels,
        target_names=[str(i) for i in range(len(set(true_labels)))],
        digits=4,
    )
    logging.info(f"\n{dataset_name} Classification Report:\n{report}")

    cm = confusion_matrix(true_labels, pred_labels)
    logging.info(f"\n{dataset_name} Confusion Matrix:")
    logging.info(cm)


def custom_collate_fn(batch):
    # Assuming your data is a list of tensors
    # Pad all tensors in the batch to the same size
    max_len = max(data.size(1) for data in batch)
    padded_batch = [torch.nn.functional.pad(data, (0, max_len - data.size(1))) for data in batch]
    return torch.stack(padded_batch)


def main():
    """
    Main function to orchestrate the training, validation, and testing of the GNN model.
    """
    print("Current directory:", os.getcwd())

    # Load datasets
    tgd_train, tgd_val, tgd_test = load_datasets(
        ROOT,
        NUM_NODES,
        NODE_FEATURE_DIM
    )

    # Create data loaders
    train_loader, val_loader, test_loader = create_loaders(
        tgd_train,
        tgd_val,
        tgd_test,
        BATCH_SIZE
    )

    # Initialize model and optimizer
    model, optimizer = initialize_model(
        in_channels=tgd_train.num_node_attributes,
        out_channels=tgd_train.num_classes,
        max_num_nodes=NUM_NODES,
        hidden_dim=HIDDEN_DIM,
        lr=LR
    )

    # Define loss function
    class_weights = torch.tensor(CLASS_WEIGHTS, device=DEVICE, dtype=torch.float)
    logging.info(f"Class weights: {class_weights}")
    loss_fn = nn.NLLLoss(weight=class_weights)

    # Train and validate the model
    best_model_path = train_and_validate(
        model,
        train_loader,
        val_loader,
        optimizer,
        loss_fn,
        PATIENCE,
        EPOCHS
    )

    # Load the best model and evaluate on train, validation, and test datasets
    if best_model_path:
        model.load_state_dict(torch.load(best_model_path))
        evaluate_model(model, train_loader, loss_fn, "Train")
        evaluate_model(model, val_loader, loss_fn, "Validation")
        evaluate_model(model, test_loader, loss_fn, "Test")
    else:
        print("No best model was saved.")


if __name__ == "__main__":
    main()
