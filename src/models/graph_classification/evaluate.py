import torch

from src.models.graph_classification.utils import (
    calculate_accuracy,
    calculate_f1_score,
    calculate_confusion_matrix,
    preprocess_predictions_and_targets,
)


def evaluate_model(model: torch.nn.Module,
                   data_loader: torch.utils.data.DataLoader,
                   device: torch.device) -> tuple[float, float, 'np.ndarray']:
    """
    Evaluate the model using accuracy, F1 score, and confusion matrix.

    Parameters:
        model (torch.nn.Module): The trained graph neural network model.
        data_loader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): Device (CPU or GPU) where the evaluation will be performed.

    Returns:
        tuple: (Accuracy score, F1 score, Confusion matrix)
    """
    model.eval()
    predictions, targets = [], []

    with torch.no_grad():
        for batch in data_loader:
            adjacency_matrix, node_features, labels = map(lambda x: x.to(device), batch)

            outputs = model(adjacency_matrix, node_features)
            predictions.append(outputs)
            targets.append(labels)

    predictions = torch.cat(predictions)
    targets = torch.cat(targets)

    predictions, targets = preprocess_predictions_and_targets(predictions, targets)

    accuracy = calculate_accuracy(predictions, targets)
    f1 = calculate_f1_score(predictions, targets)
    confusion_matrix_result = calculate_confusion_matrix(predictions, targets)

    return accuracy, f1, confusion_matrix_result
