"""
Utility functions for training GNN models.
@author Thiago Raulino Dal Pont
@date 2024-04-12
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)


def preprocess_predictions_and_targets(predictions: torch.Tensor,
                                       targets: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert model outputs and targets to numpy arrays for metric calculations.

    Parameters:
        predictions (torch.Tensor): Raw model outputs.
        targets (torch.Tensor): True labels.

    Returns:
        tuple: (Processed predictions, Processed targets)
    """
    predictions = torch.argmax(predictions, dim=1).cpu().numpy()
    targets = targets.cpu().numpy()
    return predictions, targets


def calculate_accuracy(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate accuracy given predicted labels and true labels."""
    return accuracy_score(targets, predictions)


def calculate_f1_score(predictions: np.ndarray, targets: np.ndarray) -> float:
    """Calculate weighted F1 score."""
    return f1_score(targets, predictions, average='weighted', zero_division=0.0)


def calculate_confusion_matrix(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Calculate confusion matrix."""
    return confusion_matrix(targets, predictions)


def calculate_precision(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Calculate precision score(s) for each class."""
    return precision_score(targets, predictions, average='weighted', zero_division=0.0)


def calculate_recall(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Calculate recall score(s) for each class."""
    return recall_score(targets, predictions, average='weighted', zero_division=0.0)


def calculate_roc_auc(predictions: torch.Tensor, targets: torch.Tensor) -> list[float]:
    """
    Calculate ROC AUC score for each class.

    Parameters:
        predictions (torch.Tensor): Predicted probabilities or scores.
        targets (torch.Tensor): True labels (one-hot encoded or binary).

    Returns:
        list: ROC AUC scores for each class.
    """
    predictions = predictions.cpu().numpy()
    targets = targets.cpu().numpy()

    roc_auc_scores = [
        roc_auc_score(targets[:, i], predictions[:, i])
        for i in range(predictions.shape[1])
    ]
    return roc_auc_scores
