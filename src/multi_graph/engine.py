# engine.py

import torch
from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix, roc_auc_score
from torch_scatter import scatter_add
from tqdm import tqdm
import torch.nn.functional as F

def compute_entropy_loss(attention_weights: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    """
    Calculates the entropy of attention distributions to encourage sparsity.
    A lower entropy corresponds to a more "spiky" and focused attention.

    Args:
        attention_weights (torch.Tensor): The learned signed attention weights.
        group_ids (torch.Tensor): The tensor identifying the parent node for each weight
                                  (e.g., the sentence index for each word's attention).

    Returns:
        torch.Tensor: A scalar tensor representing the mean entropy loss.
    """

    # Use the absolute value, as entropy is defined for probability distributions (non-negative)
    # Our signed_l1_norm ensures the absolute values in each group sum to 1.
    p_attn = torch.abs(attention_weights)

    # Calculate entropy for each attention value: H_i = -p_i * log(p_i)
    # Add a small epsilon to prevent log(0) -> NaN
    entropy = -p_attn * torch.log(p_attn + 1e-10)

    # Sum the entropies for each group (each sentence/document)
    grouped_entropy = scatter_add(entropy, group_ids, dim=0)

    # Return the mean entropy across all groups in the batch
    return torch.mean(grouped_entropy)


def train(model, train_loader, optimizer, criterion, entropy_weight: float = 0.005, accumulation_steps: int = 1):
    """
    Trains the model for one epoch using gradient accumulation.

    Args:
        model: The GNN model to train.
        train_loader: DataLoader for the training set.
        optimizer: The optimizer.
        criterion: The loss function.
        entropy_weight (float): The weight for the entropy regularization loss.
        accumulation_steps (int): The number of batches to accumulate gradients over before an optimizer step.
    """
    model.train()
    total_loss = 0
    total_entropy_for_epoch = 0

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc="Training")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reset gradients at the beginning of the epoch
    optimizer.zero_grad()

    for i, data in progress_bar:
        #data = data.to(device)

        # Forward pass
        out, (word_att, sent_att) = model(data)

        # 1. Calculate primary classification loss
        classification_loss = criterion(out, data['document'].y)

        # 2. Calculate auxiliary entropy loss
        word_edge_index, word_weights = word_att
        sent_edge_index, sent_weights = sent_att
        word_entropy_loss = compute_entropy_loss(word_weights, word_edge_index[1])
        sent_entropy_loss = compute_entropy_loss(sent_weights, sent_edge_index[1])
        total_entropy_loss = word_entropy_loss + sent_entropy_loss

        # 3. Combine all losses for the current mini-batch
        loss_for_batch = classification_loss + entropy_weight * total_entropy_loss

        # 4. Normalize loss for accumulation to maintain gradient magnitude
        normalized_loss = loss_for_batch / accumulation_steps

        # 5. Accumulate gradients
        normalized_loss.backward()

        # 6. Update weights after 'accumulation_steps' batches
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            optimizer.step()
            optimizer.zero_grad()

        # --- Logging ---
        # Keep track of the un-normalized loss for accurate reporting
        total_loss += loss_for_batch.item()
        total_entropy_for_epoch += total_entropy_loss.item()

        if i % 10 == 0:
            avg_epoch_loss = total_loss / (i + 1)
            avg_epoch_entropy = total_entropy_for_epoch / (i + 1)
            progress_bar.set_postfix(
                avg_loss=f"{avg_epoch_loss:.4f}",
                avg_entropy=f"{avg_epoch_entropy:.4f}"
            )

    # Return the average loss per mini-batch for the entire epoch
    return total_loss / len(train_loader)


@torch.no_grad()
def test(model, loader, full_graph_list_for_mapping, class_names: list):
    """
    Evaluates the model on a dataset and computes a comprehensive suite of classification metrics.
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    all_explanations = []
    # --- NEW: List to store detailed results for each prediction ---
    all_detailed_results = []

    graph_map = {g['document'].doc_id: g for g in full_graph_list_for_mapping}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_filenames = []

    for data in loader:
        #data = data.to(device)

        out, (word_att, sent_att) = model(data)

        # Get probabilities using softmax for multi-class ROC AUC and other metrics
        probs = F.softmax(out, dim=-1)
        pred = probs.argmax(dim=-1)

        label = data['document'].y

        all_preds.extend(pred.cpu().tolist())
        all_labels.extend(label.cpu().tolist())
        all_probs.extend(probs.cpu().tolist())

        doc_ids_in_batch = data['document'].doc_id if isinstance(data['document'].doc_id, list) else [
            data['document'].doc_id]

        all_filenames.extend(doc_ids_in_batch)

        for i, doc_id in enumerate(doc_ids_in_batch):
            original_graph_data = graph_map.get(doc_id)
            if original_graph_data:
                # --- Explanation object (for visualization) ---
                explanation = {
                    'doc_id': doc_id,
                    'graph_data': original_graph_data.to_dict(),
                    'word_to_sent_att': word_att,
                    'sent_to_doc_att': sent_att,
                    'prediction': pred[i].item(),
                    'actual': label[i].item(),
                    'node_mappings': original_graph_data.node_mappings
                }
                all_explanations.append(explanation)

                # --- NEW: Detailed prediction result object (for CSV logging) ---
                true_label_idx = label[i].item()
                pred_label_idx = pred[i].item()

                status = "N/A"
                # # Calculate TP/FP/TN/FN status for binary classification (assuming class 1 is positive)
                # if len(class_names) == 2:
                #     if true_label_idx == 1 and pred_label_idx == 1:
                #         status = "TP"
                #     elif true_label_idx == 0 and pred_label_idx == 0:
                #         status = "TN"
                #     elif true_label_idx == 0 and pred_label_idx == 1:
                #         status = "FP"
                #     elif true_label_idx == 1 and pred_label_idx == 0:
                #         status = "FN"

                detailed_result = {
                    'doc_id': doc_id,
                    'true_label': class_names[true_label_idx],
                    'predicted_label': class_names[pred_label_idx],
                    'status': status
                }
                # Add probabilities for each class dynamically
                for c_idx, c_name in enumerate(class_names):
                    detailed_result[f'prob_{c_name}'] = probs[i][c_idx].item()

                all_detailed_results.append(detailed_result)

    # --- Comprehensive Metrics Calculation ---
    metrics = {}
    if all_labels and all_preds:
        # Generate the classification report as a dictionary for easy parsing
        report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True,
                                            zero_division=0)
        metrics['classification_report'] = report_dict

        # Pull out the main averages for quick access
        metrics['accuracy'] = report_dict.get('accuracy')
        metrics['macro_avg'] = report_dict.get('macro avg')
        metrics['weighted_avg'] = report_dict.get('weighted avg')

        # Generate the confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(all_labels, all_preds).tolist()

        # Calculate ROC AUC score
        if len(class_names) > 2:
            # One-vs-Rest for multi-class
            metrics['roc_auc_score'] = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        elif len(all_probs) > 0 and len(all_probs[0]) == 2:
            # Standard for binary, needs probabilities for the positive class (class 1)
            binary_probs = [p[1] for p in all_probs]
            metrics['roc_auc_score'] = roc_auc_score(all_labels, binary_probs)

    # --- MODIFIED: Return the new list of detailed results ---
    return metrics, all_labels, all_preds, all_explanations, all_detailed_results, all_filenames
