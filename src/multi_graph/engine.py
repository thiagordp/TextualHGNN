# engine.py

import torch
from sklearn.metrics import precision_recall_fscore_support
from torch_scatter import scatter_add
from tqdm import tqdm


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


def train(model, train_loader, optimizer, criterion, entropy_weight: float = 0.005):
    model.train()
    total_loss = 0
    total_entropy_for_epoch = 0

    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training")

    for i, data in progress_bar:

        optimizer.zero_grad()

        out, (word_att, sent_att) = model(data)

        # 1. Calculate primary classification loss
        classification_loss = criterion(out, data['document'].y)

        # 2. Calculate auxiliary entropy loss
        word_edge_index, word_weights = word_att
        sent_edge_index, sent_weights = sent_att

        word_entropy_loss = compute_entropy_loss(word_weights, word_edge_index[1])
        sent_entropy_loss = compute_entropy_loss(sent_weights, sent_edge_index[1])

        # Combine entropy losses
        total_entropy_loss = word_entropy_loss + sent_entropy_loss

        # 3. Combine all losses with the weighting factor
        total_loss_batch = classification_loss + entropy_weight * total_entropy_loss

        total_loss_batch.backward()
        optimizer.step()

        total_loss += total_loss_batch.item()

        total_entropy_for_epoch += total_entropy_loss.item()
        avg_entropy = total_entropy_for_epoch / (i + 1)

        if i % 50 == 0:
            progress_bar.set_postfix(
                loss=f"{total_loss_batch.item():.4f}",
                avg_entropy=f"{avg_entropy:.4f}"
            )

    return total_loss / len(train_loader)


@torch.no_grad()
def test(model, loader, full_graph_list_for_mapping):
    model.eval()
    all_preds, all_labels = [], []
    all_explanations = []

    graph_map = {g['document'].doc_id: g for g in full_graph_list_for_mapping}

    for data in loader:
        out, (word_att, sent_att) = model(data)
        pred = out.argmax(dim=-1)
        label = data['document'].y
        all_preds.extend(pred.cpu().tolist())
        all_labels.extend(label.cpu().tolist())

        doc_ids_in_batch = data['document'].doc_id

        for i, doc_id in enumerate(doc_ids_in_batch):
            original_graph_data = graph_map.get(doc_id)

            if original_graph_data:
                # --- CHANGE: Add node_mappings as a top-level key ---
                explanation = {
                    'doc_id': doc_id,
                    'graph_data': original_graph_data.to_dict(),
                    'word_to_sent_att': word_att,
                    'sent_to_doc_att': sent_att,
                    'prediction': pred[i].item(),
                    'actual': label[i].item(),
                    'node_mappings': original_graph_data.node_mappings  # Explicitly add the mapping
                }
                all_explanations.append(explanation)

    p, r, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='macro', zero_division=0)
    acc = (torch.tensor(all_preds) == torch.tensor(all_labels)).sum().item() / len(all_labels) if all_labels else 0

    metrics = {'accuracy': acc, 'precision': p, 'recall': r, 'f1': f1}
    return metrics, all_labels, all_preds, all_explanations
