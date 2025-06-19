# engine.py

import torch
from sklearn.metrics import precision_recall_fscore_support


def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out, _ = model(data)
        loss = criterion(out, data['document'].y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
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