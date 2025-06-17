# model.py

import torch
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm
from torch_geometric.nn import SAGEConv, HeteroConv
from torch_scatter import scatter_add, scatter_sum
from torch_geometric.data import HeteroData


class HierarchicalSignedAttention(torch.nn.Module):
    # ... (code is unchanged from the version we designed)
    def __init__(self, channels: int):
        super().__init__()
        self.attention_net = Linear(2 * channels, 1)
        self.value_transform = Linear(channels, channels)

    def signed_l1_norm(self, scores: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        abs_scores = torch.abs(scores)
        sum_abs_scores = scatter_sum(abs_scores, group_ids, dim=0)
        sum_abs_scores = sum_abs_scores.clamp(min=1e-8)
        return scores / sum_abs_scores[group_ids]

    def forward(self, child_x: torch.Tensor, parent_x: torch.Tensor, edge_index: torch.Tensor):
        child_ids, parent_ids = edge_index
        parent_expanded = parent_x[parent_ids]
        child_expanded = child_x[child_ids]
        concatenated_features = torch.cat([parent_expanded, child_expanded], dim=-1)
        raw_scores = self.attention_net(concatenated_features).squeeze(-1)
        attention_weights = self.signed_l1_norm(raw_scores, parent_ids)
        transformed_child_values = self.value_transform(child_x)
        weighted_values = transformed_child_values[child_ids] * attention_weights.unsqueeze(-1)
        new_parent_x = scatter_add(weighted_values, parent_ids, dim=0, dim_size=parent_x.size(0))
        return new_parent_x, (edge_index, attention_weights)


class ExplainableHierarchicalGNN(torch.nn.Module):
    # ... (code is unchanged from the version we designed)
    def __init__(self, hidden_channels: int, out_channels: int):
        super().__init__()
        self.word_conv = SAGEConv((-1, -1), hidden_channels)
        self.sent_conv = SAGEConv((-1, -1), hidden_channels)
        self.word_hetero_conv = HeteroConv({
            ('word', 'dep', 'word'): self.word_conv,
            ('word', 'seq', 'word'): self.word_conv,
            ('word', 'same_lemma', 'word'): self.word_conv
        }, aggr='sum')
        self.sent_hetero_conv = HeteroConv({
            ('sentence', 'seq', 'sentence'): self.sent_conv,
            ('sentence', 'sim', 'sentence'): self.sent_conv
        }, aggr='sum')
        self.word_to_sent_attention = HierarchicalSignedAttention(hidden_channels)
        self.sent_to_doc_attention = HierarchicalSignedAttention(hidden_channels)
        self.norm1 = LayerNorm(hidden_channels)
        self.norm2 = LayerNorm(hidden_channels)
        self.classifier = Linear(hidden_channels, out_channels)

    def forward(self, data: HeteroData):
        x_dict, edge_index_dict = data.x_dict, data.edge_index_dict

        # Note: SAGEConv can't use edge_attr directly in this wrapper.
        # For a model that uses edge_attr, a custom wrapper or different conv layer is needed.
        # For simplicity, we are omitting edge_attr usage here, but a real implementation
        # would use a layer like GINEConv or write a custom message passing function.
        word_x_refined = self.word_hetero_conv(x_dict, edge_index_dict)
        word_x_refined = F.leaky_relu(word_x_refined['word'])

        sent_x_aggregated, word_attentions = self.word_to_sent_attention(
            child_x=word_x_refined,
            parent_x=x_dict['sentence'],
            edge_index=edge_index_dict[('word', 'belongs', 'sentence')]
        )
        sent_x_aggregated = self.norm1(sent_x_aggregated + x_dict['sentence'])

        temp_x_dict = {'sentence': sent_x_aggregated}
        sent_x_refined = self.sent_hetero_conv(temp_x_dict, edge_index_dict)
        sent_x_refined = F.leaky_relu(sent_x_refined['sentence'])

        doc_x_final, sent_attentions = self.sent_to_doc_attention(
            child_x=sent_x_refined,
            parent_x=x_dict['document'],
            edge_index=edge_index_dict[('sentence', 'belongs', 'document')]
        )
        doc_x_final = self.norm2(doc_x_final + x_dict['document'])
        out = self.classifier(doc_x_final)
        return out, (word_attentions, sent_attentions)