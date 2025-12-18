"""
GNN models declaration

"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, DenseSAGEConv, dense_diff_pool, global_mean_pool, SAGEConv, dense_mincut_pool, \
    GATConv, DenseGCNConv
import torch.nn.functional as F
from math import ceil

from torch_geometric.utils import dense_to_sparse


class GNN(nn.Module):
    # ... (code unchanged) ...
    def __init__(self):
        super(GNN, self).__init__()

    @property
    def num_parameters(self):
        """Get number of learnable parameters in model
        """
        count = 0
        for name, param in self.named_parameters():
            # print(name, param.numel())
            count += param.numel()
        return count


class GCN(GNN):
    # ... (code unchanged) ...
    def __init__(self, num_node_features, hidden_dim, num_classes, dropout=0.5):
        super(GCN, self).__init__()
        self.num_node_features = num_node_features
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout

        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, num_classes)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout_layer(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout_layer(x)
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=1)


class GraphGCN(GNN):
    # ... (code unchanged) ...
    def __init__(self, in_channels, hidden_channels, out_channels, normalize=False,
                 lin=False):
        super().__init__()

        self.conv1 = DenseGCNConv(in_channels, hidden_channels)
        self.ln1 = nn.LayerNorm(hidden_channels)
        self.conv2 = DenseGCNConv(hidden_channels, out_channels)
        self.ln2 = nn.LayerNorm(out_channels)

    def forward(self, x, adj, mask=None):
        x = self.conv1(x, adj, mask).relu()
        x = self.ln1(x)
        x = self.conv2(x, adj, mask).relu()
        x = self.ln2(x)
        return F.relu(x)


class GraphATT(GNN):
    # ... (code unchanged) ...
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.5):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.ln1 = nn.LayerNorm(hidden_channels * heads)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.ln2 = nn.LayerNorm(out_channels)

    def forward(self, x, adj, mask=None):
        batch_size, num_nodes, _ = x.size()
        edge_index, _ = dense_to_sparse(adj)
        x_reshaped = x.view(batch_size * num_nodes, -1)
        x_reshaped = self.conv1(x_reshaped, edge_index).relu()
        x = x_reshaped.view(batch_size, num_nodes, -1)
        x = self.ln1(x)
        x_reshaped = x.view(batch_size * num_nodes, -1)
        x_reshaped = self.conv2(x_reshaped, edge_index).relu()
        x = x_reshaped.view(batch_size, num_nodes, -1)
        x = self.ln2(x)
        return x


class DiffPool(GNN):
    """
    [CHANGED] 2-Level Differentiable Pooling.
    Forward pass is modified to apply classifier *before* pooling.
    """

    def __init__(self, max_num_nodes, in_channels, inner_channels, hidden_channels, out_channels,
                 decrease_proportion=0.25, softmax_assign=False):
        super().__init__()
        # ... (init method is unchanged) ...
        num_nodes_l1 = min(max(20, ceil(decrease_proportion * max_num_nodes)), 50)
        self.gnn1_pool = GraphGCN(in_channels, inner_channels, num_nodes_l1)
        self.gnn1_embed = GraphGCN(in_channels, inner_channels, hidden_channels)
        num_nodes_l2 = min(max(5, ceil(decrease_proportion * num_nodes_l1)), 10)
        self.gnn2_pool = GraphGCN(hidden_channels, inner_channels, num_nodes_l2)
        self.gnn2_embed = GraphGCN(hidden_channels, inner_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
        self.softmax_assign = softmax_assign

    def forward(self, x, adj, mask=None, debug=False):
        # --- Layer 1 ---
        s = self.gnn1_pool(x, adj, mask)
        x_embed = self.gnn1_embed(x, adj, mask)
        x = x_embed + x

        if self.softmax_assign:
            s = torch.softmax(s, dim=-1)

        s01 = s
        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)
        emb_l1 = x
        adj_l1 = adj

        # --- Layer 2 ---
        s = self.gnn2_pool(x, adj)
        x_embed = self.gnn2_embed(x, adj)
        x = x_embed + x
        if self.softmax_assign:
            s = torch.softmax(s, dim=-1)

        s12 = s
        x, adj, l2, e2 = dense_diff_pool(x, adj, s)
        emb_l2 = x
        adj_l2 = adj

        # --- [CHANGED] Final Readout (Classify THEN Pool) ---
        # Apply classifier to *each cluster embedding*
        # x shape: [B, N_L2, C_hidden]
        cluster_logits = self.lin(x)
        # cluster_logits shape: [B, N_L2, C_out]

        # Pool the logits to get the final graph prediction
        x = cluster_logits.mean(dim=1)
        # x shape: [B, C_out]

        if debug:
            return (
                F.log_softmax(x, dim=-1),
                l1 + l2, e1 + e2,
                (emb_l1, adj_l1),
                (emb_l2, adj_l2),
                (cluster_logits, None),  # [CHANGED] Pass out cluster_logits for analysis
                (s01, s12, None)
            )

        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2


class DiffPoolMinCut(GNN):
    """
    [CHANGED] 2-Level Differentiable Pooling with MinCut loss.
    Forward pass is modified to apply classifier *before* pooling.
    """

    def __init__(self, max_num_nodes, in_channels, inner_channels, hidden_channels, out_channels,
                 decrease_proportion=0.25, softmax_assign=False):
        super().__init__()

        num_nodes_l1 = min(max(20, ceil(decrease_proportion * max_num_nodes)), 50)
        self.gnn1_pool = GraphGCN(in_channels, inner_channels, num_nodes_l1)
        self.gnn1_embed = GraphGCN(in_channels, inner_channels, hidden_channels)
        num_nodes_l2 = min(max(5, ceil(decrease_proportion * num_nodes_l1)), 10)
        self.gnn2_pool = GraphGCN(hidden_channels, inner_channels, num_nodes_l2)
        self.gnn2_embed = GraphGCN(hidden_channels, inner_channels, hidden_channels)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)
        self.softmax_assign = softmax_assign

    def forward(self, x, adj, mask=None, debug=False):
        # --- Layer 1 ---
        s = self.gnn1_pool(x, adj, mask)
        x_embed = self.gnn1_embed(x, adj, mask)
        x = x_embed + x

        if self.softmax_assign:
            s = torch.softmax(s, dim=-1)

        s01 = s
        x, adj, l1, e1 = dense_mincut_pool(x, adj, s, mask)
        emb_l1 = x
        adj_l1 = adj

        # --- Layer 2 ---
        s = self.gnn2_pool(x, adj)
        x_embed = self.gnn2_embed(x, adj)
        x = x_embed + x

        if self.softmax_assign:
            s = torch.softmax(s, dim=-1)

        s12 = s
        x, adj, l2, e2 = dense_mincut_pool(x, adj, s)
        emb_l2 = x
        adj_l2 = adj

        # --- Final Readout (Classify THEN Pool) ---

        # Apply classifier to *each cluster embedding*
        # x shape: [B, N_L2, C_hidden]
        cluster_logits = self.lin(x)
        # cluster_logits shape: [B, N_L2, C_out]

        # Pool the logits to get the final graph prediction
        x = cluster_logits.mean(dim=1)
        # x shape: [B, C_out]

        if debug:
            return (
                x,  # F.log_softmax(x, dim=-1),
                -1 * (l1 + l2), e1 + e2,
                (emb_l1, adj_l1),
                (emb_l2, adj_l2),
                cluster_logits,  # [CHANGED] Pass out cluster_logits for analysis
                (s01, s12)
            )

        return x, -1 * (l1 + l2), (e1 + e2)
        # return F.log_softmax(x, dim=-1), -1 * (l1 + l2), (e1 + e2)

