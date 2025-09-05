"""
GNN models declaration

"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, DenseSAGEConv, dense_diff_pool, global_mean_pool, SAGEConv, dense_mincut_pool
import torch.nn.functional as F
from math import ceil


class GNN(nn.Module):
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
    """
    Graph Convolutional Network (GCN) implementation using PyTorch Geometric.

    Args:
        num_node_features (int): Number of input node features.
        hidden_dim (int): Dimensionality of the hidden layer.
        num_classes (int): Number of output classes.
        dropout (float, optional): Dropout probability for regularization. Default is 0.5.

    Attributes:
        num_node_features (int): Number of input node features.
        hidden_dim (int): Dimensionality of the hidden layer.
        num_classes (int): Number of output classes.
        dropout (float): Dropout probability for regularization.
        conv1 (torch_geometric.nn.conv.GCNConv): Graph convolution layer 1.
        conv2 (torch_geometric.nn.conv.GCNConv): Graph convolution layer 2.
        dropout_layer (torch.nn.Dropout): Dropout layer for regularization.

    """

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
        # First graph convolution layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout_layer(x)

        # Second graph convolution layer
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout_layer(x)

        # Third graph convolution layer
        x = self.conv3(x, edge_index)

        # Global mean pooling
        x = global_mean_pool(x, batch)

        return F.log_softmax(x, dim=1)


# In src/models/graph_classification/gnn.py

class GraphSAGE(GNN):
    """
    A simplified, 2-layer GraphSAGE module to be used as a building block.
    This version is more parameter-efficient than the original 3-layer version.
    """

    def __init__(self, in_channels, hidden_channels, out_channels, normalize=False,
                 lin=False):  # lin is kept for compatibility but not used
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, out_channels, normalize)
        self.bn2 = torch.nn.BatchNorm1d(out_channels)

    def bn(self, i, x):
        """Applies batch normalization to the dense tensor."""
        batch_size, num_nodes, num_channels = x.size()
        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        """Forward pass for the 2-layer GNN."""
        x = self.bn(1, self.conv1(x, adj, mask).relu())
        x = self.bn(2, self.conv2(x, adj, mask).relu())
        return x


class DiffPool(GNN):
    """
    Differentiable Pooling for Graph Neural Networks.

    Args:
        max_number_nodes (int): Maximum number of nodes in the graph.
        in_channels (int): Number of input features per node.
        hidden_channels (int): Number of hidden features.
        out_channels (int): Number of output features.
        dropout (float, optional): Dropout probability for regularization. Default is 0.5.

    Attributes:
        gnn1_pool (GraphSAGE): GraphSAGE pooling layer 1.
        gnn1_embed (GraphSAGE): GraphSAGE embedding layer 1.
        gnn2_pool (GraphSAGE): GraphSAGE pooling layer 2.
        gnn2_embed (GraphSAGE): GraphSAGE embedding layer 2.
        gnn3_embed (GraphSAGE): GraphSAGE embedding layer 3.
        lin1 (torch.nn.Linear): Linear layer 1.
        lin2 (torch.nn.Linear): Linear layer 2.

    """

    def __init__(self, max_num_nodes, in_channels, inner_channels, hidden_channels, out_channels,
                 decrease_proportion=0.25, softmax_assign=False):
        super().__init__()

        # --- Layer 1 ---
        num_nodes_l1 = max(1, ceil(decrease_proportion * max_num_nodes))
        # This now instantiates the new, lighter GraphSAGE
        self.gnn1_pool = GraphSAGE(in_channels, inner_channels, num_nodes_l1)
        self.gnn1_embed = GraphSAGE(in_channels, inner_channels, hidden_channels)

        # --- Layer 2 ---
        num_nodes_l2 = max(1, ceil(decrease_proportion * num_nodes_l1))
        # This also uses the new, lighter GraphSAGE
        self.gnn2_pool = GraphSAGE(hidden_channels, inner_channels, num_nodes_l2)
        #self.gnn2_embed = GraphSAGE(hidden_channels, inner_channels, hidden_channels)

        # --- Final Embedding Layer ---
        #self.gnn3_embed = GraphSAGE(hidden_channels, inner_channels, hidden_channels)

        # --- Classifier ---
        self.lin1 = torch.nn.Linear(hidden_channels, inner_channels)
        self.lin2 = torch.nn.Linear(inner_channels, out_channels)
        self.softmax_assign = softmax_assign

    def forward(self, x, adj, mask=None, debug=False):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask) + x

        if self.softmax_assign:
            s = torch.softmax(s, dim=-1)

        s01 = s

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        emb_l1 = x
        adj_l1 = adj

        s = self.gnn2_pool(x, adj)
        if self.softmax_assign:
            s = torch.softmax(s, dim=-1)
        #x = self.gnn2_embed(x, adj) + x

        s12 = s

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        emb_l2 = x
        adj_l2 = adj

        # x = self.gnn3_embed(x, adj) + x

        x = x.mean(dim=1)
        x = self.lin1(x)
        x = self.lin2(x)

        if debug:
            return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2, (emb_l1, adj_l1), (emb_l2, adj_l2), (s01, s12)

        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2


class DiffPoolMinCut(GNN):
    """
    Differentiable Pooling for Graph Neural Networks.

    Args:
        max_number_nodes (int): Maximum number of nodes in the graph.
        in_channels (int): Number of input features per node.
        hidden_channels (int): Number of hidden features.
        out_channels (int): Number of output features.
        dropout (float, optional): Dropout probability for regularization. Default is 0.5.

    Attributes:
        gnn1_pool (GraphSAGE): GraphSAGE pooling layer 1.
        gnn1_embed (GraphSAGE): GraphSAGE embedding layer 1.
        gnn2_pool (GraphSAGE): GraphSAGE pooling layer 2.
        gnn2_embed (GraphSAGE): GraphSAGE embedding layer 2.
        gnn3_embed (GraphSAGE): GraphSAGE embedding layer 3.
        lin1 (torch.nn.Linear): Linear layer 1.
        lin2 (torch.nn.Linear): Linear layer 2.

    """

    def __init__(self, max_num_nodes, in_channels, hidden_channels, out_channels, decrease_proportion=0.25):
        super().__init__()

        inner_channels = max(5, ceil(hidden_channels * 0.5))
        num_nodes = max(1, ceil(decrease_proportion * max_num_nodes))
        self.gnn1_pool = GraphSAGE(in_channels, inner_channels, num_nodes)
        # self.gnn1_embed = GraphSAGE(in_channels, max(10, int(hidden_channels*0.1)), hidden_channels, lin=False)

        num_nodes = max(1, ceil(decrease_proportion * num_nodes))
        self.gnn2_pool = GraphSAGE(hidden_channels, inner_channels, num_nodes)
        # self.gnn2_embed = GraphSAGE(3 * hidden_channels, inner_channels, hidden_channels, lin=False)

        # self.gnn3_embed = GraphSAGE(3 * hidden_channels, inner_channels, hidden_channels, lin=False)

        # self.lin1 = torch.nn.Linear(hidden_channels, inner_channels*2)
        # self.lin2 = torch.nn.Linear(inner_channels*2, out_channels)
        self.lin1 = torch.nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, adj, mask=None, debug=False):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        if debug:
            pass
            # print("Output L1")
            # print("S", s.shape)
            # print("X", x.shape)

        s01 = s

        x, adj, l1, e1 = dense_mincut_pool(x, adj, s, mask)

        emb_l1 = x
        adj_l1 = adj

        if debug:
            # print("Diffpool L1")
            # print("Emb L1", emb_l1.shape)
            # print("Adj L1", adj_l1.shape)
            pass

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        s12 = s

        if debug:
            # print("Output L2")
            # print("S", s.shape)
            # print("X", x.shape)
            pass

        x, adj, l2, e2 = dense_mincut_pool(x, adj, s)

        emb_l2 = x
        adj_l2 = adj

        if debug:
            pass
            # print("Diffpool L2")
            # print("Emb L2", emb_l2.shape)
            # print("Adj L2", adj_l2.shape)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = self.lin1(x)
        # x = self.lin1(x).relu()
        x = self.lin2(x)

        if debug:
            return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2, (emb_l1, adj_l1), (emb_l2, adj_l2), (s01, s12)

        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2
