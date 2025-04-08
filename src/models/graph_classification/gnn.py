"""
GNN models declaration

@author: Thiago Raulino Dal Pont
@date: 2024-04-12
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, DenseSAGEConv, dense_diff_pool, global_mean_pool, SAGEConv
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

        # Define graph convolution layers
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, num_classes)

        # Dropout layer for regularization
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


class GraphSAGE(GNN):
    """
    GraphSAGE GNN model.

    Args:
        in_channels (int): Number of input features per node.
        hidden_channels (int): Number of hidden features.
        out_channels (int): Number of output features.
        normalize (bool, optional): If set to True, the input features are normalized. Default is False.
        lin (bool, optional): If set to True, a linear layer is applied after the convolutions. Default is True.

    Attributes:
        conv1 (torch_geometric.nn.DenseSAGEConv): First DenseSAGE convolutional layer.
        bn1 (torch.nn.BatchNorm1d): Batch normalization layer after the first convolution.
        conv2 (torch_geometric.nn.DenseSAGEConv): Second DenseSAGE convolutional layer.
        bn2 (torch.nn.BatchNorm1d): Batch normalization layer after the second convolution.
        conv3 (torch_geometric.nn.DenseSAGEConv): Third DenseSAGE convolutional layer.
        bn3 (torch.nn.BatchNorm1d): Batch normalization layer after the third convolution.
        lin (torch.nn.Linear or None): Linear layer applied after the convolutions. None if lin is set to False.

    """

    def __init__(self, in_channels, hidden_channels, out_channels, normalize=False, lin=True):
        """
        Initialize the GNN.

        Args:
            in_channels (int): Number of input features.
            hidden_channels (int): Number of hidden units.
            out_channels (int): Number of output units.
            normalize (bool): Whether to apply normalization in SAGEConv layers.
            lin (bool): Whether to include a linear layer at the end.
        """
        super().__init__()

        self.conv1 = DenseSAGEConv(in_channels, hidden_channels, normalize)
        self.bn1 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv2 = DenseSAGEConv(hidden_channels, hidden_channels, normalize=normalize)
        self.bn2 = torch.nn.BatchNorm1d(hidden_channels)
        self.conv3 = DenseSAGEConv(hidden_channels, out_channels, normalize=normalize)
        self.bn3 = torch.nn.BatchNorm1d(out_channels)

        self.lin = torch.nn.Linear(2 * hidden_channels + out_channels, out_channels) if lin else None

    def bn(self, i, x):
        """
        Apply batch normalization to the input tensor.

        Args:
            i (int): Index of the batch normalization layer.
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Batch normalized tensor.
        """
        batch_size, num_nodes, num_channels = x.size()
        x = x.view(-1, num_channels)
        x = getattr(self, f'bn{i}')(x)
        x = x.view(batch_size, num_nodes, num_channels)
        return x

    def forward(self, x, adj, mask=None):
        """
        Forward pass of the GNN.

        Args:
            x (torch.Tensor): Input features.
            adj (torch.Tensor): Adjacency matrix.
            mask (torch.Tensor, optional): Mask for nodes.

        Returns:
            torch.Tensor: Output features.
        """
        x0 = x
        x1 = self.bn(1, self.conv1(x0, adj, mask).relu())
        x2 = self.bn(2, self.conv2(x1, adj, mask).relu())
        x3 = self.bn(3, self.conv3(x2, adj, mask).relu())

        x = torch.cat([x1, x2, x3], dim=-1)
        if self.lin is not None:
            x = self.lin(x).relu()

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

    def __init__(self, max_num_nodes, in_channels, hidden_channels, out_channels, decrease_proportion=0.25):
        super().__init__()

        inner_channels = ceil(hidden_channels * 1.0)
        num_nodes = max(1, ceil(decrease_proportion * max_num_nodes))
        self.gnn1_pool = GraphSAGE(in_channels, inner_channels, num_nodes)
        self.gnn1_embed = GraphSAGE(in_channels, inner_channels, hidden_channels, lin=False)

        num_nodes = max(1, ceil(decrease_proportion * num_nodes))
        self.gnn2_pool = GraphSAGE(3 * hidden_channels, inner_channels, num_nodes)
        self.gnn2_embed = GraphSAGE(3 * hidden_channels, inner_channels, hidden_channels, lin=False)

        self.gnn3_embed = GraphSAGE(3 * hidden_channels, inner_channels, hidden_channels, lin=False)

        self.lin1 = torch.nn.Linear(3 * hidden_channels, hidden_channels)
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

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

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

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        emb_l2 = x
        adj_l2 = adj

        if debug:
            pass
            # print("Diffpool L2")
            # print("Emb L2", emb_l2.shape)
            # print("Adj L2", adj_l2.shape)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)

        if debug:
            return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2, (emb_l1, adj_l1), (emb_l2, adj_l2), (s01, s12)

        return F.log_softmax(x, dim=-1), l1 + l2, e1 + e2
