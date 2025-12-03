"""
Utility functions for graph operations using Pytorch Geometric.

Author:
    Thiago Raulino Dal Pont

Date 19-07-24
"""

import torch
from torch_geometric.data import Data


def remove_nodes(data, node_ids, store_ids=False, device="cpu"):
    """
    Remove the specified nodes and return a new graph.

    Parameters:
    data (torch_geometric.data.Data) - The graph data object.
    node_ids (int, Tensor, iterable[int]) - The nodes to be removed.
    store_ids (bool, optional) - If True, store the original IDs of the remaining nodes and edges.

    Returns:
    torch_geometric.data.Data - The graph with nodes deleted.
    """

    # Convert node_ids to a tensor if it's not already
    if isinstance(node_ids, int):
        node_ids = torch.tensor([node_ids])
    elif isinstance(node_ids, (list, tuple)):
        node_ids = torch.tensor(node_ids)

    data = data.to(device)
    node_ids = node_ids.to(device)

    # Create a mask for the nodes to keep
    mask = torch.ones(data.num_nodes, dtype=torch.bool, device=device)

    if len(node_ids) > 0:
        mask[node_ids] = False

    # Create a new Data object for the new graph
    new_data = Data().to(device)

    # Store original node IDs if requested
    if store_ids:
        new_data.node_ids = torch.arange(data.num_nodes, device=device)[mask]

    # Copy over node features, keeping only the masked nodes
    for key, value in data.items():
        if key.startswith('x'):
            new_data[key] = value[mask]

    # Adjust the edge index to remove edges connected to the removed nodes
    edge_index = data.edge_index
    edge_mask = mask[edge_index[0]] & mask[edge_index[1]]
    new_data.edge_index = edge_index[:, edge_mask]

    # Optionally, store original edge IDs if requested
    if store_ids:
        new_data.edge_ids = torch.arange(edge_index.size(1), device=device)[edge_mask]

    # Optionally, relabel the nodes
    new_node_idx = torch.zeros(data.num_nodes, dtype=torch.long, device=device)

    new_node_idx[mask] = torch.arange(mask.sum(), device=device)
    new_data.edge_index = new_node_idx[new_data.edge_index]

    # Copy over other edge features
    for key, value in data.items():
        if key.startswith('edge') and key != 'edge_index':
            new_data[key] = value[edge_mask]

    return new_data.to(device)
