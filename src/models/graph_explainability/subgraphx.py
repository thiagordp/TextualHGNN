import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.utils import to_networkx, subgraph, remove_isolated_nodes
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import networkx as nx

from src.utils.graph_utils import remove_nodes


class MCTSNode:
    """
    A class representing a node in the Monte Carlo Tree Search (MCTS).

    Attributes:
        nodes (Tensor): The nodes contained in this MCTS node.
        num_visit (int): The number of times this node has been visited.
        total_reward (float): The total reward accumulated by this node.
        immediate_reward (float): The immediate reward of this node.
        children (list): The children nodes of this MCTS node.
    """

    def __init__(self, nodes):
        self.nodes = nodes
        self.num_visit = 0
        self.total_reward = 0.0
        self.immediate_reward = 0.0
        self.children = []

    def __repr__(self):
        return str(f"MCTSNode(\n"
                   f"   nodes={self.nodes}, \n"
                   f"   num_visit={self.num_visit},\n "
                   f"   total_reward={self.total_reward:.3f}, \n"
                   f"   immediate_reward={self.immediate_reward:.3f}, \n"
                   f"   children={len(self.children)}\n"
                   f")")


class SubgraphX(nn.Module):
    r"""SubgraphX from `On Explainability of Graph Neural Networks via Subgraph
    Explorations <https://arxiv.org/abs/2102.05152>`

    It identifies the most important subgraph from the original graph that
    plays a critical role in GNN-based graph classification.

    It employs Monte Carlo tree search (MCTS) in efficiently exploring
    different subgraphs for explanation and uses Shapley values as the measure
    of subgraph importance.

    Parameters
    ----------
    model : nn.Module
        The GNN model to explain that tackles multiclass graph classification

        * Its forward function must have the form
          :attr:`forward(self, graph, nfeat)`.
        * The output of its forward function is the logits.
    num_hops : int
        Number of message passing layers in the model
    coef : float, optional
        This hyperparameter controls the trade-off between exploration and
        exploitation. A higher value encourages the algorithm to explore
        relatively unvisited nodes. Default: 10.0
    high2low : bool, optional
        If True, it will use the "High2low" strategy for pruning actions,
        expanding children nodes from high degree to low degree when extending
        the children nodes in the search tree. Otherwise, it will use the
        "Low2high" strategy. Default: True
    num_child : int, optional
        This is the number of children nodes to expand when extending the
        children nodes in the search tree. Default: 12
    num_rollouts : int, optional
        This is the number of rollouts for MCTS. Default: 20
    node_min : int, optional
        This is the threshold to define a leaf node based on the number of
        nodes in a subgraph. Default: 3
    shapley_steps : int, optional
        This is the number of steps for Monte Carlo sampling in estimating
        Shapley values. Default: 100
    log : bool, optional
        If True, it will log the progress. Default: False
    """

    def __init__(
            self,
            model,
            num_hops,
            coef=10.0,
            high2low=True,
            num_child=12,
            num_rollouts=20,
            node_min=3,
            shapley_steps=100,
            log=False,
            device="cpu"
    ):
        super().__init__()
        self.mcts_node_maps = None
        self.kwargs = None
        self.target_class = None
        self.feat = None
        self.graph = None
        self.num_hops = num_hops
        self.coef = coef
        self.high2low = high2low
        self.num_child = num_child
        self.num_rollouts = num_rollouts
        self.node_min = node_min
        self.shapley_steps = shapley_steps
        self.log = log
        self.model = model
        self.device = device

    def shapley(self, subgraph_nodes):
        """
           Estimate the Shapley value for a given subgraph.

           Args:
               subgraph_nodes (Tensor): The nodes of the subgraph.

           Returns:
               float: The estimated Shapley value.
        """
        num_nodes = self.graph.num_nodes
        subgraph_nodes = subgraph_nodes.tolist()

        # Obtain neighboring nodes of the subgraph g_i, P'.
        local_region = subgraph_nodes
        for _ in range(self.num_hops - 1):
            neighbors = set()
            for node in local_region:
                neighbors.update(self.graph.edge_index[1, self.graph.edge_index[0] == node].tolist())
                neighbors.update(self.graph.edge_index[0, self.graph.edge_index[1] == node].tolist())
            local_region = list(set(local_region + list(neighbors)))

        split_point = num_nodes
        coalition_space = list(set(local_region) - set(subgraph_nodes)) + [split_point]

        marginal_contributions = []
        device = self.feat.device
        for _ in range(self.shapley_steps):
            permuted_space = np.random.permutation(coalition_space)
            split_idx = int(np.where(permuted_space == split_point)[0])

            selected_nodes = permuted_space[:split_idx]

            exclude_mask = torch.ones(num_nodes, device=device)
            exclude_mask[local_region] = 0.0
            exclude_mask[selected_nodes] = 1.0

            include_mask = exclude_mask.clone()
            include_mask[subgraph_nodes] = 1.0

            exclude_feat = self.feat * exclude_mask.unsqueeze(1)
            include_feat = self.feat * include_mask.unsqueeze(1)

            with torch.no_grad():
                exclude_graph = self.graph.clone()
                exclude_graph.x = exclude_feat
                exclude_probs = self.model(exclude_graph).softmax(dim=-1)
                exclude_value = exclude_probs[:, self.target_class]

                include_graph = self.graph.clone()
                include_graph.x = include_feat
                include_probs = self.model(include_graph).softmax(dim=-1)
                include_value = include_probs[:, self.target_class]

                # exclude_probs = self.model(self.graph, exclude_feat, **self.kwargs).softmax(dim=-1)
                # exclude_value = exclude_probs[:, self.target_class]
                # include_probs = self.model(self.graph, include_feat, **self.kwargs).softmax(dim=-1)
                # include_value = include_probs[:, self.target_class]
            marginal_contributions.append(include_value - exclude_value)

        return torch.cat(marginal_contributions).mean().item()

    def get_mcts_children(self, mcts_node):
        """
        Get the children of a given MCTS node by expanding the current subgraph.

        Args:
            mcts_node (MCTSNode): The MCTS node to expand.

        Returns:
            list: A list of child MCTS nodes.
        """
        if len(mcts_node.children) > 0:
            return mcts_node.children

        # mcts_node.nodes
        # subg_edge_index, subg_edge_attr = subgraph(
        #     subset=mcts_node.nodes,
        #     edge_index=self.graph.edge_index,
        #     edge_attr=self.graph.edge_attr,
        #     relabel_nodes=False,
        #     num_nodes=self.graph.num_nodes
        # )
        # subg = Data(x=self.feat[mcts_node.nodes], edge_index=subg_edge_index, edge_attr=subg_edge_attr)

        nodes_to_remove = [node for node in range(self.graph.num_nodes) if node not in mcts_node.nodes]
        subg = remove_nodes(self.graph, nodes_to_remove, store_ids=True, device=self.device)

        node_degrees = subg.edge_index[0].bincount() + subg.edge_index[1].bincount()
        k = min(subg.num_nodes, self.num_child)
        chosen_nodes = torch.topk(node_degrees, k, largest=self.high2low).indices
        # Até aqui, tudo igual.
        mcts_children_maps = dict()

        for node in chosen_nodes:

            new_subg = remove_nodes(
                subg, [node], store_ids=True
            )
            # nodes_to_include = mcts_node.nodes[mcts_node.nodes != node]
            # new_subg_edge_index, new_subg_edge_attr = subgraph(
            #     subset=nodes_to_include,
            #     edge_index=subg_edge_index,
            #     edge_attr=subg_edge_attr,
            #     relabel_nodes=True,
            #     num_nodes=subg.num_nodes
            # )
            # new_subg = Data(x=self.feat[nodes_to_include], edge_index=new_subg_edge_index, edge_attr=new_subg_edge_attr)
            # new_subg = Data(x=subg.x[new_subg_edge_index], edge_index=new_subg_edge_index)
            # new_edge_index, new_edge_attr, node_mask = remove_isolated_nodes(
            #     edge_index=new_subg.edge_index,
            #     edge_attr=new_subg.edge_attr,
            #     num_nodes=subg.num_nodes
            # )

            # nodes_to_exclude = torch.unique(new_edge_index)
            # new_x = new_subg.x[nodes_to_exclude]

            # new_subg = Data(
            #     x=new_x,
            #     edge_index=new_edge_index,
            #     edge_attr=new_edge_attr
            # )

            # Não apagou os nós.
            new_subg_nx = to_networkx(new_subg.cpu(), node_attrs=["x"], to_undirected=False)

            largest_cc_nids = list(
                max(nx.weakly_connected_components(new_subg_nx), key=len)
            )

            largest_cc_nids = new_subg.node_ids[largest_cc_nids].long()
            largest_cc_nids = subg.node_ids[largest_cc_nids].sort().values

            # largest_cc_nids = max(nx.connected_components(new_subg_nx), key=len)
            # largest_cc_nids = torch.tensor(list(largest_cc_nids), dtype=torch.long)

            if str(largest_cc_nids) not in self.mcts_node_maps:
                child_mcts_node = MCTSNode(largest_cc_nids)
                self.mcts_node_maps[str(child_mcts_node)] = child_mcts_node
            else:
                child_mcts_node = self.mcts_node_maps[str(largest_cc_nids)]

            if str(child_mcts_node) not in mcts_children_maps:
                mcts_children_maps[str(child_mcts_node)] = child_mcts_node

        mcts_node.children = list(mcts_children_maps.values())
        for child_mcts_node in mcts_node.children:
            if child_mcts_node.immediate_reward == 0:
                child_mcts_node.immediate_reward = self.shapley(child_mcts_node.nodes)

        return mcts_node.children

    def mcts_rollout(self, mcts_node):
        """
        Perform a rollout in MCTS starting from the given node.

        Args:
            mcts_node (MCTSNode): The starting MCTS node.

        Returns:
            float: The reward obtained from the rollout.
        """

        def calculate_child(nodes):
            calculations = []
            for c in nodes:
                calculations.append(
                    c.total_reward / max(c.num_visit, 1) + (self.coef * c.immediate_reward * children_visit_sum_sqrt
                                                            / (1 + c.num_visit))
                )

            max_index = calculations.index(max(calculations))

            return nodes[max_index]

        if len(mcts_node.nodes) <= self.node_min:
            return mcts_node.immediate_reward

        children_nodes = self.get_mcts_children(mcts_node)
        children_visit_sum = sum([child.num_visit for child in children_nodes])
        if children_visit_sum > 0:
            _x = 0
        children_visit_sum_sqrt = math.sqrt(children_visit_sum)
        chosen_child = calculate_child(children_nodes)

        reward = self.mcts_rollout(chosen_child)
        chosen_child.num_visit += 1
        chosen_child.total_reward += reward

        return reward

    def explain_graph(self, graph, feat, target_class, **kwargs):
        """
        Explain the prediction of the GNN model for a given graph and target class.

        Args:
            graph (Data): The input graph.
            feat (Tensor): The node features.
            target_class (int): The target class for which the explanation is sought.
            **kwargs: Additional arguments for the GNN model.

        Returns:
            Tensor: The nodes of the subgraph that best explains the prediction.
        """

        def show_mcts_nodes(root, count):

            def _print_tabs(count):
                for i in range(count):
                    logging.info("\t", end="")

            _print_tabs(count)
            logging.info(root)

            if len(mcts_node.children) == 0:
                return None

            for children in mcts_node.children:
                show_mcts_nodes(children, count + 1)

        self.model.eval()
        assert graph.num_nodes > self.node_min, f"The number of nodes in the graph {graph.num_nodes} should be bigger than {self.node_min}."

        self.graph = graph
        self.feat = feat
        self.target_class = target_class
        self.kwargs = kwargs

        self.mcts_node_maps = dict()

        root = MCTSNode(torch.arange(graph.num_nodes))
        self.mcts_node_maps[str(root)] = root

        for i in range(self.num_rollouts):
            if self.log:
                logging.info(
                    f"Rollout {i + 1:3d}/{self.num_rollouts:3d}, "
                    f"{len(self.mcts_node_maps):4d} subgraphs have been explored."
                )
            self.mcts_rollout(root)

        best_leaf = None
        best_immediate_reward = float("-inf")
        for mcts_node in self.mcts_node_maps.values():
            if len(mcts_node.nodes) > self.node_min:
                continue

            if mcts_node.immediate_reward > best_immediate_reward:
                best_leaf = mcts_node
                best_immediate_reward = best_leaf.immediate_reward

        # if self.log:
        #     print(f"Best leaf node: {best_leaf}")

        return best_leaf.__dict__
