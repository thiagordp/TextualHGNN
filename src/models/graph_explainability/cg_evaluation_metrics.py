"""
Concept Grounding evaluation metrics.
"""
import math
import re
from typing import Optional, List

import matplotlib
import networkx as nx
import networkx.algorithms.isomorphism as iso
import numpy as np
import torch
import torch.nn as nn
import tqdm
from sklearn.tree import DecisionTreeClassifier
from torch_geometric.data import DataLoader


def calculate_concept_completeness(self, multiset: bool = True, data_part: float = 1.0, seeds: List[int] = [1],
                                   plot_tree: bool = False, max_depth: Optional[int] = None,
                                   save_path: Optional[str] = None, verbose: bool = True,
                                   num_gcexplainer_clusters: Optional[int] = None, **kwargs) -> \
        torch.Tensor:
    """
    Important: For the inputs, this assumes one-hot encoding
    """
    results = []
    for seed in seeds:
        trees = []
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_set_size = math.floor(data_part * len(self.train_loader.dataset))
        test_set_size = math.floor(data_part * len(self.test_loader.dataset))
        data_loader = self.train_loader.__class__(self.train_loader.dataset[:train_set_size] +
                                                  self.test_loader.dataset[:test_set_size],
                                                  self.train_loader.batch_size)

        all_data = self.load_required_data(data_loader, 1, "joint train and test",
                                           ["x", "target", "info_pooling_assignments", "concepts", "mask"])
        # enumerate_feat = 2 ** torch.arange(all_data["x"].shape[-1],
        #                                    device=all_data["x"].device)[None, None, :]
        # Note that I can't calculate concept completeness for ASAP anyway as I have no mapping from nodes to concepts.
        # so masks is not required in determinisitc_assignemnts

        all_train_assignments = [torch.argmax(all_data["x"][:train_set_size], dim=-1)]

        all_test_assignments = [torch.argmax(all_data["x"][train_set_size:], dim=-1)]

        if True:
            all_ass = Analyzer.deterministic_concept_assignments(self.model, all_data["info_pooling_assignments"],
                                                                 [all_data["mask"]], all_data["concepts"],
                                                                 num_gcexplainer_clusters)
            all_train_assignments += [ass[:train_set_size] for ass in all_ass]
            all_test_assignments += [ass[train_set_size:] for ass in all_ass]
        else:
            all_train_assignments += Analyzer.deterministic_concept_assignments(self.model,
                                                                                [None if d is None else d[
                                                                                                        :train_set_size]
                                                                                 for d in
                                                                                 all_data["info_pooling_assignments"]],
                                                                                None, None)
            all_test_assignments += Analyzer.deterministic_concept_assignments(self.model,
                                                                               [None if d is None else d[
                                                                                                       train_set_size:]
                                                                                for d in
                                                                                all_data["info_pooling_assignments"]],
                                                                               None, None)

        result = []
        for pool_step, (train_assignments, test_assignments) in enumerate(zip(all_train_assignments,
                                                                              all_test_assignments)):
            if train_assignments is None:
                continue  # For non-pooling layers
            num_concepts = max(torch.max(train_assignments), torch.max(test_assignments)) + 1
            batched_bincount = vmap(partial(torch.bincount, minlength=num_concepts + 1))

            # [batch_size, num_concepts] (note that assignments contains -1 for masked values)
            multisets_train = batched_bincount(train_assignments + 1)[:, 1:]
            multisets_test = batched_bincount(test_assignments + 1)[:, 1:]

            if not multiset:
                multisets_train = multisets_train.bool().int()
                multisets_test = multisets_test.bool().int()

            acc, tree = self._decision_tree_acc(multisets_train.cpu(), all_data["target"][:train_set_size].squeeze(),
                                                multisets_test.cpu(), all_data["target"][train_set_size:].squeeze(),
                                                return_tree=True, random_state=seed, max_depth=max_depth)
            result.append(acc)
            trees.append(tree)
        results.append(result)
    # [num_seeds, num_mc_blocks + 1]
    results = torch.tensor(results)
    stds, means = torch.std_mean(results, dim=0)

    if verbose:
        for i in range(stds.shape[0]):
            print(f"{100 * means[i]:.2f}%+-{100 * stds[i]:.2f}")

    if plot_tree:
        for i, tree in enumerate(trees):
            fig, ax = plt.subplots(figsize=(48, 12))
            sklearn.tree.plot_tree(tree, ax=ax, proportion=True,
                                   class_names=self.dataset_wrapper.class_names, fontsize=11, impurity=True)

            # up = False
            def replace_text(obj):
                # nonlocal up
                if type(obj) == matplotlib.text.Annotation:
                    txt = obj.get_text()
                    txt = re.sub("value[^$]*class", "class", txt)
                    txt = re.sub(":", ":\n", txt)
                    obj.set_text(txt)
                    # obj.set(y=obj.xy[1]+0.02)
                # up = not up
                return obj

            # Remove "value" https://stackoverflow.com/questions/70553185/how-to-plot-tree-without-showing-samples-and-value-in-random-forest
            ax.properties()['children'] = [replace_text(i) for i in ax.properties()['children']]

            if save_path is not None:
                fig.savefig(f"{save_path}_{i}.pdf")
            plt.show()
    return results


class ConceptCompletenessCalculator:
    def __init__(self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader):
        self.model = model
        self.model.eval()  # Set model to evaluation mode
        self.train_loader = train_loader
        self.test_loader = test_loader

    def calculate_concept_completeness(self):
        # Step 1: Collect concept assignments for train and test data
        train_assignments, train_labels = self._collect_assignments(self.train_loader)
        test_assignments, test_labels = self._collect_assignments(self.test_loader)

        # Step 2: Train decision tree on concept assignments
        completeness_scores = []
        for layer in range(len(train_assignments)):
            X_train = train_assignments[layer]
            X_test = test_assignments[layer]
            y_train = train_labels
            y_test = test_labels

            tree = DecisionTreeClassifier(random_state=0)
            tree.fit(X_train, y_train)
            score = tree.score(X_test, y_test)
            completeness_scores.append(score)

        print("Concept Completeness Scores for each layer:", completeness_scores)
        return completeness_scores

    def _collect_assignments(self, loader):
        all_assignments = [[] for _ in range(2)]  # For 2 DiffPool layers
        all_labels = []

        with torch.no_grad():
            for data in loader:
                x, adj, labels = data.x, data.adj, data.y
                _, _, _, (emb_l1, _), (emb_l2, _), (s01, s12) = self.model(x, adj, debug=True)

                # Collect concept assignments (cluster IDs)
                assignments_l1 = torch.argmax(s01, dim=-1)
                assignments_l2 = torch.argmax(s12, dim=-1)

                # Convert assignments to histograms (concept presence)
                hist_l1 = torch.bincount(assignments_l1.flatten(), minlength=s01.shape[-1]).unsqueeze(0)
                hist_l2 = torch.bincount(assignments_l2.flatten(), minlength=s12.shape[-1]).unsqueeze(0)

                all_assignments[0].append(hist_l1)
                all_assignments[1].append(hist_l2)
                all_labels.append(labels)

        # Concatenate and prepare for decision tree
        all_assignments = [torch.cat(layer_assignments, dim=0).cpu().numpy() for layer_assignments in all_assignments]
        all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
        return all_assignments, all_labels


# Example Usage:
# Assuming `diffpool_model` is your pre-trained DiffPool model
# Assuming `train_loader` and `test_loader` are DataLoader objects for training and test datasets

# completeness_calculator = ConceptCompletenessCalculator(diffpool_model, train_loader, test_loader)
# completeness_scores = completeness_calculator.calculate_concept_completeness()


class ConceptCompletenessCalculatorV2:
    def __init__(self, model: nn.Module, train_loader: DataLoader, test_loader: DataLoader, device: torch.device):
        self.model = model.to(device)
        self.model.eval()  # Set model to evaluation mode
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def calculate_concept_completeness(self):
        # Step 1: Collect concept multisets for train and test data
        train_multisets, train_labels = self._collect_multisets(self.train_loader)
        test_multisets, test_labels = self._collect_multisets(self.test_loader)

        # Step 2: Train decision tree on concept multisets
        completeness_scores = []
        for layer in range(len(train_multisets)):
            X_train = train_multisets[layer]
            X_test = test_multisets[layer]
            y_train = train_labels
            y_test = test_labels

            local_metrics = []
            for seed in range(7):
                tree = DecisionTreeClassifier(random_state=seed)
                tree.fit(X_train, y_train)
                score = tree.score(X_test, y_test)
                local_metrics.append(score)

            completeness_scores.append((np.mean(local_metrics), np.std(local_metrics)))

        print("Concept Completeness Scores for each layer:", completeness_scores)
        return completeness_scores

    def _collect_multisets(self, loader):
        all_multisets = [[] for _ in range(2)]  # For 2 DiffPool layers
        all_labels = []

        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)

                x, adj, labels = data.x, data.adj, data.y

                _, _, _, (emb_l1, _), (emb_l2, _), (s01, s12) = self.model(x, adj, debug=True)

                # Collect concept assignments (cluster IDs)
                assignments_l1 = torch.argmax(s01, dim=-1)
                assignments_l2 = torch.argmax(s12, dim=-1)

                # Convert assignments to multisets (exact concept IDs)
                multiset_l1 = self._build_multiset(assignments_l1)
                multiset_l2 = self._build_multiset(assignments_l2)

                all_multisets[0].append(multiset_l1)
                all_multisets[1].append(multiset_l2)
                labels = torch.tensor([l[0] for l in labels], dtype=torch.long).to(self.device)
                all_labels.append(labels)

                break

        # Convert multisets to numpy arrays for decision tree
        all_multisets = [torch.cat(layer_multisets, dim=0).cpu().numpy() for layer_multisets in all_multisets]
        all_labels = torch.cat(all_labels, dim=0).cpu().numpy()
        return all_multisets, all_labels

    def _build_multiset(self, assignments):
        """
        Converts concept assignments to a multiset representation.
        """
        batch_size, num_nodes = assignments.shape
        multisets = []

        i = 0
        for i in tqdm.tqdm(range(batch_size), desc=f"Retrieving multisets document {i + 1}/{batch_size}"):
            concept_ids = assignments[i]
            concept_set = torch.zeros(100, dtype=torch.int)  # TODO: remove hardcoded (# of nodes L1.)
            for concept_id in concept_ids:
                concept_set[concept_id] += 1
            multisets.append(concept_set.unsqueeze(0))  # (1, num_concepts)

        return torch.cat(multisets, dim=0)  # (batch_size, num_concepts)


class ConceptConformityCalculator:
    def __init__(self, model: nn.Module, data_loader: DataLoader, num_layers=2, conformity_threshold=0.1):
        self.model = model
        self.model.eval()  # Set model to evaluation mode
        self.data_loader = data_loader
        self.num_layers = num_layers  # Number of DiffPool layers
        self.conformity_threshold = conformity_threshold

    def calculate_concept_conformity(self):
        all_conformity_scores = []

        # Step 1: Collect subgraphs for each concept in each layer
        for layer in range(self.num_layers):
            print(f"Calculating Concept Conformity for Layer {layer + 1}...")
            concept_subgraphs = self._collect_subgraphs(layer)
            conformity_scores = self._calculate_conformity(concept_subgraphs)
            all_conformity_scores.append(conformity_scores)

        return all_conformity_scores

    def _collect_subgraphs(self, layer):
        concept_subgraphs = {}
        with torch.no_grad():
            for data in self.data_loader:
                x, adj, labels = data.x, data.adj, data.y
                _, _, _, (emb_l1, _), (emb_l2, _), (s01, s12) = self.model(x, adj, debug=True)

                assignments = torch.argmax(s01, dim=-1) if layer == 0 else torch.argmax(s12, dim=-1)

                for graph_idx in range(x.size(0)):  # Iterate over each graph in the batch
                    assignment = assignments[graph_idx]
                    adj_matrix = adj[graph_idx].cpu().numpy()
                    node_features = x[graph_idx].cpu().numpy()

                    # Create NetworkX graph
                    G = nx.from_numpy_matrix(adj_matrix)
                    for node in G.nodes:
                        G.nodes[node]["concept"] = int(assignment[node].item())

                    # Collect subgraphs for each concept
                    for node in G.nodes:
                        concept_id = G.nodes[node]["concept"]

                        # Extract the k-hop subgraph for this node (full connected subgraph)
                        subgraph_nodes = [n for n in G if G.nodes[n]["concept"] == concept_id]
                        subgraph = G.subgraph(subgraph_nodes).copy()

                        if concept_id not in concept_subgraphs:
                            concept_subgraphs[concept_id] = []

                        concept_subgraphs[concept_id].append(subgraph)

        return concept_subgraphs

    def _calculate_conformity(self, concept_subgraphs):
        conformity_scores = {}
        for concept_id, subgraphs in concept_subgraphs.items():
            # Calculate frequency of each unique subgraph
            subgraph_counts = self._count_unique_subgraphs(subgraphs)
            total_count = sum(subgraph_counts.values())

            # Calculate conformity score
            dominant_count = sum(
                count for count in subgraph_counts.values()
                if count >= self.conformity_threshold * total_count
            )
            conformity_score = dominant_count / total_count if total_count > 0 else 0.0
            conformity_scores[concept_id] = conformity_score

        return conformity_scores

    def _count_unique_subgraphs(self, subgraphs):
        """
        Count occurrences of unique subgraphs using isomorphism.
        """
        unique_subgraphs = {}
        for subgraph in subgraphs:
            matched = False
            for existing_subgraph in unique_subgraphs:
                if nx.is_isomorphic(subgraph, existing_subgraph,
                                    node_match=iso.categorical_node_match("concept", None)):
                    unique_subgraphs[existing_subgraph] += 1
                    matched = True
                    break

            if not matched:
                unique_subgraphs[subgraph] = 1

        return unique_subgraphs
