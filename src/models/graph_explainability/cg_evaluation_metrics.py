# src/models/graph_explainability/cg_evaluation_metrics.py

import torch
import networkx as nx
from collections import Counter, defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, silhouette_score
import community as community_louvain # You may need to run: pip install python-louvain
import numpy as np
from torch_geometric.utils import to_networkx
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash

class ConceptCompletenessCalculator:
    """
    Calculates concept completeness by training a decision tree on the multiset of concepts in a graph.
    This version is designed to be updated iteratively during a validation loop.
    """
    def __init__(self):
        self.concept_assignments = []
        self.labels = []

    def add_item(self, y_true, concept_ids):
        """Adds a single graph's data to the calculator."""

        # Handle different label formats (scalar, one-hot vector, etc.)
        if y_true.numel() > 1:
            # If y_true is a one-hot or multi-hot vector, convert to a class index
            label = y_true.argmax().item()
        else:
            # If it's already a scalar tensor
            label = y_true.item()
        self.labels.append(label)

        self.concept_assignments.append(Counter(concept_ids.tolist()))

    def calculate(self):
        """Calculates the final completeness score based on all added items."""
        if not self.labels or not self.concept_assignments:
            return 0.0
        
        # Determine the full vocabulary of concept IDs across all graphs
        max_concept_id = 0
        for counter in self.concept_assignments:
            if counter:
                max_concept_id = max(max_concept_id, max(counter.keys()))
        
        num_features = max_concept_id + 1
        X = np.zeros((len(self.labels), num_features))

        # Create the feature matrix from the multisets
        for i, counter in enumerate(self.concept_assignments):
            for concept_id, count in counter.items():
                if concept_id < num_features:
                    X[i, concept_id] = count
        
        y = np.array(self.labels)
        
        # Train a simple, interpretable model (Decision Tree)
        dt = DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced')
        dt.fit(X, y)
        y_pred = dt.predict(X)
        
        return accuracy_score(y, y_pred)

class ConceptConformityCalculator:
    """
    Measures the purity of concepts using a graph hashing proxy for isomorphism.
    This implementation is significantly more efficient than using nx.is_isomorphic.
    """
    def __init__(self, conformity_threshold=0.1):
        self.threshold = conformity_threshold
        # Stores {concept_id: [list_of_subgraph_hashes]}
        self.concept_subgraphs = defaultdict(list)

    def add_item(self, data, concept_ids):
        """Adds a single graph's data to the calculator."""
        graph = to_networkx(data, to_undirected=True)
        
        # Group nodes by their assigned concept
        nodes_in_concept = defaultdict(list)
        for i, concept_id in enumerate(concept_ids):
            nodes_in_concept[concept_id.item()].append(i)
        
        for concept_id, nodes in nodes_in_concept.items():
            if not nodes: continue
            subgraph = graph.subgraph(nodes)
            # WL hash is a fast and effective proxy for graph isomorphism
            wl_hash = weisfeiler_lehman_graph_hash(subgraph)
            self.concept_subgraphs[concept_id].append(wl_hash)

    def calculate(self):
        """Calculates the final conformity score based on all added items."""
        total_conformity = 0
        num_concepts_evaluated = 0

        for concept_id, subgraphs in self.concept_subgraphs.items():
            if not subgraphs: continue
            
            num_concepts_evaluated += 1
            subgraph_counts = Counter(subgraphs)
            o_c = len(subgraphs) # Total number of subgraphs for this concept
            
            # Sum the counts of subgraphs that are not "noise"
            conformity_sum = sum(count for count in subgraph_counts.values() if count >= self.threshold * o_c)
            total_conformity += conformity_sum / o_c if o_c > 0 else 0
            
        # Return the average conformity over all non-empty concepts
        return total_conformity / num_concepts_evaluated if num_concepts_evaluated > 0 else 1.0

class ModularityCalculator:
    """
    Calculates the average modularity of the learned clusters across all graphs in a dataset.
    """
    def __init__(self):
        self.total_modularity = 0
        self.num_graphs = 0

    def add_item(self, data, concept_ids):
        """Adds a single graph's data to the calculator."""
        if data.num_edges == 0: return # Modularity is undefined for graphs with no edges
        
        graph = to_networkx(data, to_undirected=True)
        partition = {i: c_id.item() for i, c_id in enumerate(concept_ids)}
        
        self.total_modularity += community_louvain.modularity(partition, graph)
        self.num_graphs += 1

    def calculate(self):
        """Calculates the final average modularity."""
        avg_modularity = self.total_modularity / self.num_graphs if self.num_graphs > 0 else 0
        # Scale modularity from its typical range of [-0.5, 1] to [0, 1] for the HI-Score
        return (avg_modularity + 0.5) / 1.5

class SilhouetteScoreCalculator:
    """
    Calculates the average Silhouette Score based on node features for all graphs in a dataset.
    """
    def __init__(self):
        self.total_silhouette = 0
        self.num_graphs = 0

    def add_item(self, data, concept_ids):
        """Adds a single graph's data to the calculator."""
        node_features = data.x.cpu().numpy()
        labels = concept_ids.cpu().numpy()

        # Silhouette score is only defined if there is more than 1 cluster
        if len(np.unique(labels)) > 1:
            self.total_silhouette += silhouette_score(node_features, labels)
            self.num_graphs += 1

    def calculate(self):
        """Calculates the final average silhouette score."""
        avg_score = self.total_silhouette / self.num_graphs if self.num_graphs > 0 else 0
        # Scale silhouette from its native range of [-1, 1] to [0, 1] for the HI-Score
        return (avg_score + 1) / 2