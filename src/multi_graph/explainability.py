"""
Explainability functions

@author Thiago R Dal Pont
"""
import logging

import networkx as nx
import torch
import torch.nn.functional as F
from itertools import permutations
from collections import deque
from typing import List, Dict, Any, Optional, Set


class SubGraphConceptGrounder:
    """
    A tool to perform concept grounding on GNN-processed document graphs.

    This class extracts meaningful, human-readable concepts for phrase-level
    nodes by finding the shortest path that connects the most salient words,
    as determined by a GNN's learned attention mechanism.

    Attributes:
        top_nodes (int): The number of top-attention words to use as anchors.
        skip_nodes (int): The max number of intermediate nodes between anchors (M).
        all_combinations (bool): Whether to check all permutations of anchor words.
        edge_type (str): The type of edges to use in the search ('dep', 'seq', 'any').
    """

    def __init__(
            self,
            top_nodes: int = 3,
            skip_nodes: int = 2,
            all_combinations: bool = True,
            edge_type: str = 'dep',
    ):
        """
        Initializes the ConceptGrounder with its search configuration.

        Args:
            top_nodes: The number of top-attention words to use as anchors.
            skip_nodes: The max number of intermediate nodes between anchors (M).
            all_combinations: Whether to check all permutations of anchor words.
            edge_type: The type of edges to use in the search ('dep', 'seq', 'any').
        """

        self.top_nodes = top_nodes
        self.skip_nodes = skip_nodes
        self.all_combinations = all_combinations
        self.edge_type = edge_type

    @staticmethod
    def _find_shortest_path(
            G: nx.MultiDiGraph,
            start_node_id: str,
            end_node_id: str,
            M: int,
            edge_type: str,
            nodes_to_avoid: Optional[Set[str]] = None
    ) -> Optional[List[str]]:
        """
        Finds the shortest path between two nodes, avoiding a given set of nodes.

        This is the core search algorithm, implemented as a static method as it
        does not depend on instance state.

        Returns:
            A list of node IDs representing the path, or None if no valid path is found.
        """

        if start_node_id not in G or end_node_id not in G:
            logging.error("Start or end nodes not found in G")
            return None

        if start_node_id == end_node_id:
            return [start_node_id]

        # Initialize the visited set with the start node and any nodes to be excluded.
        # The destination node is explicitly allowed to be "re-visited" if it was in the avoid set.

        visited = {start_node_id}
        if nodes_to_avoid:
            visited.update(nodes_to_avoid - {end_node_id})

        queue = deque([(start_node_id, [start_node_id])])

        while queue:
            current_node, path = queue.popleft()

            # Pruning: if the path to the current node is already too long, skip it.
            if len(path) - 1 > M + 1:
                continue

            for _, neighbor, data in G.edges(current_node, data=True):
                if edge_type != 'any' and data.get("type") != edge_type:
                    continue

                if neighbor not in visited:
                    new_path = path + [neighbor]
                    if neighbor == end_node_id:
                        # Path found, check final constraint.
                        if len(new_path) - 2 <= M:
                            return new_path
                        # If too long, we don't return but continue searching for a shorter alternative.
                        else:
                            continue
                    visited.add(neighbor)
                    queue.append((neighbor, new_path))

        return None

    def _prepare_graph(self, nx_graph: nx.MultiDiGraph, explanation: Dict[str, Any]) -> nx.MultiDiGraph:
        """
        Injects learned attention scores from a GNN explanation object into a graph copy.
        This makes the attention data directly accessible on the graph edges.
        """
        enriched_graph = nx_graph.copy()
        node_mappings = explanation.get('node_mappings', {})
        if not node_mappings:
            return enriched_graph

        word_map_rev = {v: k for k, v in node_mappings.get('word', {}).items()}
        sent_map_rev = {v: k for k, v in node_mappings.get('sentence', {}).items()}

        word_att_edge_index, word_att_weights = explanation.get('word_to_sent_att', ([], []))
        word_att_map = {
            (word_map_rev.get(u), sent_map_rev.get(v)): w.item()
            for u, v, w in zip(word_att_edge_index[0].tolist(), word_att_edge_index[1].tolist(), word_att_weights)
        }

        for u, v, key in enriched_graph.edges(keys=True):
            if enriched_graph.edges[u, v, key].get('type') == 'belongs':
                attention_score = word_att_map.get((u, v))
                if attention_score is not None:
                    enriched_graph.edges[u, v, key]['learned_attention'] = attention_score

        return enriched_graph

    def _get_top_words_for_phrase(self, nx_graph: nx.MultiDiGraph, phrase_node_id: str) -> List[str]:
        """Gets the top N word node IDs for a phrase based on attention, using self.top_nodes."""
        word_attentions = [
            (u, abs(data.get('learned_attention', 0)))
            for u, _, data in nx_graph.in_edges(phrase_node_id, data=True)
            if nx_graph.nodes[u].get('type') == 'word'
        ]
        word_attentions.sort(key=lambda x: x[1], reverse=True)
        return [word_id for word_id, _ in word_attentions[:self.top_nodes]]

    def _find_connecting_path(self, nx_graph: nx.MultiDiGraph, ordered_words: List[str]) -> Optional[List[str]]:
        """Searches for a continuous path connecting an ordered list of words."""
        if not ordered_words: return None
        if len(ordered_words) == 1: return ordered_words

        full_path_nodes = []
        for i in range(len(ordered_words) - 1):
            start_node = ordered_words[i] if i == 0 else full_path_nodes[-1]
            end_node = ordered_words[i + 1]

            # Exclude nodes already in the path to prevent loops.
            nodes_to_avoid = set(full_path_nodes)

            segment = self._find_shortest_path(
                nx_graph, start_node, end_node, M=self.skip_nodes,
                edge_type=self.edge_type, nodes_to_avoid=nodes_to_avoid
            )

            if segment is None: return None  # The entire path is invalid if one segment fails.

            full_path_nodes.extend(segment if not full_path_nodes else segment[1:])

        return full_path_nodes

    def _rank_candidate_paths(self, nx_graph: nx.MultiDiGraph, candidate_paths: List[List[str]],
                              phrase_embedding: torch.Tensor) -> Optional[List[str]]:
        """Ranks candidate paths by their semantic similarity to the parent phrase."""
        best_path = None
        max_similarity = -1.0

        for path in candidate_paths:
            word_embeddings = [
                nx_graph.nodes[node_id]['x'] for node_id in path
                if nx_graph.nodes[node_id].get('type') == 'word' and 'x' in nx_graph.nodes[node_id]
            ]
            if not word_embeddings: continue

            # TODO: use the attention.
            avg_path_embedding = torch.mean(torch.cat(word_embeddings, dim=0), dim=0, keepdim=True)
            similarity = F.cosine_similarity(avg_path_embedding, phrase_embedding).item()

            if similarity > max_similarity:
                max_similarity = similarity
                best_path = path

        return best_path


# In a new file, e.g., `explainability.py`

import logging
import networkx as nx
import torch
import torch.nn.functional as F
from itertools import combinations
from typing import List, Dict, Any, Optional, Set

# src/multi_graph/explainability.py

import logging
import networkx as nx
from itertools import combinations
from typing import List, Optional


class SubgraphExplainer:
    """
    A tool to extract meaningful conceptual subgraphs from a GNN-annotated document graph.
    It identifies the most salient words for a sentence and finds the structural paths
    connecting them, returning the most coherent subgraph as an explanation.
    """

    def __init__(
            self,
            top_k_words: int = 4,
            max_path_length: int = 5,
            edge_type_priority: Optional[List[str]] = None
    ):
        """
        Initializes the SubgraphExplainer.

        Args:
            top_k_words (int): The number of top-attention words to use as anchors.
            max_path_length (int): The maximum number of edges in a path between any two anchor words.
            edge_type_priority (list, optional): A list of edge types to prioritize in the search.
                                                 Defaults to ['dep', 'seq'].
        """
        if edge_type_priority is None:
            edge_type_priority = ['dep', 'seq']

        self.top_k_words = top_k_words
        self.max_path_length = max_path_length
        self.edge_type_priority = edge_type_priority
        logging.info(f"SubgraphExplainer initialized with top_k={top_k_words}, max_path={max_path_length}")

    def _get_top_attention_words(self, nx_graph: nx.MultiDiGraph, sentence_node_id: str) -> List[str]:
        """Identifies the top-k most salient word nodes for a given sentence."""
        word_attentions = []
        for u, _, data in nx_graph.in_edges(sentence_node_id, data=True):
            if nx_graph.nodes[u].get('type') == 'word' and 'learned_attention' in data:
                word_attentions.append((u, abs(data['learned_attention'])))

        word_attentions.sort(key=lambda x: x[1], reverse=True)
        return [word_id for word_id, _ in word_attentions[:self.top_k_words]]

    def _find_shortest_path_between_pair(self, nx_graph: nx.MultiDiGraph, start_node: str, end_node: str) -> Optional[
        List[str]]:
        """Finds the single shortest path between two nodes, respecting priority."""
        for edge_type in self.edge_type_priority:
            try:
                view = nx.subgraph_view(nx_graph,
                                        filter_edge=lambda u, v, k: nx_graph.edges[u, v, k].get('type') == edge_type)
                path = nx.shortest_path(view, source=start_node, target=end_node)
                if len(path) - 1 <= self.max_path_length:
                    return path
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

        try:
            path = nx.shortest_path(nx_graph, source=start_node, target=end_node)
            if len(path) - 1 <= self.max_path_length:
                return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def explain_sentence(self, nx_graph: nx.MultiDiGraph, sentence_node_id: str) -> Optional[nx.MultiDiGraph]:
        """
        Main public method. Extracts the most salient conceptual subgraph for a single sentence.
        """
        if sentence_node_id not in nx_graph:
            logging.error(f"Sentence node '{sentence_node_id}' not found in the graph.")
            return None

        top_words = self._get_top_attention_words(nx_graph, sentence_node_id)
        if len(top_words) < 2:
            logging.warning("Not enough salient words found to form a connecting subgraph.")
            return None

        logging.info(f"Top anchor words for '{sentence_node_id}': {[nx_graph.nodes[n]['text'] for n in top_words]}")

        all_paths_nodes = set(top_words)
        for start_node, end_node in combinations(top_words, 2):
            path = self._find_shortest_path_between_pair(nx_graph, start_node, end_node)
            if path:
                all_paths_nodes.update(path)

        candidate_subgraph = nx_graph.subgraph(all_paths_nodes)

        connected_components = sorted(nx.connected_components(candidate_subgraph.to_undirected()), key=len,
                                      reverse=True)
        if not connected_components:
            logging.warning("Could not find any connecting paths between top words.")
            return None

        core_concept_nodes = connected_components[0]
        final_subgraph = nx_graph.subgraph(core_concept_nodes).copy()

        logging.info(
            f"Extracted conceptual subgraph with {final_subgraph.number_of_nodes()} nodes and {final_subgraph.number_of_edges()} edges.")
        return final_subgraph