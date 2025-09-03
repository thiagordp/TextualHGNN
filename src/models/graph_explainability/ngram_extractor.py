import os
import random
from collections import deque

import networkx as nx

from itertools import islice
import networkx as nx
from typing import List, Set, Tuple

import torch
from tqdm import tqdm

from src.models.graph_explainability.embeddings_oracle import EmbeddingOracle

import networkx as nx
from typing import List, Set, Tuple
from itertools import islice


class NGramExtractor:
    def __init__(self, embedding_oracle, include_edge_labels: bool = False):
        """
        Initialize the NGramExtractor with the given EmbeddingOracle.

        :param embedding_oracle: An instance of the EmbeddingOracle class.
        :param include_edge_labels: Whether to include edge labels in the n-grams.
        """
        self.embedding_oracle = embedding_oracle
        self.include_edge_labels = include_edge_labels

    def extract_ngrams_from_multidigraph(self, graph: nx.MultiDiGraph, n: int = 3) -> List[str]:
        """
        Extract n-grams (unigrams, bigrams, trigrams) from a MultiDiGraph.
        This method performs both linear and graph-based searches to find all possible n-grams.

        :param graph: A directed NetworkX MultiDiGraph with text nodes.
        :param n: Maximum n-gram size (default is 3 for up to trigrams).
        :return: A list of n-grams extracted from the graph.
        """
        ngrams = set()

        nodes = list(graph.nodes)

        for start_node in graph.nodes:
            # BFS and DFS search to generate n-grams for all paths
            # ngrams.update(self.extract_immediate_ngrams(graph, start_node, n, strategy="bfs"))
            ngrams.update(self.extract_immediate_ngrams(graph=graph, start_node=start_node, n=n))
            # ngrams.update(self.search_and_generate_ngrams(graph, start_node, n, strategy="dfs"))

        return list(ngrams)

    def search_and_generate_ngrams(self, graph: nx.MultiDiGraph, start_node: str, n: int, strategy: str = "bfs") -> Set[
        str]:
        """
        Explore all possible paths from the start_node using BFS or DFS and generate n-grams.

        :param graph: The input MultiDiGraph.
        :param start_node: The node to start the search from.
        :param n: Maximum n-gram size.
        :param strategy: 'bfs' for Breadth-First Search or 'dfs' for Depth-First Search.
        :return: A set of n-grams discovered through graph traversal.
        """
        ngrams = set()
        queue = [(start_node, [])]  # (current_node, path)

        while queue:
            current_node, path = queue.pop(0) if strategy == "bfs" else queue.pop()
            path = path + [current_node]

            # Generate n-grams from the current path
            ngrams.update(self.generate_ngrams(path, n))

            if len(path) < n:
                for _, neighbor, edge_data in graph.out_edges(current_node, data=True):
                    edge_label = edge_data.get('label', '')

                    if self.include_edge_labels and edge_label:
                        next_path = path + [edge_label, neighbor]
                    else:
                        next_path = path + [neighbor]

                    queue.append((neighbor, path))

        return ngrams

    def extract_immediate_ngrams(self, graph: nx.MultiDiGraph, start_node: str, n: int = 3) -> Set[str]:
        """
        Extract the most immediate set of n-grams from a graph using BFS, avoiding cycles.

        :param graph: A MultiDiGraph where nodes are words.
        :param start_node: The word to generate immediate n-grams from.
        :param n: Maximum n-gram size (e.g., 3 for trigrams).
        :return: A set of n-grams starting with the start_node, avoiding cycles.
        """
        ngrams = set()

        start_node = start_node

        queue = deque([(start_node, [start_node])])  # (current_node, path)

        while queue:
            current_node, path = queue.popleft()

            # Generate valid n-grams from the current path
            if 1 <= len(path) <= n:
                ngrams.add(' '.join(path))

            # Avoid extending paths beyond the desired n-gram length
            if len(path) >= n:
                continue

            # Explore immediate neighbors, avoiding cycles
            for _, neighbor, _ in graph.out_edges(current_node, data=True):

                if neighbor not in path:  # Prevent cycles by avoiding repeated words
                    queue.append((neighbor, path + [neighbor]))

        return ngrams

    def generate_ngrams(self, nodes: List[str], n: int) -> Set[str]:
        """
        Generate n-grams (up to length n) from a list of nodes.

        :param nodes: A list of nodes or tokens.
        :param n: Maximum n-gram size.
        :return: A set of n-grams.
        """
        ngrams = set()
        for size in range(1, n + 1):
            ngrams.update([' '.join(nodes[i:i + size]) for i in range(len(nodes) - size + 1)])
        return ngrams

    def process_graph_and_store_ngrams(self, graph: nx.MultiDiGraph, n: int = 3, batch_size: int = 10000,
                                       doc_id: str = ""):
        """
        Extract n-grams from the MultiDiGraph and store them in the EmbeddingOracle.

        :param graph: The input NetworkX MultiDiGraph.
        :param n: Maximum n-gram size (default is 3).
        :param batch_size: Batch size for bulk insertion into the EmbeddingOracle.
        :param doc_id: Optional document ID for tracking n-grams related to specific documents.
        """
        # print(f"Processing graph for document ID: {doc_id}...")

        # Extract n-grams using the improved graph traversal method
        ngrams = self.extract_ngrams_from_multidigraph(graph, n)

        if doc_id:
            # Prefix n-grams with doc_id for contextual embedding (Optional)
            ngrams = [f"{doc_id}:{ngram}" for ngram in ngrams]

        # print(f"Extracted {len(ngrams)} n-grams from graph (Doc ID: {doc_id}). Storing in EmbeddingOracle...")

        ngrams = [
            " ".join(word.split("::")[0] for word in text.split())
            for text in ngrams
        ]

        try:
            # Store n-grams in the EmbeddingOracle using bulk insertion
            self.embedding_oracle.add_terms_bulk(ngrams, batch_size=batch_size)
            # (f"Successfully stored n-grams for document ID: {doc_id}.")
        except Exception as e:
            print(f"Error storing n-grams for document ID: {doc_id}: {e}")


if __name__ == "__main__":

    NGRAM = 3
    BATCH_SIZE = 100000

    language = "portuguese_voto"
    DATASET = "STF_HC_Voto_Relatorio"
    log_file = f"logs/embeddings_oracle_{DATASET}.log"

    # Initialize EmbeddingOracle (from your code)
    model_path = 'data/external/embeddings/glove_legal_100.bin'
    db_path = f'data/oracle/embeddings_{DATASET}.db'
    data_folder = f"data/datasets/{DATASET}/train/interim/"

    embedding_oracle = EmbeddingOracle(model_path=model_path, db_path=db_path)

    # Initialize the NGramExtractor
    ngram_extractor = NGramExtractor(embedding_oracle)

    # Collect all .pt files in the provided data folder
    pt_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.pt')]

    random.shuffle(pt_files)

    print(f"Found {len(pt_files)} .pt files in {data_folder}. Starting processing...")

    # Iterate over each .pt file
    for pt_file in tqdm(pt_files[:10000], desc="Processing .pt files"):
        try:
            # Load the document tuple (doc_id, label, graph_nx)
            doc_id, label, graph_nx = torch.load(pt_file)

            # Check if the loaded graph is a valid NetworkX MultiDiGraph
            if not isinstance(graph_nx, nx.MultiDiGraph):
                print(f"Skipping file {pt_file}: Not a valid MultiDiGraph.")
                continue

            # Extract and store n-grams using the NGramExtractor
            ngram_extractor.process_graph_and_store_ngrams(graph_nx, n=NGRAM, batch_size=BATCH_SIZE)

        except Exception as e:
            print(f"Error processing {pt_file}: {e}")
