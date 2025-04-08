"""
Module for handling text embeddings and graph manipulation, specifically adding relationships
between nodes in a graph using text embeddings.
"""
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np
import torch
import logging

from graphviz import Digraph

logger = logging.getLogger(__name__)


def retrieve_bert_embeddings(embeddings_tokenizer, embeddings_model, target_text: str, device: str) -> np.array:
    """
    Generate embeddings for a given input text using a provided embeddings model and tokenizer.

    Args:
        embeddings_tokenizer (Tokenizer): The tokenizer to tokenize the input text.
        embeddings_model (Model): The embeddings model to generate embeddings.
        target_text (str): The input text for which embeddings are to be generated.
        device (str): The device (e.g., 'cpu', 'cuda') on which to perform inference.

    Returns:
        np.array: The embeddings generated for the input text.
    """
    with torch.no_grad():
        # Tokenize the input text and move the token IDs to the specified device.
        tokens = embeddings_tokenizer.encode(target_text, add_special_tokens=True)
        input_ids = torch.tensor([tokens]).to(device)

        # Set the model to evaluation mode and generate embeddings.
        embeddings_model.eval()
        outputs = embeddings_model(input_ids)

        # Extract the embeddings for the [CLS] token (first token) and move them to the CPU.
        embedding = outputs[0][:, 0, :].cpu().numpy()

    # Squeeze the resulting array to remove any unnecessary dimensions.
    return embedding


def retrieve_text_embeddings(text_embedding, target_text: str, device: torch.device) -> torch.Tensor:
    """
    Retrieve text embeddings using the provided TextEmbedding object.

    Args:
        text_embedding (TextEmbedding): An instance of the TextEmbedding class for embedding retrieval.
        target_text (str): The input text for which embeddings are to be retrieved.
        device (torch.device): The device to which the resulting embeddings should be moved.

    Returns:
        torch.Tensor: The embeddings as a tensor
    """
    return text_embedding.retrieve_embeddings(target_text).to(device)


def add_new_relation_to_graph(target_graph: nx.MultiDiGraph, node_1: str, node_2: str, edge: str,
                              text_embedding, device: torch.device | str):
    """
    Add a new relation between two nodes in the graph, using text embeddings as node and edge features.

    Args:
        target_graph (nx.MultiDiGraph): The graph to which the new relation is to be added.
        node_1 (str): The first node (as text).
        node_2 (str): The second node (as text).
        edge (str): The relationship or edge label (as text).
        text_embedding (TextEmbedding): An instance of the TextEmbedding class for embedding retrieval.
        device (torch.device | str): The device to which embeddings should be moved.

    Returns:
        None
    """

    def _preprocess_labels(target: str) -> str:
        """
        Preprocess the input label by removing specific unwanted characters.

        Args:
            target (str): The label to preprocess.

        Returns:
            str: The preprocessed label with unwanted characters removed.
        """
        # Define unwanted characters in a regex-friendly way.
        remove_chars = r"[]:;,/|\\_º.'º^©°'\""

        # Remove unwanted characters and replace them with a space
        target = re.sub(f"[{re.escape(remove_chars)}]", " ", target)

        target = re.sub(r'[\u200B-\u200D\uFEFF]', '', target)

        # Normalize whitespace by stripping and replacing multiple spaces with a single space
        target = re.sub(r'\s+', ' ', target).strip()

        return target

    # Preprocess the node and edge labels.
    node_1 = _preprocess_labels(node_1)
    node_2 = _preprocess_labels(node_2)
    edge = _preprocess_labels(edge)

    # Check if the node and edge labels are valid.
    if check_extracted_info(node_1, node_2, edge):

        # Add nodes to the graph if they don't already exist.
        if not target_graph.has_node(node_1):
            target_graph.add_node(node_1)
        if not target_graph.has_node(node_2):
            target_graph.add_node(node_2)

        # Generate embeddings for the nodes and edge.
        node_1_embeddings = retrieve_text_embeddings(text_embedding=text_embedding,
                                                     target_text=node_1,
                                                     device=device)
        node_2_embeddings = retrieve_text_embeddings(text_embedding=text_embedding,
                                                     target_text=node_2,
                                                     device=device)
        # edge_embeddings = retrieve_text_embeddings(text_embedding=text_embedding,
        #                                            target_text=edge,
        #                                            device=device)

        # Add the nodes with their embeddings to the graph.
        target_graph.add_node(node_1, x=node_1_embeddings)
        target_graph.add_node(node_2, x=node_2_embeddings)

        # Add the edge with its embedding and label to the graph.
        target_graph.add_edge(node_1, node_2, label=edge)


def log_corpus_oov_statistics(unk_vocab: dict, vocab: dict) -> None:
    """
    Logs statistics about the proportion of out-of-vocabulary (UNK) tokens
    in a given corpus relative to the total vocabulary.

    Parameters:
    - unk_vocab (dict): Dictionary containing unknown words and their counts.
    - vocab (dict): Dictionary containing all known words and their counts.
    """

    # Avoid division by zero by setting a fallback value of 1 when vocab is empty
    total_vocab_words = len(vocab) if vocab else 1
    total_vocab_counts = sum(vocab.values()) if vocab else 1

    #  Calculate proportions:
    # - `proportion_words`: Proportion of UNK words compared to the total vocabulary
    # - `proportion_corpus`: Proportion of UNK word counts compared to the total corpus size
    proportion_words = len(unk_vocab) / total_vocab_words
    proportion_corpus = sum(unk_vocab.values()) / total_vocab_counts

    # Structure the log data with detailed metrics and vocabulary statistics
    log_data = {
        "timestamp": datetime.now().isoformat(),  # Record the current timestamp for traceability
        "metrics": {
            "proportion_unk_words": round(proportion_words, 5),  # Proportion of UNK words
            "proportion_unk_corpus": round(proportion_corpus, 5)  # Proportion of UNK counts in the corpus
        },
        "vocab_stats": {
            "total_vocab_size": total_vocab_words,  # Total unique words in the vocabulary
            "total_corpus_size": total_vocab_counts,  # Total word occurrences in the corpus
            "unk_vocab_size": len(unk_vocab),  # Unique UNK words
            "unk_corpus_size": sum(unk_vocab.values())  # Total occurrences of UNK words
        }
    }

    # Log the data with structured formatting
    logger.info("---- UNK Stats ----")
    logger.info(json.dumps(log_data, indent=4))

    # Provide console feedback to indicate successful logging
    logger.info("Metrics logged successfully!")


def check_extracted_info(node_1: str, node_2: str, edge: str) -> bool:
    """
    Check if the extracted information from nodes and edge is valid.

    Args:
        node_1 (str): The first node label.
        node_2 (str): The second node label.
        edge (str): The edge label.

    Returns:
        bool: True if all information is valid (not None and not empty after stripping),
              False otherwise.
    """
    return all(x and x.strip() for x in [node_1, node_2, edge])


def plot_networkx_graph(
        graph: nx.DiGraph,
        dataset_name: str,
        doc_id: str,
        label: str,
        output_folder: str | Path
) -> None:
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    dot = Digraph(format='png')  # Use PNG for faster output
    dot.attr(rankdir='TB', fontsize='10', fontname='Arial', splines='false')
    dot.graph_attr['dpi'] = '100'  # Lower DPI for faster rendering

    chart_title = f"Dataset {dataset_name} - Doc {doc_id} - Label {label}"
    dot.attr(label=chart_title, labelloc='t', fontsize='16', fontcolor='black')

    for node, data in graph.nodes(data=True):
        node_label = data.get('label', str(node))
        dot.node(str(node), node_label, shape='box', color='lightgrey')  # Simplified node style

    for source, target, edge_data in graph.edges(data=True):
        dot.edge(str(source), str(target))  # Removed edge labels and colors for speed

    output_filename = f"Dataset_{dataset_name}_Doc_{doc_id}"
    dot.render(filename=output_filename, directory=str(output_folder), cleanup=True, view=False)

    logger.info(f"Graph saved to {output_folder}/{output_filename}.png")