import json
import logging
import time

import networkx as nx
from networkx.drawing.nx_agraph import to_agraph
import tempfile
import os
import webbrowser


def format_time_elapsed(start_time: float) -> str:
    """
    Format elapsed time into 'HH:MM:SS.mmm' format.

    Args:
        start_time (float): The start time as returned by `time.time()`.

    Returns:
        str: The formatted time elapsed.
    """
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours):02}:{int(minutes):02}:{seconds:06.3f}"

def load_config(lang: str, config_path: str = "configs.json") -> dict:
    with open(config_path, 'r') as f:
        all_configs = json.load(f)

    config = all_configs.get(lang.lower())
    if config is None:
        raise ValueError(f"No configuration found for language '{lang}'")
    return config

def setup_logging(log_folder='logs', log_file='diffpool_training.log'):
    os.makedirs(log_folder, exist_ok=True)
    log_path = os.path.join(log_folder, log_file)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Setting up logging in {log_path}")

def plot_multidigraph_to_pdf(G: nx.MultiDiGraph, output_path: str = None, open_pdf: bool = True) -> str:
    """
    Plots a NetworkX MultiDiGraph using GraphViz, saves it as a PDF, and optionally opens it.

    Parameters:
    -----------
    G : nx.MultiDiGraph
        The directed multigraph to plot.
    output_path : str, optional
        Path to save the PDF. If not provided, a temporary file is created.
    open_pdf : bool
        Whether to open the PDF after saving.

    Returns:
    --------
    str
        The path to the saved PDF file.
    """
    if not isinstance(G, nx.MultiDiGraph):
        raise TypeError("Input graph must be a networkx.MultiDiGraph.")

    # Convert the graph to AGraph (Graphviz representation)
    A = to_agraph(G)
    A.graph_attr.update(dpi="300")  # Optional: adjust rendering resolution

    # Create output path if not provided
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)

    # Layout and render
    A.layout(prog="dot")  # You can use 'dot', 'neato', 'fdp', etc.
    A.draw(output_path)

    if open_pdf:
        webbrowser.open(f"file://{os.path.abspath(output_path)}")

    return output_path
