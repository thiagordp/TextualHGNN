# src/multi_graph/dataset.py

import os
import os.path as osp
from datetime import datetime
from pathlib import Path
from typing import Union, List, Tuple

import torch
import logging

from torch_geometric.data import OnDiskDataset
from tqdm import tqdm
import random

# We need the builder class definition
from graph_builder import DocumentGraphBuilder
from src.multi_graph.data_preprocessing import preprocess_text
from src.multi_graph.visualization import visualize_structural_graph


class MultiLevelGraphOnDiskDataset(OnDiskDataset):

    def __init__(self, root: str, split: str, builder: DocumentGraphBuilder, transform=None):
        """
        Custom OnDiskDataset for our multi-level heterogeneous graphs.

        Args:
            root (str): The root directory where the dataset (raw/ and processed/) is stored.
            split (str): The dataset split, e.g., "train" or "test".
            builder (DocumentGraphBuilder): An initialized instance of our graph builder.
            transform (callable, optional): A function/transform. Defaults to None.
        """

        self.class_map = None
        self.split = split
        self.builder = builder

        super().__init__(root, transform)

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.split, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.split, 'processed')

    @property
    def nx_graph_dir(self) -> str:
        return osp.join(self.root, self.split, 'processed_nx')

    @property
    def raw_file_names(self):
        try:
            return os.listdir(self.raw_dir)
        except FileNotFoundError:
            return []

    @property
    def processed_file_names(self):
        # A simple way to check if processing is done is to look for a marker file.
        if osp.exists(osp.join(self.processed_dir, 'processing_done.marker')):
            # If done, the "processed files" are all the .pt files.
            return [f for f in os.listdir(self.processed_dir) if f.endswith('.pt')]
        return []  # Otherwise, processing needs to happen.

    def process(self):
        """
        This method now handles the full pipeline:
        1. Loads raw text files.
        2. Pre-processes (cleans) the text.
        3. Runs the DocumentGraphBuilder.
        4. Saves each final HeteroData object to disk.
        """

        logging.info(f"Processed data not found. Starting graph processing for '{self.split}' split.")

        logging.info(f"'{self.processed_dir}' not found. Starting full data processing for '{self.split}' split...")
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.nx_graph_dir, exist_ok=True)

        # 1. Discover class folders and load raw documents
        raw_docs_to_process = []
        class_names = sorted([d.name for d in os.scandir(self.raw_dir) if d.is_dir()])
        self.class_map = {name: i for i, name in enumerate(class_names)}
        torch.save(self.class_map, osp.join(self.root, self.split, 'processed',  'class_map.pt'))

        for class_name in self.class_map.keys():
            class_path = osp.join(self.raw_dir, class_name)
            for filename in os.listdir(class_path):
                if filename.endswith(".txt"):
                    file_path = osp.join(class_path, filename);
                    with open(file_path, 'r', encoding='utf-8') as f:
                        raw_docs_to_process.append((filename, f.read(), class_name))

        if not raw_docs_to_process:
            raise FileNotFoundError(f"No raw .txt files found in {self.raw_dir}")

        # Shuffle before processing to make sampling (if any) random
        random.shuffle(raw_docs_to_process)
        raw_docs_to_process = raw_docs_to_process[:200]

        logging.info(f"Cleaning and preprocessing {len(raw_docs_to_process)} raw text documents.")

        cleaned_docs = [(fname, preprocess_text(text), cname)
                        for fname, text, cname in
                        tqdm(raw_docs_to_process, desc="Preprocessing text")]

        # 3. Use the builder to convert cleaned documents to graphs
        # We only need the (filename, text) tuples for the builder.
        nx_graphs, hetero_graphs, valid_filenames = self.builder.process_documents(
            [(filename, text) for filename, text, _ in cleaned_docs],
        )

        viz_output_dir = Path(self.root) / "visualizations" / "structural" / self.split
        viz_output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Structural visualizations will be saved to '{viz_output_dir}'")

        filename_to_class = {fname: cname for fname, _, cname in cleaned_docs}
        for nx_graph, filename in tqdm(zip(nx_graphs, valid_filenames), total=len(valid_filenames),
                                       desc="Generating structural visualizations"):
            class_name = filename_to_class.get(filename, "unknown")
            output_html_name = f"{class_name.capitalize()}_{Path(filename).stem}_structural.html"
            output_path = viz_output_dir / output_html_name

            visualize_structural_graph(
                nx_graph,
                self.builder.dep_label_map,
                str(output_path)
            )

        # Create a map to get the label for each valid filename
        filename_to_label = {fname: self.class_map[cname] for fname, _, cname in cleaned_docs}

        # 4. Save both the HeteroData object and the NetworkX object
        logging.info(f"Saving {len(hetero_graphs)} processed graphs to '{self.processed_dir}'.")
        for i, h_graph in enumerate(tqdm(hetero_graphs, desc=f"Saving '{self.split}' graphs")):
            filename = valid_filenames[i]
            label = filename_to_label[filename]

            h_graph['document'].y = torch.tensor(label)
            h_graph['document'].doc_id = filename

            # Save the HeteroData object for the GNN
            torch.save(h_graph, osp.join(self.processed_dir, f'data_{i}.pt'))
            # Save the NetworkX object for visualization
            torch.save(nx_graphs[i], osp.join(self.nx_graph_dir, f'{filename}.nx'))

        # Create a marker file to indicate processing is complete
        with open(osp.join(self.processed_dir, 'processing_done.marker'), 'w', encoding='utf-8') as f:
            f.write(f"{datetime.now().isoformat()}")

    def len(self):
        # The length of the dataset is the number of saved .pt files
        return len(self.processed_file_names)

    def get(self, idx: int):
        """Loads the pre-processed graph object at a given index."""
        data = torch.load(osp.join(self.processed_dir, f'data_{idx:07d}.pt'))
        return data
