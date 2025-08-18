"""
Python class for the TextGraph dataset based on PyTorch Geometric's OnDiskDataset class.

This class provides a convenient way to handle large-scale text-based graph datasets
stored on disk. It includes methods for initializing the dataset, defining the structure
of raw and processed directories, and retrieving various dataset properties.

Attributes:
    root (str): The root directory where the dataset is stored.
    transform (callable, optional): A function/transform that takes in an object and returns a transformed version.
    pre_filter (callable, optional): A function that takes in a Data object and returns a boolean value,
                                     indicating whether the data object should be included in the final dataset.
    backend (str): The backend storage system to use (default is "sqlite").

Methods:
    raw_file_names(): Lists the raw file names that are expected to be found in the raw directory.
    raw_dir() -> str: Returns the path to the raw data directory.
    processed_dir() -> str: Returns the path to the processed data directory.
    __repr__() -> str: Returns a string representation of the dataset, including its class name and length.
    get_class_name() -> str: Retrieves the class name of the dataset.
    num_node_labels() -> int: Returns the number of node labels in the dataset.
    num_node_attributes() -> int: Returns the number of node attributes in the dataset.
    num_edge_labels() -> int: Returns the number of edge labels in the dataset.
    num_edge_attributes() -> int: Returns the number of edge attributes in the dataset.

Example usage:
    dataset = TextGraphDatasetOnDisk(root='/path/to/dataset')
    print(dataset)
    print(f"Class Name: {dataset.get_class_name()}")
"""
import glob
import os
import os.path as osp
import pickle
import random
import shutil
from pathlib import Path
from typing import Callable, Optional, List, Dict, Any

import torch
from networkx.classes import MultiDiGraph
from networkx.classes.reportviews import DiMultiDegreeView, OutMultiEdgeView
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import OnDiskDataset, Data
from torch_geometric.data.data import BaseData
from torch_geometric.utils import from_networkx
from tqdm import tqdm

from src.data.text_graph_dataset_parsers import Text2DP, Text2GraphDataset
from networkx.classes.coreviews import MultiAdjacencyView
torch.serialization.add_safe_globals([MultiDiGraph, DiMultiDegreeView, MultiAdjacencyView, OutMultiEdgeView])

import logging
import time

logger = logging.getLogger(__name__)


class TextGraphDatasetOnDisk(OnDiskDataset):
    split_mapping = {
        'train': 'train',
        'validation': 'valid',
        'test': 'test-dev',
        'holdout': 'test-challenge',
    }

    def __init__(
            self,
            root: str,
            split: str = "train",
            transform: Optional[Callable] = None,
            pre_filter: Optional[Callable] = None,
            backend: str = 'sqlite',
            node_feature_size: int = 768,
            max_num_nodes: int=1100,
            batch_size: int = 16,
            lang="english"
    ):
        """
        Initializes the TextGraphDatasetOnDisk.

        Args:
            root (str): The root directory where the dataset is stored.
            transform (callable, optional): A function/transform that takes in an object and returns a transformed version.
            pre_filter (callable, optional): A function that takes in a Data object and returns a boolean value,
                                             indicating whether the data object should be included in the final dataset.
            backend (str): The backend storage system to use (default is "sqlite").
        """
        self.encoded_labels = None
        self.label_encoder = None
        self.labels = None
        self.proc_numbers = None
        self.max_num_nodes = max_num_nodes
        assert split in ['train', 'validation', 'test', 'holdout']

        schema = {
            'doc_id': int,
            'x': dict(dtype=torch.float, size=(-1, node_feature_size)),
            # 'tokens': List[str],
            'edge_index': dict(dtype=torch.int64, size=(2, -1)),
            'y': int,
        }

        self.text2graph_parser = None
        self.root = root
        self.split = split
        self.transform = transform
        self.pre_filter = pre_filter
        self.backend = backend
        self.batch_size = batch_size
        self.node_feature_size = node_feature_size
        self.lang = lang

        metadata_path = Path(self.processed_dir) / 'metadata.pkl'
        if metadata_path.exists():
            self.load_metadata_and_encoder(metadata_path)

        self.doc_ids = set()

        super().__init__(
            root=root,
            transform=transform,
            pre_filter=pre_filter,
            backend=backend,
            schema=schema
        )

    @property
    def data_paths(self):
        pattern = osp.join(self.interim_dir, "*.pt")
        available_processed_files = glob.glob(pattern)

        return available_processed_files

    # def len(self):
    #    return len(self.data_paths)

    @property
    def raw_dir(self) -> str:
        """
        Returns the path to the raw data directory.

        Returns:
            str: The path to the raw data directory.
        """
        return osp.join(self.root, self.split, 'raw')

    @property
    def interim_dir(self) -> str:
        """
        Returns the path to the interim data directory.

        Returns:
            str: The path to the interim data directory.
        """
        return osp.join(self.root, self.split, 'interim')

    @property
    def processed_dir(self) -> str:
        """
        Returns the path to the processed data directory.

        Returns:
            str: The path to the processed data directory.
        """
        return osp.join(self.root, self.split, 'processed')

    @property
    def get_class_name(self):
        """
        Retrieves the class name of the dataset.

        Returns:
            str: The class name.
        """
        return self.__class__.__name__

    @property
    def num_classes(self) -> int:
        return len(self.label_encoder.classes_)

    @property
    def num_node_attributes(self) -> int:
        """
        Returns the number of node attributes in the dataset.

        Returns:
            int: The number of node attributes (default is -1, meaning undefined).
        """
        return self.node_feature_size

    def process(self) -> None:
        """
         Processes the raw data files and converts them into a list of `Data` objects.
         Called automatically by PyG when processed files are not available
        """
        print("Split: ", self.split)
        self.text2graph_parser = Text2DP(lang=self.lang)

        input_dir = osp.join(self.raw_dir)
        output_dir = osp.join(self.interim_dir)

        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        print("Text->Graph Parsing")
        self.process_text2graph(
            input_path=input_dir,
            output_path=output_dir
        )

        input_dir = osp.join(self.interim_dir)
        output_dir = osp.join(self.processed_dir)

        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        print("Graph->Dataset Parsing")
        self.process_graph2dataset(
            input_path=input_dir,
            output_path=output_dir,
            plot_graph=True
        )

    def process_text2graph(self, input_path=None, output_path=None, batch_size=1000):
        logger.info(f"Starting Text->Graph processing for {self.split} split...")
        start_time = time.time()

        logger.info(f"Input Path: {input_path}, Output Path: {output_path}")
        logger.info(f"Batch Size: {batch_size}")

        text_to_graph_dp = Text2GraphDataset(
            text_to_graph_parser=self.text2graph_parser,
            path_to_corpus=Path(input_path),
            path_to_output=Path(output_path),
        )

        text_to_graph_dp.load_corpus()
        total_samples = len(text_to_graph_dp.corpus)
        logger.info(f"Total samples to process: {total_samples}")

        # Process in batches of 5000
        for batch_index, start_idx in enumerate(range(0, total_samples, batch_size)):
            end_idx = min(start_idx + batch_size, total_samples)
            logger.info(f"Processing batch {batch_index} from {start_idx} to {end_idx}...")

            batch_corpus = text_to_graph_dp.corpus[start_idx:end_idx]
            parsed_graphs = text_to_graph_dp.parse(batch_corpus)

            text_to_graph_dp.build_pyg_dataset_individual(
                parsed_graphs=parsed_graphs,
                output_path=output_path,
                batch_index=batch_index  # Pass the batch index for unique file naming
            )

        logger.info(f"Text->Graph processing completed in {time.time() - start_time:.2f}s")

    def load_metadata_and_encoder(self, target_path):

        with open(target_path, "rb") as f:
            metadata = pickle.load(f)

        self.proc_numbers = metadata["proc_numbers"]
        self.labels = metadata["labels"]
        self.label_encoder = LabelEncoder()
        self.encoded_labels = [label for label in self.label_encoder.fit_transform(self.labels)]

    def process_graph2dataset(self, input_path=None, output_path=None, plot_graph: bool = False):
        data_list: List[Data] = []
        iterator = enumerate(self.data_paths)
        num_docs = len(self.data_paths)

        metadata_path = Path(input_path) / 'metadata.pkl'
        self.load_metadata_and_encoder(metadata_path)

        shutil.copy(metadata_path, output_path)

        graphs_path = Path(input_path).parent / "graphs"
        os.makedirs(graphs_path, exist_ok=True)

        for i, data_path in tqdm(iterator, total=num_docs):

            try:
                doc_id, y, graph_nx = torch.load(data_path, weights_only=True)
            except ValueError as e:
                print(f"Error while reading doc in {data_path}. Exception: {e}")
                continue


            data = from_networkx(graph_nx).cpu()

            if data.num_nodes > self.max_num_nodes:
                logging.warning(f"Ignoring doc {doc_id} due to max number of nodes (> {self.max_num_nodes}")
                continue

            doc_id = int(doc_id.replace(".txt", ""))
            if doc_id in self.doc_ids:
                print(f"Ops! Document {doc_id} already processed.")
                continue
            self.doc_ids.add(doc_id)

            y_encoded = self.label_encoder.transform([y])

            data.y = torch.tensor(y_encoded, dtype=torch.int64).cpu()
            data.doc_id = torch.tensor(doc_id, dtype=torch.int64).cpu()
            # data.tokens = list(graph_nx.nodes(data=False))

            # if plot_graph and random.random() < 0.05: # Eventually, draw some graphs.
            #     logging.info("Plotting graph")
            #     plot_networkx_graph(
            #         graph=graph_nx,
            #         dataset_name=str(input_path).split("/")[2],
            #         doc_id=str(doc_id),
            #         label=y,
            #         output_folder=graphs_path
            #     )

            data_list.append(data)

            if (i + 1) % self.batch_size == 0 or (i + 1) == num_docs:
                self.extend(data_list)
                data_list = []

    def run_text2graph(self, path_to_corpus, output_path=None, return_graph=True):
        """
        Converts a text corpus to a graph dataset using the specified text-to-graph parser.

        Args:
           path_to_corpus (str): Path to the input text corpus.
           output_path (str, optional): Path to save the output graph dataset. If not specified, defaults to None.
           return_graph (bool, optional): If True, returns the parsed graph dataset. Defaults to True.

        Returns:
           Optional[PyGDataset]: The parsed PyG (PyTorch Geometric) graph dataset if return_graph is True.

        This method performs the following steps:
           1. Initializes a Text2GraphDataset instance with the specified parser and corpus path.
           2. Loads the corpus from the specified path.
           3. Parses the loaded corpus into graph representations.
           4. Builds and optionally saves a PyG (PyTorch Geometric) dataset from the parsed graphs.

        Note:
           The `text2graph_parser` attribute of the class instance should be set to a valid text-to-graph parser before calling this method.
        """
        text_to_graph_dp = Text2GraphDataset(
            text_to_graph_parser=self.text2graph_parser,
            path_to_corpus=path_to_corpus,
            path_to_output=output_path,
        )
        print("Loading corpus")
        text_to_graph_dp.load_corpus()

        print("Starting parsing")
        parsed_graphs = text_to_graph_dp.parse()

        print("Creating PyG Dataset")
        text_to_graph_dp.build_pyg_dataset_individual(
            parsed_graphs=parsed_graphs,
            output_path="dp_jec_2020_pyg"
        )

    def get(self, idx: int) -> BaseData:
        """
        Gets the data object at index `idx`.

        Args:
            idx (int): The index of the data object to retrieve.

        Returns:
            BaseData: The data object at the specified index.
        """
        return self.deserialize(self.db.get(idx))

    def serialize(self, data: BaseData) -> Dict[str, Any]:
        """
        Serializes a `Data` object to a dictionary.

        Args:
            data (BaseData): The `Data` object to serialize.

        Returns:
            Dict[str, Any]: The serialized data.
        """
        assert isinstance(data, Data)
        return dict(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            # tokens=data.tokens,
            y=data.y,
            doc_id=data.doc_id,
        )

    def deserialize(self, data: Dict[str, Any]) -> Data:
        """
        Deserializes a dictionary to a `Data` object.

        Args:
            data (Dict[str, Any]): The dictionary to deserialize.

        Returns:
            Data: The deserialized `Data` object.
        """
        return Data.from_dict(data)

    def __repr__(self) -> str:
        """
        Returns a string representation of the dataset, including its class name and length.

        Returns:
            str: A string representation of the dataset.
        """
        return (f'{self.get_class_name}(num_graphs={self.len()}, num_node_features={self.num_node_attributes}, '
                f'num_classes={self.num_classes})')
