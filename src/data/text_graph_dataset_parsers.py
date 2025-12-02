"""
Classes for Text to Graph parsing

@date March 26th, 2024
"""
import json
import logging
import os
import pickle
import random
import re
import time
import math
from collections import defaultdict
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Callable, Optional, Tuple, Any, Hashable, Set, Dict

import gensim
import networkx as nx
import numpy as np
import pygraphviz as pgv
import spacy
import torch
import torch_geometric
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from networkx import MultiDiGraph
from sklearn.preprocessing import LabelEncoder
from spacy import Language
from spacy.tokens import Doc, Token
from torch_geometric.data import Dataset, Data
from torch_geometric.deprecation import deprecated
from torch_geometric.transforms import ToDense
from torch_geometric.utils import from_networkx
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
from transformers.models.bert.modeling_bert import BertEmbeddings

from src.data.preprocessing import preprocessing_legal_pt
from src.data.utils import add_new_relation_to_graph
import nltk
from nltk.tokenize import WordPunctTokenizer

from collections import Counter

from src.multi_graph.visualization import visualize_structural_graph

load_dotenv()

MAX_CORPUS_SIZE = int(os.environ.get("MAX_CORPUS_SIZE")) if os.environ.get("MAX_CORPUS_SIZE") else 5

nltk.download('punkt')


class Text2Graph(ABC):
    """
    An interface for parsing text data into a graph representation.
    """

    @property
    @abstractmethod
    def text_embedding(self):
        """Should return the TextEmbedding object."""
        pass

    @property
    @abstractmethod
    def nlp(self):
        """Should return the spaCy NLP object."""
        pass

    @abstractmethod
    def parse_corpus(self, corpus: List[str], preprocessing_fn: Callable = None) -> List[nx.MultiDiGraph]:
        """
        Parse the input text corpus into a graph representation.

        Parameters:
        - corpus (list of str): The text corpus to parse.
        - preprocessing_fn (function, optional): A function for preprocessing the text before parsing.
                                                 Default is None.

        Returns:
        - dp_graphs (List[nx.MultiDiGraph]): The graph representations of the parsed text.
        """
        pass

    @abstractmethod
    def parse_document(self, text: str, preprocessing_fn: Callable = None) -> nx.MultiDiGraph:
        """
        Parse an individual document into a graph representation.

        Parameters:
        - text (str): The text document to parse.
        - preprocessing_fn (function, optional): A function for preprocessing the text before parsing.
                                                 Default is None.

        Returns:
        - graph (nx.MultiDiGraph): The graph representation of the parsed document.
        """
        pass


class GloveEmbedding:
    """
    A class to handle GloVe embeddings, providing functionality to load embeddings and
    convert text to an embedding matrix.
    """

    def __init__(self, glove_file: str):
        """
        Initialize the GloveEmbedding class with the path to a GloVe file.

        Args:
            glove_file (str): The file path to the GloVe embeddings.
        """
        self.unk_embedding = None  # Embedding for unknown words
        self.glove_dim = None  # Dimensionality of GloVe embeddings
        self.glove_model = None  # Gensim KeyedVectors model to store GloVe embeddings
        self.glove_file_path = glove_file  # Path to the GloVe file
        self.known_vocab = {}
        self.unk_vocab = {}
        self.unk_token = "<unk>"

        self.load_embeddings()

    def load_embeddings(self) -> gensim.models.KeyedVectors:
        """
        Load GloVe embeddings using Gensim and set the relevant attributes.

        Returns:
            gensim.models.KeyedVectors: The loaded GloVe embeddings.
        """
        # Load GloVe embeddings in Word2Vec format using Gensim
        # self.glove_model = gensim.models.KeyedVectors.load_word2vec_format(self.glove_file_path, binary=False,
        #                                                                    no_header=False)
        self.glove_model = gensim.models.KeyedVectors.load(self.glove_file_path, mmap='r')

        # Set the dimensionality of the embeddings
        self.glove_dim = self.glove_model.vector_size

        # Create an embedding for unknown words based on the mean of all embeddings.

        if self.unk_token in self.glove_model.key_to_index:
            unk_vector = torch.tensor(self.glove_model[self.unk_token])
        else:
            all_vectors = {word: torch.tensor(self.glove_model[word]) for word in self.glove_model.key_to_index[:10000]}
            unk_vector = torch.mean(torch.stack(list(all_vectors.values())), dim=0) if all_vectors else torch.zeros(
                self.glove_dim)
        self.unk_embedding = unk_vector

        # TODO: Add all word embeddings to Embeddings DB.
        return self.glove_model

    def text_to_embedding(self, text: str, word_embeddings: bool = True) -> torch.Tensor:
        """
        Convert a text string into a matrix of GloVe embeddings.

        Args:
            text (str): The input text to be converted into embeddings.
            word_embeddings (bool, optional): If True, returns the mean of the word embeddings.
                                              If False, returns a matrix of word embeddings.
                                              Defaults to True.

        Returns:
            torch.Tensor: A matrix of shape (number_of_words, glove_dim), where each row
                          corresponds to the GloVe embedding of a word in the text.
                          If word_embeddings is True, returns a tensor of shape (glove_dim,).
        """
        # Tokenize the text into words using the WordPunctTokenizer
        words = text.lower().strip().split()

        # Initialize an empty list to store the embedding vectors
        embedding_matrix = []

        # Iterate over each word in the tokenized text
        for word in words:
            # Check if the word is in the GloVe model
            if word in self.glove_model:
                # Retrieve the GloVe embedding for the word
                embedding_vector = torch.tensor(self.glove_model[word])

                if word in self.known_vocab:
                    # Increment the count of known words
                    self.known_vocab[word] += 1
                else:
                    # Add the word to the known vocab and set its count to 1
                    self.known_vocab[word] = 1
            else:
                # Use the random embedding for out-of-vocabulary words
                embedding_vector = self.unk_embedding.clone().detach()

                if word in self.unk_vocab:
                    # Increment the count of unknown words
                    self.unk_vocab[word] += 1
                else:
                    # Add the word to the unknown vocab and set its count to 1
                    self.unk_vocab[word] = 1

            # Append the embedding vector to the list
            embedding_matrix.append(embedding_vector)

        if not embedding_matrix:
            # Return a zero vector of the correct dimension if text is empty
            return torch.zeros(self.glove_dim, dtype=torch.float)

        # If word_embeddings is True, return the mean of the word embeddings
        if word_embeddings:
            # Stack the list of embedding vectors into a tensor and calculate the mean
            return torch.mean(torch.stack(embedding_matrix, dim=0), dim=0)

        # If word_embeddings is False, return the matrix of word embeddings
        # Stack the list of embedding vectors into a tensor
        return torch.stack(embedding_matrix)

    def retrieve_embedding_meaning(self, target_embedding: torch.tensor):
        # TODO: implement retrieve_embedding_meaning ()
        pass

    @property
    def retrieve_unk_vocab(self):
        return self.unk_vocab

    @property
    def retrieve_known_vocab(self):
        return self.known_vocab


class TextEmbedding:
    """
    A class that provides a unified interface for retrieving embeddings from either GloVe or BERT.
    """

    def __init__(self, model_name: str = "glove", file_path: str = None):
        """
        Initialize the TextEmbedding class with a specified embedding model.

        Args:
            model_name (str): The name of the embedding model to use ('glove' or 'bert').
            file_path (str): The path to the GloVe or BERT configuration file.
        """

        # Validate the model name
        if model_name not in ("glove", "bert"):
            raise Exception("Embedding model must be either 'glove' or 'bert'")

        # Initialize the appropriate embedding model
        if model_name == "glove":
            self.model = GloveEmbedding(file_path)
        else:
            self.model = None

    def retrieve_embeddings(self, text: str) -> torch.Tensor:
        """
        Retrieve embeddings for the given text using the specified model.

        Args:
            text (str): The input text for which embeddings are to be retrieved.

        Returns:
            torch.Tensor | np.array: The embeddings for the input text.
        """
        return self.model.text_to_embedding(text)

    def retrieve_vocab_known_and_unk(self) -> Tuple[dict, dict]:
        return self.model.retrieve_unk_vocab, self.model.retrieve_known_vocab


class Text2DP(Text2Graph):
    """
    Builds a "Graph of Words" where each node is a unique (lemma, POS) pair.
    All occurrences of a word map to the *same* node.
    """

    def __init__(self, lang="english", max_num_nodes=1000):

        self.spacy_models = {
            "italian": "it_core_news_lg",
            "english": "en_core_web_lg",
            "portuguese": "pt_core_news_lg",
            "portuguese_voto": "pt_core_news_lg",
        }
        self.embeddings_path = {
            "italian": "data/external/embeddings/itwiki_20180420_100d.bin",
            "english": "data/external/embeddings/glove_stanford_100d.bin",
            # "english": "data/external/embeddings/enwiki_20180420_100d.bin",
            "portuguese": "data/external/embeddings/glove_legal_100.bin",
            "portuguese_voto": "data/external/embeddings/glove_legal_100.bin",
        }

        self.lang = lang
        self._nlp = spacy.load(self.spacy_models[lang])

        deps = self.nlp.get_pipe("parser").labels
        self.dep_label_map = {label: idx for idx, label in enumerate(deps)}
        self.max_num_nodes = max_num_nodes

        self.pos_to_ignore: Set[str] = {
            "PUNCT",  # Punctuation
            # "ADP",  # Adposition (prepositions, postpositions)
            # "SPACE",  # Whitespace
            # "SYM",  # Symbol
            # "X",  # Other
            # "DET",
            # "CCONJ"
        }

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Word Embeddings
        self._text_embedding = TextEmbedding(
            model_name="glove",
            file_path=self.embeddings_path[self.lang]
        )

    @property
    def nlp(self):
        """Implement property from Text2Graph interface"""
        return self._nlp

    @property
    def text_embedding(self):
        """Implement property from Text2Graph interface"""
        return self._text_embedding

    def _is_valid_token(self, token: spacy.tokens.Token) -> bool:
        """Centralized check to validate a token."""
        return token.pos_ not in self.pos_to_ignore and token.text.strip()

    def parse_corpus(self, corpus: List[str], preprocessing_fn: Optional[Callable] = None) -> List[nx.MultiDiGraph]:
        """
        Parse a corpus of text into a list of Dependency Parsing graph representations.

        This method iterates over each document in the corpus, applies a preprocessing function (if provided),
        parses the document into a Dependency Parsing graph representation using the `parse_document` method,
        and collects the resulting graphs into a list.

        Args:
            corpus (List[str]): The text corpus to parse.
            preprocessing_fn (Optional[Callable], optional): A function for preprocessing the text before parsing.
                                                             Defaults to None.

        Returns:
            List[nx.MultiDiGraph]: The Dependency Parsing graph representations of the parsed documents.
                                   Each element in the list represents a graph corresponding to a document
                                   in the corpus.
        """
        # Initialize an empty list to store the parsed graphs
        graphs = []

        # Iterate over each document in the corpus using tqdm for progress tracking
        for text in tqdm(corpus):
            # Parse the document into a Dependency Parsing graph representation
            graph = self.parse_document(text, preprocessing_fn=preprocessing_fn)

            # Append the parsed graph to the list
            if graph is not None:
                graphs.append(graph)

        # Return the list of parsed graphs
        return graphs

    def parse_document(self, text: str, preprocessing_fn: Optional[Callable] = None) -> nx.MultiDiGraph | None:
        """
        Parse an individual document into a Knowledge Graph representation, 
        considering both dependency parsing and sequential edges.

        Args:
            text (str): The text document to parse.
            preprocessing_fn (Callable, optional): A preprocessing function for the input text. Defaults to None.

        Returns:
            nx.MultiDiGraph: The enriched Knowledge Graph representation of the parsed document.
        """
        if not text or not text.strip():
            logging.warning("Input text is empty or contains only whitespace. Skipping.")
            return None

        if preprocessing_fn:
            text = preprocessing_fn(text, self.nlp)

        # Initialize an empty graph
        graph = nx.MultiDiGraph()
        doc = self.nlp(text)

        # Add both dependency parsing and sequential edges
        self._add_dependency_edges(graph, doc)
        # self._add_sequential_edges(graph, doc) # TODO: test with and without sequential edges.

        # After edges are added
        if graph.number_of_nodes() >= self.max_num_nodes:
            logging.warning(
                f"Graph grew too large ({graph.number_of_nodes()} nodes > {self.max_num_nodes}). Skipping graph.")
            return None

        return graph

    def _add_dependency_edges(self, graph: nx.MultiDiGraph, doc: spacy.tokens.Doc) -> None:
        """Add dependency parsing edges to the graph for valid tokens."""
        for token in doc:
            # ️ Ensure both token and its head are valid before adding the edge
            if self._is_valid_token(token) and self._is_valid_token(token.head) and token != token.head:
                status = add_new_relation_to_graph(
                    target_graph=graph,
                    node_1_lemma=token.head.text,
                    node_1_pos=token.head.pos_,  # Pass the head's POS tag
                    node_2_lemma=token.text,
                    node_2_pos=token.pos_,  # Pass the token's POS tag
                    edge=token.dep_,
                    text_embedding=self.text_embedding,
                    device=self.device,
                    max_num_nodes=self.max_num_nodes
                )

                if not status:
                    logging.info("Stopping adding dep edges")
                    break

    def _add_sequential_edges(self, graph: nx.MultiDiGraph, doc: spacy.tokens.Doc) -> None:
        """Add sequential edges to the graph to represent the natural word order."""
        # Use the centralized filter for cleaner code
        filtered_tokens = [token for token in doc if self._is_valid_token(token)]

        # TODO: Test both with lemma and without lemma.
        for current_token, next_token in zip(filtered_tokens[:-1], filtered_tokens[1:]):
            status = add_new_relation_to_graph(
                graph,
                current_token.lemma_,  # Use lemma for normalization
                next_token.lemma_,
                "sequence",
                self.text_embedding,
                self.device,
                max_num_nodes=self.max_num_nodes
            )

            if not status:
                break


class PhraseSubgraphBuilder(Text2Graph):
    """
    Builds a "Graph of Occurrences" where each node is a unique TOKEN (word occurrence).
    Edges are created for:
    1. Syntactic dependencies (within a sentence), weighted by PMI.
    2. Sequential links between sentence roots (weight=1.0).
    """

    def __init__(self, nlp: Language, text_embedding: TextEmbedding,
                 max_num_nodes=1000, min_pmi_threshold=0.0, use_pmi:bool=False):
        """
        Initializes the builder.

        Args:
            nlp (Language): A loaded spaCy model.
            text_embedding (TextEmbedding): The loaded TextEmbedding object from your repo.
            max_num_nodes (int): Max nodes for the graph.
            min_pmi_threshold (float): Minimum PMI to create an edge.
        """
        self._nlp = nlp
        self._text_embedding = text_embedding
        self.max_num_nodes = max_num_nodes
        self.min_pmi = min_pmi_threshold  # [NEW]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pmi: Dict[Tuple[str, str], float] = {}  # [NEW]
        self.dep_to_ignore: Set[str] = {"punct"}  # [NEW]

        deps = self._nlp.get_pipe("parser").labels
        self.dep_label_map = {label: idx for idx, label in enumerate(deps)}

        # [STANDARD] Replicate the token filters from Text2DP
        self.pos_to_ignore: Set[str] = {
            "PUNCT"
        }
        self.use_pmi=use_pmi


    @property
    def nlp(self):
        """Implement property from Text2Graph interface"""
        return self._nlp

    @property
    def text_embedding(self):
        """Implement property from Text2Graph interface"""
        return self._text_embedding

    def _is_valid_token(self, token: Token) -> bool:
        """[STANDARD] Centralized check to validate a token."""
        return token.pos_ not in self.pos_to_ignore and token.text.strip()

    def _preprocess_node_text(self, text: str) -> str:
        return text.lower().strip()

    def _is_valid_dependency(self, h, c, d) -> bool:
        if not h or not c or h == c: return False
        if d in self.dep_to_ignore: return False
        return True

    def compute_pmi(self, corpus_docs: List[Doc]):
        """Calculates PMI from a list of pre-processed spaCy Docs."""
        logging.info("Calculating PMI for PhraseSubgraphBuilder...")
        co_occur, word_count = defaultdict(int), defaultdict(int)
        total_edges = 0
        for doc in tqdm(corpus_docs, desc="1/2: Counting co-occurrences"):
            for token in doc:
                # Use preprocessed text for PMI keys
                h = self._preprocess_node_text(token.head.text)
                c = self._preprocess_node_text(token.text)
                if not self._is_valid_dependency(h, c, token.dep_): continue

                co_occur[(h, c)] += 1
                word_count[h] += 1
                word_count[c] += 1
                total_edges += 1

        if total_edges == 0:
            logging.warning("No valid dependency edges found to compute PMI.")
            return

        pmi_values = {}
        for (w1, w2), count in tqdm(co_occur.items(), desc="2/2: Calculating PMI"):
            p_w1 = word_count[w1] / total_edges
            p_w2 = word_count[w2] / total_edges
            p_w1_w2 = count / total_edges
            pmi_val = math.log2(p_w1_w2 / (p_w1 * p_w2 + 1e-9))
            pmi_values[(w1, w2)] = pmi_val

        self.pmi = pmi_values
        logging.info(f"PMI calculation complete. Found {len(self.pmi)} valid pairs.")

    # --- End of [NEW] PMI Methods ---

    def parse_corpus(self, corpus: List[str], preprocessing_fn: Optional[Callable] = None) -> List[nx.MultiDiGraph]:
        """Implement abstract method"""
        graphs = []
        for text in tqdm(corpus, desc="Parsing corpus (PhraseSubgraphBuilder)"):
            graph = self.parse_document(text, preprocessing_fn=preprocessing_fn)
            if graph is not None:
                graphs.append(graph)
        return graphs

    def parse_document(self, text: str, preprocessing_fn: Optional[Callable] = None) -> nx.MultiDiGraph | None:
        """
        [UPGRADED LOGIC]
        Parse an individual document into a "graph of occurrences" representation,
        now with PMI-weighted dependency edges.
        """
        if not text or not text.strip():
            logging.warning("Input text is empty or contains only whitespace. Skipping.")
            return None

        if preprocessing_fn:
            text = preprocessing_fn(text, self.nlp)

        G = nx.MultiDiGraph()
        doc = self.nlp(text)

        # --- Pass 1: Create Nodes (one per token) ---
        for token in doc:
            if not self._is_valid_token(token):
                continue

            if G.number_of_nodes() >= self.max_num_nodes:
                logging.warning(f"Graph capacity reached ({self.max_num_nodes}). Stopping node addition.")
                break

            node_id = token.i
            node_text = self._preprocess_node_text(token.text)
            embedding = self.text_embedding.retrieve_embeddings(node_text)

            # ONLY add the 'x' attribute.
            # The 'lemma' and 'pos' attributes are strings and
            # are confusing the from_networkx converter.
            G.add_node(
                node_id,
                x=embedding,
                token=token.text.strip().lower()
                # lemma=token.lemma_.lower().strip(), # <-- REMOVED
                # pos=token.pos_                     # <-- REMOVED
            )

        if G.number_of_nodes() == 0:
            return None

        # --- Pass 2: Create PMI-Weighted Dependency Edges (Directional) ---
        for token in doc:
            head_id = token.head.i
            token_id = token.i

            if head_id == token_id or not G.has_node(head_id) or not G.has_node(token_id):
                continue

            h = self._preprocess_node_text(token.head.text)
            c = self._preprocess_node_text(token.text)

            if not self._is_valid_dependency(h, c, token.dep_):
                continue

            if self.use_pmi:
                pmi_val = max(0.0, self.pmi.get((h, c), 0))
                edge_value = pmi_val + 1
            else:
                edge_value = 1
            G.add_edge(token_id, head_id, label=token.dep_, weight=edge_value)

        # --- Pass 3: Create Stitching Edges (Directional, weight=1.0) ---
        prev_root_id = None
        for sent in doc.sents:
            current_root_id = sent.root.i

            if not G.has_node(current_root_id):
                continue

            if prev_root_id is not None:
                G.add_edge(prev_root_id, current_root_id, label="next_root", weight=1.0)
                G.add_edge(current_root_id, prev_root_id, label="prev_root", weight=1.0)

            prev_root_id = current_root_id

        # Remove all isolated nodes.
        # Find isolates (nodes with degree 0: no incoming and no outgoing edges in directed graphs)
        isolated_nodes = list(nx.isolates(G))
        # nx.isolates returns an iterator, so cast to list first :contentReference[oaicite:1]{index=1}

        # Remove them
        # avoid passing the generator directly to remove_nodes_from to prevent “dictionary changed size” errors. :contentReference[oaicite:2]{index=2}
        G.remove_nodes_from(isolated_nodes)
        return G


class Text2GraphDataset:
    """
    Class for managing datasets and parsing text data into dp_graphs.
    Now includes a helper to get processed docs for PMI calculation.

    Attributes:
        text_to_graph_parser (Text2Graph): An instance of Text2Graph or its subclass used for parsing text into dp_graphs.
        path_to_corpus (Path): The path to the directory containing the text corpus.

    Methods:
        load_corpus(): Loads the text corpus from the specified directory.
        parse(corpus): Parses the loaded corpus into dp_graphs using the provided text_to_graph_parser.
        build_pyg_dataset(): Builds a PyTorch Geometric (PyG) dataset from the parsed dp_graphs.
    """

    def __init__(self, text_to_graph_parser: Text2Graph, path_to_corpus: Path, path_to_output: Path,
                 preprocessing_fn: Callable = None):
        """
        Initialize the Text2GraphDataset.

        Args:
            text_to_graph_parser (Text2Graph): An instance of Text2Graph or its subclass used for parsing text into dp_graphs.
            path_to_corpus (Path): The path to the directory containing the text corpus.
            path_to_output (Path): The path to the directory where the output will be saved.

        Attributes:
            kg_dataset: The knowledge graph dataset (initialized as None).
            text_to_graph_parser: The text to graph parser instance.
            path_to_corpus: The path to the text corpus directory.
            path_to_output: The path to the output directory.
            labels: The labels for the dataset (initialized as None).
            corpus: The text corpus (initialized as None).
            parsed_graphs: The parsed graphs (initialized as None).
        """
        # Initialize the knowledge graph dataset as None
        self.kg_dataset = None

        # Store the text to graph parser instance
        self.text_to_graph_parser = text_to_graph_parser

        # Store the path to the text corpus directory
        self.path_to_corpus = path_to_corpus

        # Store the path to the output directory
        self.path_to_output = path_to_output

        # Initialize the labels, corpus, and parsed graphs as None
        self.class_names = None
        self.corpus = None
        self.parsed_graphs = None
        self.preprocessing_fn = preprocessing_fn
        self.nlp = self.text_to_graph_parser.nlp

    def get_processed_docs(self) -> List[Doc]:
        """
        Pre-processes the entire corpus and returns a list of spaCy Docs.
        Used for corpus-wide calculations like PMI.
        """
        if not self.corpus:
            logging.warning("Corpus not loaded. Call load_corpus() first.")
            return []

        processed_docs = []
        desc = "Pre-processing texts for PMI"
        for file_name, label, text in tqdm(self.corpus, desc=desc):
            clean_text = text
            if self.preprocessing_fn:
                clean_text = self.preprocessing_fn(text, self.nlp)
            processed_docs.append(self.nlp(clean_text))
        return processed_docs

    def load_corpus(self) -> None:
        """
        Load the text corpus from the specified directory.

        Returns:
           tuple: A tuple containing the corpus and labels.
                  - corpus (list): List of tuples, each containing the label and text content.
                  - labels (list): List of labels corresponding to each text sample.
        """
        corpus = []
        labels = []

        def _read_document(target_path: Path, label_name: str):
            """
            Read the content of a text document and append it to the corpus.

            Parameters:
              target_path (Path): The path to the text document to be read.
            """
            with open(target_path, "r", encoding="utf-8") as file:
                file_name = os.path.basename(target_path)
                text = file.read()
                corpus.append((file_name, label_name, text))

        for label_dir in self.path_to_corpus.iterdir():
            if label_dir.is_dir():
                label = label_dir.name

                if label == "ignored":
                    continue

                labels.append(label)

                for file_path in sorted(label_dir.glob("*.txt")):
                    _read_document(file_path, label)

        self.corpus = random.sample(corpus, 200)
        #self.corpus = corpus
        self.class_names = labels

    def parse(self, corpus=None):
        """
        Parse the provided or loaded corpus into graphs using the text_to_graph_parser.
        Calculate and log OOV statistics for each document.

        Args:
            corpus (list, optional): A batch of (proc_number, label, text) tuples to parse.
            vocabulary (set, optional): The known vocabulary to calculate OOV statistics.

        Returns:
            list: Parsed graphs in the format (proc_number, label, text, graph).
        """

        if corpus is None:
            corpus = self.corpus

        parsed_graphs = []
        # Use the parser's class name in the description
        desc = f"Parsing batch ({self.text_to_graph_parser.__class__.__name__})"
        for proc_number, label, text in tqdm(corpus, desc=desc):
            # Note: preprocessing_fn is now applied *inside* parse_document
            graph = self.text_to_graph_parser.parse_document(
                text.strip(),
                preprocessing_fn=self.preprocessing_fn
            )

            if graph is None:
                continue
            isolated_nodes = list(nx.isolates(graph))
            if len(isolated_nodes) > 0:
                logging.info(f'{"=" * 50} {proc_number} {"=" * 50}')
                logging.info("Isolated nodes")
                logging.info(isolated_nodes)

            parsed_graphs.append((proc_number, label, text, graph))

        return parsed_graphs

    def generate_graph_images(self, graphs, output_folder: Path):
        """
        Generates graph images for a list of NetworkX dp_graphs using Graphviz.

        Args:
        - dp_graphs (list): List of NetworkX dp_graphs.
        - output_folder (str): Path to the folder where images will be stored.

        Returns:
        - None
        """

        def _save_graphviz_graph(G: nx.MultiDiGraph, dot_path: Path, image_path: Path):
            nx.drawing.nx_pydot.write_dot(G, dot_path)

            try:
                # Read the DOT file
                A = pgv.AGraph(image_path, directed=True)

                # Render the graph (choose a suitable format)
                A.draw(output_path_png, prog="circo")

            except Exception as e:
                logging.info(f"Error rendering graph: {e}")

        def _save_nx_graph(G, image_path: Path):
            plt.figure(figsize=(16, 9))
            pos = nx.spring_layout(G)

            nx.draw_networkx_nodes(G, pos)
            nx.draw_networkx_labels(G, pos)
            nx.draw_networkx_edges(G, pos, edge_color='r', arrows=True)
            plt.title(f"Document {proc_number} | Label: {label}")
            plt.tight_layout()
            plt.savefig(image_path, dpi=300, format="png")

        # Create the output folder if it does not exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        logging.info("Plotting")

        # Iterate over each graph
        for idx, graph_tuple in tqdm(enumerate(graphs)):

            proc_number, label, text, G = graph_tuple
            output_path_dot = output_folder / f"graph_{proc_number}.dot"
            output_path_png = output_folder / f"graph_{proc_number}.png"

            logging.info(f"Proc: {proc_number},\tLabel: {label}\tImage: {output_path_png}")

            try:
                _save_graphviz_graph(G, output_path_dot, output_path_png)
            except Exception as e:
                logging.error(f"Error while export graphviz: {e}. Saving using NetworkX.")
                _save_nx_graph(G, output_path_png)

    def build_pyg_dataset_individual(self, input_filename=None, parsed_graphs=None, output_path=None, batch_index=0):
        """
        Builds a PyTorch Geometric (PyG) dataset from parsed graphs, saving each graph individually.

        Args:
            input_filename (str or Path, optional): Path to the input file containing parsed graphs.
            parsed_graphs (list, optional): Parsed graphs to process.
            output_path (str or Path, optional): Directory to save the individual graph files.
            batch_index (int, optional): Index of the current batch to avoid filename collisions.
        """
        if input_filename is None and parsed_graphs is None:
            raise RuntimeError("input_filename or parsed_graphs should be provided.")

        if input_filename is not None:
            dataset_path = Path(input_filename)
            with open(dataset_path, "rb") as f:
                self.parsed_graphs = pickle.load(f)
        else:
            self.parsed_graphs = parsed_graphs

        output_dir = self.path_to_output if output_path is None else Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        proc_numbers = []
        labels = []

        start_index = batch_index * len(self.parsed_graphs)
        for i, (proc_number, label, text, graph) in tqdm(enumerate(self.parsed_graphs), desc="Storing graphs."):
            try:
                # Ensure unique filenames across batches
                doc_number = start_index + i
                graph_file_path = output_dir / f"graph_{doc_number:07d}.pt"
                proc_numbers.append(proc_number)
                labels.append(label)
                torch.save((proc_number, label, graph), graph_file_path)

                pyvis_base_path = Path(str(output_dir).replace("interim", "visualization"))
                os.makedirs(pyvis_base_path, exist_ok=True)
                pyvis_path = pyvis_base_path / f"graph_{doc_number:07d}.html"

                # visualize_l0()

            except Exception as e:
                logging.info(f"Failed to save graph {proc_number}: {e}")
                continue

        # Save metadata incrementally with duplicate checking
        metadata_path = output_dir / "metadata.pkl"
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)

            # Avoid duplicates in metadata
            new_proc_numbers = set(proc_numbers) - set(metadata["proc_numbers"])
            if new_proc_numbers:
                metadata["proc_numbers"].extend(list(new_proc_numbers))
                metadata["labels"].extend(
                    [label for i, label in enumerate(labels) if proc_numbers[i] in new_proc_numbers])
        else:
            metadata = {
                "proc_numbers": proc_numbers,
                "labels": labels
            }

        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

        logging.info(f"Batch {batch_index} processed and saved successfully!")

    def save_graphs_individually(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, data in enumerate(self.kg_dataset.data_list):
            file_path = output_dir / f"graph_{i}.pt"
            torch.save(data, file_path)
