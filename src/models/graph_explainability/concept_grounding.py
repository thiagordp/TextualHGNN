import json
import logging
from pathlib import Path

import torch
from matplotlib import pyplot as plt
from torch_geometric.data import Data
from tqdm import tqdm
import seaborn as sns

from src.models.graph_classification.gnn import DiffPool
from src.models.graph_explainability.embeddings_oracle import EmbeddingOracle
from src.models.graph_explainability.llm_oracle import LLMOracle


class ConceptGrounding:

    def __init__(self,
                 graph_model: DiffPool,
                 embedding_oracle: EmbeddingOracle,
                 llm_oracle: LLMOracle,
                 graph: Data,
                 hyper_nodes_to_explain: int = 10,
                 nodes_per_hyper_node: int = 5,
                 original_raw_file_path: str = "",
                 language: str = "english"
                 ):
        self.graph_model = graph_model
        self.graph = graph
        self.embedding_oracle = embedding_oracle  # This is fixed for L0
        self.llm_oracle = llm_oracle
        self.hyper_nodes_to_explain = hyper_nodes_to_explain
        self.nodes_per_hyper_node = nodes_per_hyper_node
        self.explanation = None
        self.raw_file_path = Path(original_raw_file_path)
        self.raw_file_content = None
        self.language = language

        self._init_model()

    def _init_model(self):
        self.data_sample_id = int.from_bytes(self.graph.doc_id, byteorder='little')

        with torch.no_grad():
            self.prediction = self.graph_model(self.graph.x, self.graph.adj, debug=True)

        y_pred, obj1, obj2, g_layer1, g_layer2, s = self.prediction
        y_pred = torch.softmax(y_pred, dim=1)
        s01, s12 = s
        x_l1, adj_l1 = g_layer1
        x_l2, adj_l2 = g_layer2

        self.s01 = s01.squeeze(0)
        self.s12 = s12.squeeze(0)
        self.x_l0 = self.graph.x.squeeze(0)
        self.adj_l0 = self.graph.adj.squeeze(0)
        self.x_l1 = x_l1.squeeze(0)
        self.adj_l1 = adj_l1.squeeze(0)
        self.x_l2 = x_l2.squeeze(0)
        self.adj_l2 = adj_l2.squeeze(0)
        self.y_pred = torch.argmax(y_pred, dim=1).item()
        self.y_test = int.from_bytes(self.graph.y, byteorder='little')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hardcoded for now.
        if self.language == 'english':
            labels = {0: "negative", 1: 'positive'}
        else:
            labels = {0: "NotReleased", 1: 'Released'}

        label = labels[self.y_test]

        self.raw_file_content = open(self.raw_file_path / f"{label}/{self.data_sample_id}.txt", "r").read().strip()

        logging.info(f"Analyzing File {self.data_sample_id} | y_test {self.y_test} | y_pred {self.y_pred}")

    def ground_concepts_at_layer_one(self, method="semantic_similarity"):
        if method not in ("semantic_similarity", "l0_similarity", "llm"):
            raise ValueError("Invalid method. Choose from: semantic_similarity, l0_similarity, llm.")

        top_assignments_s12, top_nodes_s12 = self._get_top_most_nodes(layer=1, k=10)

    def retrieve_relevant_hypernodes_and_corresponding_nodes(self, top_l1: int = 10, top_l0: int = 5):

        # Step 1: Compute the top L1 indices based on row sums
        row_sums = self.s12.sum(dim=1)
        top_l1_indices = torch.topk(row_sums, top_l1).indices.tolist()

        # Step 2: Compute the top L0 indices for each L1 index
        top_l0_indices_per_l1 = {}
        s01 = self.s01

        for m in top_l1_indices:
            column_values = s01[:, m]
            _, top_k_indices = torch.topk(column_values, top_l0, dim=0)
            top_l0_indices_per_l1[m] = top_k_indices.tolist()


        # logging.info("Top L0 Indices per L1:")
        # logging.info(json.dumps(top_l0_indices_per_l1, indent=3))

        # Step 3: Create a dictionary with hypernodes and corresponding nodes
        relevant_data = {}

        for l1_index in top_l1_indices:
            # Retrieve the relevant hypernode (a row from x_l1)
            hypernode = self.x_l1[l1_index, :].tolist()  # Shape: (100,)

            # Retrieve the relevant nodes (rows from x_l0)
            l0_indices = top_l0_indices_per_l1.get(l1_index, [])
            nodes = self.x_l0[l0_indices, :].tolist()  # Shape: (top_l0, 100)

            # Build the dictionary entry
            relevant_data[l1_index] = {
                "hypernode": hypernode,
                "nodes": nodes
            }

        # logging.info("\nRelevant Data (Structured as Dictionary):")
        # logging.info(json.dumps(relevant_data, indent=3))

        return relevant_data

    # Get the biggest hypernodes from S12.
    def _get_top_most_nodes(self, layer: int = 1, k: int = 10):
        if layer == 1:
            row_sums = self.s12.sum(dim=1)
            top_k_indices = torch.topk(row_sums, k).indices

            tok_k_rows = self.s12[top_k_indices]
            return tok_k_rows, top_k_indices
        elif layer == 0:
            row_sums = self.s01.sum(dim=1)
            top_k_indices = torch.topk(row_sums, k).indices

            tok_k_rows = self.s01[top_k_indices]
            return tok_k_rows, top_k_indices
        else:
            raise ValueError("Invalid layer. Choose from: 1, 2.")

    def _get_nodes_l0(self, nodes_l0):
        terms_l0 = []

        for node_l0 in nodes_l0:
            input_embedding = torch.tensor([node_l0], device=self.device)
            term = self.embedding_oracle.concept_grounding_from_embeddings(
                input_embedding,
                num_terms_to_retrieve=1,
                similarity_threshold=0.9999
            )
            terms_l0.extend(term['terms'] if 'terms' in term else term)

        return terms_l0

    def concept_grounding(self):

        nodes_l1_to_nodes_l0 = self.retrieve_relevant_hypernodes_and_corresponding_nodes(top_l1=self.hyper_nodes_to_explain, top_l0=self.nodes_per_hyper_node)
        nodes_l1_to_words_l0 = {'explanation': {}}

        for node in tqdm(nodes_l1_to_nodes_l0, desc="Grounding L1"):
            hyper_node = nodes_l1_to_nodes_l0[node]["hypernode"][:100]
            hyper_node = torch.tensor(hyper_node, device=self.device)

            nodes_l0 = nodes_l1_to_nodes_l0[node]["nodes"]

            # For this embeddings oracle is required.
            terms_l0 = self._get_nodes_l0(nodes_l0)

            logging.info(f"Nodes from L0 under analysis: {terms_l0}")

            terms_cg_methods = {}

            # methods = ["top_l0", "llm_search", "semantic_search_l0", "semantic_search_l1"]
            # methods = ["top_l0", "llm_search"]
            methods = ["top_l0", "llm_search"]
            for cg_method in methods:
                logging.info("Starting with method " + cg_method)

                if cg_method == "top_l0":
                    term_l0  = terms_l0[:1]
                    terms_cg_methods["top_l0"] = term_l0
                    logging.info(f"Terms (top_l0): {term_l0}")

                if cg_method == "semantic_search_l0":
                    term_l1 = self.embedding_oracle.concept_grounding_from_words(
                        terms_l0,
                        num_terms_to_retrieve=3,
                        similarity_threshold=0.9999
                    )
                    logging.info(f"Terms (Method L0): {term_l1}")
                    terms_cg_methods[cg_method] = term_l1['terms'] if 'terms' in term_l1 else term_l1

                elif cg_method == "semantic_search_l1":
                    term_l1 = self.embedding_oracle.concept_grounding_from_embeddings(
                        hyper_node,
                        num_terms_to_retrieve=3,
                        similarity_threshold=None
                    )
                    logging.info(f"Terms (Method L1): {term_l1}")

                    terms_cg_methods[cg_method] = term_l1['terms'] if 'terms' in term_l1 else term_l1
                elif cg_method == "llm_search":
                    term_l1 = self.llm_oracle.concept_grounding_from_words(
                        terms_l0,
                        num_terms_to_retrieve=3,
                        similarity_threshold=0.9
                    )
                    logging.info(f"Terms (Method LLM): {term_l1}")
                    terms_cg_methods[cg_method] = term_l1['terms'] if 'terms' in term_l1 else term_l1

            nodes_l1_to_words_l0['explanation'][node] = {
                "words_l0": terms_l0,
                "words_cg_methods": terms_cg_methods
            }

        nodes_l1_to_words_l0["y_pred"] = self.y_pred
        nodes_l1_to_words_l0["y_test"] = self.y_test
        nodes_l1_to_words_l0["original_content"] = self.raw_file_content

        self.explanation = nodes_l1_to_words_l0

    def save_explanation(self, output_file):
        logging.info(f"Storing to {output_file}")

        with open(output_file, "w") as f:
            f.write(json.dumps(self.explanation, indent=4))
        # with open(output_file, 'w') as outfile:
        #     json.dump(self.explanation, outfile, indent=4)
