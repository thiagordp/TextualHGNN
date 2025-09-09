import json
import logging
from pathlib import Path

import spacy
import torch
from matplotlib import pyplot as plt
from torch_geometric.data import Data
from tqdm import tqdm
import seaborn as sns

from src.data.preprocessing import preprocessing_legal_pt
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
        self.spacy_models = {
            "italian": "it_core_news_lg",
            "english": "en_core_web_lg",
            "portuguese": "pt_core_news_lg",
            "portuguese_voto": "pt_core_news_lg",
        }
        self.nlp = spacy.load(self.spacy_models[self.language])

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
        elif self.language == "italian":
            labels = {0: "NotReleased", 1: 'Released'}
        else:
            labels = {0: "Preso", 1: 'Solto'}
        label = labels[self.y_test]

        self.raw_file_content = open(self.raw_file_path / f"{label}/{self.data_sample_id}.txt", "r").read().strip()

        # Preprocess and print the document content
        preprocessed_content = preprocessing_legal_pt(self.raw_file_content, self.nlp)
        logging.info(f"--- Preprocessed Document Content ---")
        logging.info(preprocessed_content)
        logging.info(f"------------------------------------")

        logging.info(f"Analyzing File {self.data_sample_id} | y_test {self.y_test} | y_pred {self.y_pred}")

    def ground_concepts_at_layer_one(self, method="semantic_similarity"):
        if method not in ("semantic_similarity", "l0_similarity", "llm"):
            raise ValueError("Invalid method. Choose from: semantic_similarity, l0_similarity, llm.")

        top_assignments_s12, top_nodes_s12 = self._get_top_most_nodes(layer=1, k=10)

    def retrieve_relevant_hypernodes_and_corresponding_nodes(self, l1_threshold: float = 0.5,
                                                             l0_threshold: float = 0.5):

        # Step 1: Compute the top L1 indices based on a threshold
        top_l1_indices = (self.s12 > l1_threshold).any(dim=1).nonzero(as_tuple=True)[0].tolist()

        # Step 2: Compute the top L0 indices for each L1 index based on a threshold
        top_l0_indices_per_l1 = {}
        s01 = self.s01

        for m in top_l1_indices:
            column_values = s01[:, m]
            top_k_indices = (column_values > l0_threshold).nonzero(as_tuple=True)[0]
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
                "nodes": nodes,
                "l0_indices": l0_indices
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

    def _get_nodes_l0(self, nodes_l0, l0_indices, l1_index):
        terms_with_scores = {}
        for node_l0, original_index in zip(nodes_l0, l0_indices):
            input_embedding = torch.tensor([node_l0], device=self.device)
            result = self.embedding_oracle.concept_grounding_from_embeddings(
                input_embedding,
                num_terms_to_retrieve=1,
                similarity_threshold=0.9999
            )
            if 'terms' in result and result['terms']:
                term = result['terms'][0]
                # Get the assignment value from the s01 matrix
                assignment_value = self.s01[original_index, l1_index].item()
                terms_with_scores[term] = assignment_value

        # Sort the dictionary by value in descending order
        sorted_terms = sorted(terms_with_scores.items(), key=lambda item: item[1], reverse=True)
        return dict(sorted_terms)

    def concept_grounding(self):

        nodes_l1_to_nodes_l0 = self.retrieve_relevant_hypernodes_and_corresponding_nodes(l1_threshold=0.5,
                                                                                         l0_threshold=0.1)
        nodes_l1_to_words_l0 = {'explanation': {}}

        for node_index, node_data in tqdm(nodes_l1_to_nodes_l0.items(), desc="Grounding L1"):
            hyper_node = node_data["hypernode"][:100]
            hyper_node = torch.tensor(hyper_node, device=self.device)

            nodes_l0 = node_data["nodes"]
            l0_indices = node_data["l0_indices"]

            # For this embeddings oracle is required.
            terms_l0_with_scores = self._get_nodes_l0(nodes_l0, l0_indices, node_index)

            logging.info("Nodes from L0 under analysis:")
            logging.info(json.dumps(terms_l0_with_scores, indent=4, ensure_ascii=False))

            # New: Log L1 to L2 assignment values, sorted
            l1_to_l2_assignments = self.s12[node_index, :].tolist()
            sorted_l1_to_l2 = sorted(l1_to_l2_assignments, reverse=True)
            #logging.info(f"L1 Node {node_index} to L2 assignments (sorted): {sorted_l1_to_l2}")

            terms_l0 = list(terms_l0_with_scores.keys())

            terms_cg_methods = {}

            methods = ["top_l0", "llm_search", "semantic_search_l0", "semantic_search_l1"]
            for cg_method in methods:
                logging.info("Starting with method " + cg_method)

                if cg_method == "top_l0":
                    term_l0 = terms_l0[:1]
                    terms_cg_methods["top_l0"] = term_l0
                    logging.info(f"Terms (top_l0): {term_l0}")

                elif cg_method == "semantic_search_l0":
                    term_l1 = self.embedding_oracle.concept_grounding_from_words(
                        terms_l0,
                        num_terms_to_retrieve=1,
                        similarity_threshold=0.9999
                    )
                    if 'terms' in term_l1 and 'similarities' in term_l1:
                        terms_with_sim = dict(zip(term_l1['terms'], term_l1['similarities'][0]))
                        logging.info(f"Terms (Method L0): {json.dumps(terms_with_sim, indent=4, ensure_ascii=False)}")
                    else:
                        logging.info(f"Terms (Method L0): {term_l1}")
                    terms_cg_methods[cg_method] = term_l1['terms'] if 'terms' in term_l1 else term_l1

                elif cg_method == "semantic_search_l1":
                    term_l1 = self.embedding_oracle.concept_grounding_from_embeddings(
                        hyper_node,
                        num_terms_to_retrieve=1,
                        similarity_threshold=None
                    )
                    if 'terms' in term_l1 and 'similarities' in term_l1:
                        terms_with_sim = dict(zip(term_l1['terms'], term_l1['similarities'][0]))
                        logging.info(f"Terms (Method L1): {json.dumps(terms_with_sim, indent=4, ensure_ascii=False)}")
                    else:
                        logging.info(f"Terms (Method L1): {term_l1}")
                    terms_cg_methods[cg_method] = term_l1['terms'] if 'terms' in term_l1 else term_l1
                elif cg_method == "llm_search":
                    term_l1 = self.llm_oracle.concept_grounding_from_words(
                        terms_l0,
                        num_terms_to_retrieve=1,
                        similarity_threshold=0.9
                    )
                    if 'terms' in term_l1 and 'similarities' in term_l1:
                        terms_with_sim = dict(zip(term_l1['terms'], term_l1['similarities']))
                        logging.info(f"Terms (Method LLM): {json.dumps(terms_with_sim, indent=4, ensure_ascii=False)}")
                    else:
                        logging.info(f"Terms (Method LLM): {term_l1}")
                    terms_cg_methods[cg_method] = term_l1['terms'] if 'terms' in term_l1 else term_l1


            nodes_l1_to_words_l0['explanation'][node_index] = {
                "words_l0": terms_l0_with_scores,
                "l1_to_l2_assignments": sorted_l1_to_l2,
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