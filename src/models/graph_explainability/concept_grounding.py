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
                 hyper_nodes_to_explain: int | None = 10,
                 nodes_per_hyper_node: int | None = 5,
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

        self.l0_names = [str(i) for i in range(self.x_l0.shape[0])]
        self.l1_names = [f"C1_{i}" for i in range(self.x_l1.shape[0])]
        self.l2_names = [f"C2_{i}" for i in range(self.x_l2.shape[0])]

        self.y_test = int.from_bytes(self.graph.y, byteorder='little')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.y_pred = y_pred.max(dim=1)[1].cpu()

        y_tensor = self.graph.y
        if isinstance(y_tensor, list):
            y_tensor = torch.tensor(y_tensor, device=self.device, dtype=torch.float)

        # Convert one-hot labels to class indices
        if y_tensor.ndim > 1 and y_tensor.shape[1] > 1:
            y_indices = y_tensor.sum(dim=1)
        elif y_tensor.shape[0] > 1:
            y_indices = y_tensor.sum(dim=0)
        else:
            y_indices = y_tensor.long()  # Ensure it's integer type

        logging.info("=" * 80)
        logging.info(y_indices, y_indices.ndim, type(y_indices))
        logging.info("=" * 80)

        self.y_test = int(y_indices.view(-1).cpu())

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
                                                             l0_threshold: float = 0.1):

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

    def _ground_all_l0_nodes(self):
        """
        Iterates through all L0 node embeddings and finds their corresponding
        word from the embedding oracle. Populates self.l0_names.
        """
        logging.info("Starting L0 concept grounding for all nodes...")
        cutoff_idx = (self.x_l0.abs().sum(dim=1) > 1e-5).nonzero().max().item() + 1

        l0_node_indices = list(range(cutoff_idx))#[:2]
        for i in tqdm(l0_node_indices, desc="Grounding L0 Nodes"):
            # if i > 7:
            #     break

            node_embedding = self.x_l0[i].unsqueeze(0)
            result = self.embedding_oracle.concept_grounding_from_embeddings(
                node_embedding,
                num_terms_to_retrieve=1,
                similarity_threshold=0.999  # Use a high threshold for direct matches
            )
            if 'terms' in result and result['terms']:
                self.l0_names[i] = result['terms'][0]
            else:
                self.l0_names[i] = f"UNK_{i}"  # Mark as unknown if not found
        logging.info("Finished L0 concept grounding.")

    def _get_nodes_l0(self, l0_indices, l1_index, top_k=None):
        """
        Retrieves the pre-computed L0 names and their assignment scores for a
        given L1 cluster.
        """
        terms_with_scores = {}

        # Get assignment values for all relevant L0 nodes to the specific L1 cluster
        for node_l0_index in l0_indices:
            # Retrieve the already grounded name
            term = self.l0_names[node_l0_index]
            assignment_value = self.s01[node_l0_index, l1_index].item()
            terms_with_scores[term] = assignment_value

        # Sort the dictionary by assignment score in descending order
        sorted_terms = sorted(terms_with_scores.items(), key=lambda item: item[1], reverse=True)

        if top_k:
            sorted_terms = sorted_terms[:top_k]

        return dict(sorted_terms)

    def concept_grounding(self):

        # --- STEP 1: Ground all L0 nodes first ---
        self._ground_all_l0_nodes()

        # --- STEP 2: Ground L1 nodes based on the now-known L0 names ---
        nodes_l1_to_nodes_l0 = self.retrieve_relevant_hypernodes_and_corresponding_nodes(l1_threshold=0.01,
                                                                                         l0_threshold=0.01)

        i = 0
        for node_index, node_data in tqdm(nodes_l1_to_nodes_l0.items(), desc="Grounding L1 Concepts"):
            # i += 1
            # if i > 4:
            #     break
            l0_indices = node_data["l0_indices"]

            # This method now just looks up names and scores, it doesn't call the oracle
            terms_l0_with_scores = self._get_nodes_l0(l0_indices, node_index, top_k=3)
            terms_l0 = list(terms_l0_with_scores.keys())

            if terms_l0:
                # Assign the most representative L0 word as the name for the L1 cluster
                self.l1_names[node_index] = terms_l0[0]

        # --- STEP 3: Ground L2 nodes based on the now-known L1 names ---
        logging.info("Grounding L2 concepts...")
        best_l1_indices_for_l2 = torch.argmax(self.s12, dim=0)

        for l2_idx, l1_idx in enumerate(best_l1_indices_for_l2):
            l1_idx_item = l1_idx.item()
            if l1_idx_item < len(self.l1_names):
                # The name for the L2 super-cluster is the name of its most representative L1 cluster
                self.l2_names[l2_idx] = self.l1_names[l1_idx_item]

        logging.info(f"Grounded L2 Names: {self.l2_names}")

        self.explanation = nodes_l1_to_nodes_l0

    def save_explanation(self, output_file):
        logging.info(f"Storing to {output_file}")
        with open(output_file, "w") as f:
            f.write(json.dumps(self.explanation, indent=4, ensure_ascii=False))
