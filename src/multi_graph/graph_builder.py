# graph_builder.py

import spacy
import torch
import networkx as nx
import numpy as np
import logging

from torch_geometric.data import HeteroData
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity, one_hot

# Local imports
from src.multi_graph.config import COMMON_DEP_LABELS

# --- NEW: Schemas define the required attributes for valid graph components ---
NODE_SCHEMA = {
    'word': {'type', 'text', 'x', 'contextual_x', 'position_in_sentence'},
    'sentence': {'type', 'text', 'x', 'sentence_index'},
    'document': {'type', 'text', 'x', 'filename'}
}

EDGE_SCHEMA = {
    'dep': {'type', 'feature'},
    'seq': {'type', 'feature'},
    'same_lemma': {'type', 'feature'},
    'sim': {'type', 'feature'},
    'belongs': {'type'}  # 'belongs' edges have no required features
}


def get_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    """Generates sinusoidal positional encodings."""
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    if d_model % 2 != 0:
        pe[:, 1::2] = torch.cos(position * div_term)[:, :pe[:, 1::2].shape[1]]
    else:
        pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class DocumentGraphBuilder:
    """
    A class to build multi-level, heterogeneous graphs from textual documents.
    This version implements our final hybrid architecture design and includes
    a validation and pruning step to ensure data integrity.
    """

    def __init__(self, spacy_model: str, embedding_model: str, similarity_threshold: float):
        logging.info("Initializing DocumentGraphBuilder...")
        self.nlp = spacy.load(spacy_model)
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model)
        self.similarity_threshold = similarity_threshold
        self.dep_label_map = {label: i for i, label in enumerate(COMMON_DEP_LABELS)}
        self.used_deps = {label: 0 for label in COMMON_DEP_LABELS}
        self.embedding_dim = self.model.config.hidden_size
        logging.info(f"Models and configuration loaded successfully. Embedding dim: {self.embedding_dim}")

    def _get_batch_embedding(self, texts: list[str]) -> torch.Tensor:
        """Generates embeddings for a batch of texts."""
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

    def process_documents(self, documents: list[tuple[str, str]]) -> tuple[list[nx.MultiDiGraph], list[HeteroData]]:
        """
        Processes documents, validates each resulting graph, and returns lists of
        only the valid graphs.
        """
        valid_nx_graphs = []
        valid_hetero_graphs = []

        for i, (filename, doc_text) in enumerate(documents):
            logging.info(f"--- Processing Document {i + 1}/{len(documents)} ({filename}) ---")

            # 1. Build the graph for one document
            nx_graph, hetero_graph = self.build_graphs_for_document(doc_text, filename, doc_id=i)

            # 2. If graph building resulted in an empty graph, skip it
            if not hetero_graph.node_types:
                continue

            # 3. Validate the final HeteroData object
            if self._validate_hetero_data(hetero_graph, filename):
                valid_nx_graphs.append(nx_graph)
                valid_hetero_graphs.append(hetero_graph)

        return valid_nx_graphs, valid_hetero_graphs

    def _validate_hetero_data(self, data: HeteroData, filename: str) -> bool:
        """
        Performs integrity checks on a HeteroData object. Returns True if valid.
        """
        is_valid = True
        error_messages = []

        # Check Node-Level Integrity
        for node_type in data.node_types:
            if 'x' not in data[node_type]:
                is_valid = False
                error_messages.append(f"Node type '{node_type}' is missing feature matrix 'x'.")
                continue
            if data[node_type].num_nodes != data[node_type].x.size(0):
                is_valid = False
                error_messages.append(f"Node type '{node_type}': Mismatch between number of nodes and features.")

        # Check Edge-Level Integrity
        for edge_type in data.edge_types:
            edge_index = data[edge_type].edge_index
            src, _, dst = edge_type
            if edge_index.numel() > 0:
                if edge_index[0].max() >= data[src].num_nodes:
                    is_valid = False
                    error_messages.append(f"Edge type '{edge_type}': Source index out of bounds.")
                if edge_index[1].max() >= data[dst].num_nodes:
                    is_valid = False
                    error_messages.append(f"Edge type '{edge_type}': Destination index out of bounds.")
            if 'edge_attr' in data[edge_type] and data[edge_type].edge_index.size(1) != data[edge_type].edge_attr.size(
                    0):
                is_valid = False
                error_messages.append(f"Edge type '{edge_type}': Mismatch between number of edges and edge attributes.")

        if not is_valid:
            logging.error(f"Graph for document '{filename}' failed validation and will be skipped.")
            for msg in error_messages:
                logging.error(f"- {msg}")
        return is_valid

    def _validate_and_prune_graph(self, nx_graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
        """
        Checks all nodes against a schema, logs detailed information about any invalid
        nodes and their connections, and then prunes them to ensure graph integrity.
        """
        nodes_to_remove = []

        # First, identify all nodes that are invalid
        for node, data in nx_graph.nodes(data=True):
            node_type = data.get('type')
            if not node_type or node_type not in NODE_SCHEMA:
                nodes_to_remove.append(node)
                continue

            required_attrs = NODE_SCHEMA[node_type]
            if not required_attrs.issubset(data.keys()):
                nodes_to_remove.append(node)

        # If we found invalid nodes, log detailed context before removing them
        if nodes_to_remove:
            logging.warning(f"Found {len(nodes_to_remove)} invalid node(s) to be pruned. Logging details:")

            for node_id in nodes_to_remove:
                node_data = nx_graph.nodes[node_id]
                log_message = [f"  - Pruning Node ID: '{node_id}'"]
                log_message.append(f"    - Attributes: {node_data}")

                # Log incoming connections (predecessors)
                in_edges = list(nx_graph.in_edges(node_id, data=True))
                if in_edges:
                    log_message.append(f"    - Connected FROM ({len(in_edges)} edge(s)):")
                    for u, _, edge_data in in_edges:
                        log_message.append(
                            f"      - Node '{u}' (type: {nx_graph.nodes[u].get('type', 'N/A')}) via edge: {edge_data}")

                # Log outgoing connections (successors)
                out_edges = list(nx_graph.out_edges(node_id, data=True))
                if out_edges:
                    log_message.append(f"    - Connected TO ({len(out_edges)} edge(s)):")
                    for _, v, edge_data in out_edges:
                        log_message.append(
                            f"      - Node '{v}' (type: {nx_graph.nodes[v].get('type', 'N/A')}) via edge: {edge_data}")

                # Print the detailed multi-line log message for the node
                logging.warning("\n".join(log_message))

            # Finally, remove the nodes from the graph
            nx_graph.remove_nodes_from(nodes_to_remove)
            logging.info(f"Pruning complete. Removed {len(nodes_to_remove)} invalid node(s).")

        return nx_graph

    def build_graphs_for_document(self, doc_text: str, filename: str, doc_id: int) -> tuple[
        nx.MultiDiGraph, HeteroData]:
        """Constructs both graph representations for a single document."""
        spacy_doc = self.nlp(doc_text)
        sentences = list(spacy_doc.sents)
        nx_graph = nx.MultiDiGraph(doc_id=doc_id, text=doc_text, filename=filename)
        doc_node_id = f"doc_{doc_id}"

        content_words = [tok.text for tok in spacy_doc if not ((tok.is_stop and tok.pos_ != "PRP") or tok.is_punct)]
        unique_content_words = sorted(list(set(content_words)))
        word_embeddings_map = {}
        if unique_content_words:
            word_embeddings_map = {word: emb for word, emb in
                                   zip(unique_content_words, self._get_batch_embedding(unique_content_words))}

        max_sent_len = max((len(list(s)) for s in sentences), default=0)
        positional_encodings = get_positional_encoding(max_sent_len, self.embedding_dim)

        lemma_to_nodes = {}
        processed_sentences = []

        for sent_idx, sent in enumerate(sentences):
            potential_word_nodes = []
            contextual_embs_for_sent = []

            for token_idx_in_sent, token in enumerate(sent):
                word_node_id = f"word_{doc_id}_{token.i}"
                contextual_emb = torch.zeros(self.embedding_dim)
                is_content_word = not ((token.is_stop and token.pos_ != "PRP") or token.is_punct)

                if not is_content_word:
                    final_emb = torch.zeros(1, self.embedding_dim)
                else:
                    contextual_emb = word_embeddings_map.get(token.text, torch.zeros(self.embedding_dim))
                    pos_emb = positional_encodings[token_idx_in_sent]
                    final_emb = (contextual_emb + pos_emb).unsqueeze(0)
                    contextual_embs_for_sent.append(contextual_emb)
                    if token.lemma_ not in lemma_to_nodes:
                        lemma_to_nodes[token.lemma_] = []
                    lemma_to_nodes[token.lemma_].append(word_node_id)

                potential_word_nodes.append({
                    'id': word_node_id, 'type': 'word', 'text': token.text, 'x': final_emb,
                    'contextual_x': contextual_emb.unsqueeze(0), 'position_in_sentence': token_idx_in_sent
                })

            if len(contextual_embs_for_sent) == 0:
                continue

            word_nodes_in_sent = []
            for node_data in potential_word_nodes:
                node_id = node_data.pop('id')
                nx_graph.add_node(node_id, **node_data)
                word_nodes_in_sent.append(node_id)

            final_sent_emb = torch.mean(torch.stack(contextual_embs_for_sent), dim=0).unsqueeze(0)
            sent_node_id = f"sent_{doc_id}_{sent_idx}"
            nx_graph.add_node(sent_node_id, type='sentence', text=sent.text, x=final_sent_emb, sentence_index=sent_idx)

            for wnid in word_nodes_in_sent:
                nx_graph.add_edge(wnid, sent_node_id, type='belongs')
            processed_sentences.append({'id': sent_node_id, 'embedding': final_sent_emb})

        if not processed_sentences:
            logging.warning(f"Document '{filename}' resulted in no processable sentences. Creating empty graph.")
            nx_graph.add_node(doc_node_id, type='document', text=f"DOC_{doc_id}",
                              x=torch.zeros((1, self.embedding_dim)), filename=filename)
            return nx_graph, self.to_hetero_data(nx_graph)

        final_sentence_embeddings = torch.cat([s['embedding'] for s in processed_sentences])
        final_doc_embedding = torch.mean(final_sentence_embeddings, dim=0, keepdim=True)
        nx_graph.add_node(doc_node_id, type='document', text=f"DOC_{doc_id}", x=final_doc_embedding, filename=filename)

        for sent_data in processed_sentences:
            nx_graph.add_edge(sent_data['id'], doc_node_id, type='belongs')

        for lemma, nodes in lemma_to_nodes.items():
            if len(nodes) > 1:
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        u_context = nx_graph.nodes[nodes[i]]['contextual_x']
                        v_context = nx_graph.nodes[nodes[j]]['contextual_x']
                        sim = round(cosine_similarity(u_context, v_context).item(), 4)
                        feature = torch.tensor([sim])
                        nx_graph.add_edge(nodes[i], nodes[j], type='same_lemma', feature=feature)
                        nx_graph.add_edge(nodes[j], nodes[i], type='same_lemma', feature=feature)

        for token in spacy_doc:
            u_id = f"word_{doc_id}_{token.i}"
            if token.i + 1 < len(spacy_doc) and spacy_doc[token.i + 1].sent == token.sent:
                v_id = f"word_{doc_id}_{token.i + 1}"
                nx_graph.add_edge(u_id, v_id, type='seq', feature=torch.tensor([1.0]))
            if token.head != token and nx_graph.has_node(f"word_{doc_id}_{token.head.i}"):
                v_id = f"word_{doc_id}_{token.head.i}"
                dep_label = token.dep_
                dep_idx = self.dep_label_map.get(dep_label, self.dep_label_map['dep'])
                dep_feature = one_hot(torch.tensor(dep_idx), num_classes=len(self.dep_label_map)).float()
                nx_graph.add_edge(u_id, v_id, type='dep', feature=dep_feature)
                self.used_deps[COMMON_DEP_LABELS[dep_idx]] += 1

        sentence_node_ids = [s['id'] for s in processed_sentences]
        for i in range(len(sentence_node_ids)):
            if i + 1 < len(sentence_node_ids):
                nx_graph.add_edge(sentence_node_ids[i], sentence_node_ids[i + 1], type='seq',
                                  feature=torch.tensor([1.0]))
            for j in range(i + 1, len(sentence_node_ids)):
                sim = cosine_similarity(nx_graph.nodes[sentence_node_ids[i]]['x'],
                                        nx_graph.nodes[sentence_node_ids[j]]['x']).item()
                if sim > self.similarity_threshold:
                    nx_graph.add_edge(sentence_node_ids[i], sentence_node_ids[j], type='sim',
                                      feature=torch.tensor([sim]))

        # --- NEW: Final validation and pruning step before returning ---
        nx_graph = self._validate_and_prune_graph(nx_graph)
        hetero_data = self.to_hetero_data(nx_graph)

        logging.info(f"Final validated graph: {nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges.")
        return nx_graph, hetero_data

    def to_hetero_data(self, nx_graph: nx.MultiDiGraph) -> HeteroData:
        """
        Converts a NetworkX graph to a HeteroData object.
        This version contains the fix for the indexing bug and a final
        validation step that repairs any remaining edge inconsistencies.
        """

        data = HeteroData()

        node_types = {d['type'] for _, d in nx_graph.nodes(data=True) if 'type' in d}
        node_mappings = {}
        for ntype in node_types:
            # First, get all nodes of the current type
            nodes_of_type = [n for n, d in nx_graph.nodes(data=True) if d.get('type') == ntype]
            # Then, enumerate that filtered list to create dense indices (0 to N-1)
            node_mappings[ntype] = {node_id: i for i, node_id in enumerate(nodes_of_type)}

        for ntype, mapping in node_mappings.items():
            nodes_of_type = list(mapping.keys())
            if not nodes_of_type: continue

            # This logic assumes all nodes of a type have the same set of attributes.
            # Our new validation step helps ensure this is true.
            node_attrs_names = list(nx_graph.nodes[nodes_of_type[0]].keys())
            for attr_key in node_attrs_names:
                if attr_key == 'type' or attr_key == 'contextual_x': continue  # Skip helper attributes

                values = [nx_graph.nodes[n][attr_key] for n in nodes_of_type]

                try:
                    if isinstance(values[0], torch.Tensor):
                        # Use cat for features like 'x'
                        data[ntype][attr_key] = torch.cat(values, dim=0) if attr_key == 'x' else torch.tensor(values)
                    else:
                        data[ntype][attr_key] = values
                except (TypeError, ValueError, RuntimeError):
                    data[ntype][attr_key] = values

        for u, v, attrs in nx_graph.edges(data=True):
            src_type, dst_type = attrs.get('source_type'), attrs.get('target_type')
            if not src_type: src_type = nx_graph.nodes[u].get('type')
            if not dst_type: dst_type = nx_graph.nodes[v].get('type')

            if not src_type or not dst_type: continue

            edge_type = attrs['type']
            edge_tuple = (src_type, edge_type, dst_type)
            if u not in node_mappings[src_type] or v not in node_mappings[dst_type]: continue

            src_idx, dst_idx = node_mappings[src_type][u], node_mappings[dst_type][v]

            if 'edge_index' not in data[edge_tuple]:
                data[edge_tuple].edge_index = torch.empty((2, 0), dtype=torch.long)

            edge_index = torch.tensor([[src_idx], [dst_idx]], dtype=torch.long)
            data[edge_tuple].edge_index = torch.cat([data[edge_tuple].edge_index, edge_index], dim=1)

            if 'feature' in attrs:
                if 'edge_attr' not in data[edge_tuple]: data[edge_tuple].edge_attr = []
                data[edge_tuple].edge_attr.append(attrs['feature'])

        for store in data.edge_stores:
            if 'edge_attr' in store:
                try:
                    # Stack all feature tensors for this edge type
                    store.edge_attr = torch.stack(store.edge_attr, dim=0)
                except (TypeError, ValueError, RuntimeError) as e:
                    logging.warning(f"Could not convert edge_attr for {store} to tensor: {e}")

        # This will fix any invalid edges before the object is returned.
        data = self._fix_and_validate_heterodata(data)

        return data

    def _fix_and_validate_heterodata(self, data: HeteroData) -> HeteroData:
        """
        Checks for and removes invalid edges from a HeteroData object.
        An edge is invalid if its source or destination index is out of bounds.
        """
        for edge_type in data.edge_types:
            edge_index = data[edge_type].edge_index
            if edge_index.numel() == 0:
                continue

            src_node_type, _, dst_node_type = edge_type
            num_src_nodes = data[src_node_type].num_nodes
            num_dst_nodes = data[dst_node_type].num_nodes

            # Create a boolean mask for valid edges
            valid_mask = (edge_index[0] < num_src_nodes) & (edge_index[1] < num_dst_nodes)

            num_original_edges = edge_index.size(1)
            num_valid_edges = valid_mask.sum().item()

            if num_valid_edges < num_original_edges:
                logging.warning(
                    f"Edge type '{edge_type}': Found and removed {num_original_edges - num_valid_edges} invalid edges "
                    f"(pointing to non-existent nodes)."
                )
                # Filter the edge_index and any associated attributes
                data[edge_type].edge_index = edge_index[:, valid_mask]
                if 'edge_attr' in data[edge_type]:
                    data[edge_type].edge_attr = data[edge_type].edge_attr[valid_mask]

        return data
