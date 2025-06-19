# graph_builder.py

import spacy
import torch
import networkx as nx
import numpy as np
import logging

from torch_geometric.data import HeteroData
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity, one_hot

# Local imports
from src.multi_graph.config import COMMON_DEP_LABELS

# --- NEW: Schemas define the required attributes for valid graph components ---
NODE_SCHEMA = {
    'word': {'type', 'text', 'x', 'contextual_x', 'position_in_sentence', 'position_in_doc'},
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

    def process_documents(self, documents: list[tuple[str, str]]) -> tuple[
        list[nx.MultiDiGraph], list[HeteroData], list[str]]:
        """
        Processes documents, validates each resulting graph, and returns lists of
        only the valid graphs and their corresponding filenames.
        """
        valid_nx_graphs, valid_hetero_graphs, valid_filenames = [], [], []

        progress_bar = tqdm(enumerate(documents), total=len(documents), desc="Building graphs")

        for i, (filename, doc_text) in progress_bar:
            progress_bar.set_postfix_str(f"File: {filename}", refresh=True)
            nx_graph, hetero_graph = self.build_graphs_for_document(doc_text, filename, doc_id=i)
            if hetero_graph.node_types:
                valid_nx_graphs.append(nx_graph)
                valid_hetero_graphs.append(hetero_graph)
                valid_filenames.append(filename)

        return valid_nx_graphs, valid_hetero_graphs, valid_filenames

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

    def _validate_and_prune_nx_graph(self, nx_graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
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

        if nodes_to_remove:
            logging.warning(f"Found {len(nodes_to_remove)} invalid node(s) to be pruned. Logging details:")
            nx_graph.remove_nodes_from(nodes_to_remove)
            logging.info(
                f"Pruning complete. Removed {len(nodes_to_remove)}/{nx_graph.number_of_nodes()} invalid node(s).")

        return nx_graph

    def _fix_and_validate_heterodata(self, data: HeteroData, filename: str) -> HeteroData:
        """
        Checks for and removes invalid edges from a HeteroData object.
        An edge is invalid if its source or destination index is out of bounds.
        """
        for edge_type in data.edge_types:
            if 'edge_index' not in data[edge_type]: continue
            edge_index = data[edge_type].edge_index
            if edge_index.numel() == 0: continue

            src_node_type, _, dst_node_type = edge_type
            num_src_nodes = data[src_node_type].num_nodes
            num_dst_nodes = data[dst_node_type].num_nodes

            # Create a boolean mask for valid edges
            valid_mask = (edge_index[0] < num_src_nodes) & (edge_index[1] < num_dst_nodes)
            num_original_edges = edge_index.size(1)
            num_valid_edges = valid_mask.sum().item()

            if num_valid_edges < num_original_edges:
                logging.warning(
                    f"In '{filename}', edge type '{edge_type}': Found and removed "
                    f"{num_original_edges - num_valid_edges} invalid edges pointing to non-existent nodes."
                )
                data[edge_type].edge_index = edge_index[:, valid_mask]
                if 'edge_attr' in data[edge_type]:
                    data[edge_type].edge_attr = data[edge_type].edge_attr[valid_mask]
        return data

    def build_graphs_for_document(self, doc_text: str, filename: str, doc_id: int) -> tuple[
        nx.MultiDiGraph, HeteroData]:
        spacy_doc = self.nlp(doc_text)
        sentences = list(spacy_doc.sents)
        nx_graph = nx.MultiDiGraph(doc_id=doc_id, text=doc_text, filename=filename)
        doc_node_id = f"doc_{doc_id}"
        content_words = [tok.text for tok in spacy_doc if not ((tok.is_stop and tok.pos_ != "PRP") or tok.is_punct)]
        unique_content_words = sorted(list(set(content_words)))
        word_embeddings_map = {word: emb for word, emb in zip(unique_content_words, self._get_batch_embedding(
            unique_content_words))} if unique_content_words else {}
        max_sent_len = max((len(list(s)) for s in sentences), default=0)
        positional_encodings = get_positional_encoding(max_sent_len, self.embedding_dim)
        lemma_to_nodes, processed_sentences = {}, []

        for sent_idx, sent in enumerate(sentences):
            potential_word_nodes, contextual_embs_for_sent = [], []
            for token_idx_in_sent, token in enumerate(sent):
                word_node_id = f"word_{doc_id}_{token.i}"
                contextual_emb = torch.zeros(self.embedding_dim)
                is_content_word = not ((token.is_stop and token.pos_ != "PRP") or token.is_punct)
                if not is_content_word:
                    final_emb = torch.zeros(1, self.embedding_dim)
                else:
                    contextual_emb = word_embeddings_map.get(token.text, torch.zeros(self.embedding_dim))
                    pos_emb = positional_encodings[
                        token_idx_in_sent] if token_idx_in_sent < max_sent_len else torch.zeros(self.embedding_dim)
                    final_emb = (contextual_emb + pos_emb).unsqueeze(0)
                    contextual_embs_for_sent.append(contextual_emb)
                    if token.lemma_ not in lemma_to_nodes: lemma_to_nodes[token.lemma_] = []
                    lemma_to_nodes[token.lemma_].append(word_node_id)
                potential_word_nodes.append({
                    'id': word_node_id, 'type': 'word', 'text': token.text, 'x': final_emb,
                    'contextual_x': contextual_emb.unsqueeze(0), 'position_in_sentence': token_idx_in_sent,
                    'position_in_doc': token.i
                })
            if not contextual_embs_for_sent: continue
            word_nodes_in_sent = []
            for node_data in potential_word_nodes:
                node_id = node_data.pop('id');
                nx_graph.add_node(node_id, **node_data);
                word_nodes_in_sent.append(node_id)
            final_sent_emb = torch.mean(torch.stack(contextual_embs_for_sent), dim=0).unsqueeze(0)
            sent_node_id = f"sent_{doc_id}_{sent_idx}"
            nx_graph.add_node(sent_node_id, type='sentence', text=sent.text, x=final_sent_emb, sentence_index=sent_idx)
            for wnid in word_nodes_in_sent: nx_graph.add_edge(wnid, sent_node_id, type='belongs')
            processed_sentences.append({'id': sent_node_id, 'embedding': final_sent_emb})

        if not processed_sentences:
            nx_graph.add_node(doc_node_id, type='document', text=f"DOC_{doc_id}",
                              x=torch.zeros((1, self.embedding_dim)), filename=filename)
            return nx_graph, self.to_hetero_data(nx_graph, filename)

        final_sentence_embeddings = torch.cat([s['embedding'] for s in processed_sentences])
        final_doc_embedding = torch.mean(final_sentence_embeddings, dim=0, keepdim=True)
        nx_graph.add_node(doc_node_id, type='document', text=f"DOC_{doc_id}", x=final_doc_embedding, filename=filename)
        for sent_data in processed_sentences: nx_graph.add_edge(sent_data['id'], doc_node_id, type='belongs')

        for lemma, nodes in lemma_to_nodes.items():
            if len(nodes) > 1:
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        u_context = nx_graph.nodes[nodes[i]]['contextual_x'];
                        v_context = nx_graph.nodes[nodes[j]]['contextual_x']
                        sim = round(cosine_similarity(u_context, v_context).item(), 4);
                        feature = torch.tensor([sim])
                        nx_graph.add_edge(nodes[i], nodes[j], type='same_lemma', feature=feature);
                        nx_graph.add_edge(nodes[j], nodes[i], type='same_lemma', feature=feature)

        for token in spacy_doc:
            u_id = f"word_{doc_id}_{token.i}"
            if not nx_graph.has_node(u_id): continue
            if token.i + 1 < len(spacy_doc) and spacy_doc[token.i + 1].sent == token.sent and nx_graph.has_node(
                    f"word_{doc_id}_{token.i + 1}"):
                v_id = f"word_{doc_id}_{token.i + 1}";
                nx_graph.add_edge(u_id, v_id, type='seq', feature=torch.tensor([1.0]))
            if token.head != token and nx_graph.has_node(f"word_{doc_id}_{token.head.i}"):
                v_id = f"word_{doc_id}_{token.head.i}"
                dep_idx = self.dep_label_map.get(token.dep_, self.dep_label_map['dep'])
                dep_feature = one_hot(torch.tensor(dep_idx), num_classes=len(self.dep_label_map)).float()
                nx_graph.add_edge(u_id, v_id, type='dep', feature=dep_feature)

        sentence_node_ids = [s['id'] for s in processed_sentences]
        for i in range(len(sentence_node_ids)):
            if i + 1 < len(sentence_node_ids): nx_graph.add_edge(sentence_node_ids[i], sentence_node_ids[i + 1],
                                                                 type='seq', feature=torch.tensor([1.0]))
            for j in range(i + 1, len(sentence_node_ids)):
                sim = cosine_similarity(nx_graph.nodes[sentence_node_ids[i]]['x'],
                                        nx_graph.nodes[sentence_node_ids[j]]['x']).item()
                if sim > self.similarity_threshold: nx_graph.add_edge(sentence_node_ids[i], sentence_node_ids[j],
                                                                      type='sim', feature=torch.tensor([sim]))

        nx_graph = self._validate_and_prune_nx_graph(nx_graph)
        return nx_graph, self.to_hetero_data(nx_graph, filename)

    def to_hetero_data(self, nx_graph: nx.MultiDiGraph, filename: str) -> HeteroData:
        """
        Converts a NetworkX graph to a HeteroData object.
        UPDATED: Now correctly attaches the node_mappings dictionary to the final object.
        """
        data = HeteroData()
        node_types = {d.get('type') for _, d in nx_graph.nodes(data=True) if 'type' in d}
        node_mappings = {}
        for ntype in node_types:
            nodes_of_type = sorted([n for n, d in nx_graph.nodes(data=True) if d.get('type') == ntype])
            node_mappings[ntype] = {node_id: i for i, node_id in enumerate(nodes_of_type)}

        data.node_mappings = node_mappings

        for ntype, mapping in node_mappings.items():
            nodes_of_type = list(mapping.keys())
            if not nodes_of_type: continue

            node_attrs_names = list(nx_graph.nodes[nodes_of_type[0]].keys())
            for attr_key in node_attrs_names:
                if attr_key == 'contextual_x': continue  # Skip helper attributes

                values = [nx_graph.nodes[n].get(attr_key) for n in nodes_of_type]
                try:
                    if isinstance(values[0], torch.Tensor):
                        data[ntype][attr_key] = torch.cat(values, dim=0) if attr_key == 'x' else torch.stack(values)
                    else:
                        data[ntype][attr_key] = values
                except (TypeError, ValueError, RuntimeError):
                    data[ntype][attr_key] = values

        # This loop copies all edge information
        for u, v, attrs in nx_graph.edges(data=True):
            src_type = nx_graph.nodes[u].get('type');
            dst_type = nx_graph.nodes[v].get('type')
            if not src_type or not dst_type: continue
            edge_type = attrs['type'];
            edge_tuple = (src_type, edge_type, dst_type)
            if u not in node_mappings[src_type] or v not in node_mappings[dst_type]: continue
            src_idx, dst_idx = node_mappings[src_type][u], node_mappings[dst_type][v]
            if 'edge_index' not in data[edge_tuple]: data[edge_tuple].edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_index = torch.tensor([[src_idx], [dst_idx]], dtype=torch.long)
            data[edge_tuple].edge_index = torch.cat([data[edge_tuple].edge_index, edge_index], dim=1)
            if 'feature' in attrs:
                if 'edge_attr' not in data[edge_tuple]: data[edge_tuple].edge_attr = []
                data[edge_tuple].edge_attr.append(attrs['feature'])

        # This loop correctly formats edge_attr tensors
        for store in data.edge_stores:
            if 'edge_attr' in store:
                try:
                    store.edge_attr = torch.stack(store.edge_attr, dim=0)
                except (TypeError, ValueError, RuntimeError) as e:
                    logging.warning(f"Could not stack edge_attr for {store}, attempting to concatenate. Error: {e}")
                    try:
                        store.edge_attr = torch.cat([attr for attr in store.edge_attr if attr.numel() > 0], dim=0)
                    except Exception as e_inner:
                        logging.error(f"Could not process edge_attr for {store} after fallback: {e_inner}")

        data = self._fix_and_validate_heterodata(data, filename)
        return data
