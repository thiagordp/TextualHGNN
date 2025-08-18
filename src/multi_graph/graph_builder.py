# graph_builder.py
import logging
from pathlib import Path
from typing import List

import networkx as nx
import numpy as np
import pandas as pd
import spacy
import torch
from spacy.tokens import Doc, Span
from torch.nn.functional import cosine_similarity, one_hot
from torch_geometric.data import HeteroData
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

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

from spacy.language import Language

@Language.component("split_on_semicolon")
def _split_on_semicolon(doc):
    for token in doc[:-1]:
        if token.text in [";", ":"]:
            doc[token.i + 1].is_sent_start = True
    return doc


# Dicionário de gatilhos que identificamos a partir dos exemplos.
# Pode ser definido como uma constante no módulo.
CLAUSE_TRIGGERS = {
    "GERUNDS": ["considerando", "tendo", "sendo", "caber", "restringir", "comparecendo", "visando"],
    "RELATIVES": ["que", "o que", "qual seja", "cujo", "cuja"],
    "CONNECTORS": ["isso pois", "nesse sentido", "de mais a mais", "destarte", "outrossim"]
}


class DocumentGraphBuilder:
    """
    A class to build multi-level, heterogeneous graphs from textual documents.
    This version implements our final hybrid architecture design and includes
    a validation and pruning step to ensure data integrity.
    """

    def __init__(self, nlp_spacy_model, embedding_model: str, similarity_threshold: float):
        logging.info("Initializing DocumentGraphBuilder...")
        self.nlp = nlp_spacy_model

        if not self.nlp.has_pipe("split_on_semicolon"):
            self.nlp.add_pipe("split_on_semicolon", before="parser")

        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        self.model = AutoModel.from_pretrained(embedding_model)
        self.similarity_threshold = similarity_threshold
        self.dep_label_map = {label: i for i, label in enumerate(COMMON_DEP_LABELS)}
        self.used_deps = {label: 0 for label in COMMON_DEP_LABELS}
        self.embedding_dim = self.model.config.hidden_size

        config = self.model.config
        self.graph_stats = []
        logging.info("--- Embedding Model Details ---")
        logging.info(f"  - Model Type:                {config.name_or_path}")
        logging.info(f"  - Model Type:                {config.model_type}")
        logging.info(f"  - Hidden Size/Embedding Dim: {config.hidden_size}")
        logging.info(f"  - Number of Layers:          {config.num_hidden_layers}")
        logging.info(f"  - Number of Attention Heads: {config.num_attention_heads}")
        logging.info(f"  - Vocabulary Size:           {config.vocab_size}")
        logging.info("---------------------------------")

    # --- NEW: Public method to display aggregated statistics ---
    def display_graph_statistics(self, output_filepath: Path):
        """
        Calculates and logs the median and standard deviation of graph structural properties
        across all documents processed. Saves the raw data to a specified Excel file.
        """
        if not self.graph_stats:
            logging.warning("No graph statistics were collected. Cannot display.")
            return

        df = pd.DataFrame(self.graph_stats)
        logging.info("\n--- Graph Statistics Summary ---")
        # Use to_string() to ensure the table format is respected in the log
        print(df.describe().to_string())
        logging.info("--------------------------------\n")

        # Ensure the parent directory exists
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(output_filepath, index=False)
        logging.info(f"Saved detailed graph statistics to '{output_filepath}'")

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

    def _refine_sentence_splits(self, doc: Doc, max_length: int = 80) -> List[Span]:
        """
        Recebe um Doc e refina a segmentação. Quebra sentenças que excedem
        max_length usando gatilhos linguísticos. Esta é a Etapa 2 da nossa estratégia híbrida.

        Args:
            doc: O Doc processado pelo spaCy (que já passou pela quebra por ';').
            max_length: O número máximo de tokens para uma sentença ser considerada "aceitável".

        Returns:
            Uma lista de Spans, onde cada Span é uma sentença final e mais curta.
        """
        final_sentences = []
        # doc.sents aqui já contém as sentenças quebradas por ponto e vírgula
        for sent in doc.sents:
            if len(sent) <= max_length:
                final_sentences.append(sent)
                continue

            # Se a sentença ainda for muito longa, aplicamos a quebra agressiva.
            split_indices = []
            for i in range(len(sent) - 2):  # Itera até o antepenúltimo para checar locuções
                token = sent[i]

                # O padrão principal que procuramos é uma vírgula
                if token.text != ",":
                    continue

                # Após a vírgula, verificamos nossos gatilhos
                next_token = sent[i + 1]
                next_two_tokens_text = sent[i + 1: i + 3].text.lower()

                # Condição 1: É um gerúndio?
                if next_token.lemma_.lower() in CLAUSE_TRIGGERS["GERUNDS"]:
                    split_indices.append(next_token.i)  # O ponto de quebra é o próprio token

                # Condição 2: É um pronome relativo?
                elif next_token.text.lower() in CLAUSE_TRIGGERS["RELATIVES"]:
                    # Verificação extra para 'que': só quebrar se não for seguido por um verbo no infinitivo
                    # para evitar quebrar "..., que fazer"
                    if next_token.text.lower() == 'que':
                        # Verifica se há um token seguinte para evitar erro de índice
                        if (i + 2) < len(sent):
                            token_after_que = sent[i + 2]
                            # A forma correta de checar o infinitivo é via morfologia.
                            # "VerbForm=Inf" é a anotação para infinitivos.
                            # Também verificamos se o token é de fato um verbo.
                            is_infinitive = "Inf" in token_after_que.morph.get("VerbForm") and token_after_que.pos_ == 'VERB'

                            # SÓ quebramos a sentença se NÃO for um verbo no infinitivo.
                            if not is_infinitive:
                                split_indices.append(next_token.i)
                    else:
                        split_indices.append(next_token.i)

                # Condição 3: É um dos nossos conectivos lógicos de duas palavras?
                elif next_two_tokens_text in CLAUSE_TRIGGERS["CONNECTORS"]:
                    split_indices.append(sent[i + 3].i)  # Quebra após o conector

            # Se encontramos pontos de quebra, fatiamos a sentença original
            if split_indices:
                start_idx = sent.start
                for point_idx in sorted(list(set(split_indices))):  # Usa set para evitar duplicatas
                    # Garante que o ponto de quebra esteja dentro dos limites da sentença
                    if start_idx < point_idx < sent.end:
                        final_sentences.append(doc[start_idx:point_idx])
                        start_idx = point_idx
                # Adiciona o último segmento da sentença
                final_sentences.append(doc[start_idx:sent.end])
            else:
                # Se mesmo após a lógica agressiva não foi possível quebrar,
                # mantemos a sentença longa para não perder a informação.
                final_sentences.append(sent)

        return final_sentences

    def build_graphs_for_document(self, doc_text: str, filename: str, doc_id: int) -> tuple[
        nx.MultiDiGraph, HeteroData]:

        spacy_doc = self.nlp(doc_text)

        if self.nlp.meta["lang"] == "pt":
            sentences = self._refine_sentence_splits(spacy_doc, max_length=30)
        else:
            sentences = spacy_doc.sents

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

        sents_per_doc = len(sentences)
        words_per_sent = [len(list(s)) for s in sentences]

        num_word_nodes = len([n for n, d in nx_graph.nodes(data=True) if d.get('type') == 'word'])
        num_sent_nodes = len([n for n, d in nx_graph.nodes(data=True) if d.get('type') == 'sentence'])

        num_dep_edges = len([e for e in nx_graph.edges(data=True) if e[2].get('type') == 'dep'])
        num_lemma_edges = len([e for e in nx_graph.edges(data=True) if e[2].get('type') == 'same_lemma'])

        # Split sequence edges by node type
        num_word_seq_edges = len([e for e in nx_graph.edges(data=True) if
                                  e[2].get('type') == 'seq' and nx_graph.nodes[e[0]].get('type') == 'word'])
        num_sent_seq_edges = len([e for e in nx_graph.edges(data=True) if
                                  e[2].get('type') == 'seq' and nx_graph.nodes[e[0]].get('type') == 'sentence'])
        num_sent_sim_edges = len([e for e in nx_graph.edges(data=True) if e[2].get('type') == 'sim'])

        self.graph_stats.append({
            'filename': filename,
            'sentences_per_document': sents_per_doc,
            'words_per_sentence_avg': np.mean(words_per_sent) if words_per_sent else 0,
            'word_nodes': num_word_nodes,
            'sentence_nodes': num_sent_nodes,
            'dep_edges': num_dep_edges,
            'word_seq_edges': num_word_seq_edges,
            'same_lemma_edges': num_lemma_edges,
            'sent_seq_edges': num_sent_seq_edges,
            'sent_sim_edges': num_sent_sim_edges,
        })

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
