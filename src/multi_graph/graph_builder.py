# graph_builder.py

import spacy
import torch
import networkx as nx
import numpy as np
import logging

import tqdm
from torch_geometric.data import HeteroData
from transformers import AutoTokenizer, AutoModel
from torch.nn.functional import cosine_similarity, one_hot

# Local imports
from config import COMMON_DEP_LABELS


# (get_positional_encoding function remains the same)
def get_positional_encoding(max_len: int, d_model: int) -> torch.Tensor:
    # ... (code is unchanged)
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    if d_model % 2 != 0:
        pe[:, 1::2] = torch.cos(position * div_term)[:, :pe[:, 1::2].shape[1]]
    else:
        pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# graph_builder.py

# ... (imports and other functions remain the same) ...

class DocumentGraphBuilder:
    # ... (__init__ and _get_batch_embedding are unchanged) ...
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

    # --- CHANGE: Updated to accept a list of (filename, text) tuples ---
    def process_documents(self, documents: list[tuple[str, str]]) -> tuple[list[nx.MultiDiGraph], list[HeteroData]]:
        """Processes a list of documents and returns their graph representations."""
        nx_graphs = []
        hetero_graphs = []
        for i, (filename, doc_text) in enumerate(documents):
            print(f"--- Processing Document {i + 1}/{len(documents)} ({filename}) ---")
            nx_graph, hetero_graph = self.build_graphs_for_document(doc_text, filename, doc_id=i)
            nx_graphs.append(nx_graph)
            hetero_graphs.append(hetero_graph)
        return nx_graphs, hetero_graphs

    # --- CHANGE: Updated to accept a filename ---
    def build_graphs_for_document(self, doc_text: str, filename: str, doc_id: int) -> tuple[
        nx.MultiDiGraph, HeteroData]:
        """Constructs both graph representations for a single document."""

        def _is_not_content(t):
            ignore_pos = ["PRP", "PRON", "POS"]
            return (t.is_stop and t.pos_ not in ignore_pos) or t.is_punct

        spacy_doc = self.nlp(doc_text)
        sentences = list(spacy_doc.sents)
        nx_graph = nx.MultiDiGraph(doc_id=doc_id, text=doc_text, filename=filename)
        doc_node_id = f"doc_{doc_id}"

        # ... (rest of the build process is unchanged until document node creation) ...
        # ... (word and sentence node creation) ...
        content_words = [tok.text for tok in spacy_doc if not _is_not_content(tok)]
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
            word_nodes_in_sent = []
            contextual_embs_for_sent = []

            for token_idx_in_sent, token in enumerate(sent):
                word_node_id = f"word_{doc_id}_{token.i}"
                contextual_emb = torch.zeros(self.embedding_dim)

                if _is_not_content(token):
                    final_emb = torch.zeros(1, self.embedding_dim)
                else:
                    contextual_emb = word_embeddings_map.get(token.text, torch.zeros(self.embedding_dim))
                    pos_emb = positional_encodings[token_idx_in_sent]
                    final_emb = (contextual_emb + pos_emb).unsqueeze(0)
                    contextual_embs_for_sent.append(contextual_emb)
                    if token.lemma_ not in lemma_to_nodes:
                        lemma_to_nodes[token.lemma_] = []
                    lemma_to_nodes[token.lemma_].append(word_node_id)

                nx_graph.add_node(
                    word_node_id, type='word', text=token.text, x=final_emb,
                    contextual_x=contextual_emb.unsqueeze(0), position_in_sentence=token_idx_in_sent
                )
                word_nodes_in_sent.append(word_node_id)

            if not contextual_embs_for_sent: continue

            final_sent_emb = torch.mean(torch.stack(contextual_embs_for_sent), dim=0).unsqueeze(0)
            sent_node_id = f"sent_{doc_id}_{sent_idx}"
            nx_graph.add_node(sent_node_id, type='sentence', text=sent.text, x=final_sent_emb, sentence_index=sent_idx)

            for wnid in word_nodes_in_sent:
                nx_graph.add_edge(wnid, sent_node_id, type='belongs')
            processed_sentences.append({'id': sent_node_id, 'embedding': final_sent_emb})

        if not processed_sentences:
            logging.warning(f"Document {doc_id} resulted in no processable sentences.")
            nx_graph.add_node(doc_node_id, type='document', text=f"DOC_{doc_id}",
                              x=torch.zeros((1, self.embedding_dim)), filename=filename)
            return nx_graph, self.to_hetero_data(nx_graph)

        final_sentence_embeddings = torch.cat([s['embedding'] for s in processed_sentences])
        final_doc_embedding = torch.mean(final_sentence_embeddings, dim=0, keepdim=True)

        # --- CHANGE: Added 'filename' attribute to the document node ---
        nx_graph.add_node(doc_node_id, type='document', text=f"DOC_{doc_id}", x=final_doc_embedding, filename=filename)

        # ... (rest of the function is unchanged) ...
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

        logging.info(f"Constructed graph: {nx_graph.number_of_nodes()} nodes, {nx_graph.number_of_edges()} edges.")
        return nx_graph, self.to_hetero_data(nx_graph)

    # (to_hetero_data method is unchanged, it will automatically handle the new `filename` attribute)
    def to_hetero_data(self, nx_graph: nx.MultiDiGraph) -> HeteroData:
        # ... (code is unchanged)
        data = HeteroData()
        node_types = {d['type'] for _, d in nx_graph.nodes(data=True) if 'type' in d}
        node_mappings = {
            ntype: {n: i for i, (n, d) in enumerate(nx_graph.nodes(data=True)) if d.get('type') == ntype}
            for ntype in node_types
        }

        for ntype, mapping in node_mappings.items():
            nodes_of_type = list(mapping.keys())
            if not nodes_of_type: continue

            if 'x' in nx_graph.nodes[nodes_of_type[0]]:
                data[ntype].x = torch.cat([nx_graph.nodes[n]['x'] for n in nodes_of_type], dim=0)

            other_attrs = {k for k in nx_graph.nodes[nodes_of_type[0]].keys() if k not in ['type', 'x', 'contextual_x']}
            for attr in other_attrs:
                values = [nx_graph.nodes[n][attr] for n in nodes_of_type]
                data[ntype][attr] = values

        for u, v, attrs in nx_graph.edges(data=True):
            src_type, dst_type = nx_graph.nodes[u].get('type'), nx_graph.nodes[v].get('type')
            if not src_type or not dst_type: continue

            edge_type = attrs['type']
            edge_tuple = (src_type, edge_type, dst_type)
            if u not in node_mappings[src_type] or v not in node_mappings[dst_type]: continue

            src_idx, dst_idx = node_mappings[src_type][u], node_mappings[dst_type][v]

            if 'edge_index' not in data[edge_tuple]:
                data[edge_tuple].edge_index = torch.empty((2, 0), dtype=torch.long)

            edge_index = torch.tensor([[src_idx], [dst_idx]], dtype=torch.long)
            data[edge_tuple].edge_index = torch.cat([data[edge_tuple].edge_index, edge_index], dim=1)

            for attr_key, attr_val in attrs.items():
                if attr_key == 'type': continue
                if attr_key not in data[edge_tuple]: data[edge_tuple][attr_key] = []
                data[edge_tuple][attr_key].append(attr_val)

        for store in data.edge_stores:
            key = 'edge_attr'  # Our static features are all named 'feature'
            if key in store:
                value = store.pop('feature')
                try:
                    store[key] = torch.stack(value, dim=0)
                except (TypeError, ValueError, RuntimeError) as e:
                    logging.warning(f"Could not convert edge attribute '{key}' to tensor: {e}")
                    # Fallback for mixed-shape tensors (like one-hot vs scalars)
                    if isinstance(value, list):
                        store[key] = value
        return data