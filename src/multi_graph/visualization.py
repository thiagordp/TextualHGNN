# visualization.py

import logging
import torch
import networkx as nx
import matplotlib.colors as mcolors
from pyvis.network import Network
from torch_geometric.data import HeteroData

import logging
import torch
import networkx as nx
import matplotlib.colors as mcolors
from pyvis.network import Network
from src.models.graph_explainability.concept_grounding import ConceptGrounding


def visualize_structural_graph(nx_graph: nx.MultiDiGraph, dep_label_map: dict, output_filename: str):
    """
    Creates an interactive visualization of the initial graph structure,
    styling edges based on their pre-computed static features.
    """
    # logging.info(f"Creating structural graph visualization, saving to {output_filename}...")

    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", cdn_resources='remote')

    node_color_map = {'document': '#d62828', 'sentence': '#003049', 'word': '#f77f00', 'unknown': '#e0e0e0'}
    edge_color_map = {'belongs': '#2a9d8f', 'seq': '#e76f51', 'dep': '#8d99ae', 'same_lemma': '#6a4c93',
                      'sim': '#118ab2'}
    rev_dep_map = {v: k for k, v in dep_label_map.items()}

    for node, data in nx_graph.nodes(data=True):
        node_type = data.get('type', 'unknown');
        group = node_type
        label = data.get('text', 'N/A')

        if node_type == 'sentence':
            title = f"ID: {node}\nType: Sentence\nText: {label}"
            label = f"SENT_{data.get('sentence_index', '?')}"
        elif node_type == 'word':
            title = f"ID: {node}\nType: Word\nPos: {data.get('position_in_sentence', '?')}\nText: {label}"
        else:  # Document or Unknown
            title = f"ID: {node}\nType: {node_type.capitalize()}"

        size = 30 if node_type == 'document' else 20 if node_type == 'sentence' else 10 + len(str(label))
        net.add_node(node, label=str(label), title=title, color=node_color_map.get(node_type, 'grey'), size=size,
                     group=group)

    for u, v, data in nx_graph.edges(data=True):
        edge_type = data.get('type');
        group = edge_type
        if not edge_type: continue

        width = 1.5;
        color = edge_color_map.get(edge_type, 'grey')
        title = f"Type: {edge_type.capitalize()}"

        if 'feature' in data:
            feature_val = data['feature']
            if torch.is_tensor(feature_val) and feature_val.numel() == 1:  # Scalar features
                width = 1 + (feature_val.item() * 4.0)
                title += f"\nFeature: {feature_val.item():.4f}"
            elif edge_type == 'dep':  # Vector feature
                width = 1
                dep_idx = torch.argmax(feature_val).item()
                dep_label = rev_dep_map.get(dep_idx, 'UNKNOWN')
                title = f"Type: Dependency\nLabel: {dep_label}"

        net.add_edge(u, v, title=title, color=color, width=width, group=group)

    net.toggle_physics(True)
    net.show_buttons(filter_=['nodes', 'edges'])
    net.save_graph(output_filename)
    # logging.info(f"Successfully saved structural graph to {output_filename}.")


def visualize_learned_attentions(nx_graph: nx.MultiDiGraph, explanation: dict, dep_label_map: dict,
                                 output_filename: str):
    """
    Injects learned attention weights into a NetworkX graph and creates an
    interactive visualization, styling edges based on attention.
    """
    # logging.info(f"Creating learned attention visualization for {explanation['doc_id']}...")

    # 1. Create a copy to avoid modifying the original graph object
    enriched_graph = nx_graph.copy()

    # 2. Create Mappings to look up attention weights

    # print(f"Keys inside explanation: {explanation.keys()}")
    # print(f"Keys inside explanation['graph_data']: {explanation['graph_data'].keys()}")

    node_mappings = explanation['node_mappings']

    word_map_rev = {v: k for k, v in node_mappings['word'].items()}
    sent_map_rev = {v: k for k, v in node_mappings['sentence'].items()}
    doc_map_rev = {v: k for k, v in node_mappings['document'].items()}

    word_att_edge_index, word_att_weights = explanation['word_to_sent_att']
    word_att_map = {(word_map_rev.get(u), sent_map_rev.get(v)): w.item() for u, v, w in
                    zip(word_att_edge_index[0].tolist(), word_att_edge_index[1].tolist(), word_att_weights)}

    sent_att_edge_index, sent_att_weights = explanation['sent_to_doc_att']
    sent_att_map = {(sent_map_rev.get(u), doc_map_rev.get(v)): w.item() for u, v, w in
                    zip(sent_att_edge_index[0].tolist(), sent_att_edge_index[1].tolist(), sent_att_weights)}

    # 3. Inject the learned attention scores into the graph copy
    for u, v, key in enriched_graph.edges(keys=True):
        if enriched_graph.edges[u, v, key].get('type') == 'belongs':
            attention_score = word_att_map.get((u, v)) or sent_att_map.get((u, v))
            if attention_score is not None:
                enriched_graph.edges[u, v, key]['learned_attention'] = attention_score

    # 4. Setup Pyvis Network and Styling
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", cdn_resources='remote')
    node_color_map = {'document': '#d62828', 'sentence': '#003049', 'word': '#f77f00'}
    static_edge_color_map = {'seq': '#e76f51', 'dep': '#8d99ae', 'same_lemma': '#6a4c93', 'sim': '#118ab2'}
    cmap = mcolors.LinearSegmentedColormap.from_list("attention_cmap",
                                                     ["#e63946", "#adb5bd", "#2a9d8f"])  # Red -> Grey -> Green

    # 5. Add Nodes to Pyvis Graph
    for node, data in enriched_graph.nodes(data=True):
        node_type = data.get('type', 'unknown');
        group = node_type;
        label = data.get('text', 'N/A')
        if node_type == 'sentence':
            title = f"ID: {node}\nType: Sentence\nText: {label}"
            label = f"SENT_{data.get('sentence_index', '?')}"
        elif node_type == 'word':
            title = f"ID: {node}\nType: Word\nPos: {data.get('position_in_sentence', '?')}\nText: {label}"
        else:
            title = f"ID: {node}\nType: {node_type.capitalize()}"
        size = 30 if node_type == 'document' else 20 if node_type == 'sentence' else 10 + len(str(label))
        net.add_node(node, label=str(label), title=title, color=node_color_map.get(node_type, 'grey'), size=size,
                     group=group)

    # 6. Add Edges to Pyvis Graph, with conditional styling
    for u, v, data in enriched_graph.edges(data=True):
        edge_type = data.get('type');
        group = edge_type
        if not edge_type: continue

        attention_score = data.get('learned_attention')
        if attention_score is not None:
            # Style hierarchical edges based on learned attention
            magnitude = abs(attention_score)
            color_val = (attention_score + 1) / 2.0
            color_hex = mcolors.to_hex(cmap(color_val))
            width = 1 + magnitude * 8
            title = f"Type: Belongs\nLearned Attention: {attention_score:+.4f}"
            net.add_edge(u, v, title=title, color=color_hex, width=width, group=group)
        else:
            # Style static edges based on pre-computed features
            width = 1.5
            color = static_edge_color_map.get(edge_type, 'grey')
            title = f"Type: {edge_type.capitalize()}"
            net.add_edge(u, v, title=title, width=width, group=group, color=color)

    net.toggle_physics(True)
    net.show_buttons(filter_=['nodes', 'edges'])
    net.save_graph(output_filename)
    # logging.info(f"Successfully saved learned attention graph to {output_filename}.")


def visualize_diffpool_explanation(cg: ConceptGrounding, nx_graph: nx.MultiDiGraph, output_filename: str,
                                   assignment_threshold: float = 0.05):
    """
    Creates an interactive visualization of a DiffPool explanation.

    UPDATED:
    - Only shows assignment edges with a weight > assignment_threshold.
    """
    logging.info(f"Creating DiffPool explanation visualization for {cg.data_sample_id}...")

    # --- 1. Setup Pyvis Network and Styling ---
    net = Network(height="900px", width="100%", bgcolor="#222222", font_color="white", cdn_resources='remote')
    node_color_map = {'document': '#d62828', 'L2': '#fca311', 'L1': '#003049', 'word': '#f77f00'}
    edge_color_map = {'belongs': '#2a9d8f', 'seq': '#e76f51', 'dep': '#8d99ae'}

    # --- 2. Add Nodes for All Levels ---
    # This logic assumes 'node_map' exists on the graph nodes from the data loader
    l0_id_to_idx_map = {node_id: data['node_map'] for node_id, data in nx_graph.nodes(data=True) if 'node_map' in data}

    for node_id, data in nx_graph.nodes(data=True):
        word_text = data.get('text', 'N/A')
        net.add_node(node_id, label=word_text, title=f"L0: {word_text}",
                     color=node_color_map['word'], size=12, group='word')

    # Level 1 nodes with Concept Grounding labels
    num_l1_nodes = cg.x_l1.shape[0]
    for i in range(num_l1_nodes):
        node_id = f"L1_{i}"
        l1_explanation = cg.explanation.get('explanation', {}).get(str(i))
        display_label = f"L1_{i}"
        tooltip_title = f"Abstract Node L1_{i}"
        if l1_explanation:
            cg_methods = l1_explanation.get('words_cg_methods', {})
            display_label = cg_methods.get('llm_search', [f"L1_{i}"])[0]
            tooltip_title = f"Abstract Node L1_{i}\n\n--- Grounded Concepts ---\n"
            tooltip_title += f"LLM-based: {cg_methods.get('llm_search')}\n"
            tooltip_title += f"Top L0 Word: {cg_methods.get('top_l0')}\n"
            tooltip_title += f"--- Assigned Words ---\n{l1_explanation.get('words_l0', [])}"
        net.add_node(node_id, label=display_label, title=tooltip_title,
                     color=node_color_map['L1'], size=25, group='L1')

    # Level 2 and Document Nodes
    num_l2_nodes = cg.x_l2.shape[0]
    for i in range(num_l2_nodes):
        node_id = f"L2_{i}"
        net.add_node(node_id, label=f"L2_{i}", title=f"Abstract Node L2_{i}",
                     color=node_color_map['L2'], size=35, group='L2')
        net.add_edge(node_id, "DOC", color=edge_color_map['belongs'], width=2, group='belongs')

    net.add_node("DOC", label="Document", title="Final Graph Representation",
                 color=node_color_map['document'], size=45, group='document')

    # --- 3. Add Edges with Updated Thresholding Logic ---
    l0_idx_to_id_map = {v: k for k, v in l0_id_to_idx_map.items()}

    # L0 -> L1 assignment edges
    assignment_matrix_s01 = cg.s01
    for l0_idx in range(assignment_matrix_s01.shape[0]):
        for l1_idx in range(assignment_matrix_s01.shape[1]):
            weight = assignment_matrix_s01[l0_idx, l1_idx].item()
            # --- CHANGE: Use the configurable assignment_threshold ---
            if weight > assignment_threshold:
                l0_node_id, l1_node_id = l0_idx_to_id_map.get(l0_idx), f"L1_{l1_idx}"
                if l0_node_id:
                    net.add_edge(l0_node_id, l1_node_id,
                                 title=f"Belongs (Assignment: {weight:.4f})",
                                 width=0.5 + weight * 6,
                                 opacity=0.4 + weight * 0.6,
                                 color=edge_color_map['belongs'],
                                 group='belongs')

    # L1 -> L2 assignment edges
    assignment_matrix_s12 = cg.s12
    for l1_idx in range(assignment_matrix_s12.shape[0]):
        for l2_idx in range(assignment_matrix_s12.shape[1]):
            weight = assignment_matrix_s12[l1_idx, l2_idx].item()
            # --- CHANGE: Use the configurable assignment_threshold ---
            if weight > assignment_threshold:
                net.add_edge(f"L1_{l1_idx}", f"L2_{l2_idx}",
                             title=f"Belongs (Assignment: {weight:.4f})",
                             width=1 + weight * 5,
                             color=edge_color_map['belongs'],
                             opacity=0.7,
                             group='belongs')

    # Original word-word edges for context
    for u, v, data in nx_graph.edges(data=True):
        edge_type = data.get('type')
        if edge_type in ['dep', 'seq']:
            net.add_edge(u, v, color=edge_color_map.get(edge_type, 'grey'), width=0.5, dashes=True,
                         title=f"Type: {edge_type}", group=edge_type)

    net.toggle_physics(True)
    net.show_buttons(filter_=True)
    net.save_graph(output_filename)

    logging.info(f"Successfully saved DiffPool explanation graph to {output_filename}.")


def visualization_l0(nx_graph: nx.MultiDiGraph, output_filename: str) -> None:
    from pyvis.network import Network

    net = Network(
        height="900px",
        width="100%",
        bgcolor="#222222",
        font_color="white",
        directed=True,        # <-- mark graph as directed
        cdn_resources='remote'
    )

    node_color_map = {'document': '#d62828', 'L2': '#fca311', 'L1': '#003049', 'word': '#f77f00'}
    edge_color_map = {'belongs': '#2a9d8f', 'seq': '#e76f51', 'dep': '#8d99ae', 'sent_seq': '#e76f51'}

    # Add nodes
    for node_id, data in nx_graph.nodes(data=True):
        node_label = data.get('token', data.get('text', node_id))
        node_pos   = data.get('pos', '')
        net.add_node(
            n_id=node_id,
            label=node_label,
            title=node_pos,
            color=node_color_map.get('word', '#f77f00'),
            size=12
        )

    # Determine max weight for scaling widths
    max_weight = 0.0
    for u, v, key, data in nx_graph.edges(keys=True, data=True):
        w = data.get('weight', 0.0)
        if w > max_weight:
            max_weight = w

    # Add edges (with arrows)
    for u, v, key, data in nx_graph.edges(keys=True, data=True):
        edge_type = data.get('label', data.get('dep', ''))
        weight    = data.get('weight', 0.0)

        if max_weight > 0:
            width = (weight / max_weight) * 9 + 1  # width range [1,10]
        else:
            width = 1

        net.add_edge(
            source=u,
            to=v,
            color=edge_color_map.get(edge_type, 'grey'),
            width=width,
            title=f"Type: {edge_type}\nWeight: {weight:.3f}",
            arrowStrikethrough=False  # <-- makes arrow head clean
        )

    net.toggle_physics(True)
    net.show_buttons(filter_=True)
    net.save_graph(output_filename)

