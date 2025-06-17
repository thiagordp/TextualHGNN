# visualization.py

import logging

import networkx as nx
import numpy as np
import torch
from pyvis.network import Network
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


def visualize_explanatory_graph(nx_graph: nx.MultiDiGraph):
    """
    Draws a high-quality, explanatory graph.
    UPDATED: Visualizes static edge features instead of pre-computed attention.
    """
    fig, ax = plt.subplots(figsize=(70, 30))
    pos = {}

    # --- 1. Define Color and Style Maps (No Change) ---
    node_color_map = {'document': '#d62828', 'sentence': '#003049', 'word': '#f77f00'}
    edge_color_map = {
        'belongs': '#2a9d8f', 'seq': '#e76f51', 'dep': '#8d99ae',
        'same_lemma': '#6a4c93', 'sim': '#118ab2'
    }

    # --- 2-3. Positioning Logic (No Change) ---
    # (The existing logic for node positioning remains effective)
    nodes_by_type = {'document': [], 'sentence': [], 'word': []}
    sent_to_words = {}
    for node, data in nx_graph.nodes(data=True):
        nodes_by_type[data['type']].append(node)
        if data['type'] == 'sentence':
            sent_to_words[node] = []
    for u, v, data in nx_graph.edges(data=True):
        if data.get('type') == 'belongs' and nx_graph.nodes[u]['type'] == 'word':
            sent_to_words[v].append(u)
    y_coords = {'document': 1.0, 'sentence': 0.0, 'word': -1.0}
    pos[nodes_by_type['document'][0]] = (0.5, y_coords['document'])
    sorted_sents = sorted(nodes_by_type['sentence'], key=lambda n: nx_graph.nodes[n]['sentence_index'])
    num_sentences = len(sorted_sents)
    total_plot_width = 0.9
    left_margin = (1 - total_plot_width) / 2
    if num_sentences > 1:
        group_space_ratio = 0.9
        total_group_width = total_plot_width * group_space_ratio
        total_gap_width = total_plot_width * (1 - group_space_ratio)
        single_group_width = total_group_width / num_sentences
        single_gap_width = total_gap_width / (num_sentences - 1)
    else:
        single_group_width = total_plot_width
        single_gap_width = 0
    current_x = left_margin
    for sent_node in sorted_sents:
        x_start = current_x
        x_end = x_start + single_group_width
        sent_x = (x_start + x_end) / 2
        pos[sent_node] = (sent_x, y_coords['sentence'])
        words_in_sent = sorted(sent_to_words[sent_node], key=lambda n: nx_graph.nodes[n]['position_in_doc'])
        num_words = len(words_in_sent)
        padding = 0.1 * single_group_width
        word_x_coords = np.linspace(x_start + padding, x_end - padding, num_words) if num_words > 1 else [sent_x]
        y_base = y_coords['word']
        y_stagger_offset = 0.2
        for j, word_node in enumerate(words_in_sent):
            y_pos = y_base if j % 2 == 0 else y_base - y_stagger_offset
            pos[word_node] = (word_x_coords[j], y_pos)
        rect = Rectangle((x_start, y_coords['word'] - y_stagger_offset - 0.1),
                         width=single_group_width, height=0.4 + y_stagger_offset,
                         facecolor='#f8f9fa', edgecolor='#ced4da', linestyle='--', alpha=0.6, zorder=0)
        ax.add_patch(rect)
        current_x = x_end + single_gap_width

    # --- 4. Labels and Sizes (No Change) ---
    labels, node_colors, node_sizes = {}, [], []
    for node, data in nx_graph.nodes(data=True):
        node_colors.append(node_color_map[data['type']])
        text_label = data['text']
        if data['type'] == 'word':
            labels[node] = text_label
            node_sizes.append(1000 + len(text_label) * 250)
        elif data['type'] == 'sentence':
            labels[node] = f"SENT_{data['sentence_index']}"
            node_sizes.append(5000)
        else:
            labels[node] = text_label
            node_sizes.append(6000)

    # --- 5. Draw Graph Elements (UPDATED LOGIC) ---
    nx.draw_networkx_nodes(nx_graph, pos, node_color=node_colors, node_size=node_sizes, alpha=1.0, ax=ax)
    nx.draw_networkx_labels(nx_graph, pos, labels=labels, font_size=9, font_weight='bold', font_color='white', ax=ax)

    for u, v, data in nx_graph.edges(data=True):
        edge_type = data['type']
        edge_color = edge_color_map.get(edge_type, '#b7b7a4')

        # --- CHANGE: Updated edge style logic ---
        # 'belongs' edges are now fixed style, as their weight is learned.
        if edge_type == 'belongs':
            alpha = 0.9
            lw = 2.5
        # For other edges, use the static feature to determine style.
        else:
            alpha = 0.7
            # Use the scalar feature for line width if it exists
            feature_val = data.get('feature', torch.tensor([1.0])).item()
            lw = 1.0 + (feature_val * 3.0)  # Scale feature (0-1) to a visible width

        rad = 0.15
        if hash(u) > hash(v): rad = -rad
        if nx_graph.nodes[u]['type'] == nx_graph.nodes[v]['type']: rad = 0.2

        ax.annotate("", xy=pos[v], xycoords='data', xytext=pos[u], textcoords='data',
                    arrowprops=dict(arrowstyle="->", color=edge_color, alpha=alpha, lw=lw,
                                    shrinkA=35, shrinkB=35, patchA=None, patchB=None,
                                    connectionstyle=f"arc3,rad={rad}"))

    # --- 6. Create Legend (UPDATED LOGIC) ---
    legend_handles = []
    for node_type, color in node_color_map.items():
        legend_handles.append(Line2D([0], [0], marker='o', color='w', label=f'Node: {node_type.capitalize()}',
                                     markersize=15, markerfacecolor=color))
    for edge_type, color in edge_color_map.items():
        # --- CHANGE: Removed mention of attention ---
        label = f'Edge: {edge_type.capitalize()}'
        if edge_type in ['sim', 'same_lemma']: label += ' (Width = Feature)'
        legend_handles.append(Line2D([0], [0], color=color, lw=4, label=label))

    ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, -0.02),
              fancybox=True, shadow=True, ncol=4, fontsize=14)
    ax.set_title(f"Explanatory Graph Structure (Doc ID: {nx_graph.graph.get('doc_id', 'N/A')})", size=28, pad=20)
    plt.subplots_adjust(bottom=0.1, top=0.95)
    plt.axis('off')
    plt.show()

# (visualize_interactive_graph code is unchanged from the last correct version)

def visualize_interactive_graph(nx_graph: nx.MultiDiGraph, dep_label_map: dict,
                                output_filename: str = "interactive_graph.html"):
    """
    Creates a beautiful, interactive, physics-based graph visualization.

    FINAL ROBUST VERSION:
    - Uses direct pyvis methods to reliably generate the filter UI.
    - Sacrifices some advanced physics tuning for stability.
    """
    logging.info(f"Creating robust interactive graph, saving to {output_filename}...")

    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white",
                  notebook=False, cdn_resources='remote')

    node_color_map = {'document': '#d62828', 'sentence': '#003049', 'word': '#f77f00'}
    edge_color_map = {
        'belongs': '#2a9d8f', 'seq': '#e76f51', 'dep': '#8d99ae',
        'same_lemma': '#6a4c93', 'sim': '#118ab2'
    }
    rev_dep_map = {v: k for k, v in dep_label_map.items()}

    # Add nodes and assign them to a 'group' for filtering
    for node, data in nx_graph.nodes(data=True):
        node_type, label = data['type'], data['text']
        group = node_type
        if node_type == 'sentence':
            sent_idx = data.get('sentence_index', 'N/A')
            label = f"SENT_{sent_idx}"
            title = f"ID: {node}\nType: Sentence\nText: {data['text']}"
        elif node_type == 'word':
            word_pos = data.get('position_in_sentence', 'N/A')
            title = f"ID: {node}\nType: Word\nPosition in Sent: {word_pos}\nText: {label}"
        else:
            title = f"ID: {node}\nType: Document"
        size = 30 if node_type == 'document' else 20 if node_type == 'sentence' else 10 + len(str(label))
        net.add_node(node, label=str(label), title=title, color=node_color_map.get(node_type, 'grey'), size=size,
                     group=group)

    # Add edges and assign them to a 'group' for filtering
    for u, v, data in nx_graph.edges(data=True):
        edge_type = data['type']
        group = edge_type
        if edge_type == 'belongs':
            width = 4
            title = "Type: Belongs (Learned Weight)"
        elif edge_type == 'dep':
            width = 1
            feature_vec = data.get('feature')
            if feature_vec is not None:
                dep_idx = torch.argmax(feature_vec).item()
                dep_label = rev_dep_map.get(dep_idx, 'UNKNOWN')
                title = f"Type: Dependency\nLabel: {dep_label}"
            else:
                title = "Type: Dependency"
        else:
            feature_val = data.get('feature', torch.tensor([1.0]))
            scalar_feature = feature_val.item()
            width = 1 + (scalar_feature * 4.0)
            title = f"Type: {edge_type.capitalize()}\nFeature: {scalar_feature:.4f}"
        net.add_edge(u, v, title=title, color=edge_color_map.get(edge_type, 'grey'), width=width, group=group)

    # --- CHANGE: Using direct, robust methods to enable UI ---
    # 1. Enable physics
    net.toggle_physics(True)

    # 2. Explicitly request the filter UI for nodes and edges
    net.show_buttons(filter_=['nodes', 'edges'])

    try:
        net.save_graph(output_filename)
        logging.info("Successfully saved interactive graph.")
    except Exception as e:
        logging.error(f"Could not save interactive graph: {e}")