# main_stf_preprocessing.py
import logging
import os
from datetime import datetime
import random
from pathlib import Path
import pickle
import gc  # Import the garbage collector

import numpy as np
import spacy
import torch
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from spacy import Language
from torch_geometric.loader import DataLoader
import nltk
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from src.multi_graph.config import ModelConfig, TrainingConfig, DataConfig, VisualizationConfig
from src.multi_graph.data_preprocessing import preprocessing_legal_pt
from src.multi_graph.engine import train, test
from src.multi_graph.explainability import SubgraphExplainer
from src.multi_graph.graph_builder import DocumentGraphBuilder
from src.multi_graph.model import ExplainableHierarchicalGNN
from src.multi_graph.visualization import visualize_learned_attentions, visualize_structural_graph

# --- Setup ---
os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)

timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
script_name = Path(__file__).stem
log_filename = f"logs/{script_name}_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
    force=True
)
logging.info("Logging setup complete.")


def load_documents_from_disk(base_path_str: str, class_map: dict, samples_per_class: int) -> dict:
    base_path = Path(base_path_str)
    docs_by_class = {}
    logging.info("\n--- Loading Documents From Disk ---")
    for class_name, sub_dir in class_map.items():
        class_path = base_path / sub_dir
        if not class_path.exists():
            logging.warning(f"Directory not found: {class_path}. Skipping class '{class_name}'.")
            continue
        logging.info(f"Loading from: {class_path}")
        file_paths = sorted(list(class_path.glob("*.txt")))
        docs_by_class[class_name] = []
        for file_path in tqdm.tqdm(file_paths[:samples_per_class], desc=f"Loading '{class_name}' files"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                docs_by_class[class_name].append((file_path.name, content))
            except Exception as e:
                logging.error(f"Could not read file {file_path}: {e}")
    return docs_by_class


@Language.component("set_custom_boundaries_advanced")
def set_custom_boundaries_advanced(doc):
    for token in doc[:-1]:
        if token.text in [";", ":"]:
            doc[token.i + 1].is_sent_start = True
    return doc


def generate_attention_visualizations(
        explanations: list,
        nx_graphs: list,
        class_names: list,
        viz_cfg: VisualizationConfig,
        data_cfg: DataConfig,
        model_param_str: str,
        run_timestamp: str,
        builder: DocumentGraphBuilder
):
    """Generates and saves interactive visualizations of learned attention."""
    if not explanations:
        logging.warning("No explanations were generated, skipping visualization.")
        return

    logging.info(f"\n--- Generating {len(explanations)} Explainability Visualizations ---")

    attention_viz_dir = Path(
        viz_cfg.VISUALIZATION_FOLDER) / data_cfg.DATASET_NAME / "attention" / f"run_{run_timestamp}_{model_param_str}"
    attention_viz_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Attention visualizations for this run will be saved to '{attention_viz_dir}'")

    nx_graph_map_for_viz = {g.graph['filename']: g for g in nx_graphs}

    for explanation in tqdm.tqdm(explanations, desc="Generating Explainability Visualizations"):
        doc_filename = explanation['doc_id']
        original_nx_graph = nx_graph_map_for_viz.get(doc_filename)

        if not original_nx_graph:
            logging.warning(f"Could not find original NetworkX graph for doc_id: {doc_filename} to visualize.")
            continue

        pred_class = class_names[explanation['prediction']]
        actual_class = class_names[explanation['actual']]
        is_correct_str = "CORRECT" if pred_class == actual_class else "WRONG"

        viz_filename = (
            f"{Path(doc_filename).stem}_"
            f"PRED_{pred_class}_ACTUAL_{actual_class}_{is_correct_str}.html"
        )
        output_html_path = attention_viz_dir / viz_filename

        visualize_learned_attentions(
            nx_graph=original_nx_graph,
            explanation=explanation,
            dep_label_map=builder.dep_label_map,
            output_filename=str(output_html_path)
        )


def _create_param_string(data_cfg: DataConfig, model_cfg: ModelConfig = None, train_cfg: TrainingConfig = None) -> str:
    """Creates a standardized string of hyperparameters for filenames."""
    embedding_name = data_cfg.EMBEDDING_MODEL.split('/')[-1].replace('-', '_')
    parts = [embedding_name]
    if model_cfg:
        parts.append(f"H{model_cfg.HIDDEN_CHANNELS}")
    if train_cfg:
        parts.append(f"LR{train_cfg.LEARNING_RATE}")
        parts.append(f"EW{train_cfg.ENTROPY_WEIGHT}")
    return "_".join(parts)


def log_results_to_csv(filepath: Path, metrics: dict, data_cfg: DataConfig, model_cfg: ModelConfig,
                       train_cfg: TrainingConfig):
    """Appends the results of a training run to a CSV file, flattening nested metric dictionaries."""

    # Start with the basic run info
    results_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'dataset_name': data_cfg.DATASET_NAME,
        'embedding_model': data_cfg.EMBEDDING_MODEL,
        'similarity_threshold': data_cfg.SIMILARITY_THRESHOLD,
        'hidden_channels': model_cfg.HIDDEN_CHANNELS,
        'learning_rate': train_cfg.LEARNING_RATE,
        'entropy_weight': train_cfg.ENTROPY_WEIGHT,
        'epochs': train_cfg.EPOCHS,
        'batch_size': train_cfg.BATCH_SIZE,
        'patience': train_cfg.PATIENCE,
        'roc_auc_score': metrics.get('roc_auc_score')
    }

    # Flatten the classification report for detailed logging
    report = metrics.get('classification_report', {})
    for key, value in report.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                # Sanitize keys for CSV header (e.g., 'macro avg' -> 'macro_avg')
                clean_key = key.replace(' ', '_')
                results_data[f"{clean_key}_{sub_key}"] = sub_value
        else:
            results_data[key] = value

    df = pd.DataFrame([results_data])

    file_exists = filepath.exists()
    df.to_csv(filepath, mode='a', header=not file_exists, index=False)
    logging.info(f"Logged training results to '{filepath}'")


def log_model_summary(model: torch.nn.Module):
    """Logs the total number of trainable parameters in a model."""
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Model Initialized. Trainable Parameters: {trainable_params:,}")


def save_detailed_predictions_to_csv(filepath: Path, detailed_results: list):
    """Saves the detailed per-prediction results to a CSV file."""
    if not detailed_results:
        logging.warning("No detailed prediction results to save.")
        return

    df = pd.DataFrame(detailed_results)
    df.to_excel(filepath, index=False)
    logging.info(f"Saved detailed prediction results to '{filepath}'")


def inject_learned_attentions(nx_graph: nx.MultiDiGraph, explanation: dict) -> nx.MultiDiGraph:
    """Helper function to inject learned attention weights into a NetworkX graph."""
    enriched_graph = nx_graph.copy()
    node_mappings = explanation['graph_data']['node_mappings']
    word_map_rev = {v: k for k, v in node_mappings['word'].items()}
    sent_map_rev = {v: k for k, v in node_mappings['sentence'].items()}
    doc_map_rev = {v: k for k, v in node_mappings['document'].items()}

    word_att_edge_index, word_att_weights = explanation['word_to_sent_att']
    word_att_map = {(word_map_rev.get(u), sent_map_rev.get(v)): w.item() for u, v, w in
                    zip(word_att_edge_index[0].tolist(), word_att_edge_index[1].tolist(), word_att_weights)}

    sent_att_edge_index, sent_att_weights = explanation['sent_to_doc_att']
    sent_att_map = {(sent_map_rev.get(u), doc_map_rev.get(v)): w.item() for u, v, w in
                    zip(sent_att_edge_index[0].tolist(), sent_att_edge_index[1].tolist(), sent_att_weights)}

    for u, v, key in enriched_graph.edges(keys=True):
        if enriched_graph.edges[u, v, key].get('type') == 'belongs':
            attention_score = word_att_map.get((u, v)) or sent_att_map.get((u, v))
            if attention_score is not None:
                enriched_graph.edges[u, v, key]['learned_attention'] = attention_score
    return enriched_graph


def preprocess_and_save_raw_files(
        base_path_str: str,
        interim_path_str: str,
        class_map: dict,
        samples_per_class: int
) -> dict:
    """
    Checks if preprocessed text exists. If not, it loads raw text from base_path,
    preprocesses it, and saves the clean text to interim_path.
    On subsequent runs, it loads the clean text directly from interim_path.
    """
    raw_path = Path(base_path_str)
    processed_text_path = Path(interim_path_str)
    docs_by_class = {}

    # Check if the data has already been preprocessed and can be loaded directly
    if processed_text_path.exists() and any(processed_text_path.iterdir()):
        logging.info(f"--- Loading PRE-PROCESSED text from '{processed_text_path}' ---")
        for class_name in class_map.keys():
            class_dir = processed_text_path / class_name
            if not class_dir.exists(): continue
            docs_by_class[class_name] = []
            file_paths = list(class_dir.glob("*.txt"))
            for file_path in tqdm.tqdm(file_paths[:samples_per_class], desc=f"Loading clean '{class_name}' text"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    docs_by_class[class_name].append((file_path.name, f.read()))
    else:
        # If not preprocessed, run the one-time cleaning and saving pipeline
        logging.info(f"--- No preprocessed text found. Running ONE-TIME preprocessing. ---")
        logging.info(f"Raw data source: '{raw_path}'")
        logging.info(f"Clean text will be saved to: '{processed_text_path}'")

        for class_name, sub_dir_name in class_map.items():
            # Correctly join the base path with the subdirectory name
            raw_class_dir = raw_path / sub_dir_name
            processed_class_dir = processed_text_path / class_name  # Save into flat class folders
            processed_class_dir.mkdir(parents=True, exist_ok=True)

            if not raw_class_dir.exists(): continue

            docs_by_class[class_name] = []
            file_paths = list(raw_class_dir.glob("*.txt"))
            random.shuffle(file_paths)

            for file_path in tqdm.tqdm(file_paths[:samples_per_class], desc=f"Preprocessing '{class_name}' text"):
                with open(file_path, 'r', encoding='utf-8') as f:
                    raw_content = f.read()

                clean_content = preprocessing_legal_pt(raw_content)
                docs_by_class[class_name].append((file_path.name, clean_content))

                with open(processed_class_dir / file_path.name, 'w', encoding='utf-8') as f_out:
                    f_out.write(clean_content)

    return docs_by_class


def run():
    """Main function to run the entire pipeline with a caching mechanism."""
    # --- 1. Configuration ---
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()
    data_cfg = DataConfig()
    viz_cfg = VisualizationConfig()

    # --- 2. Data Caching Logic ---
    processed_data_dir = Path(f"data/datasets/{data_cfg.DATASET_NAME}/full/processed_graphs")
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    cache_file_path = processed_data_dir / "processed_data.pkl"
    timestamp_str = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
    # The main metadata file now acts as the cache check
    metadata_cache_path = processed_data_dir / "metadata.pkl"
    model_param_str = _create_param_string(data_cfg, model_cfg, train_cfg)

    if not getattr(data_cfg, 'FORCE_PREPROCESSING', False) and metadata_cache_path.exists():
        logging.info(f"Found cached metadata at '{metadata_cache_path}'. Loading pre-processed graph data.")
        with open(metadata_cache_path, 'rb') as f:
            metadata = pickle.load(f)

        all_labels = metadata['labels']
        all_filenames = metadata['filenames']
        builder = metadata['builder']
        class_names = metadata['class_names']
        model_cfg.OUT_CHANNELS = len(class_names)

        # Load graphs iteratively
        all_hetero_graphs = []
        all_nx_graphs = []
        hetero_dir = processed_data_dir / "hetero_graphs"
        nx_dir = processed_data_dir / "nx_graphs"

        for i in tqdm.tqdm(range(len(all_filenames)), desc="Loading cached graphs"):
            hetero_path = hetero_dir / f"{i}.pt"
            nx_path = nx_dir / f"{i}.pkl"

            if hetero_path.exists():
                all_hetero_graphs.append(torch.load(hetero_path))
            else:
                logging.error(f"Missing cached file: {hetero_path}. Re-run with FORCE_PREPROCESSING=True.")
                return

            if nx_path.exists():
                with open(nx_path, 'rb') as f:
                    all_nx_graphs.append(pickle.load(f))
            else:
                logging.error(f"Missing cached file: {nx_path}. Re-run with FORCE_PREPROCESSING=True.")
                return

        logging.info(f"Successfully loaded {len(all_hetero_graphs)} graphs from cache.")
    else:
        logging.info("No cached data found or FORCE_PREPROCESSING is True. Starting memory-efficient pipeline.")

        # --- Load and Preprocess Data From Disk (if no cache) ---
        logging.info("\n--- Cleaning and Preprocessing Raw Text ---")
        nlp_pt = spacy.load(DataConfig.SPACY_MODEL)
        if not nlp_pt.has_pipe("set_custom_boundaries_advanced"):
            nlp_pt.add_pipe("set_custom_boundaries_advanced", before="parser")

        # --- CHANGE: Call the new "process-or-load" function ---
        docs_by_class = preprocess_and_save_raw_files(
            base_path_str=data_cfg.BASE_DATA_PATH,
            interim_path_str=data_cfg.INTERIM_DATA_PATH,
            class_map=data_cfg.CLASS_SUBDIRECTORIES,
            samples_per_class=data_cfg.SAMPLES_PER_CLASS
        )

        builder = DocumentGraphBuilder(
            nlp_spacy_model=nlp_pt,
            embedding_model=data_cfg.EMBEDDING_MODEL,
            similarity_threshold=data_cfg.SIMILARITY_THRESHOLD
        )

        class_names = list(sorted(docs_by_class.keys()))
        if not class_names:
            logging.error("No data was loaded. Exiting.")
            return

        all_hetero_graphs, all_labels, all_filenames, all_nx_graphs = [], [], [], []

        model_cfg.OUT_CHANNELS = len(class_names)

        output_viz_dir = Path(viz_cfg.VISUALIZATION_FOLDER) / data_cfg.DATASET_NAME
        output_viz_dir.mkdir(exist_ok=True)
        logging.info(f"Interactive graph visualizations will be saved to '{output_viz_dir}/'")

        # Process files iteratively to save memory
        for class_idx, (class_name, docs) in enumerate(docs_by_class.items()):
            print(f"\n--- Building Graphs for Class: {class_name} ---")

            nx_graphs_for_class, hetero_graphs_for_class, processed_filenames = builder.process_documents(docs)

            all_nx_graphs.extend(nx_graphs_for_class)
            all_hetero_graphs.extend(hetero_graphs_for_class)
            all_labels.extend([class_idx] * len(processed_filenames))
            all_filenames.extend(processed_filenames)

            del nx_graphs_for_class, hetero_graphs_for_class, processed_filenames
            gc.collect()

        logging.info("\n--- Generating Structural Graph Visualizations ---")
        structural_viz_dir = Path(viz_cfg.VISUALIZATION_FOLDER) / data_cfg.DATASET_NAME / "structural"
        structural_viz_dir.mkdir(parents=True, exist_ok=True)

        filename_to_class_map = {fname: cname for cname, docs in docs_by_class.items() for fname, _ in docs}

        for nx_graph in tqdm.tqdm(all_nx_graphs, desc="Generating Structural Visualizations"):
            original_filename = nx_graph.graph['filename']
            class_name = filename_to_class_map.get(original_filename, "unknown")

            viz_filename = (
                f"{timestamp_str}_{class_name}_{Path(original_filename).stem}_"
                f"structural_{model_param_str}.html"
            )
            output_path = structural_viz_dir / viz_filename

            visualize_structural_graph(
                nx_graph,
                dep_label_map=builder.dep_label_map,
                output_filename=str(output_path)
            )

        del docs_by_class

        # --- Generate structural visualizations with descriptive names ---
        stats_dir = Path("results")
        param_str = _create_param_string(data_cfg, model_cfg, train_cfg)
        stats_filename = f"graph_stats_{param_str}_{timestamp_str}.xlsx"
        stats_filepath = stats_dir / stats_filename
        builder.display_graph_statistics(stats_filepath)

        hetero_dir = processed_data_dir / "hetero_graphs"
        nx_dir = processed_data_dir / "nx_graphs"
        hetero_dir.mkdir(exist_ok=True)
        nx_dir.mkdir(exist_ok=True)

        logging.info("Iterative cache saving complete.")

    # --- 3. Assign IDs and Labels ---
    for i, graph in enumerate(all_hetero_graphs):
        graph['document'].y = torch.tensor(all_labels[i])
        graph['document'].doc_id = all_filenames[i]

    logging.info("--- Splitting Dataset ---")
    indices = list(range(len(all_hetero_graphs)))
    train_indices, test_val_indices = train_test_split(indices, test_size=0.3, random_state=42, stratify=all_labels)
    val_indices, test_indices = train_test_split(test_val_indices, test_size=0.5, random_state=42,
                                                 stratify=[all_labels[i] for i in test_val_indices])
    train_graphs, val_graphs, test_graphs = (
        [all_hetero_graphs[i] for i in train_indices],
        [all_hetero_graphs[i] for i in val_indices],
        [all_hetero_graphs[i] for i in test_indices]
    )

    # --- MEMORY SAVING: Keep only the NetworkX graphs needed for the test set visualizations ---
    logging.info("Optimizing memory by retaining only necessary NetworkX graphs for testing.")
    test_nx_graphs = [all_nx_graphs[i] for i in test_indices]
    del all_nx_graphs, all_hetero_graphs
    gc.collect()

    train_loader = DataLoader(train_graphs, batch_size=train_cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=train_cfg.BATCH_SIZE)
    test_loader = DataLoader(test_graphs, batch_size=train_cfg.BATCH_SIZE)
    logging.info(f"Dataset split: Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)}")

    # --- 5. Initialize Model and Training Components ---
    model = ExplainableHierarchicalGNN(
        initial_channels=builder.embedding_dim,
        hidden_channels=model_cfg.HIDDEN_CHANNELS,
        out_channels=model_cfg.OUT_CHANNELS
    )

    log_model_summary(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.LEARNING_RATE)
    train_labels = [g['document'].y.item() for g in train_graphs]
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(weights, dtype=torch.float)
    logging.info(f"Calculated class weights for loss function: {class_weights}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    model.to(device)

    # --- 6. Training or Loading Model ---
    model_param_str = _create_param_string(data_cfg, model_cfg, train_cfg)
    model_filename = f"{model_param_str}.pth"
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / model_filename

    if not getattr(train_cfg, 'FORCE_TRAINING', False) and model_path.exists():
        logging.info(f"Found existing trained model at '{model_path}'. Loading weights and skipping training.")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        logging.info("No trained model found or FORCE_TRAINING is True. Starting training loop.")
        best_val_f1 = 0
        patience_counter = 0

        for epoch in range(1, train_cfg.EPOCHS + 1):
            loss = train(model, train_loader, optimizer, criterion, entropy_weight=train_cfg.ENTROPY_WEIGHT)
            val_metrics, _, _, _, _, _ = test(model, val_loader, val_graphs, class_names=class_names)
            val_f1 = val_metrics.get('weighted_avg', {}).get('f1-score', 0)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), model_path)
                patience_counter = 0
            else:
                patience_counter += 1
            logging.info(
                f"Epoch {epoch:02d}, Loss: {loss:.4f}, Val Acc: {val_metrics.get('accuracy', 0):.4f}, Val F1: {val_f1:.4f}")
            if patience_counter >= train_cfg.PATIENCE:
                logging.info(f"Early stopping at epoch {epoch}.")
                break
        logging.info(f"Training finished. Best model saved to '{model_path}'.")

    logging.info("\n--- Final Evaluation on Test Set ---")

    model.load_state_dict(torch.load(model_path, weights_only=True))
    test_metrics, true_labels, pred_labels, final_explanations, detailed_results, _ = test(model, test_loader,
                                                                                           test_graphs,
                                                                                           class_names=class_names)

    logging.info(f"Test Accuracy: {test_metrics.get('accuracy', 'N/A'):.4f}")
    logging.info(f"Test F1-Score (Weighted): {test_metrics.get('weighted_avg', {}).get('f1-score', 'N/A'):.4f}")
    logging.info(f"Test ROC AUC Score: {test_metrics.get('roc_auc_score', 'N/A'):.4f}")

    full_report_text = classification_report(true_labels, pred_labels, target_names=class_names, zero_division=0)
    logging.info(f"\nClassification Report:\n{full_report_text}")
    cm = confusion_matrix(true_labels, pred_labels)
    logging.info(f"\nConfusion Matrix:\n{cm}")

    results_csv_path = Path("results") / "training_log.csv"
    log_results_to_csv(results_csv_path, test_metrics, data_cfg, model_cfg, train_cfg)

    predictions_csv_path = Path("results") / f"predictions_{model_param_str}_{timestamp_str}.xlsx"
    save_detailed_predictions_to_csv(predictions_csv_path, detailed_results)

    generate_attention_visualizations(
        explanations=final_explanations,
        nx_graphs=test_nx_graphs,
        class_names=class_names,
        viz_cfg=viz_cfg,
        data_cfg=data_cfg,
        model_param_str=model_param_str,
        run_timestamp=timestamp_str,
        builder=builder
    )

    # --- Visualize Full Attention Graph for a sample ---
    explanation_to_viz = final_explanations[0]
    doc_filename = explanation_to_viz['doc_id']
    nx_graph_map_for_viz = {g.graph['filename']: g for g in test_nx_graphs}
    original_nx_graph = nx_graph_map_for_viz.get(doc_filename)

    if original_nx_graph:
        logging.info(f"--- Full Explainability Analysis for Doc: {doc_filename} ---")
        logging.info(
            f"Predicted: '{class_names[explanation_to_viz['prediction']]}', Actual: '{class_names[explanation_to_viz['actual']]}'")

        # output_html_path = output_viz_dir / f"{Path(doc_filename).stem}_full_attention.html"
        enriched_graph = inject_learned_attentions(original_nx_graph, explanation_to_viz)
        # visualize_learned_attentions(enriched_graph, builder.dep_label_map, str(output_html_path))
        # logging.info(f"Saved full attention graph to {output_html_path}")

        # --- NEW: Conceptual Subgraph Explanation ---
        logging.info(f"\n--- Extracting Conceptual Subgraph for a Key Sentence ---")
        explainer = SubgraphExplainer(top_k_words=4, max_path_length=5)

        # Find the sentence with the highest absolute attention to explain
        sent_att_edge_index, sent_att_weights = explanation_to_viz['sent_to_doc_att']
        if sent_att_weights.numel() > 0:
            top_sent_pyg_idx = sent_att_edge_index[0][torch.argmax(torch.abs(sent_att_weights))].item()
            sent_map_rev = {v: k for k, v in explanation_to_viz['graph_data']['node_mappings']['sentence'].items()}
            top_sent_nx_id = sent_map_rev.get(top_sent_pyg_idx)

            if top_sent_nx_id:
                conceptual_subgraph = explainer.explain_sentence(enriched_graph, top_sent_nx_id)

                if conceptual_subgraph:
                    subgraph_output_path = output_viz_dir / f"{Path(doc_filename).stem}_conceptual_subgraph.html"
                    # We can reuse the same visualization function for the smaller subgraph
                    visualize_learned_attentions(conceptual_subgraph, builder.dep_label_map, str(subgraph_output_path))
                    logging.info(f"Saved conceptual subgraph visualization to {subgraph_output_path}")


if __name__ == '__main__':
    run()
