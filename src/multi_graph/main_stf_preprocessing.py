# main_stf.py
import logging
import os
from datetime import datetime
import random
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import spacy
import torch
import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from spacy import Language
from torch_geometric.loader import DataLoader
import nltk

from src.multi_graph.config import ModelConfig, TrainingConfig, DataConfig, VisualizationConfig
from src.multi_graph.data_preprocessing import preprocessing_legal_pt
from src.multi_graph.engine import train, test
from src.multi_graph.graph_builder import DocumentGraphBuilder
from src.multi_graph.model import ExplainableHierarchicalGNN
from src.multi_graph.visualization import visualize_learned_attentions, visualize_structural_graph

# --- Setup ---
os.makedirs("logs", exist_ok=True)
os.makedirs("results", exist_ok=True)  # Ensure results directory exists
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
    force=True  # This will remove and re-add handlers, ensuring your config is applied.
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
    model_param_str = _create_param_string(data_cfg, model_cfg, train_cfg)

    if not getattr(data_cfg, 'FORCE_PREPROCESSING', False) and cache_file_path.exists():
        logging.info(f"Found cached data at '{cache_file_path}'. Loading pre-processed graphs.")
        with open(cache_file_path, 'rb') as f:
            cached_data = pickle.load(f)
        all_hetero_graphs = cached_data['hetero_graphs']
        all_labels = cached_data['labels']
        all_filenames = cached_data['filenames']
        all_nx_graphs = cached_data['nx_graphs']
        builder = cached_data['builder']
        class_names = cached_data['class_names']
        model_cfg.OUT_CHANNELS = len(class_names)
        logging.info("Successfully loaded data from cache.")
    else:
        logging.info("No cached data found or FORCE_PREPROCESSING is True. Starting memory-efficient pipeline.")
        # --- Load and Preprocess Data From Disk (if no cache) ---
        raw_docs_by_class = load_documents_from_disk(
            data_cfg.BASE_DATA_PATH,
            data_cfg.CLASS_SUBDIRECTORIES,
            data_cfg.SAMPLES_PER_CLASS
        )
        class_names = list(sorted(raw_docs_by_class.keys()))
        model_cfg.OUT_CHANNELS = len(class_names)

        logging.info("\n--- Cleaning and Preprocessing Raw Text ---")
        nlp_pt = spacy.load(DataConfig.SPACY_MODEL)
        if not nlp_pt.has_pipe("set_custom_boundaries_advanced"):
            nlp_pt.add_pipe("set_custom_boundaries_advanced", before="parser")

        docs_by_class = {}
        for class_name, docs_with_filenames in raw_docs_by_class.items():
            cleaned_docs = [
                (filename, preprocessing_legal_pt(content, nlp_spacy=nlp_pt))
                for filename, content in tqdm.tqdm(docs_with_filenames, desc=f"Cleaning '{class_name}' docs")
            ]
            docs_by_class[class_name] = cleaned_docs

        # --- Build Graphs (if no cache) ---
        builder = DocumentGraphBuilder(
            spacy_model=data_cfg.SPACY_MODEL,
            embedding_model=data_cfg.EMBEDDING_MODEL,
            similarity_threshold=data_cfg.SIMILARITY_THRESHOLD
        )

        all_hetero_graphs, all_labels, all_filenames, all_nx_graphs = [], [], [], []
        for class_idx, (class_name, docs) in enumerate(docs_by_class.items()):
            logging.info(f"\n--- Building Graphs for Class: {class_name} ---")
            nx_graphs_for_class, hetero_graphs_for_class, processed_filenames = builder.process_documents(docs)
            all_nx_graphs.extend(nx_graphs_for_class)
            all_hetero_graphs.extend(hetero_graphs_for_class)
            all_labels.extend([class_idx] * len(processed_filenames))
            all_filenames.extend(processed_filenames)

        # --- NEW: Generate structural visualizations with descriptive names ---
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

        stats_dir = Path("results")

        stats_filename = f"graph_stats_{model_param_str}_{timestamp_str}.xlsx"
        stats_filepath = stats_dir / stats_filename
        builder.display_graph_statistics(stats_filepath)

        # --- Save the processed data to cache ---
        logging.info(f"Saving processed data to cache file: '{cache_file_path}'")
        data_to_cache = {
            'hetero_graphs': all_hetero_graphs, 'labels': all_labels, 'filenames': all_filenames,
            'nx_graphs': all_nx_graphs, 'builder': builder, 'class_names': class_names
        }
        with open(cache_file_path, 'wb') as f:
            pickle.dump(data_to_cache, f)
        logging.info("Cache saving complete.")

    # --- 3. Assign IDs and Labels ---
    for i, graph in enumerate(all_hetero_graphs):
        graph['document'].y = torch.tensor(all_labels[i])
        graph['document'].doc_id = all_filenames[i]

    # --- 4. Create Datasets and DataLoaders ---
    logging.info("--- Splitting Dataset ---")
    indices = list(range(len(all_hetero_graphs)))
    train_indices, test_val_indices = train_test_split(indices, test_size=0.3, random_state=42, stratify=all_labels)
    val_indices, test_indices = train_test_split(test_val_indices, test_size=0.5, random_state=42,
                                                 stratify=[all_labels[i] for i in test_val_indices])
    train_graphs = [all_hetero_graphs[i] for i in train_indices]
    val_graphs = [all_hetero_graphs[i] for i in val_indices]
    test_graphs = [all_hetero_graphs[i] for i in test_indices]
    test_nx_graphs = [all_nx_graphs[i] for i in test_indices]
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
    logging.info(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.LEARNING_RATE)
    train_labels = [g['document'].y.item() for g in train_graphs]
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(weights, dtype=torch.float)
    logging.info(f"Calculated class weights for loss function: {class_weights}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    model.to(device)

    # --- 6. Training or Loading Model ---
    embedding_model_name = data_cfg.EMBEDDING_MODEL.split('/')[-1].replace('-', '_')
    model_param_str = _create_param_string(data_cfg, model_cfg, train_cfg)
    model_filename = f"{model_param_str}.pth"
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / model_filename

    if not getattr(train_cfg, 'FORCE_TRAINING', False) and model_path.exists():
        logging.info(f"Found existing trained model at '{model_path}'. Loading weights and skipping training.")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        logging.info("No trained model found. Starting training loop.")
        best_val_f1 = 0
        patience_counter = 0
        for epoch in range(1, train_cfg.EPOCHS + 1):
            loss = train(model, train_loader, optimizer, criterion, entropy_weight=train_cfg.ENTROPY_WEIGHT)
            val_metrics, _, _, _, _, _ = test(model, val_loader, val_graphs, class_names=class_names)
            val_f1 = val_metrics.get('macro_avg', {}).get('f1-score', 0)
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
    test_metrics, true_labels, pred_labels, final_explanations, detailed_results, _ = test(model, test_loader,
                                                                                           test_graphs,
                                                                                           class_names=class_names)

    logging.info(f"Test Accuracy: {test_metrics.get('accuracy', 'N/A'):.4f}")
    logging.info(f"Test F1-Score (MAcro): {test_metrics.get('macro_avg', {}).get('f1-score', 'N/A'):.4f}")
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


if __name__ == '__main__':
    run()
