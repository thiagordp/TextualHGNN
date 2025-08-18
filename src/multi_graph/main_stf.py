# main.py
import logging
import os
from datetime import datetime
import random
from pathlib import Path  # Use pathlib for robust path handling

import numpy as np
import spacy
import torch
import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from spacy import Language
from torch_geometric.loader import DataLoader

import nltk

# Local imports from our new modules
from src.multi_graph.config import ModelConfig, TrainingConfig, DataConfig
from src.multi_graph.data_preprocessing import preprocess_text, preprocessing_legal_pt
from src.multi_graph.engine import train, test
from src.multi_graph.graph_builder import DocumentGraphBuilder
from src.multi_graph.main_stf_preprocessing import log_results_to_csv, save_detailed_predictions_to_csv, \
    generate_attention_visualizations
from src.multi_graph.model import ExplainableHierarchicalGNN
from src.multi_graph.config import VisualizationConfig
from src.multi_graph.visualization import visualize_structural_graph, visualize_learned_attentions

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)

# Generate timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
log_filename = f"logs/multi-level-graph_{Path(__file__).stem}_{timestamp}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()],
    force=True
)
logging.info("Logging setup complete.")


def load_documents_from_disk(base_path_str: str, class_map: dict, samples_per_class: int) -> dict:
    """
    Loads text documents from specified subdirectories.

    Args:
        base_path_str (str): The root directory of the dataset.
        class_map (dict): A dictionary mapping class names to subdirectories.
        samples_per_class (int): The number of files to load per class.

    Returns:
        A dictionary mapping class names to a list of (filename, content) tuples.
    """
    base_path = Path(base_path_str)
    docs_by_class = {}
    print("\n--- Loading Documents From Disk ---")

    for class_name, sub_dir in class_map.items():
        class_path = base_path / sub_dir
        if not class_path.exists():
            logging.warning(f"Directory not found: {class_path}. Skipping class '{class_name}'.")
            continue

        print(f"Loading from: {class_path}")
        file_paths = sorted(list(class_path.glob("*.txt")))
        # random.shuffle(file_paths)  # Shuffle for random sampling

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
        # Quebra a sentença no ponto e vírgula OU nos dois-pontos
        if token.text in [";", ":"]:
            doc[token.i + 1].is_sent_start = True
    return doc


# --- NEW: Helper function for creating descriptive filenames ---
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


def run():
    """Main function to run the entire pipeline."""

    # --- 1. Configuration ---
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()
    data_cfg = DataConfig()
    viz_cfg = VisualizationConfig()

    timestamp_str = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")

    # --- 2. Load and Preprocess Data From Disk ---
    raw_docs_by_class = load_documents_from_disk(
        data_cfg.BASE_DATA_PATH,
        data_cfg.CLASS_SUBDIRECTORIES,
        data_cfg.SAMPLES_PER_CLASS
    )
    class_names = list(sorted(raw_docs_by_class.keys()))
    model_cfg.OUT_CHANNELS = len(class_names)  # Dynamically set output channels

    print("\n--- Cleaning and Preprocessing Raw Text ---")

    nlp_pt = spacy.load(DataConfig.SPACY_MODEL)
    nlp_pt.add_pipe("set_custom_boundaries_advanced", before="parser")

    docs_by_class = {}
    for class_name, docs_with_filenames in raw_docs_by_class.items():
        cleaned_docs = [
            (filename, preprocessing_legal_pt(content, nlp_spacy=nlp_pt))
            for filename, content in tqdm.tqdm(docs_with_filenames, desc=f"Cleaning '{class_name}' docs")
        ]
        docs_by_class[class_name] = cleaned_docs

    # --- 3. Build Graphs ---
    builder = DocumentGraphBuilder(
        nlp_spacy_model=nlp_pt,
        embedding_model=data_cfg.EMBEDDING_MODEL,
        similarity_threshold=data_cfg.SIMILARITY_THRESHOLD
    )

    # --- CHANGE: Create a dedicated directory for visualization outputs ---
    output_viz_dir = Path(viz_cfg.VISUALIZATION_FOLDER) / data_cfg.DATASET_NAME
    output_viz_dir.mkdir(exist_ok=True)
    logging.info(f"Interactive graph visualizations will be saved to '{output_viz_dir}/'")

    # This is the section you provided, now updated
    all_hetero_graphs, all_labels, all_filenames, all_nx_graphs = [], [], [], []

    for class_idx, (class_name, docs) in enumerate(docs_by_class.items()):
        print(f"\n--- Building Graphs for Class: {class_name} ---")

        # --- Unpack all three return values from the builder ---
        nx_graphs_for_class, hetero_graphs_for_class, processed_filenames = builder.process_documents(docs)

        # --- Loop through the generated graphs and create a visualization for each ---
        # We zip the nx_graphs with the original 'docs' list to match each graph with its filename.

        # for nx_graph, (original_filename, _) in tqdm.tqdm(zip(nx_graphs_for_class, docs),
        #                                                   total=len(docs),
        #                                                   desc="Creating graph visualizations"):
        #     # Create a clean output filename (e.g., "123_4.html") from the original ("123_4.txt")
        #     output_html_name = f"{class_name.capitalize()}_{Path(original_filename).stem}.html"
        #     output_path = output_viz_dir / output_html_name
        #
        #     # Call the visualization function to save the HTML file
        #     visualize_structural_graph(
        #         nx_graph,
        #         dep_label_map=builder.dep_label_map,
        #         output_filename=str(output_path)  # pyvis expects a string path
        #     )

        # All graphs returned are guaranteed to be valid
        all_nx_graphs.extend(nx_graphs_for_class)
        all_hetero_graphs.extend(hetero_graphs_for_class)
        all_labels.extend([class_idx] * len(processed_filenames))
        all_filenames.extend(processed_filenames)

    stats_dir = Path("results")
    param_str = _create_param_string(data_cfg, model_cfg, train_cfg)
    stats_filename = f"graph_stats_{param_str}_{timestamp_str}.xlsx"
    stats_filepath = stats_dir / stats_filename
    builder.display_graph_statistics(stats_filepath)

    # --- Assign the actual filename as the doc_id ---
    for i, graph in enumerate(all_hetero_graphs):
        graph['document'].y = torch.tensor(all_labels[i])  # The label should be a scalar tensor
        graph['document'].doc_id = all_filenames[i]  # The ID should be a string

    logging.info("--- Splitting Dataset ---")
    indices = list(range(len(all_hetero_graphs)))

    # --- 4. Create Datasets and DataLoaders ---
    train_indices, test_val_indices = train_test_split(indices, test_size=0.3, random_state=42, stratify=all_labels)
    val_indices, test_indices = train_test_split(test_val_indices, test_size=0.5, random_state=42,
                                                 stratify=[all_labels[i] for i in test_val_indices])

    train_graphs = [all_hetero_graphs[i] for i in train_indices]
    val_graphs = [all_hetero_graphs[i] for i in val_indices]
    test_graphs = [all_hetero_graphs[i] for i in test_indices]

    # Create the corresponding split for the NetworkX graphs needed for visualization
    test_nx_graphs = [all_nx_graphs[i] for i in test_indices]

    train_loader = DataLoader(train_graphs, batch_size=train_cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=train_cfg.BATCH_SIZE)
    test_loader = DataLoader(test_graphs, batch_size=train_cfg.BATCH_SIZE)
    logging.info(f"Dataset split: Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)}")

    # --- 5. Initialize Model and Training Components ---
    model = ExplainableHierarchicalGNN(
        initial_channels=builder.embedding_dim,  # e.g., 384
        hidden_channels=model_cfg.HIDDEN_CHANNELS,  # e.g., 64
        out_channels=model_cfg.OUT_CHANNELS
    )

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Trainable Parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.LEARNING_RATE)

    # --- NEW: Calculate class weights for the loss function ---
    train_labels = [g['document'].y.item() for g in train_graphs]
    weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(weights, dtype=torch.float)
    class_weights = torch.clamp_min(class_weights, min=1.0)

    logging.info(f"Calculated class weights for loss function: {class_weights}")

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # --- 6. Training Loop with Early Stopping ---
    best_val_f1 = 0
    patience_counter = 0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model= model.to(device)
    model_param_str = _create_param_string(data_cfg, model_cfg, train_cfg)
    model_filename = f"{model_param_str}.pth"
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / model_filename

    for epoch in range(1, train_cfg.EPOCHS + 1):
        loss = train(
            model,
            train_loader,
            optimizer,
            criterion,
            entropy_weight=train_cfg.ENTROPY_WEIGHT
        )

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

    # --- 7. Final Evaluation and Explainability ---
    logging.info("\n--- Final Evaluation on Test Set ---")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    # Pass test_graphs to align explanations with the correct original data
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

if __name__ == '__main__':
    run()
