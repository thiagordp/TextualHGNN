# main.py
import logging
import os
from datetime import datetime
import random
from pathlib import Path  # Use pathlib for robust path handling

import torch
import tqdm
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

# Local imports from our new modules
from src.multi_graph.config import ModelConfig, TrainingConfig, DataConfig
from src.multi_graph.data_preprocessing import preprocess_text
from src.multi_graph.engine import train, test
from src.multi_graph.graph_builder import DocumentGraphBuilder
from src.multi_graph.model import ExplainableHierarchicalGNN
from src.multi_graph.config import VisualizationConfig
from src.multi_graph.visualization import visualize_structural_graph, visualize_learned_attentions

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)
# Generate timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H.%M.%S")
log_filename = f"logs/multi-level-graph_{timestamp}.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_filename),
        logging.StreamHandler()  # Optional: also log to console
    ]
)

# Example usage
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
        file_paths = list(class_path.glob("*.txt"))
        random.shuffle(file_paths)  # Shuffle for random sampling

        docs_by_class[class_name] = []
        for file_path in tqdm.tqdm(file_paths[:samples_per_class], desc=f"Loading '{class_name}' files"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                docs_by_class[class_name].append((file_path.name, content))
            except Exception as e:
                logging.error(f"Could not read file {file_path}: {e}")

    return docs_by_class


def run():
    """Main function to run the entire pipeline."""

    # --- 1. Configuration ---
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()
    data_cfg = DataConfig()
    viz_cfg = VisualizationConfig()

    # --- 2. Load and Preprocess Data From Disk ---
    raw_docs_by_class = load_documents_from_disk(
        data_cfg.BASE_DATA_PATH,
        data_cfg.CLASS_SUBDIRECTORIES,
        data_cfg.SAMPLES_PER_CLASS
    )
    class_names = list(sorted(raw_docs_by_class.keys()))
    model_cfg.OUT_CHANNELS = len(class_names)  # Dynamically set output channels

    print("\n--- Cleaning and Preprocessing Raw Text ---")
    docs_by_class = {}
    for class_name, docs_with_filenames in raw_docs_by_class.items():
        cleaned_docs = [
            (filename, preprocess_text(content))
            for filename, content in tqdm.tqdm(docs_with_filenames, desc=f"Cleaning '{class_name}' docs")
        ]
        docs_by_class[class_name] = cleaned_docs

    # --- 3. Build Graphs ---
    builder = DocumentGraphBuilder(
        spacy_model=data_cfg.SPACY_MODEL,
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

        # --- FIX: Unpack all three return values from the builder ---
        nx_graphs_for_class, hetero_graphs_for_class, processed_filenames = builder.process_documents(docs)

        # --- Loop through the generated graphs and create a visualization for each ---
        # We zip the nx_graphs with the original 'docs' list to match each graph with its filename.

        for nx_graph, (original_filename, _) in tqdm.tqdm(zip(nx_graphs_for_class, docs),
                                                     total=len(docs),
                                                     desc="Creating graph visualizations"):
            # Create a clean output filename (e.g., "123_4.html") from the original ("123_4.txt")
            output_html_name = f"{class_name.capitalize()}_{Path(original_filename).stem}.html"
            output_path = output_viz_dir / output_html_name

            # Call the visualization function to save the HTML file
            visualize_structural_graph(
                nx_graph,
                dep_label_map=builder.dep_label_map,
                output_filename=str(output_path)  # pyvis expects a string path
            )

        # All graphs returned are guaranteed to be valid
        all_nx_graphs.extend(nx_graphs_for_class)
        all_hetero_graphs.extend(hetero_graphs_for_class)
        all_labels.extend([class_idx] * len(processed_filenames))
        all_filenames.extend(processed_filenames)

    # --- Assign the actual filename as the doc_id ---
    for i, graph in enumerate(all_hetero_graphs):
        graph['document'].y = torch.tensor(all_labels[i])  # The label should be a scalar tensor
        graph['document'].doc_id = all_filenames[i]  # The ID should be a string

    logging.info("--- Splitting Dataset ---")
    indices = list(range(len(all_hetero_graphs)))

    # --- 4. Create Datasets and DataLoaders ---
    train_indices, test_val_indices = train_test_split(indices, test_size=0.4, random_state=42, stratify=all_labels)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()

    # --- 6. Training Loop with Early Stopping ---
    best_val_f1 = 0
    patience_counter = 0

    for epoch in range(1, train_cfg.EPOCHS + 1):
        loss = train(model, train_loader, optimizer, criterion)
        val_metrics, _, _, _ = test(model, val_loader, val_graphs)

        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1

        logging.info(f"Epoch {epoch:02d}, "
              f"Loss: {loss:.4f}, "
              f"Val Acc:  {val_metrics['accuracy']:.4f}, "
              f"Val Prec: {val_metrics['precision']:.4f}, "
              f"Val Rec:  {val_metrics['recall']:.4f}, "
              f"Val F1:   {val_metrics['f1']:.4f}")

        if patience_counter >= train_cfg.PATIENCE:
            logging.info(f"Early stopping at epoch {epoch}.")
            break

    # --- 7. Final Evaluation and Explainability ---
    logging.info("\n--- Final Evaluation on Test Set ---")
    model.load_state_dict(torch.load('best_model.pth', weights_only=True))
    # Pass test_graphs to align explanations with the correct original data
    test_metrics, true_labels, pred_labels, final_explanations = test(model, test_loader, test_graphs)
    logging.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}, F1-Score: {test_metrics['f1']:.4f}")
    print("Final explanations:", final_explanations)

    if final_explanations:
        output_viz_dir = Path(viz_cfg.VISUALIZATION_FOLDER) / data_cfg.DATASET_NAME
        output_viz_dir.mkdir(parents=True, exist_ok=True)

        # This map now uses the correctly created test_nx_graphs list
        nx_graph_map_for_viz = {g.graph['filename']: g for g in test_nx_graphs}

        explanation_to_viz = final_explanations[0]
        doc_filename = explanation_to_viz['doc_id']
        original_nx_graph = nx_graph_map_for_viz.get(doc_filename)

        if original_nx_graph:
            logging.info(f"--- Explainability Analysis for Doc: {doc_filename} ---")
            logging.info(
                f"Predicted: '{class_names[explanation_to_viz['prediction']]}', Actual: '{class_names[explanation_to_viz['actual']]}'")

            output_html_path = output_viz_dir / f"{Path(doc_filename).stem}_learned_attention.html"
            visualize_learned_attentions(original_nx_graph, explanation_to_viz, builder.dep_label_map,
                                         str(output_html_path))


if __name__ == '__main__':
    run()
