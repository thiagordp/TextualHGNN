# main.py
import logging
import random
from pathlib import Path  # Use pathlib for robust path handling

import torch
import tqdm
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

# Local imports from our new modules
from config import ModelConfig, TrainingConfig, DataConfig
from data_preprocessing import preprocess_text
from engine import train, test
from graph_builder import DocumentGraphBuilder
from model import ExplainableHierarchicalGNN
from visualization import visualize_interactive_graph


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

    graphs, labels, filenames = [], [], []
    for class_idx, (class_name, docs) in enumerate(docs_by_class.items()):
        print(f"\n--- Building Graphs for Class: {class_name} ---")
        contents = [content for _, content in docs]
        current_filenames = [filename for filename, _ in docs]

        _, hetero_graphs = builder.process_documents(docs)

        graphs.extend(hetero_graphs)
        labels.extend([class_idx] * len(hetero_graphs))
        filenames.extend(current_filenames)

    # --- CHANGE: Assign the actual filename as the doc_id ---
    for i, graph in enumerate(graphs):
        graph['document'].y = torch.tensor([labels[i]])
        graph['document'].doc_id = [filenames[i]]

    # --- 4. Create Datasets and DataLoaders ---
    train_graphs, test_graphs = train_test_split(graphs, test_size=0.4, random_state=42, stratify=labels)
    val_graphs, test_graphs = train_test_split(test_graphs, test_size=0.5, random_state=42,
                                               stratify=[g['document'].y.item() for g in test_graphs])

    train_loader = DataLoader(train_graphs, batch_size=train_cfg.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=train_cfg.BATCH_SIZE)
    test_loader = DataLoader(test_graphs, batch_size=train_cfg.BATCH_SIZE)
    print(f"\nDataset split: Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)}")

    # --- 5. Initialize Model and Training Components ---
    model = ExplainableHierarchicalGNN(
        hidden_channels=model_cfg.HIDDEN_CHANNELS,
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

        if epoch % 5 == 0:
            print(
                f"Epoch {epoch:02d}, Loss: {loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}, Val F1: {val_metrics['f1']:.4f}")

        if patience_counter >= train_cfg.PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    # --- 7. Final Evaluation and Explainability ---
    print("\n--- Final Evaluation on Test Set ---")
    model.load_state_dict(torch.load('best_model.pth'))
    test_metrics, true_labels, pred_labels, final_explanations = test(model, test_loader, test_graphs)

    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}, F1-Score: {test_metrics['f1']:.4f}")

    if final_explanations:
        explanation = final_explanations[0]
        graph = explanation['graph_data']
        # The graph object needs to be converted back to networkx for visualization
        nx_graph = graph.to_networkx()

        print(f"\n--- Explainability Analysis for Doc: {explanation['doc_id']} ---")
        print(f"Predicted: '{class_names[explanation['prediction']]}', Actual: '{class_names[explanation['actual']]}'")

        output_html = f"{Path(explanation['doc_id']).stem}_explanation.html"
        visualize_interactive_graph(nx_graph, builder.dep_label_map, output_html)
        print(f"Saved interactive visualization to {output_html}")


if __name__ == '__main__':
    run()