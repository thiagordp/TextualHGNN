"""
Complete pipeline for:
1. Building text graphs from the IMDB dataset (using TextGraphBuilder)
2. ** RUNNING HYPERPARAMETER OPTIMIZATION (Optuna) ** to find the best
   GNN (GAT vs. GCN), model size, and training parameters.
3. Training the final best model and evaluating it.

Best Practice Upgrades:
- ** NEW: Optuna HPO ** for lr, weight_decay, dropout, model_type,
           hidden_channels, and num_heads.
- ** NEW: GCNGraphClassifier ** as an alternative to GAT.
- ** NEW: Edge Weight Usage **: GCN uses PMI weights; GAT does not.
- ** NEW: HPO Persistence **: Study is saved to 'hpo_study.db'.
- Kept: GAT, Early Stopping, LR Scheduler, Grad Clipping, AdamW, PMI Caching.
"""
import logging
# --- Part 0: Imports ---
import math
import glob
import random
import warnings
import pickle
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional

# Third-party ML imports
import networkx as nx
import numpy as np
import gensim
import spacy
import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, BatchNorm
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.utils import coalesce, remove_self_loops
from spacy.tokens import Doc
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ** NEW: HPO Import **
import optuna

# Local (assumed) imports
from src.data.preprocessing import preprocessing_imdb

# --- Part 1: Configuration Constants ---

# Paths
EMBEDDING_PATH = Path("data/external/embeddings/glove_legal_100.bin")
DATASET_PATH_PATTERN = "data/datasets/IMDB/train/raw/{sentiment}/*.txt"
PMI_CACHE_PATH = Path("data/processed/pmi_cache.pkl")

# TextGraphBuilder Config
SPACY_MODEL = 'en_core_web_lg'
DEP_TO_IGNORE: Set[str] = {"punct"}

# GNN Training Config
RANDOM_SEED = 42
# Keep this low for fast HPO, increase for final run
NUM_SAMPLES_TO_PROCESS = 20000
MIN_PMI_THRESHOLD = 0.0
TEST_SIZE = 0.2
VAL_SIZE = 0.2
BATCH_SIZE = 32
NUM_EPOCHS = 100  # Max epochs (will be stopped early)
GRAD_CLIP_MAX_NORM = 1.0

# HPO Configuration
N_TRIALS = 50  # Number of HPO trials to run
HPO_STORAGE_DB = "sqlite:///hpo_study.db"
HPO_STUDY_NAME = "gnn_imdb_v3"


# --- Part 2: Text-to-Graph Builder Class (Unchanged) ---
# (Class code is identical to previous step, omitted for brevity)
class TextGraphBuilder:
    def __init__(self, embedding_path: Path, spacy_model: str, dep_to_ignore: Set[str]):
        logging.info(f"Loading spaCy model '{spacy_model}'...")
        self.nlp = spacy.load(spacy_model)
        logging.info(f"Loading GloVe vectors from '{embedding_path}'...")
        if not embedding_path.exists():
            raise FileNotFoundError(f"Embedding file not found at: {embedding_path}")
        self.glove_vectors = gensim.models.KeyedVectors.load(str(embedding_path), mmap='r')
        self.mean_vec = torch.tensor(np.mean(self.glove_vectors.vectors, axis=0))
        self.embedding_dim = self.glove_vectors.vector_size
        self.dep_to_ignore = dep_to_ignore
        self.pmi: Dict[Tuple[str, str], float] = {}

    def _preprocess_node_text(self, text: str) -> str:
        return text.lower().strip()

    def _is_valid_dependency(self, h, c, d) -> bool:
        if not h or not c or h == c: return False
        if d in self.dep_to_ignore: return False
        return True

    def get_embedding(self, token_text: str) -> torch.Tensor:
        token = self._preprocess_node_text(token_text)
        if token in self.glove_vectors:
            return torch.tensor(self.glove_vectors[token])
        elif "<unk>" in self.glove_vectors:
            return torch.tensor(self.glove_vectors["<unk>"])
        else:
            return self.mean_vec.clone()

    def compute_pmi(self, corpus_docs: List[Doc]):
        logging.info("Calculating PMI...")
        co_occur, word_count = defaultdict(int), defaultdict(int)
        total_edges = 0
        for doc in tqdm(corpus_docs, desc="1/2: Counting co-occurrences"):
            for token in doc:
                h, c = self._preprocess_node_text(token.head.text), self._preprocess_node_text(token.text)
                if not self._is_valid_dependency(h, c, token.dep_): continue
                co_occur[(h, c)] += 1;
                word_count[h] += 1;
                word_count[c] += 1;
                total_edges += 1
        if total_edges == 0: return
        pmi_values = {}
        for (w1, w2), count in tqdm(co_occur.items(), desc="2/2: Calculating PMI"):
            p_w1 = word_count[w1] / total_edges;
            p_w2 = word_count[w2] / total_edges
            p_w1_w2 = count / total_edges
            pmi_val = math.log2(p_w1_w2 / (p_w1 * p_w2 + 1e-9))
            # pmi_values[(w1, w2)] = 0  # Ignore for now PMI.
            pmi_values[(w1, w2)] = pmi_val
        self.pmi = pmi_values
        logging.info(f"PMI calculation complete. Found {len(self.pmi)} valid pairs.")

    def build_graph_from_doc(self, doc: Doc, min_pmi: float = 0.0) -> nx.MultiDiGraph:
        G = nx.MultiDiGraph();
        node_id_map = {}
        for token in doc:
            if token.dep_ in self.dep_to_ignore: continue
            node_id = f"{token.lemma_}_{token.i}"
            node_id_map[token] = node_id
            G.add_node(node_id, feature=self.get_embedding(self._preprocess_node_text(token.text)))
        for token in doc:
            if token not in node_id_map or token.head not in node_id_map: continue
            h, c = self._preprocess_node_text(token.head.text), self._preprocess_node_text(token.text)
            if not self._is_valid_dependency(h, c, token.dep_): continue
            pmi_val = self.pmi.get((h, c))
            if pmi_val is None or pmi_val < min_pmi: continue
            G.add_edge(node_id_map[token.head], node_id_map[token], weight=pmi_val + 1)
        prev_root_id = None
        for sent in doc.sents:
            if sent.root in node_id_map:
                root_id = node_id_map[sent.root]
                if prev_root_id is not None:
                    G.add_edge(prev_root_id, root_id, weight=1.0)
                    G.add_edge(root_id, prev_root_id, weight=1.0)
                prev_root_id = root_id
        return G


# --- Part 3: Data Loading & Conversion (Upgraded) ---

def load_raw_texts(path_pattern: str, label: str) -> List[Tuple[str, str]]:
    # (Identical to previous step, omitted for brevity)
    filenames = glob.glob(path_pattern.format(sentiment=label))
    dataset = []
    for f in tqdm(filenames, desc=f"Reading {label} reviews"):
        try:
            with open(f, 'r', encoding='utf-8') as infile:
                dataset.append((infile.read(), label))
        except Exception as e:
            pass
    return dataset


def nx_to_pyg_data(G: nx.MultiDiGraph, label: float) -> Data:
    """
    ** UPDATED **
    Converts a NetworkX graph into a PyTorch Geometric Data object.
    Now extracts `edge_weight` and stores it in `edge_attr`.
    """
    node_map = {node_id: i for i, node_id in enumerate(G.nodes())}
    x_list = [G.nodes[node_id]['feature'] for node_id in node_map.keys()]
    x = torch.stack(x_list).float()

    edge_index_list = []
    edge_attr_list = []  # ** NEW: Store weights here **

    for u, v, data in G.edges(data=True):
        u_idx, v_idx = node_map.get(u), node_map.get(v)
        if u_idx is None or v_idx is None: continue

        weight = data.get('weight', 1.0)

        # Add edge in both directions
        edge_index_list.append([u_idx, v_idx])
        edge_attr_list.append(weight)
        edge_index_list.append([v_idx, u_idx])
        edge_attr_list.append(weight)

    if not edge_index_list:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0,), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    # Coalesce to sum/average duplicate edge weights (e.g., from undirected)
    edge_index, edge_attr = coalesce(edge_index, edge_attr,
                                     num_nodes=x.size(0),
                                     reduce='mean')

    # Remove self-loops (GNN layers will add them back if needed)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)

    y = torch.tensor([label], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


# --- Part 4: GNN Implementations (Upgraded) ---

class GATGraphClassifier(nn.Module):
    """
    A GAT-based model. `forward` is updated to accept `edge_attr`
    (even though it's not used) to have a consistent API with GCN.
    """

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_heads: int, dropout: float):
        super().__init__()
        self.dropout = dropout

        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout)
        self.bn1 = BatchNorm(hidden_channels * num_heads)
        self.conv2 = GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout)
        self.bn2 = BatchNorm(hidden_channels * num_heads)

        classifier_in_dim = (hidden_channels * num_heads) * 2
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, hidden_channels), nn.ReLU(),
            nn.Dropout(p=dropout), nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # ** edge_attr is ignored, GAT learns its own attention **
        x = self.conv1(x, edge_index);
        x = self.bn1(x);
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index);
        x = self.bn2(x);
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_graph = torch.cat([x_mean, x_max], dim=-1)
        return self.classifier(x_graph)


class GCNGraphClassifier(nn.Module):
    """
    ** NEW **
    A GCN-based model that USES the `edge_attr` (PMI weights).
    """

    def __init__(self, in_channels: int, hidden_channels: int,
                 out_channels: int, dropout: float):
        super().__init__()
        self.dropout = dropout

        # GCNConv can use edge_weight (passed as edge_attr)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.bn1 = BatchNorm(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = BatchNorm(hidden_channels)

        classifier_in_dim = hidden_channels * 2
        self.classifier = nn.Sequential(
            nn.Linear(classifier_in_dim, hidden_channels), nn.ReLU(),
            nn.Dropout(p=dropout), nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # ** edge_attr IS USED here as edge_weight **
        x = self.conv1(x, edge_index, edge_weight=edge_attr);
        x = self.bn1(x);
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight=edge_attr);
        x = self.bn2(x);
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_graph = torch.cat([x_mean, x_max], dim=-1)
        return self.classifier(x_graph)


# --- Part 5: Training & Evaluation (Upgraded) ---

class EarlyStopper:
    # (Identical to previous step, omitted for brevity)
    def __init__(self, patience: int = 10, mode: str = "min", delta: float = 0.0):
        self.patience = patience;
        self.mode = mode;
        self.delta = delta
        self.counter = 0;
        self.best_score = float('inf') if mode == "min" else float('-inf')
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        is_better = False
        if self.mode == "min":
            is_better = score < (self.best_score - self.delta)
        else:
            is_better = score > (self.best_score + self.delta)
        if is_better:
            self.best_score = score;
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience: self.should_stop = True
        return is_better


def train_one_epoch(model: nn.Module, loader: DataLoader,
                    criterion: nn.Module, optimizer: torch.optim.Optimizer,
                    device: torch.device) -> float:
    """
    ** UPDATED **
    Passes `data.edge_attr` to the model.
    """
    model.train()
    total_loss = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(out.squeeze(), data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             criterion: nn.Module, device: torch.device) -> Tuple[float, float, float]:
    """
    ** UPDATED **
    Passes `data.edge_attr` to the model.
    """
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = criterion(out.squeeze(), data.y)
        total_loss += loss.item() * data.num_graphs
        preds = (torch.sigmoid(out.squeeze()) > 0.5).float()
        all_preds.append(preds.cpu())
        all_labels.append(data.y.cpu())
    avg_loss = total_loss / len(loader.dataset)
    all_preds = torch.cat(all_preds);
    all_labels = torch.cat(all_labels)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, acc, f1


# --- Part 6: HPO Objective Function ---

def objective(trial: optuna.Trial, train_loader: DataLoader,
              val_loader: DataLoader, device: torch.device,
              in_channels: int) -> float:
    """
    The main HPO objective function.
    Optuna will call this for N_TRIALS.
    """
    # === 1. Suggest Hyperparameters ===
    model_type = trial.suggest_categorical('model_type', ['gcn', 'gat'])
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.2, 0.6, step=0.01)
    hidden_channels = trial.suggest_categorical('hidden_channels', [64, 100, 128])

    if model_type == 'gat':
        num_heads = trial.suggest_categorical('num_heads', [1, 2, 4])
        model = GATGraphClassifier(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=1,
            num_heads=num_heads,
            dropout=dropout
        ).to(device)
    else:
        model = GCNGraphClassifier(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=1,
            dropout=dropout
        ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopper = EarlyStopper(patience=10, mode='min')

    # === 2. Training & Pruning Loop ===
    best_val_loss = float('inf')
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        # Report to Optuna for pruning
        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Local early stopping for this trial
        if early_stopper(val_loss) and early_stopper.should_stop:
            break

    return best_val_loss  # Optuna will minimize this value


# --- Part 7: Main Execution (Completely Refactored) ---

def main():
    """
    Main function to run the full pipeline:
    Data Loading -> HPO -> Final Evaluation
    """
    # --- 1. Setup ---
    random.seed(RANDOM_SEED);
    np.random.seed(RANDOM_SEED);
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(RANDOM_SEED)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    nltk.download('punkt', quiet=True);
    nltk.download('stopwords', quiet=True)
    warnings.filterwarnings("ignore", message="Using slow pure-python")

    # --- 2. Load Graph Builder & Data ---
    try:
        graph_builder = TextGraphBuilder(EMBEDDING_PATH, SPACY_MODEL, DEP_TO_IGNORE)
    except FileNotFoundError as e:
        logging.info(e)
        return

    logging.info("\n--- 1. Loading and Preprocessing Data ---")
    positive_data = load_raw_texts(DATASET_PATH_PATTERN, "positive")
    negative_data = load_raw_texts(DATASET_PATH_PATTERN, "negative")
    dataset = (positive_data + negative_data)
    random.shuffle(dataset);
    dataset = dataset[:NUM_SAMPLES_TO_PROCESS]

    processed_data: List[Tuple[Doc, str]] = []
    for text, label in tqdm(dataset, desc="Preprocessing texts"):
        clean_text = preprocessing_imdb(text, nlp=graph_builder.nlp)
        processed_data.append((graph_builder.nlp(clean_text), label))

    # --- 3. Compute or Load PMI ---
    logging.info("\n--- 2. Computing or Loading PMI ---")
    if PMI_CACHE_PATH.exists():
        logging.info(f"Loading cached PMI from {PMI_CACHE_PATH}...")
        with open(PMI_CACHE_PATH, 'rb') as f:
            graph_builder.pmi = pickle.load(f)
    else:
        logging.info("No cached PMI found. Calculating from scratch...")
        graph_builder.compute_pmi([doc for doc, label in processed_data])
        PMI_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(PMI_CACHE_PATH, 'wb') as f:
            pickle.dump(graph_builder.pmi, f)

    # --- 4. Build Graphs & Split Data ---
    logging.info("\n--- 3. Building Graphs & Splitting Data ---")
    label_map = {"positive": 1.0, "negative": 0.0}
    pyg_data_list = []
    for doc, label in tqdm(processed_data, desc="Building PyG graphs"):
        G = graph_builder.build_graph_from_doc(doc=doc, min_pmi=MIN_PMI_THRESHOLD)
        if G.number_of_nodes() > 0:
            pyg_data_list.append(nx_to_pyg_data(G, label_map[label]))

    train_val_data, test_data = train_test_split(
        pyg_data_list, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    train_data, val_data = train_test_split(
        train_val_data, test_size=VAL_SIZE / (1.0 - TEST_SIZE), random_state=RANDOM_SEED
    )
    logging.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

    in_channels = pyg_data_list[0].x.shape[1]

    # --- 5. Run Hyperparameter Optimization (HPO) ---
    logging.info(f"\n--- 4. Starting Optuna HPO ({N_TRIALS} trials) ---")
    storage = optuna.storages.RDBStorage(url=HPO_STORAGE_DB, heartbeat_interval=60, grace_period=120)
    study = optuna.create_study(
        study_name=HPO_STUDY_NAME,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )

    objective_with_data = lambda trial: objective(
        trial, train_loader, val_loader, device, in_channels
    )

    study.optimize(objective_with_data, n_trials=N_TRIALS)

    logging.info("HPO complete. Best trial:")
    logging.info(f"  Value (Val Loss): {study.best_value:.4f}")
    logging.info("  Params: ")
    for key, value in study.best_params.items():
        logging.info(f"    {key}: {value}")

    # --- 6. Final Evaluation with Best Model ---
    logging.info("\n--- 5. Training Final Model with Best Params ---")
    best_params = study.best_params

    # Build the best model
    if best_params['model_type'] == 'gat':
        model = GATGraphClassifier(
            in_channels=in_channels,
            hidden_channels=best_params['hidden_channels'],
            out_channels=1,
            num_heads=best_params['num_heads'],
            dropout=best_params['dropout']
        ).to(device)
    else:
        model = GCNGraphClassifier(
            in_channels=in_channels,
            hidden_channels=best_params['hidden_channels'],
            out_channels=1,
            dropout=best_params['dropout']
        ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(),
                      lr=best_params['lr'],
                      weight_decay=best_params['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopper = EarlyStopper(patience=10, mode='min')

    best_model_state = None
    best_val_loss = float('inf')

    # Run one final training pass to get the best *model state*
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        logging.info(f"Epoch: {epoch:03d} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if early_stopper(val_loss):
            best_val_loss = val_loss
            best_model_state = model.state_dict()

        if early_stopper.should_stop:
            logging.info(f"Stopping early after {epoch} epochs.")
            break

    # --- 7. Load best model state and run on Test set ---
    logging.info("\n--- 6. Final Evaluation on Test Set ---")
    if best_model_state:
        model.load_state_dict(best_model_state)
    else:
        logging.info("Warning: No best model state found. Using last model state.")

    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)

    logging.info("=" * 30)
    logging.info(f"**Final Test Results**")
    logging.info(f"  Best Val Loss: {best_val_loss:.4f}")
    logging.info(f"  Test Loss: {test_loss:.4f}")
    logging.info(f"  Test Acc:  {test_acc:.4f}")
    logging.info(f"  Test F1:   {test_f1:.4f}")
    logging.info("=" * 30)


if __name__ == "__main__":
    main()
