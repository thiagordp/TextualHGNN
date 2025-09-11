"""

@author Thiago R Dal Pont
"""
import json
import logging
from pickletools import decimalnl_short

import torch
import tqdm
from sklearn.metrics import classification_report
from torch import nn
import torch_geometric.transforms as T

from src.data.text_graph_dataset_ondisk import TextGraphDatasetOnDisk
from src.models.graph_classification.gnn import DiffPool
from src.models.graph_classification.train_and_evaluate import load_datasets, initialize_model, calculate_class_weights, \
    train_and_validate, create_loaders
from src.utils.general_utils import load_config, setup_logging
from src.utils.models_utils import get_timestamp


PARAMS_GRIDSEARCH = {
    'LR': [1e-4], 'INNER_DIM': [32], 'BATCH_SIZE': [4],
    'SOFTMAX_ASSIGN': [True], "DECREASE_PROPORTION": [0.1]
}

LANG = "portuguese_voto"
MODEL_PATH = f"models/grid_search/STF_HC_Voto_Relatorio/best_model_DiffPool_20250903_135039_lr0.0001_hd_32_bs4_dec0.1_lk10.0_en0.01_rc0.1_ct0.1_bl0.1_rp0.1_l20.01_rep02.pth"
DATASET = "STF_HC_Voto_Relatorio"

CONFIG = load_config(LANG, "src/utils/config.json")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
timestamp = get_timestamp()
weights = torch.load(MODEL_PATH)

EMBEDDINGS_DATABASE_PATH = f"data/oracle/embeddings_{DATASET}.db"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT = f"data/datasets/{DATASET}"
DOCUMENTS_TO_EXPLAIN = 6

setup_logging(log_file=f"testing_model_experiment_gridsearch_{CONFIG['DATASET']}_loss_{timestamp}.log")
logging.info(f"============  STARTING EXPERIMENT {CONFIG['DATASET']}  ============")
logging.info(f"CONFIG: \n{json.dumps(CONFIG, indent=3)}")


def main():
    diffpool_model = DiffPool(
        max_num_nodes=3000,
        in_channels=100,
        hidden_channels=100,
        out_channels=2,
        inner_channels=32,
        softmax_assign=True,
        decrease_proportion=0.1,
    )

    diffpool_model.load_state_dict(weights)
    diffpool_model = diffpool_model.to(device=DEVICE)

    tgd_test = TextGraphDatasetOnDisk(
        root=ROOT,
        split="validation",
        batch_size=1,
        node_feature_size=100,
        transform=T.ToDense(num_nodes=3000),
        max_num_nodes=3000,
        lang=LANG
    )
    tgd_train = TextGraphDatasetOnDisk(
        root=ROOT,
        split="train",
        batch_size=1,
        node_feature_size=100,
        transform=T.ToDense(num_nodes=3000),
        max_num_nodes=3000,
        lang=LANG
    )
    tgd_val = TextGraphDatasetOnDisk(
        root=ROOT,
        split="validation",
        batch_size=1,
        node_feature_size=100,
        transform=T.ToDense(num_nodes=3000),
        max_num_nodes=3000,
        lang=LANG
    )


    _, _, test_loader = create_loaders(tgd_train, tgd_val, tgd_test, batch_size=1)

    y_preds = []
    y_tests = []
    for data in test_loader:
        data = data.to(device=DEVICE)

        with torch.no_grad():
            prediction = diffpool_model(data.x, data.adj, debug=True)

        logits, obj1, obj2, g_layer1, g_layer2, s = prediction

        y_pred = logits.max(dim=1)[1]
        y_preds.append(y_pred.cpu())

        y_tensor = data.y
        if isinstance(y_tensor, list):
            y_tensor = torch.tensor(y_tensor, device=DEVICE, dtype=torch.float)

        # Convert one-hot labels to class indices
        if y_tensor.ndim > 1 and y_tensor.shape[1] > 1:
            y_indices = y_tensor.sum(dim=1)
        else:
            y_indices = y_tensor.long()  # Ensure it's integer type

        y_tests.append(y_indices.view(-1).cpu())

    print(f"y-preds: {y_preds}")
    print(f"y_tests: {y_tests}")

    print(classification_report(y_tests, y_preds, digits=3))


if __name__ == "__main__":
    main()
