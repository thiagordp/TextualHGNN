"""

@author Thiago R Dal Pont
"""
import json
import logging

import torch
import torch_geometric.transforms as T
from sklearn.metrics import classification_report, f1_score, accuracy_score
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from src.data.text_graph_dataset_ondisk import TextGraphDatasetOnDisk
from src.models.graph_classification.gnn import DiffPool
from src.models.graph_classification.train_and_evaluate import create_loaders
from src.models.graph_explainability.cg_evaluation_metrics import ConceptCompletenessCalculator, \
    ConceptConformityCalculator, ModularityCalculator, SilhouetteScoreCalculator
from src.utils.general_utils import load_config, setup_logging
from src.utils.models_utils import get_timestamp

PARAMS_GRIDSEARCH = {
    'LR': [1e-4], 'INNER_DIM': [32], 'BATCH_SIZE': [4],
    'SOFTMAX_ASSIGN': [True], "DECREASE_PROPORTION": [0.1]
}

LANG = "portuguese_voto"
MODEL_PATH = f"models/grid_search/STF_HC_Voto_Relatorio/best_model_DiffPool_20250903_135039_lr0.0001_hd_32_bs4_dec0.1_lk0.0_en0.0_rc0.0_ct0.0_bl0.0_rp0.0_l20.01_rep00.pth"
# MODEL_PATH = f"models/grid_search/STF_HC_Voto_Relatorio/best_model_DiffPool_20250903_135039_lr0.0001_hd_32_bs4_dec0.1_lk10.0_en0.01_rc0.1_ct0.1_bl0.1_rp0.1_l20.01_rep02.pth"
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

    logging.info("Starting..")
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

    logging.info("Loaded model")

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

    logging.info("Loaded dataset")
    train_loader, val_loader, test_loader = create_loaders(tgd_train, tgd_val, tgd_test, batch_size=1)

    diffpool_model.eval()
    all_preds, all_labels = [], []
    # Initialize metric calculators for the epoch
    completeness_calc = ConceptCompletenessCalculator()
    conformity_calc = ConceptConformityCalculator()
    modularity_calc = ModularityCalculator()
    silhouette_calc = SilhouetteScoreCalculator()

    logging.info("Starting experiment")
    with torch.no_grad():
        for data in test_loader:
            data = data.to(DEVICE)
            model_outputs = diffpool_model(data.x, data.adj, data.mask, debug=True)
            logits, _, _, _, _, (s01, s12) = model_outputs
            pred = logits.max(dim=1)[1]

            all_preds.append(pred.cpu())

            y_tensor = data.y
            if isinstance(y_tensor, list):
                y_tensor = torch.tensor(y_tensor, device=DEVICE, dtype=torch.float)

            # Convert one-hot labels to class indices
            if y_tensor.ndim > 1 and y_tensor.shape[1] > 1:
                y_indices = y_tensor.sum(dim=1)
            else:
                y_indices = y_tensor.long()  # Ensure it's integer type

            all_labels.append(y_indices.view(-1).cpu())

            # Get concept assignments (we focus on the first layer for interpretability)
            concept_ids_batch = s01.argmax(dim=-1)

            # The 'batch' attribute maps each node to its graph in the batch
            batch_vector = data.batch if hasattr(data, 'batch') else torch.zeros(s01.size(1), dtype=torch.long,
                                                                                 device=DEVICE)
            num_graphs_in_batch = data.x.size(0)

            for i in range(num_graphs_in_batch):
                num_nodes = int(data.mask[i].sum())
                if num_nodes == 0: continue

                x_i = data.x[i, :num_nodes]
                adj_i = data.adj[i, :num_nodes, :num_nodes]

                # Ensure y_i is a tensor for the Data object
                y_val = data.y[i]

                if isinstance(y_val, list):
                    # Handle case where y is a list of lists/tensors
                    y_i = torch.tensor(y_val, device=DEVICE, dtype=torch.long)
                elif isinstance(y_val, (int, float)):
                    # Handle case where y is a flat list of numbers
                    y_i = torch.tensor([y_val], device=DEVICE, dtype=torch.long)
                else:  # It's already a tensor
                    y_i = y_val

                y_i = y_i.unsqueeze(dim=0)
                if y_i.ndim > 1 and y_i.shape[1] > 1:
                    y_i = y_i.sum(dim=1)
                else:
                    y_i = y_i.long()  # Ensure it's integer type

                edge_index_i = dense_to_sparse(adj_i)[0]
                single_graph_data = Data(x=x_i.clone(), edge_index=edge_index_i.clone(), y=y_i.clone())

                single_graph_concepts = concept_ids_batch[i, :num_nodes]

                completeness_calc.add_item(y_i, single_graph_concepts)
                conformity_calc.add_item(single_graph_data, single_graph_concepts)
                modularity_calc.add_item(single_graph_data, single_graph_concepts)
                silhouette_calc.add_item(single_graph_data, single_graph_concepts)

    # Calculate final metrics for the epoch
    logging.info("Calculated completeness score")
    preds, labels = torch.cat(all_preds).numpy(), torch.cat(all_labels).numpy()

    with torch.no_grad():
        total_params = 0
        sum_abs_params = 0.0
        for param in diffpool_model.parameters():
            if param.requires_grad:
                total_params += param.numel()
                sum_abs_params += torch.sum(torch.abs(param.data)).item()

    avg_abs_params = sum_abs_params / total_params if total_params > 0 else 0

    macro_f1 = f1_score(labels, preds, average='macro', zero_division=0)
    acc = accuracy_score(labels, preds)
    completeness = completeness_calc.calculate()
    conformity = conformity_calc.calculate()
    modularity = modularity_calc.calculate()
    silhouette = silhouette_calc.calculate()

    # Calculate HI-Score and E-Score
    hi_score = (completeness + conformity) / 2
    e_score = (2 * macro_f1 * hi_score) / (macro_f1 + hi_score) if (macro_f1 + hi_score) > 0 else 0

    print(f"y-preds: {preds}")
    print(f"y_tests: {labels}")

    logging.info("Classification report")
    logging.info(classification_report(labels, preds, digits=3))


if __name__ == "__main__":
    main()
