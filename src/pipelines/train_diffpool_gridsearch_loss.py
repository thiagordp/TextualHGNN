import itertools
import logging

import pandas as pd
from src.utils.models_utils import get_timestamp
import torch
import tqdm
from torch import nn
import json
import random

from src.models.graph_classification.train_and_evaluate import load_datasets, initialize_model, calculate_class_weights, \
    train_and_validate, create_loaders, test
from src.utils.general_utils import load_config, setup_logging

PARAMS_GRIDSEARCH = {
    "english": {
        'LR': [0.0001],
        'INNER_DIM': [64],
        'BATCH_SIZE': [16],
        'SOFTMAX_ASSIGN': [True],
        "DECREASE_PROPORTION": [0.1]
    },
    "italian": {
        'LR': [0.0001],
        'INNER_DIM': [32],
        'BATCH_SIZE': [4],
        'SOFTMAX_ASSIGN': [True],
        "DECREASE_PROPORTION": [0.05]
    },
    "portuguese": {
        'LR': [1e-3, 1e-4],
        'INNER_DIM': [16, 32, 64],
        'BATCH_SIZE': [2, 4, 8],
        'SOFTMAX_ASSIGN': [True],
        "DECREASE_PROPORTION": [0.02, 0.05, 0.1]
    }
}

import itertools
import random


def generate_loss_configs(sample_size=300, seed=42):
    magnitudes = [1e-2, 1e-1, 1e0, 1e1]

    full_grid = []
    i = 0
    for entropy, link, recon, contrast, balance, repel in itertools.product(magnitudes, repeat=6):
        config = {
            "id": f"cfg_{i + 1:05d}",
            "link": link * 100,
            "entropy": entropy,
            "reconstruction": recon,
            "contrastive": contrast,
            "balance": balance,
            "repel": repel
        }
        full_grid.append(config)
        i += 1

    # Random subset
    if sample_size and  len(full_grid) > sample_size:
        random.seed(seed)
        sampled_grid = random.sample(full_grid, k=sample_size - 1)

        # TODO: check duplicated ids.

        sampled_grid.append({
            "id": f"cfg_{0:05d}",
            "link": 0,
            "entropy": 0,
            "reconstruction": 0.0,
            "contrastive": 0.0,
            "balance": 0.0,
            "repel": 0.0
        })

        # Sort the sampled configurations by 'id'
        sampled_grid.sort(key=lambda x: x['id'])
        return sampled_grid

    # If the full grid is smaller than the sample size / sort it by 'id'
    full_grid.sort(key=lambda x: x['id'])
    return full_grid


LOSS_CONFIG_GRID = generate_loss_configs(sample_size=5)

# LANG = "italian"
# LANG = "english"
LANG = "portuguese"
PARAM_GRID = PARAMS_GRIDSEARCH[LANG]
CONFIG = load_config(LANG, "src/utils/config.json")

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

setup_logging(log_file=f"training_model_experiment_gridsearch_{CONFIG['DATASET']}_loss.log")
logging.info(f"============  STARTING EXPERIMENT {CONFIG['DATASET']}  ============")
logging.info(f"CONFIG: \n{json.dumps(CONFIG, indent=3)}")


def grid_search():
    results = []
    best_config = None
    best_hybrid_score = best_val_f1 = 0

    param_combinations = list(itertools.product(
        PARAM_GRID['LR'],
        PARAM_GRID['INNER_DIM'],
        PARAM_GRID['BATCH_SIZE'],
        PARAM_GRID['SOFTMAX_ASSIGN'],
        PARAM_GRID["DECREASE_PROPORTION"]
    ))

    iteration = 0
    num_combinations = len(param_combinations) * len(LOSS_CONFIG_GRID)
    timestamp = get_timestamp()

    for lr, inner_dim, batch_size, softmax_assign, decrease_proportion in tqdm.tqdm(param_combinations):

        inner_dim = int(inner_dim)
        batch_size = int(batch_size)
        softmax_assign = bool(softmax_assign)
        decrease_proportion = float(decrease_proportion)

        for loss_config in LOSS_CONFIG_GRID:
            loss_id = loss_config["id"]
            iteration += 1

            logging.info("\n" + "=" * 100)
            logging.info(f"[{iteration}/{num_combinations}] Grid Search Configuration")
            logging.info(f"Model Params:\n"
                         f"  LR={lr}\n"
                         f"  INNER_DIM={inner_dim}\n"
                         f"  BATCH_SIZE={batch_size}\n"
                         f"  SOFTMAX_ASSIGN={softmax_assign}\n"
                         f"  DECREASE_PROPORTION={decrease_proportion}")
            logging.info(f"Loss Config ({loss_id}):")
            for k in ["link", "entropy", "reconstruction", "contrastive", "balance", "repel"]:
                logging.info(f"  {k.upper():<12} = {loss_config.get(k, 0.0)}")

            CONFIG.update({
                'LR': lr,
                'INNER_DIM': int(inner_dim),
                'BATCH_SIZE': int(batch_size),
                'SOFTMAX_ASSIGN': bool(softmax_assign),
                'DECREASE_PROPORTION': float(decrease_proportion)
            })

            tgd_train, tgd_val, tgd_test = load_datasets(
                root=CONFIG["ROOT"],
                max_num_nodes=CONFIG["NUM_NODES"],
                node_feature_size=CONFIG["NODE_FEATURE_DIM"],
                lang=LANG
            )

            train_loader, val_loader, test_loader = create_loaders(
                tgd_train, tgd_val, tgd_test, batch_size=batch_size
            )

            model, optimizer = initialize_model(
                in_channels=tgd_train.num_node_attributes,
                out_channels=tgd_train.num_classes,
                max_num_nodes=CONFIG["NUM_NODES"],
                lr=lr,
                hidden_dim=CONFIG["HIDDEN_DIM"],
                inner_dim=inner_dim,
                softmax_assign=softmax_assign,
                decrease_proportion=decrease_proportion,
                device=DEVICE
            )

            class_weights = calculate_class_weights(tgd_train, device=DEVICE)

            # CrossEntropyLoss with:
            # - class_weights: adjusts loss per class to handle imbalance
            # - label_smoothing=0.2: smooths targets by replacing one-hot label y_i = 1 with y_i = 1 - ε,
            #   and y_j ≠ i with y_j = ε / (K - 1), where ε = 0.2 and K = number of classes
            loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.2)

            best_model_path, val_macro_f1, avg_completeness, hybrid_score = train_and_validate(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                patience=CONFIG["PATIENCE"],
                epochs=CONFIG["EPOCHS"],
                lr=lr,
                hidden_dim=CONFIG["HIDDEN_DIM"],
                batch_size=batch_size,
                use_softmax=softmax_assign,
                timestamp=timestamp,
                decrease_prop=decrease_proportion,
                dataset_name=CONFIG["DATASET"],
                device=DEVICE,
                grid_search=True,
                verbose=False,
                loss_config=loss_config
            )

            if best_model_path:
                result_row = {
                    "LR": lr,
                    "INNER_DIM": inner_dim,
                    "BATCH_SIZE": batch_size,
                    "SOFTMAX_ASSIGN": softmax_assign,
                    "DECREASE_PROPORTION": decrease_proportion,
                    "LOSS_CONFIG_ID": loss_config["id"],
                    **{k: v for k, v in loss_config.items() if k != "id"},
                    "VAL_MACRO_F1": round(val_macro_f1, 4),
                    "COMPLETENESS": round(avg_completeness, 4),
                    "HYBRID_SCORE": round(hybrid_score, 4),
                    "MODEL_PATH": best_model_path,
                }

                # Append all loss weights with consistent formatting
                # for key in ["link", "entropy", "reconstruction", "contrastive", "balance", "repel"]:
                #    result_row[f"LOSS_{key.upper()}"] = loss_config.get(key, 0.0)

                results.append(result_row)

                # Track best by hybrid score
                if hybrid_score > best_hybrid_score:
                    best_hybrid_score = hybrid_score
                    best_val_f1 = val_macro_f1
                    best_config = result_row

            logging.info(f"[Result] ID: {loss_id} | F1: {val_macro_f1:.4f} | "
                         f"Comp: {avg_completeness:.4f} | Hybrid: {hybrid_score:.4f}")

    # Save all results to CSV
    df = pd.DataFrame(results)
    csv_path = f"gridsearch_results_{CONFIG['DATASET']}_loss.xlsx"
    df.to_excel(csv_path, index=False)
    logging.info(f"\nGrid search results saved to: {csv_path}")
    logging.info(f"\nBest configuration by hybrid score:\n{json.dumps(best_config, indent=4)}")


if __name__ == "__main__":
    grid_search()
