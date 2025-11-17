# src/pipelines/train_diffpool_gridsearch_loss.py

import json
import logging
import os
import time

import pandas as pd
import torch
import tqdm
from torch import nn

from src.data.preprocessing import preprocessing_imdb, preprocessing_legal_pt_voto_relatorio
from src.models.graph_classification.train_and_evaluate import load_datasets, initialize_model, calculate_class_weights, \
    train_and_validate, create_loaders
from src.utils.general_utils import load_config, setup_logging
from src.utils.models_utils import get_timestamp

# Options: "graph_of_words", "phrase_subgraphs"
GRAPH_BUILDER_TYPE = "graph_of_words"

PARAMS_GRIDSEARCH = {
    "english": {
        'LR': [0.001], 'INNER_DIM': [32], 'BATCH_SIZE': [8, 32],
        'SOFTMAX_ASSIGN': [True], "DECREASE_PROPORTION": [0.1]
    },
    "italian": {
        'LR': [0.0001], 'INNER_DIM': [32], 'BATCH_SIZE': [4],
        'SOFTMAX_ASSIGN': [True], "DECREASE_PROPORTION": [0.05]
    },
    "portuguese": {
        'LR': [1e-4], 'INNER_DIM': [32, 64], 'BATCH_SIZE': [8],
        'SOFTMAX_ASSIGN': [True], "DECREASE_PROPORTION": [0.02, 0.1]
    },
    "portuguese_voto": {
        'LR': [1e-4], 'INNER_DIM': [16, 32], 'BATCH_SIZE': [4, 8],
        'SOFTMAX_ASSIGN': [True], "DECREASE_PROPORTION": [0.01, 0.05, 0.1]
    }
}

PREPROCESSING_FNS = {
    "english": preprocessing_imdb,
    "portuguese_voto": preprocessing_legal_pt_voto_relatorio
}

import itertools
import random


def generate_loss_configs(sample_size=100, seed=42):
    """
    Generates a structured and sampled grid of loss configurations.
    Each loss term can have its own set of possible magnitudes.
    """
    loss_terms = ["link", "entropy", "reconstruction", "contrastive", "balance", "repel"]

    # Define possible values for each loss term separately
    loss_values = {
        "link": [1000, 5000, 10000],  # will also be scaled by 100
        "entropy": [0.001, 0.01, 0.1, 1.0],
        "reconstruction": [0.1, 1.0],
        "contrastive": [0.01, 0.1, 1.0],
        "balance": [0.001, 0.01, 0.1, 1.0],
        "repel": [0.001, 0.01, 0.1, 1.0]
    }

    scale_loss = {}  # special scaling for 'link'

    generated_configs = []
    config_id = 1

    # Hierarchical generation: from single loss terms to all terms active
    for k in range(1, len(loss_terms) + 1):
        for active_terms in itertools.combinations(loss_terms, k):
            # Get all possible weight combinations for the active terms
            term_values = [loss_values[term] for term in active_terms]
            for weights in itertools.product(*term_values):
                config = {term: 0.0 for term in loss_terms}
                for i, term in enumerate(active_terms):
                    config[term] = weights[i] * scale_loss[term] if term in scale_loss else weights[i]

                config['id'] = f"cfg_{config_id:05d}"
                config['l2'] = 0.05

                generated_configs.append(config)
                config_id += 1

    # Baseline configuration
    final_configs = [{
        "link": 0.0, "entropy": 0.0, "reconstruction": 0.0,
        "contrastive": 0.0, "balance": 0.0, "repel": 0.0, "l2": 0.05, "id": f"cfg_{0:05d}"
    }]

    random.seed(seed)
    random.shuffle(generated_configs)
    final_configs.extend(generated_configs)

    # Sampling if too many configs
    if len(final_configs) > sample_size:
        baseline = final_configs[0]
        num_isolated_combinations = sum(len(loss_values[term]) for term in loss_terms)
        isolated_configs = final_configs[1:]
        random.shuffle(isolated_configs)

        sampled_configs = [baseline] + isolated_configs[:num_isolated_combinations]

    else:
        sampled_configs = final_configs

    sampled_configs.sort(key=lambda x: x['id'])
    return sampled_configs


LOSS_CONFIG_GRID = generate_loss_configs(sample_size=200)
LANG = "english"
# LANG = "portuguese_voto"
PARAM_GRID = PARAMS_GRIDSEARCH[LANG]
PREPROCESSING_FN=PREPROCESSING_FNS[LANG]
CONFIG = load_config(LANG, "src/utils/config.json")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
timestamp = get_timestamp()

setup_logging(log_file=f"training_model_experiment_gridsearch_{CONFIG['DATASET']}_loss_{timestamp}.log")
logging.info(f"============  STARTING EXPERIMENT {CONFIG['DATASET']}  ============")
logging.info(f"CONFIG: \n{json.dumps(CONFIG, indent=3)}")


def grid_search():
    results = []
    best_config_results = None
    best_e_score = -1.0

    param_combinations = list(itertools.product(
        PARAM_GRID['LR'], PARAM_GRID['INNER_DIM'], PARAM_GRID['BATCH_SIZE'],
        PARAM_GRID['SOFTMAX_ASSIGN'], PARAM_GRID["DECREASE_PROPORTION"]
    ))

    NUM_REPETITIONS = 3
    num_combinations = len(param_combinations) * len(LOSS_CONFIG_GRID) * NUM_REPETITIONS

    pbar = tqdm.tqdm(total=num_combinations, desc="Grid Search Progress")

    for lr, inner_dim, batch_size, softmax_assign, decrease_proportion in param_combinations:
        for loss_config in LOSS_CONFIG_GRID:
            for repetition in range(NUM_REPETITIONS):
                pbar.update(1)
                loss_id = loss_config["id"]

                logging.info("\n" + "=" * 100)
                logging.info(f"Starting Run {pbar.n}/{num_combinations} | Loss Config ID: {loss_id}")
                logging.info(f"Model Params:\n"
                             f"  LR={lr}\n"
                             f"  INNER_DIM={inner_dim}\n"
                             f"  BATCH_SIZE={batch_size}\n"
                             f"  SOFTMAX_ASSIGN={softmax_assign}\n"
                             f"  DECREASE_PROPORTION={decrease_proportion}"
                             f"  Repetition={repetition}")
                logging.info(f"Loss Config ({loss_id}):")
                for k in ["link", "entropy", "reconstruction", "contrastive", "balance", "repel", "l2"]:
                    logging.info(f"  {k.upper():<12} = {loss_config.get(k, 0.0)}")

                # --- DATA and MODEL SETUP ---
                tgd_train, tgd_val, tgd_test = load_datasets(
                    root=CONFIG["ROOT"], max_num_nodes=CONFIG["NUM_NODES"],
                    node_feature_size=CONFIG["NODE_FEATURE_DIM"], lang=LANG,
                    preprocessing_fn=PREPROCESSING_FN
                )
                train_loader, val_loader, _ = create_loaders(tgd_train, tgd_val, tgd_test, batch_size=batch_size)
                model, optimizer = initialize_model(
                    in_channels=tgd_train.num_node_attributes, out_channels=tgd_train.num_classes,
                    max_num_nodes=CONFIG["NUM_NODES"], lr=lr, hidden_dim=CONFIG["HIDDEN_DIM"],
                    inner_dim=inner_dim, softmax_assign=softmax_assign, decrease_proportion=decrease_proportion,
                    device=DEVICE, l2=loss_config["l2"]
                )
                class_weights = calculate_class_weights(tgd_train, device=DEVICE)
                loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

                # --- TRAIN AND VALIDATE ---
                best_model_path, best_epoch_results = train_and_validate(
                    model=model, train_loader=train_loader, val_loader=val_loader,
                    optimizer=optimizer, loss_fn=loss_fn, patience=CONFIG["PATIENCE"],
                    epochs=CONFIG["EPOCHS"], lr=lr, hidden_dim=CONFIG["INNER_DIM"],
                    batch_size=batch_size, use_softmax=softmax_assign, timestamp=timestamp,
                    decrease_prop=decrease_proportion, dataset_name=CONFIG["DATASET"],
                    device=DEVICE, grid_search=True, verbose=False, loss_config=loss_config,
                    repetition=repetition
                )

                # --- LOGGING AND RESULTS ---
                if best_model_path:
                    current_e_score = best_epoch_results.get("e_score", 0.0)

                    # Combine model params and results into a single row
                    result_row = {
                        "LR": lr, "INNER_DIM": inner_dim, "BATCH_SIZE": batch_size,
                        "SOFTMAX_ASSIGN": softmax_assign, "DECREASE_PROPORTION": decrease_proportion,
                        "LOSS_CONFIG_ID": loss_config["id"],
                        "Repetition": repetition,
                        **{f"loss_{k}": v for k, v in loss_config.items() if k != "id"},
                        **{k: round(v, 4) for k, v in best_epoch_results.items() if isinstance(v, (int, float))},
                        "MODEL_PATH": best_model_path
                    }

                    logging.info(
                        f"[Result] ID: {loss_id} | Best E-Score: {current_e_score:.4f} at epoch {best_epoch_results.get('best_epoch')}")

                    if current_e_score > best_e_score:
                        best_e_score = current_e_score
                        best_config_results = result_row
                        results.append(result_row)

                        logging.info(f"New overall best E-Score found: {best_e_score:.4f}")

                if results:
                    save_results(results, timestamp, best_config_results, verbose=False)

                logging.info("Sleeping for 10s...")
                time.sleep(10)

    pbar.close()

    # --- SAVE FINAL RESULTS ---
    if results:
        save_results(results, timestamp, best_config_results, verbose=True)


def save_results(target, timestamp, best_config_results, verbose=True):
    df = pd.DataFrame(target)
    # Ensure consistent column order
    cols_order = ["LOSS_CONFIG_ID", "Repetition", "E_SCORE", "HI_SCORE", "accuracy", "MACRO_F1", "COMPLETENESS",
                  "CONFORMITY",
                  "MODULARITY", "SILHOUETTE"]
    cols_order += ["LR", "INNER_DIM", "DECREASE_PROPORTION", "BATCH_SIZE", "MODEL_PATH"]
    # Add loss columns dynamically
    loss_cols = [f"loss_{k}" for k in LOSS_CONFIG_GRID[0] if k != 'id']
    cols_order += loss_cols

    # Reorder DataFrame, converting metrics to uppercase for clarity
    df.rename(columns={k: k.upper() for k in df.columns}, inplace=True)
    final_cols = [c.upper() for c in cols_order if c.upper() in df.columns]
    df = df[final_cols]

    results_dir = "grid_search_results"
    os.makedirs(results_dir, exist_ok=True)
    excel_path = os.path.join(results_dir, f"gridsearch_results_{CONFIG['DATASET']}_{timestamp}.xlsx")
    df.to_excel(excel_path, index=False)
    if verbose:
        logging.info(f"\nGrid search complete. All results saved to: {excel_path}")
        logging.info(f"\nBest Overall Configuration:\n{json.dumps(best_config_results, indent=4)}")


if __name__ == "__main__":
    grid_search()
