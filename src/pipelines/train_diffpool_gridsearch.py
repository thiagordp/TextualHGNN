import itertools
import logging

import pandas as pd
from src.utils.models_utils import get_timestamp
import torch
import tqdm
from torch import nn
import json

from src.models.graph_classification.train_and_evaluate import load_datasets, initialize_model, calculate_class_weights, \
    train_and_validate, create_loaders, test
from src.utils.general_utils import load_config, setup_logging

PARAMS_GRIDSEARCH = {
    "english": {
        'LR': [0.00001,0.0001],
        'INNER_DIM': [8, 16, 32],
        'BATCH_SIZE': [32, 64, 128],
        'SOFTMAX_ASSIGN': [True],
        "DECREASE_PROPORTION": [0.1, 0.15, 0.2]
    },
    "italian": {
        'LR': [0.001],
        'INNER_DIM': [16],
        'BATCH_SIZE': [4],
        'SOFTMAX_ASSIGN': [True],
        "DECREASE_PROPORTION": [0.05]
    },
    "portuguese": {
        'LR': [1e-3, 1e-4, 1e-5],
        'INNER_DIM': [16, 32, 64],
        'BATCH_SIZE': [2, 4, 8, 16, 32],
        'SOFTMAX_ASSIGN': [True],
        "DECREASE_PROPORTION": [0.001, 0.01, 0.02, 0.05, 0.1]
    }
}

"""
PARAM_GRID = {
    'LR': [0.001],
    'INNER_DIM': [16],
    'BATCH_SIZE': [1, 2, 4, 8, 16],
    'SOFTMAX_ASSIGN': [True],
    "DECREASE_PROPORTION": [0.05]
}
"""

LANG = "portuguese"
# LANG = "italian"
# LANG = "english"
PARAM_GRID = PARAMS_GRIDSEARCH[LANG]
CONFIG = load_config(LANG, "src/utils/config.json")

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

setup_logging(log_file=f"training_model_experiment_gridsearch_{CONFIG['DATASET']}.log")
logging.info(f"============  STARTING EXPERIMENT {CONFIG['DATASET']}  ============")
logging.info(f"CONFIG: \n{json.dumps(CONFIG, indent=3)}")


def grid_search():
    results = []
    best_val_f1 = 0
    best_config = None

    param_combinations = list(itertools.product(
        PARAM_GRID['LR'],
        PARAM_GRID['INNER_DIM'],
        PARAM_GRID['BATCH_SIZE'],
        PARAM_GRID['SOFTMAX_ASSIGN'],
        PARAM_GRID["DECREASE_PROPORTION"]
    ))

    iteration = 0
    num_combinations = len(param_combinations)

    timestamp = get_timestamp()

    for lr, inner_dim, batch_size, softmax_assign, decrease_proportion in tqdm.tqdm(param_combinations):
        iteration += 1
        logging.info(f"")

        inner_dim = int(inner_dim)
        batch_size = int(batch_size)
        softmax_assign = bool(softmax_assign)
        decrease_proportion = float(decrease_proportion)

        logging.info(
            f"||||||||||||||||||||||| Testing combination {iteration} out of {num_combinations}  ||||||||||||||||||||||||||||||")
        logging.info(
            f"Testing configuration: lr={lr}, inner_dim={inner_dim}, BATCH_SIZE={batch_size}, SOFTMAX_ASSIGN={softmax_assign}, Decrease Proportion={decrease_proportion}")
        logging.info(
            "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")

        CONFIG['LR'] = lr
        CONFIG['INNER_DIM'] = inner_dim
        CONFIG['BATCH_SIZE'] = batch_size
        CONFIG['SOFTMAX_ASSIGN'] = softmax_assign
        CONFIG['DECREASE_PROPORTION'] = decrease_proportion

        tgd_train, tgd_val, tgd_test = load_datasets(
            root=CONFIG["ROOT"],
            max_num_nodes=CONFIG["NUM_NODES"],
            node_feature_size=CONFIG["NODE_FEATURE_DIM"],
            lang=LANG
        )

        input("Press Enter to continue...")

        train_loader, val_loader, test_loader = create_loaders(
            tgd_train,
            tgd_val,
            tgd_test,
            batch_size=CONFIG["BATCH_SIZE"]
        )

        model, optimizer = initialize_model(
            in_channels=tgd_train.num_node_attributes,
            out_channels=tgd_train.num_classes,
            max_num_nodes=CONFIG["NUM_NODES"],
            lr=CONFIG["LR"],
            hidden_dim=CONFIG["HIDDEN_DIM"],
            inner_dim=CONFIG["INNER_DIM"],
            softmax_assign=CONFIG["SOFTMAX_ASSIGN"],
            decrease_proportion=CONFIG["DECREASE_PROPORTION"],
            device=DEVICE
        )

        class_weights = calculate_class_weights(tgd_train, device=DEVICE)
        loss_fn = nn.NLLLoss(weight=class_weights)

        best_model_path, val_macro_f1, avg_completeness, hybrid_score = train_and_validate(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            patience=CONFIG["PATIENCE"],
            epochs=CONFIG["EPOCHS"],
            lr=CONFIG["LR"],
            hidden_dim=CONFIG["HIDDEN_DIM"],
            batch_size=CONFIG["BATCH_SIZE"],
            use_softmax=CONFIG["SOFTMAX_ASSIGN"],
            timestamp=timestamp,
            decrease_prop=CONFIG["DECREASE_PROPORTION"],
            dataset_name=CONFIG["DATASET"],
            device=DEVICE,
            grid_search=True,
            verbose=False,
        )

        if best_model_path:
            model.load_state_dict(torch.load(best_model_path))

            val_acc, val_macro_f1, _, _, val_loss, _ = test(
                model=model,
                loader=val_loader,
                device=DEVICE,
                loss_fn=loss_fn,
                verbose=False
            )

            results.append({
                "LR": lr,
                "INNER_DIM": inner_dim,
                "BATCH_SIZE": batch_size,
                "SOFTMAX_ASSIGN": softmax_assign,
                "DECREASE_PROPORTION": decrease_proportion,
                "VAL_MACRO_F1": val_macro_f1,
                "VAL_ACC": val_acc,
                "VAL_LOSS": val_loss,
                "VAL_CONCEPT_COMP": avg_completeness,
                "VAL_HYBRID_SCORE": hybrid_score,
                "MODEL_PATH": best_model_path
            })

            if val_macro_f1 > best_val_f1:
                best_val_f1 = val_macro_f1
                best_config = results[-1]

        logging.info(
            f"Finished config LR={lr}, inner_dim={inner_dim}, BATCH_SIZE={batch_size}, SOFTMAX_ASSIGN={softmax_assign}, DECREASE_PROPORTION={decrease_proportion}")

    # Save all results to CSV
    df = pd.DataFrame(results)
    csv_path = f"gridsearch_results_{CONFIG['DATASET']}.xlsx"
    df.to_excel(csv_path, index=False)
    logging.info(f"Grid search results saved to: {csv_path}")
    logging.info(f"Best configuration: \n{json.dumps(best_config, indent=4)}")


if __name__ == "__main__":
    grid_search()