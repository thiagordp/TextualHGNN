# src/pipelines/train_diffpool_gridsearch_loss_complete.py

import json
import logging
import os
import time

import pandas as pd
import torch
import torch.nn as nn
import optuna
from optuna.trial import TrialState
from torch import nn

from src.data.preprocessing import preprocessing_imdb, preprocessing_legal_pt_voto_relatorio, preprocess_text_merged
from src.models.graph_classification.train_and_evaluate import load_datasets, initialize_model, calculate_class_weights, \
    train_and_validate, create_loaders, test
from src.utils.general_utils import load_config, setup_logging
from src.utils.models_utils import get_timestamp

PREPROCESSING_FNS = {
    "english": preprocess_text_merged,
    # "english": preprocessing_imdb,
    "portuguese_voto": preprocessing_legal_pt_voto_relatorio
}

# --- Configuration ---
# LANG = "portuguese_voto"
LANG = "english"
PREPROCESSING_FN = PREPROCESSING_FNS[LANG]
CONFIG = load_config(LANG, "src/utils/config.json")
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
timestamp = get_timestamp()
# GRAPH_BUILDER_TYPE = "graph_of_words"  # Or "phrase_subgraphs"
GRAPH_BUILDER_TYPE = "phrase_subgraphs"  # Or "phrase_subgraphs"
USE_PMI = False

# Optuna Configuration
N_TRIALS = 2  # Total number of HPO trials to run
FINAL_TRAIN_EPOCHS = 50  # Epochs for final retraining
HPO_STORAGE_DB = f"sqlite:///hpo_results_{CONFIG['DATASET']}.db"
HPO_STUDY_NAME = f"diffpool_loss_hpo_{CONFIG['DATASET']}_{timestamp}"
RESULTS_CSV_PATH = f"hpo_results_{CONFIG['DATASET']}_{timestamp}.csv"

setup_logging(log_file=f"hpo_experiment_{CONFIG['DATASET']}_loss_{timestamp}.log")
logging.info(f"============  STARTING HPO EXPERIMENT {CONFIG['DATASET']}  ============")
logging.info(f"CONFIG: \n{json.dumps(CONFIG, indent=3)}")
logging.info(f"DEVICE: {DEVICE}")
logging.info(f"GRAPH BUILDER TYPE: {GRAPH_BUILDER_TYPE}")
logging.info(f"Optuna Storage: {HPO_STORAGE_DB}")
logging.info(f"Optuna Study Name: {HPO_STUDY_NAME}")
logging.info(f"Use PMI: {USE_PMI}")


def objective(trial: optuna.Trial, tgd_train, tgd_val) -> float:
    """
    The main objective function for Optuna.
    A single trial will:
    1. Suggest model and loss hyperparameters.
    2. Load data, initialize model, and train.
    3. Return the 'E-Score' (our IAPS metric) for Optuna to maximize.
    """
    try:
        # --- 1. Suggest Model Hyperparameters ---
        lr = trial.suggest_float("LR", 1e-5, 1e-3, log=True)
        inner_dim = trial.suggest_categorical("INNER_DIM", [32, 64, 128])
        batch_size = trial.suggest_categorical("BATCH_SIZE", [16, 32, 64])
        softmax_assign = trial.suggest_categorical("SOFTMAX_ASSIGN", [True])
        decrease_proportion = trial.suggest_float("DECREASE_PROPORTION", 0.01, 0.2, log=True)

        # --- 2. Suggest Loss Hyperparameters ---
        # We replace the complex generator with Optuna's categorical sampler,
        # which can learn to pick 0.0 (i.e., "turn off" the loss).
        loss_config = {}
        loss_config["l2"] = trial.suggest_float("loss_l2", 1e-5, 1e-2, log=True)
        loss_config["link"] = trial.suggest_categorical("loss_link", [0.0, 0.001, 0.01, 0.1, 1.0])
        loss_config["entropy"] = trial.suggest_categorical("loss_entropy", [0.0, 0.001, 0.01, 0.1, 1.0])
        loss_config["reconstruction"] = trial.suggest_categorical("loss_reconstruction", [0.0, 0.001, 0.01, 0.1, 1.0])
        loss_config["contrastive"] = trial.suggest_categorical("loss_contrastive", [0.0, 0.001, 0.01, 0.1, 1.0])
        loss_config["balance"] = trial.suggest_categorical("loss_balance", [0.0, 0.001, 0.01, 0.1, 1.0])
        loss_config["repel"] = trial.suggest_categorical("loss_repel", [0.0, 0.001, 0.01, 0.1, 1.0])

        # Add loss_id for model naming compatibility
        loss_config["id"] = f"trial_{trial.number}"

        logging.info("\n" + "=" * 100)
        logging.info(f"Starting Trial {trial.number}/{N_TRIALS}")
        logging.info(f"Model Params:\n"
                     f"  LR={lr}\n"
                     f"  INNER_DIM={inner_dim}\n"
                     f"  BATCH_SIZE={batch_size}\n"
                     f"  SOFTMAX_ASSIGN={softmax_assign}\n"
                     f"  DECREASE_PROPORTION={decrease_proportion}")
        logging.info(f"Loss Config:")
        for k, v in loss_config.items():
            if k != 'id': logging.info(f"  {k.upper():<16} = {v}")

        # --- 3. DATA and MODEL SETUP ---
        # Data is already loaded; just create new loaders for the trial's batch size
        train_loader, val_loader, _ = create_loaders(tgd_train, tgd_val, None, batch_size=batch_size)

        model, optimizer = initialize_model(
            in_channels=tgd_train.num_node_attributes, out_channels=tgd_train.num_classes,
            max_num_nodes=CONFIG["NUM_NODES"], lr=lr, hidden_dim=CONFIG["HIDDEN_DIM"],
            inner_dim=inner_dim, softmax_assign=softmax_assign, decrease_proportion=decrease_proportion,
            device=DEVICE, l2=loss_config["l2"]
        )

        # Class weights are static, but we re-calculate for clarity (it's fast)
        class_weights = calculate_class_weights(tgd_train, device=DEVICE)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

        # --- 4. TRAIN AND VALIDATE ---
        best_model_path, best_epoch_results = train_and_validate(
            model=model, train_loader=train_loader, val_loader=val_loader,
            optimizer=optimizer, loss_fn=loss_fn, patience=CONFIG["PATIENCE"],
            epochs=CONFIG["EPOCHS"], lr=lr,
            hidden_dim=inner_dim,  # Pass inner_dim for logging/model naming
            batch_size=batch_size, use_softmax=softmax_assign, timestamp=timestamp,
            decrease_prop=decrease_proportion, dataset_name=CONFIG["DATASET"],
            device=DEVICE, grid_search=True, verbose=False, loss_config=loss_config,
            repetition=trial.number  # Use trial number as repetition ID
        )

        # --- 5. Return Objective Metric (E_SCORE) ---
        # This is your "IAPS" metric
        e_score = best_epoch_results.get("e_score", 0.0)

        logging.info(
            f"[Result] Trial {trial.number} finished. Best E-Score: {e_score:.4f} at epoch {best_epoch_results.get('best_epoch')}")

        # Store key results for later analysis
        trial.set_user_attr("e_score", e_score)
        trial.set_user_attr("hi_score", best_epoch_results.get("hi_score", 0.0))
        trial.set_user_attr("macro_f1", best_epoch_results.get("macro_f1", 0.0))
        trial.set_user_attr("completeness", best_epoch_results.get("completeness", 0.0))
        trial.set_user_attr("conformity", best_epoch_results.get("conformity", 0.0))
        trial.set_user_attr("model_path", best_model_path)

        return e_score

    except Exception as e:
        logging.error(f"Trial {trial.number} failed with exception: {e}", exc_info=True)
        # Prune or return a very bad value (0.0) if the trial fails (e.g., OOM)
        return 0.0


def save_optuna_results(study: optuna.Study, csv_path: str):
    """Saves the complete Optuna study results to a CSV file."""
    try:
        df = study.trials_dataframe()

        # Clean up column names for readability
        df.columns = df.columns.str.replace("params_", "", regex=False)
        df.columns = df.columns.str.replace("user_attrs_", "", regex=False)

        df.to_csv(csv_path, index=False)
        logging.info(f"\nOptuna study results saved to: {csv_path}")

        logging.info("\nBest Trial:")
        best = study.best_trial
        logging.info(f"  Number: {best.number}")
        logging.info(f"  Value (E-Score): {best.value:.4f}")
        logging.info("  Params:")
        for k, v in best.params.items():
            logging.info(f"    {k}: {v}")
        logging.info("  Metrics:")
        for k, v in best.user_attrs.items():
            logging.info(f"    {k}: {v}")

    except Exception as e:
        logging.error(f"Failed to save Optuna results: {e}", exc_info=True)


def main():
    """Main execution function to run HPO."""

    logging.info("Loading datasets (this may take a moment)...")
    # --- 1. Load Data ONCE ---
    tgd_train, tgd_val, tgd_test = load_datasets(
        root=CONFIG["ROOT"], max_num_nodes=CONFIG["NUM_NODES"],
        node_feature_size=CONFIG["NODE_FEATURE_DIM"], lang=LANG,
        preprocessing_fn=PREPROCESSING_FN,
        graph_builder_type=GRAPH_BUILDER_TYPE,
        use_pmi=USE_PMI
    )
    logging.info("Datasets loaded successfully.")
    logging.info(f"Train samples: {len(tgd_train)}, Val samples: {len(tgd_val)}, Test samples: {len(tgd_test)}")

    # --- 2. Create or Load Optuna Study ---
    study = optuna.create_study(
        study_name=HPO_STUDY_NAME,
        storage=HPO_STORAGE_DB,
        direction="maximize",  # We want to MAXIMIZE the E-Score
        load_if_exists=True  # This allows yo   u to resume a stopped study
    )

    # --- 3. Run HPO ---
    logging.info(f"Starting/Resuming HPO study. Running for {N_TRIALS} trials.")
    obj_with_data = lambda trial: objective(trial, tgd_train, tgd_val)
    study.optimize(obj_with_data, n_trials=N_TRIALS, show_progress_bar=True)

    # --- 4. Save Final Results ---
    logging.info("HPO complete.")
    save_optuna_results(study, RESULTS_CSV_PATH)

    # --- [NEW] 5. Retrain and Test Best Model ---
    logging.info("\n" + "=" * 100)
    logging.info("STARTING FINAL TRAINING WITH BEST PARAMETERS...")

    best_params = study.best_params
    # Re-create the best loss_config from params
    loss_config = {"id": "best_run"}
    for k in best_params:
        if k.startswith("loss_"):
            loss_config[k.replace("loss_", "")] = best_params[k]

    logging.info(f"Best HPO Params:\n{json.dumps(best_params, indent=2)}")

    # --- 6. Create Final DataLoaders ---
    # We train on the same train/val splits used during HPO
    final_train_loader, final_val_loader, final_test_loader = create_loaders(
        tgd_train, tgd_val, tgd_test, batch_size=best_params['BATCH_SIZE']
    )

    # --- 7. Initialize Final Model ---
    final_model, final_optimizer = initialize_model(
        in_channels=tgd_train.num_node_attributes,
        out_channels=tgd_train.num_classes,
        max_num_nodes=CONFIG["NUM_NODES"],
        lr=best_params['LR'],
        hidden_dim=CONFIG["HIDDEN_DIM"],
        inner_dim=best_params['INNER_DIM'],
        softmax_assign=best_params['SOFTMAX_ASSIGN'],
        decrease_proportion=best_params['DECREASE_PROPORTION'],
        device=DEVICE,
        l2=loss_config["l2"]
    )

    final_loss_fn = nn.CrossEntropyLoss(
        weight=calculate_class_weights(tgd_train, device=DEVICE),
        label_smoothing=0.1
    )

    # --- 8. Retrain for 50 Epochs ---
    logging.info(f"Retraining for {FINAL_TRAIN_EPOCHS} epochs...")

    best_model_path, best_epoch_results = train_and_validate(
        model=final_model,
        train_loader=final_train_loader,
        val_loader=final_val_loader,
        optimizer=final_optimizer,
        loss_fn=final_loss_fn,
        patience=FINAL_TRAIN_EPOCHS,  # Set patience = epochs to run full duration
        epochs=FINAL_TRAIN_EPOCHS,  # Set epochs to 50
        lr=best_params['LR'],
        hidden_dim=best_params['INNER_DIM'],
        batch_size=best_params['BATCH_SIZE'],
        use_softmax=best_params['SOFTMAX_ASSIGN'],
        timestamp=f"{timestamp}_final",  # Add a new timestamp
        decrease_prop=best_params['DECREASE_PROPORTION'],
        dataset_name=CONFIG["DATASET"],
        device=DEVICE,
        grid_search=False,  # This is not a grid search
        verbose=True,  # We want to see the epoch logs
        loss_config=loss_config,
        repetition=0
    )

    logging.info(f"Final training complete. Best model saved to: {best_model_path}")

    # --- 9. Test on Test Set ---
    logging.info("\n" + "=" * 100)
    logging.info("EVALUATING BEST MODEL ON **TEST SET**...")

    # Load the best model state from the retraining
    final_model.load_state_dict(torch.load(best_model_path))

    test_acc, test_f1, _, _, test_loss, test_report = test(
        final_model,
        final_test_loader,
        final_loss_fn,
        DEVICE,
        verbose=False
    )

    logging.info("=" * 50)
    logging.info("           FINAL TEST RESULTS           ")
    logging.info("=" * 50)
    logging.info(f"  Test Accuracy: {test_acc:.4f}")
    logging.info(f"  Test Macro F1: {test_f1:.4f}")
    logging.info(f"  Test Loss:     {test_loss:.4f}")
    logging.info(
        f"  (Best Val E-Score was: {best_epoch_results.get('e_score', 0.0):.4f} at epoch {best_epoch_results.get('best_epoch')})")
    logging.info(f"  Classification Report:\n{json.dumps(test_report, indent=2)}")


if __name__ == "__main__":
    main()
