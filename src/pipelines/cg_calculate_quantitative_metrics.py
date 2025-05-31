import torch
from src.models.graph_classification.gnn import DiffPool
from src.utils.general_utils import load_config, setup_logging
from src.models.graph_classification.train_and_evaluate import load_datasets, create_loaders
from src.models.graph_explainability.cg_evaluation_metrics import ConceptCompletenessCalculatorV2
import logging

setup_logging(
    log_folder="logs",
    log_file="cg_calculate_quantitative_metrics.log"
)


LANG = "english"

CONFIG = load_config(LANG, "src/utils/config.json")
if LANG == "english":
    DATASET = "IMDB"
    # MODEL_PATH = f"models/IMDB_DiffPool_20250224_234935_lr1e-05_valmacrof1score0.8500_epoch011.pth"
    # MODEL_PATH = f"models/IMDB_DiffPool_20250413_200733_lr0.0001_valmacrof1score0.8070_epoch097.pth"
    MODEL_PATH = f"models/grid_search/IMDB/IMDB_DiffPool_20250425_142400_lr0.005_hd100_bs64_softmaxTrue_decrease_prop0.1_valmacrof1score0.7986_epoch014.pth"
    EMBEDDING_PATH = "data/external/embeddings/enwiki_20180420_100d.bin"
    max_num_nodes = 1000
else:
    DATASET = "Imprisonment-IT"
    max_num_nodes = 1000
    # MODEL_PATH = f"models/Imprisonment-IT_DiffPool_20250225_214311_lr1e-05_valmacrof1score0.6441_epoch035.pth"
    MODEL_PATH = f"models/grid_search/Imprisonment-IT/Imprisonment-IT_DiffPool_20250424_213829_lr0.0001_hd100_bs4_softmaxTrue_decrease_prop0.05_valmacrof1score0.7183_epoch027.pth"
    EMBEDDING_PATH = "data/external/embeddings/itwiki_20180420_100d.bin"
DEVICE = torch.device("cpu")

logging.info("================================================================================================")
logging.info(f"Lang: {LANG}")
logging.info(f"Dataset: {DATASET}")
logging.info(f"Max Num Nodes: {max_num_nodes}")
logging.info(f"Model Path: {MODEL_PATH}")
logging.info(f"Embedding Path: {EMBEDDING_PATH}")

# Imprisonment dims
# diffpool_model = DiffPool(
#     max_num_nodes=max_num_nodes,
#     in_channels=100,
#     hidden_channels=100,
#     out_channels=2,
#     inner_channels=16,
#     softmax_assign=True,
#     decrease_proportion=0.05,
# )

# IMDB dims
diffpool_model = DiffPool(
    max_num_nodes=max_num_nodes,
    in_channels=100,
    hidden_channels=100,
    out_channels=2,
    inner_channels=64,
    softmax_assign=True,
    decrease_proportion=0.1,
)


weights = torch.load(MODEL_PATH)
diffpool_model.load_state_dict(weights)
diffpool_model = diffpool_model.to(device=DEVICE)
logging.info("Loaded model")

tgd_train, tgd_val, tgd_test = load_datasets(
    root=CONFIG["ROOT"],
    max_num_nodes=CONFIG["NUM_NODES"],
    node_feature_size=CONFIG["NODE_FEATURE_DIM"],
    lang=LANG
)



train_loader, val_loader, test_loader = create_loaders(
    tgd_train,
    tgd_val,
    tgd_test,
    batch_size=1000
)
logging.info("Loaded datasets and dataloaders")

logging.info("calculating Metrics")
completeness_calculator = ConceptCompletenessCalculatorV2(diffpool_model, train_loader, test_loader, DEVICE)

completeness_scores = completeness_calculator.calculate_concept_completeness()



# Assuming `diffpool_model` is your pre-trained DiffPool model
# Assuming `train_loader` and `test_loader` are DataLoader objects for training and test datasets


logging.info("\nFinal Concept Completeness Scores (per layer):")
for layer, (avg_score, std_score) in enumerate(list(completeness_scores), start=1):
    logging.info(f"Layer {layer}: {avg_score:.4f} ± {std_score:.4f}")
