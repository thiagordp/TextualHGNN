import logging

from src.utils.general_utils import setup_logging, load_config
from src.utils.models_utils import get_timestamp

LANG = "portuguese_voto"
# LANG = "italian"
CONFIG = load_config(LANG, "src/utils/config.json")

# if LANG == "english":
#     DATASET = "IMDB"
#     # MODEL_PATH = f"models/IMDB_DiffPool_20250224_234935_lr1e-05_valmacrof1score0.8500_epoch011.pth"
#     # MODEL_PATH = f"models/IMDB_DiffPool_20250413_200733_lr0.0001_valmacrof1score0.8070_epoch097.pth"
#     MODEL_PATH = f"models/grid_search/IMDB/IMDB_DiffPool_20250425_142400_lr0.005_hd100_bs64_softmaxTrue_decrease_prop0.1_valmacrof1score0.7986_epoch014.pth"
#     EMBEDDING_PATH = "data/external/embeddings/enwiki_20180420_100d.bin"
#     max_num_nodes = 1000
# elif LANG == "italian":
#     DATASET = "Imprisonment-IT"
#     max_num_nodes = 1000
#     # MODEL_PATH = f"models/Imprisonment-IT_DiffPool_20250225_214311_lr1e-05_valmacrof1score0.6441_epoch035.pth"
#     MODEL_PATH = f"models/grid_search/Imprisonment-IT/Imprisonment-IT_DiffPool_20250424_213829_lr0.0001_hd100_bs4_softmaxTrue_decrease_prop0.05_valmacrof1score0.7183_epoch027.pth"
#     EMBEDDING_PATH = "data/external/embeddings/itwiki_20180420_100d.bin"
# else:
#     # Best model
MODEL_PATH = f"models/grid_search/STF_HC_Voto_Relatorio/best_model_DiffPool_20250903_135039_lr0.0001_hd_32_bs4_dec0.1_lk10.0_en0.01_rc0.1_ct0.1_bl0.1_rp0.1_l20.01_rep02.pth"
DATASET = "STF_HC_Voto_Relatorio"
max_num_nodes = 3000
EMBEDDING_PATH = 'data/external/embeddings/glove_legal_100.bin'

timestamp = get_timestamp()
setup_logging(log_file=f"cg_dataset_oi_Diffpool_{timestamp}.log")


logging.info("testing")