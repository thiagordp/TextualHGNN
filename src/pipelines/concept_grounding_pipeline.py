import json
import logging
import os
from pathlib import Path

import torch
from tqdm import tqdm

from src.data.text_graph_dataset_ondisk import TextGraphDatasetOnDisk
from src.models.graph_classification.gnn import DiffPool
from src.models.graph_explainability.concept_grounding import ConceptGrounding
from src.models.graph_explainability.embeddings_oracle import EmbeddingOracle
from src.models.graph_explainability.llm_oracle import LLMOracle
import torch_geometric.transforms as T

from src.utils.general_utils import plot_multidigraph_to_pdf, setup_logging, load_config


def main():

    # LANG = "english"
    LANG = "italian"
    CONFIG = load_config(LANG, "src/utils/config.json")

    if LANG == "english":
        DATASET = "IMDB"
        # MODEL_PATH = f"models/IMDB_DiffPool_20250224_234935_lr1e-05_valmacrof1score0.8500_epoch011.pth"
        # MODEL_PATH = f"models/IMDB_DiffPool_20250413_200733_lr0.0001_valmacrof1score0.8070_epoch097.pth"
        MODEL_PATH = f"models/selected_models/IMDB_DiffPool_20250425_142400_lr0.005_hd100_bs32_softmaxTrue_decrease_prop0.05_valmacrof1score0.7984_epoch008.pth"
        EMBEDDING_PATH = "data/external/embeddings/enwiki_20180420_100d.bin"
        max_num_nodes = 1000
    else:
        DATASET = "Imprisonment-IT"
        max_num_nodes = 1000
        # MODEL_PATH = f"models/Imprisonment-IT_DiffPool_20250225_214311_lr1e-05_valmacrof1score0.6441_epoch035.pth"
        MODEL_PATH = f"models/grid_search/Imprisonment-IT/Imprisonment-IT_DiffPool_20250424_213829_lr0.0001_hd100_bs4_softmaxTrue_decrease_prop0.05_valmacrof1score0.7183_epoch027.pth"
        EMBEDDING_PATH = "data/external/embeddings/itwiki_20180420_100d.bin"

    setup_logging(log_file=f"cg_dataset_{DATASET}.log")
    logging.info(f"============  STARTING CG EXPERIMENT {DATASET}  ============")
    logging.info(f"CONFIG: \n{json.dumps(CONFIG, indent=3)}\n")
    logging.info(f"MODEL CHECKPOINT: {MODEL_PATH}")

    weights = torch.load(MODEL_PATH)

    EMBEDDINGS_DATABASE_PATH = f"data/oracle/embeddings_{DATASET}.db"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ROOT = f"data/datasets/{DATASET}"
    DOCUMENTS_TO_EXPLAIN = 5

    # TODO: check the inputs below based on the loaded model.
    diffpool_model = DiffPool(
        max_num_nodes=max_num_nodes,
        in_channels=100,
        hidden_channels=100,
        out_channels=2,
        inner_channels=16,
        softmax_assign=True,
        decrease_proportion=0.05,
    )

    diffpool_model.load_state_dict(weights)
    diffpool_model = diffpool_model.to(device=DEVICE)

    tgd_test = TextGraphDatasetOnDisk(
        root=ROOT,
        split="test",
        batch_size=1,
        node_feature_size=100,
        transform=T.ToDense(num_nodes=1000),
        max_num_nodes=1000,
        lang=LANG
    )

    oracle = EmbeddingOracle(
        model_path=EMBEDDING_PATH,
        db_path=EMBEDDINGS_DATABASE_PATH,
        batch_size=500000
    )

    PROMPT_LLM_ORACLE_PATH = "data/prompts/llm_oracle.txt"

    # Retrieve the API key from environment variables for security
    api_key = os.getenv("TOGETHER_API_KEY")
    llm_oracle = LLMOracle(
        system_prompt=PROMPT_LLM_ORACLE_PATH,
        language=LANG,
        api_key=api_key,
    )

    for doc_index in tqdm(range(DOCUMENTS_TO_EXPLAIN), desc="Explaining Documents"):
        data_element = tgd_test[doc_index].to(device=DEVICE)
        cg = ConceptGrounding(
            graph_model=diffpool_model,
            embedding_oracle=oracle,
            llm_oracle=llm_oracle,
            graph=data_element,
            hyper_nodes_to_explain=5,
            nodes_per_hyper_node=5,
            original_raw_file_path=f"{ROOT}/test/raw/" ,
            language=LANG
        )

        cg.concept_grounding()

        model_checkpoint_file = MODEL_PATH.split('/')[-1].replace(".pth","")
        output_path = f"data/explanations/{DATASET}/{model_checkpoint_file}"
        os.makedirs(output_path, exist_ok=True)
        cg.save_explanation(Path(output_path) / f"{cg.data_sample_id}.json")

        nx_graphs = Path(ROOT) / "interim"
        stored_graph_paths = nx_graphs.glob("*.pt")
        stored_graph_paths = list(stored_graph_paths)

        # TODO: Make this more efficient.
        for graph_data_path in tqdm(stored_graph_paths, desc="Analyzing graphs"):
            doc_name, label, graph_nx = torch.load(graph_data_path)

            if doc_name.find(str(cg.data_sample_id)) != -1:
                plot_multidigraph_to_pdf(graph_nx, output_path=output_path + f"/graph_{cg.data_sample_id}.pdf")
                break


if __name__ == "__main__":
    main()
