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
# Best model
MODEL_PATH = f"models/grid_search/STF_HC_Voto_Relatorio/best_model_DiffPool_20250903_135039_lr0.0001_hd_32_bs4_dec0.1_lk10.0_en0.01_rc0.1_ct0.1_bl0.1_rp0.1_l20.01_rep02.pth"
DATASET = "STF_HC_Voto_Relatorio"
max_num_nodes = 3000
EMBEDDING_PATH = 'data/external/embeddings/glove_legal_100.bin'

timestamp = get_timestamp()
setup_logging(log_file=f"cg_dataset_{DATASET}_Diffpool_pipeline_{timestamp}.log")

import glob
import json
import logging
import os
from pathlib import Path

import networkx as nx
import torch

from src.data.text_graph_dataset_ondisk import TextGraphDatasetOnDisk
from src.models.graph_classification.gnn import DiffPool
from src.models.graph_explainability.concept_grounding import ConceptGrounding
from src.models.graph_explainability.embeddings_oracle import EmbeddingOracle
from src.models.graph_explainability.llm_oracle import LLMOracle
import torch_geometric.transforms as T


def main():
    logging.info(f"============  STARTING CG EXPERIMENT {DATASET}  ============")
    logging.info(f"CONFIG: \n{json.dumps(CONFIG, indent=3)}\n")
    logging.info(f"MODEL CHECKPOINT: {MODEL_PATH}")

    weights = torch.load(MODEL_PATH)

    EMBEDDINGS_DATABASE_PATH = f"data/oracle/embeddings_{DATASET}.db"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ROOT = f"data/datasets/{DATASET}"
    DOCUMENTS_TO_EXPLAIN = 6

    # TODO: check the inputs below based on the loaded model.
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
        split="test",
        batch_size=1,
        node_feature_size=100,
        transform=T.ToDense(num_nodes=3000),
        max_num_nodes=3000,
        lang=LANG
    )

    logging.info(f"Test set size: {len(tgd_test)}")

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
    nx_graphs = Path(ROOT) / "test" / "interim"
    nx_graphs = str(nx_graphs) + "/*.pt"

    stored_graph_paths = glob.glob(nx_graphs)
    stored_graph_paths = list(stored_graph_paths)
    logging.info(f"NX Graphs at '{nx_graphs}' with: {len(stored_graph_paths)} NX graphs.")

    docs_explained = 0
    doc_index = 1

    max_per_label = int(DOCUMENTS_TO_EXPLAIN / 2)
    predicted_per_label = {}
    while sum(predicted_per_label.values()) < DOCUMENTS_TO_EXPLAIN and doc_index < len(tgd_test):

        doc_index += 1

        logging.info(f"Explaining Document {docs_explained} out of {DOCUMENTS_TO_EXPLAIN}")

        model_checkpoint_file = MODEL_PATH.split('/')[-1].replace(".pth", "")
        output_path = f"data/explanations/{DATASET}/{model_checkpoint_file}"

        data_element = tgd_test[doc_index * -1].to(device=DEVICE)
        cg = ConceptGrounding(
            graph_model=diffpool_model,
            embedding_oracle=oracle,
            llm_oracle=llm_oracle,
            graph=data_element,
            hyper_nodes_to_explain=5,
            nodes_per_hyper_node=5,
            original_raw_file_path=f"{ROOT}/test/raw/",
            language=LANG
        )

        if cg.y_test in predicted_per_label and predicted_per_label[cg.y_test] > max_per_label:
            logging.warning(f"Skipping label {cg.y_test} because it has already been explained.")
            continue
        if cg.y_test != cg.y_pred:
            logging.warning(f"Skipping wrong prediction (Doc id: {cg.data_sample_id}")
            continue

        graph = retrieve_graph(stored_graph_paths, str(cg.data_sample_id), output_path=output_path)

        if graph is not None and graph.number_of_nodes() <= 1000:

            if cg.y_test not in predicted_per_label:
                predicted_per_label[cg.y_test] = 1
            else:
                predicted_per_label[cg.y_test] += 1

            docs_explained += 1
            node_to_token_map = {i: token for i, token in enumerate(graph.nodes())}

            cg.concept_grounding()

            os.makedirs(output_path, exist_ok=True)
            cg.save_explanation(Path(output_path) / f"{cg.data_sample_id}.json")
            # interactive_output_path = Path(output_path) / f"{cg.data_sample_id}_visualization.html"
            #
            # # In pipelines/concept_grounding_pipeline.py
            # visualize_diffpool_explanation(
            #     cg=cg,
            #     nx_graph=graph,
            #     output_filename=str(interactive_output_path),
            #     assignment_threshold=0.1  # Example of using the new parameter
            # )
        else:
            logging.info("Skipping graph > 300")


def retrieve_graph(list_of_paths_to_graphs, sample_id, output_path) -> nx.MultiDiGraph | None:
    for graph_data_path in list_of_paths_to_graphs:
        doc_name, label, graph_nx = torch.load(graph_data_path)

        if doc_name.find(sample_id) >= 0:
            if graph_nx.number_of_nodes() <= 300:
                # plot_multidigraph_to_pdf(
                #     graph_nx,
                #     output_path=output_path + f"/graph_{sample_id}.pdf",
                #     open_pdf=False
                # )
                return graph_nx
            else:
                return None
    return None


if __name__ == "__main__":
    main()
