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


def main():

    LANG = "italian"

    if LANG == "english":
        DATASET = "IMDB"
        MODEL_PATH = f"models/IMDB_DiffPool_20250224_234935_lr1e-05_valmacrof1score0.8500_epoch011.pth"
        EMBEDDING_PATH = "data/external/embeddings/enwiki_20180420_100d.bin"
        max_num_nodes = 1100
    else:
        DATASET = "Imprisonment-IT"
        max_num_nodes = 1000
        MODEL_PATH = f"models/Imprisonment-IT_DiffPool_20250225_214311_lr1e-05_valmacrof1score0.6441_epoch035.pth"
        EMBEDDING_PATH = "data/external/embeddings/itwiki_20180420_100d.bin"

    weights = torch.load(MODEL_PATH)

    EMBEDDINGS_DATABASE_PATH = f"data/oracle/embeddings_{DATASET}.db"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ROOT = f"data/datasets/{DATASET}"
    DOCUMENTS_TO_EXPLAIN = 5

    diffpool_model = DiffPool(
        max_num_nodes=max_num_nodes,
        in_channels=100,
        hidden_channels=100,
        out_channels=2
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
            hyper_nodes_to_explain=10,
            nodes_per_hyper_node=5,
            original_raw_file_path=f"{ROOT}/test/raw/" ,
            language=LANG
        )

        cg.concept_grounding()

        output_path = f"data/explanations/{DATASET}"
        os.makedirs(output_path, exist_ok=True)
        cg.save_explanation(Path(output_path) / f"{cg.data_sample_id}.json")


if __name__ == "__main__":
    main()
