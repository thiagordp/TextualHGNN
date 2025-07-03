# config.py
# config.py

from dataclasses import dataclass
import os


@dataclass
class ModelConfig:
    HIDDEN_CHANNELS: int = 64
    OUT_CHANNELS: int = 2


@dataclass
class TrainingConfig:
    BATCH_SIZE: int = 1
    LEARNING_RATE: float = 0.001
    EPOCHS: int = 150
    PATIENCE: int = 25
    # A small value is recommended to start, e.g., 0.005
    ENTROPY_WEIGHT: float = 0.005
    FORCE_TRAINING: bool = True


from dataclasses import dataclass, field


@dataclass
class DataConfig:
    # --- CHANGE: Define base path and subdirectory mapping ---
    # Assumes a common structure like the IMDB dataset
    DATASET_NAME:str="STF_HC"
    # DATASET_NAME:str="IMDB"
    BASE_DATA_PATH: str =f"data/datasets/{DATASET_NAME}"

    # # Use field(default_factory=...) for mutable defaults like dicts
    # CLASS_SUBDIRECTORIES: dict = field(default_factory=lambda: {
    #     "positive": "train/raw/positive",
    #     "negative": "train/raw/negative"
    # })
    CLASS_SUBDIRECTORIES: dict = field(default_factory=lambda: {
        "preso": "full/raw/Preso",
        "solto": "full/raw/Solto"
    })

    # For demonstration, limit the number of files per class
    SAMPLES_PER_CLASS: int = 25

    # SPACY_MODEL: str = 'en_core_web_lg'
    SPACY_MODEL: str = 'pt_core_news_lg'

    # EMBEDDING_MODEL: str = 'sentence-transformers/all-MiniLM-L6-v2'
    # EMBEDDING_MODEL: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
    # SIMILARITY_THRESHOLD: float = 0.99

    # EMBEDDING_MODEL: str = 'rufimelo/Legal-BERTimbau-base'

    EMBEDDING_MODEL: str = 'cnmoro/bert-tiny-embeddings-english-portuguese'
    SIMILARITY_THRESHOLD: float = 0.998

    FORCE_PREPROCESSING:bool = True
    # EMBEDDING_MODEL: str = 'sentence-transformers/LaBSE'


@dataclass
class VisualizationConfig:
    VISUALIZATION_FOLDER = "data/graph_visualizations"

# Dependency labels are a core part of the graph definition
COMMON_DEP_LABELS = [
    "nsubj", "obj", "iobj", "csubj", "ccomp", "xcomp", "obl", "amod", "advmod",
    "det", "case", "mark", "nmod", "acl", "conj", "cc", "punct", "root", "dep"
]