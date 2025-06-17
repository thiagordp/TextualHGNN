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
    EPOCHS: int = 50
    PATIENCE: int = 10


from dataclasses import dataclass, field


@dataclass
class DataConfig:
    # --- CHANGE: Define base path and subdirectory mapping ---
    # Assumes a common structure like the IMDB dataset
    BASE_DATA_PATH: str = "data/datasets/IMDB"

    # Use field(default_factory=...) for mutable defaults like dicts
    CLASS_SUBDIRECTORIES: dict = field(default_factory=lambda: {
        "positive": "train/raw/positive",
        "negative": "train/raw/negative"
    })

    # For demonstration, limit the number of files per class
    SAMPLES_PER_CLASS: int = 200

    SPACY_MODEL: str = 'en_core_web_lg'
    EMBEDDING_MODEL: str = 'sentence-transformers/all-MiniLM-L6-v2'
    SIMILARITY_THRESHOLD: float = 0.85


# Dependency labels are a core part of the graph definition
COMMON_DEP_LABELS = [
    "nsubj", "obj", "iobj", "csubj", "ccomp", "xcomp", "obl", "amod", "advmod",
    "det", "case", "mark", "nmod", "acl", "conj", "cc", "punct", "root", "dep"
]