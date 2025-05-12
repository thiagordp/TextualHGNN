# Textual Hierarchical Graph Neural Networks

A modular, scalable pipeline for transforming raw textual data into graph-structured datasets using dependency and sequential parsing. Designed to support large-scale graph-based learning tasks in natural language processing (NLP) and built upon PyTorch Geometric’s `OnDiskDataset`.

## Overview

This repository introduces a dataset processing framework that converts labeled text corpora into graph representations suitable for graph neural networks (GNNs). It includes functionality for:

- Parsing raw text into graphs using SpaCy dependency trees and sequential context
- Efficient on-disk storage and retrieval of graphs via PyTorch Geometric
- Model training using GCN, GraphSAGE, and DiffPool
- Explainability via semantic grounding using embedding-based oracles and LLMs


## Key Features

- **Graph-Based Representation**: Constructs `MultiDiGraph` structures from text, integrating syntactic and sequential relations.
- **Efficient Storage**: Leverages `OnDiskDataset` to scale to large corpora without memory constraints.
- **Multilingual Support**: Currently supports English, Portuguese, and Italian.
- **Explainability**: Maps learned hypernodes back to interpretable concepts using embeddings and LLM-guided oracles.
- **Modular Architecture**: Easily extendable for different parsing strategies, embeddings, and explainability approaches.


## Directory Structure

```
src/
├── data/
│   ├── text_graph_dataset_ondisk.py       # Dataset processing and serialization
│   ├── text_graph_dataset_parsers.py      # Text to graph conversion logic
│   └── utils.py                           # Embedding and graph utilities
├── models/
│   ├── graph_classification/              # GNN models and evaluation
│   └── graph_explainability/              # Concept grounding and interpretability
```

## Dataset Format

Expected raw data structure:

```
/data/
└── train/
    ├── class_A/
    │   ├── doc1.txt
    │   └── ...
    ├── class_B/
    │   ├── doc2.txt
    │   └── ...
```

Each `.txt` file is parsed into a graph using SpaCy dependency parsing and token sequencing, then serialized as a PyG `Data` object.

---

## Example Usage

```python
from src.data.text_graph_dataset_ondisk import TextGraphDatasetOnDisk

dataset = TextGraphDatasetOnDisk(root='/path/to/data', split='train')
print(dataset)
```

---

## GNN Models

Implemented in `src/models/graph_classification/gnn.py`:

- `GCN`: Graph Convolutional Network
- `GraphSAGE`: Sample and Aggregate-based inductive model
- `DiffPool`: Differentiable graph pooling for hierarchical representations

Metrics supported: Accuracy, F1-score, Precision, Recall, Confusion Matrix.


## Concept Grounding & Explainability

The `ConceptGrounding` class interprets GNN predictions by:

- Tracing active hypernodes in `DiffPool`
- Mapping node embeddings to known terms (via GloVe) or concepts (via LLMs)
- Providing a human-readable explanation dictionary per document

## Requirements

- Python 3.9+
- PyTorch, PyTorch Geometric
- SpaCy (with models: `en_core_web_lg`, etc.)
- Gensim, NetworkX, NLTK
- Transformers (optional for BERT/LLM explainability)
- SQLite (for embedding DB)

Install dependencies via:

```bash
pip install -r requirements.txt
```

