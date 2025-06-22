# Textual Hierarchical Graph Neural Networks

A modular, scalable pipeline for transforming raw textual data into graph-structured datasets using dependency and sequential parsing. Designed to support large-scale graph-based learning tasks in natural language processing (NLP) and built upon PyTorch Geometric’s `OnDiskDataset`.

## Overview

This repository introduces a dataset processing framework that converts labelled text corpora into graph representations suitable for graph neural networks (GNNs). It includes functionality for:

  - Parsing raw text into graphs using SpaCy dependency trees and sequential context.
  - Efficient on-disk storage and retrieval of graphs via PyTorch Geometric.
  - Model training using GCN, GraphSAGE, and DiffPool.
  - Explainability via semantic grounding using embedding-based oracles and LLMs.

## Key Features

  - **Graph-Based Representation**: Constructs `MultiDiGraph` structures from text, integrating syntactic and sequential relations.
  - **Efficient Storage**: Leverages `OnDiskDataset` to scale to large corpora without memory constraints.
  - **Multilingual Support**: Currently supports English, Portuguese, and Italian.
  - **Explainability**: Maps learned hypernodes back to interpretable concepts using embeddings and LLM-guided oracles.
  - **Modular Architecture**: Easily extendable for different parsing strategies, embeddings, and explainability approaches.

## Installation

To set up the environment, please use the provided `environment.yml` file with Conda:

```bash
conda env create -f environment.yml
conda activate phd_env
```

This will install all the necessary dependencies, including PyTorch, PyTorch Geometric, SpaCy, and Transformers.

## Usage

The `src/pipelines` directory contains several scripts to run the core functionalities of this project.

### 1\. Text to Graph Conversion

To convert a raw text corpus into a graph dataset, run the `text_to_graph_pipeline.py` script. You will need to configure the `DATASET` and `LANG` variables within the script to point to your data.

```bash
python src/pipelines/text_to_graph_pipeline.py
```

### 2\. Model Training

This repository supports training for different GNN models.

  - **DiffPool**: To train the DiffPool model, use the `train_diffpool_pipeline.py` script.
  - **GCN**: For the GCN baseline, run `gcn_baseline_pipeline.py`.
  - **BERT**: A BERT baseline is also available in `bert_baseline_pipeline.py`.

Example of running the DiffPool training:

```bash
python src/pipelines/train_diffpool_pipeline.py
```

### 3\. Hyperparameter Search

Grid search for hyperparameters can be performed using the `train_diffpool_gridsearch.py` and `train_diffpool_gridsearch_loss.py` scripts. The results of these searches are stored in `.xlsx` files, which can be analyzed with the `gridsearch_analysis.py` script.

### 4\. Explainability

The explainability pipeline uses concept grounding to interpret the model's predictions. This can be run using the `concept_grounding_pipeline.py` script.

## Project Structure

```
.
├── data/                     # Datasets and external resources
│   ├── datasets/
│   ├── evaluation/
│   ├── explanations/
│   └── prompts/
├── notebooks/                # Jupyter notebooks for analysis and visualisation
├── src/                      # Source code
│   ├── data/                 # Data loading and processing
│   ├── models/               # GNN models and explainability methods
│   ├── multi_graph/          # Multi-level graph construction and models
│   ├── pipelines/            # End-to-end pipelines for different tasks
│   └── utils/                # Utility functions
├── README.md                 # This file
└── environment.yml           # Conda environment file
```

## Explainability

This project includes a novel approach to GNN explainability through concept grounding. The `src/models/graph_explainability` directory contains the core components:

  - **`concept_grounding.py`**: The main module for generating explanations.
  - **`llm_oracle.py`**: Uses a Large Language Model to generate concepts for hypernodes.
  - **`embeddings_oracle.py`**: An embedding-based oracle for concept grounding.
  - **`cg_evaluation_metrics.py`**: Implements metrics like "Concept Completeness" to evaluate the quality of explanations.

## Citation

If you use this work, please cite:

```
@misc{textualhgnn,
  author = {Thiago Raulino {Dal Pont}},
  title = {Textual Hierarchical Graph Neural Networks},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/thiagordp/TextualHGNN}}
}
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
