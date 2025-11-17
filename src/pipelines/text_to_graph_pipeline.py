import logging
import os
import time
from torch_geometric.loader import DenseDataLoader
from src.data.text_graph_dataset_ondisk import TextGraphDatasetOnDisk
import torch_geometric.transforms as T
from datetime import datetime
from src.data.utils import log_corpus_oov_statistics
from src.utils.general_utils import format_time_elapsed

# Set up logging
# DATASET = "IMDB"
DATASET = "STF_HC_Voto_Relatorio"
# LANG = "english"
LANG = "portuguese_voto"

ROOT = f"data/datasets/{DATASET}"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file = f"logs/experiment_text2graph_{DATASET}_{timestamp}.log"


logging.info(f"==================== START OF NEW EXPERIMENT: {DATASET} ====================")


def main():
    logging.info("Starting experiment...")
    start_time = time.time()

    logging.info(f"Using dataset at {ROOT}")

    for split in ['train', 'validation', 'test']:
        logging.info(f"Processing {split} split...")

        start_split_time = time.time()
        tgd = TextGraphDatasetOnDisk(
            root=ROOT,
            split=split,
            batch_size=1,
            node_feature_size=100,
            transform=T.ToDense(num_nodes=3000),
            max_num_nodes=1000,
            lang=LANG
        )

        logging.info(f"Dataset Loaded - Split: {split}, Graphs: {len(tgd)}, Batch Size: {tgd.batch_size}")

        # Calculate and log corpus-wide OOV statistics
        logging.info(f"Calculating and logging corpus-wide OOV statistics...")
        unk_vocab, known_vocab = tgd.text2graph_parser.text_embedding.retrieve_vocab_known_and_unk()
        vocabulary = {**unk_vocab, **known_vocab}

        logging.info(f"Number of unique words in vocabulary: {len(vocabulary)}")
        log_corpus_oov_statistics(unk_vocab, vocabulary)
        logging.info("UNK tokens")
        logging.info(unk_vocab)

        # Assuming `vocabulary` is available in the TextGraphDatasetOnDisk instance
        loader = DenseDataLoader(tgd, batch_size=32, shuffle=True)

        del loader, tgd
        logging.info(f"Finished processing {split} split in {format_time_elapsed(start_split_time)}")

    logging.info(f"Experiment completed in {format_time_elapsed(start_time)}")


if __name__ == '__main__':
    main()
