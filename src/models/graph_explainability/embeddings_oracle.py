import random
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from typing import List, Tuple, Dict
from gensim.models import KeyedVectors
from torch.nn.functional import cosine_similarity
import pickle
from tqdm import tqdm
import gc
import logging
import time
import torch

from src.models.graph_explainability.base_oracle import BaseOracle

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class EmbeddingOracle(BaseOracle):
    def __init__(self, model_path: str, db_path: str = "/mnt/data/embeddings.db", batch_size: int = 2000000):
        self.db_path = db_path
        self.batch_size = batch_size
        self.model = KeyedVectors.load(model_path, mmap='r')
        self.dimension = self.model.vector_size
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize the database and load embeddings
        self._create_table()
        self._initialize_database()

    # Create the SQLite table
    def _create_table(self):
        self.cursor.execute("DROP INDEX IF EXISTS idx_term;")
        self.cursor.execute("""
               CREATE TABLE IF NOT EXISTS embeddings (
                   term TEXT PRIMARY KEY,
                   embedding BLOB
               )
           """)
        self.conn.commit()

    def _initialize_database(self):
        existing_terms = set(term[0] for term in self.cursor.execute("SELECT term FROM embeddings").fetchall())
        new_terms = list(set(self.model.index_to_key) - existing_terms)

        random.shuffle(new_terms)
        print(f"Starting batch insertion of {len(new_terms)} new terms into the database...")

        self.cursor.execute("PRAGMA synchronous = OFF")
        self.cursor.execute("PRAGMA journal_mode = MEMORY")

        try:
            for i in tqdm(range(0, len(new_terms), self.batch_size), desc="Initializing db with word embeddings"):
                batch_terms = new_terms[i:i + self.batch_size]
                batch_data = [
                    (term, pickle.dumps(self.model[term].astype(np.float32)))
                    for term in batch_terms
                ]
                self._bulk_insert(batch_data)
                gc.collect()

        except Exception as e:
            print(f"Error during batch insertion: {e}")
        finally:
            self.cursor.execute("PRAGMA synchronous = FULL")
            self.cursor.execute("PRAGMA journal_mode = WAL")

        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_term ON embeddings(term);")
        self.conn.commit()
        # print("Batch insertion completed.")

    def _bulk_insert(self, batch_data: List[Tuple[str, bytes]]):
        try:
            query = "INSERT OR REPLACE INTO embeddings (term, embedding) VALUES (?, ?)"
            self.cursor.executemany(query, batch_data)
            self.conn.commit()
        except Exception as e:
            print(f"Error during bulk insertion: {e}")

    def add_term(self, term: str):
        words = term.split()
        valid_embeddings = [self.model[word] for word in words if word in self.model]

        if valid_embeddings:
            average_embedding = np.mean(valid_embeddings, axis=0)
            self._insert_embedding(term, average_embedding)
            print(f"Added term '{term}' with averaged embedding to the database.")
        else:
            print(f"No valid words found in the term '{term}'. Nothing added to the database.")

    def _insert_embedding(self, term: str, embedding: np.ndarray):
        embedding_blob = pickle.dumps(embedding.astype(np.float32))
        self.cursor.execute(
            "INSERT OR REPLACE INTO embeddings (term, embedding) VALUES (?, ?)",
            (term, embedding_blob)
        )
        self.conn.commit()

    def add_terms_bulk(self, terms: List[str], batch_size: int = 10000):
        batch_data = []
        existing_terms = set(term[0] for term in self.cursor.execute("SELECT term FROM embeddings").fetchall())

        for term in tqdm(terms, desc="Processing terms for bulk addition"):
            if term in existing_terms:
                continue

            words = term.lower().strip().split()
            valid_embeddings = [self.model[word] for word in words if word in self.model]

            if valid_embeddings:
                average_embedding = np.mean(valid_embeddings, axis=0)
                embedding_blob = pickle.dumps(average_embedding.astype(np.float32))
                batch_data.append((term, embedding_blob))

            if len(batch_data) >= batch_size:
                self._bulk_insert(batch_data)
                batch_data = []

        if batch_data:
            self._bulk_insert(batch_data)

    def retrieve_embedding(self, term: str) -> torch.Tensor:
        result = self.cursor.execute(
            "SELECT embedding FROM embeddings WHERE term = ?", (term,)
        ).fetchone()
        if result:
            embedding = pickle.loads(result[0])
            return torch.tensor(embedding, dtype=torch.float32, device=self.device)
        else:
            print(f"Embedding for term '{term}' not found in the database.")
            return None

    def _get_all_embeddings(self) -> Tuple[List[str], torch.Tensor]:
        results = self.cursor.execute("SELECT term, embedding FROM embeddings").fetchall()
        terms = [row[0] for row in results]
        embeddings = torch.stack([torch.tensor(pickle.loads(row[1]), dtype=torch.float32) for row in results])
        return terms, embeddings.to(self.device)

    def retrieve_top_k_by_embeddings(self, embedding: torch.Tensor, k: int, similarity_threshold: float = 0.9999) -> \
            List[str]:
        """
        Retrieve the top-k most similar terms using GPU for fast computation.
        This method processes embeddings in batches to avoid memory overflow.
        """
        top_k_results = []
        offset = 0

        while offset <= 5 * self.batch_size:
            query = f"SELECT term, embedding FROM embeddings LIMIT {self.batch_size} OFFSET {offset}"
            results = self.cursor.execute(query).fetchall()

            if not results:
                break

            terms = [row[0] for row in results]
            embeddings = torch.stack(
                [torch.tensor(pickle.loads(row[1]), dtype=torch.float32) for row in results]
            ).to(self.device)

            # Compute cosine similarities in the batch
            similarities = cosine_similarity(embedding.unsqueeze(0), embeddings).squeeze(0)

            # Get top-k within this batch
            batch_top_k_indices = torch.topk(similarities, min(k, len(terms))).indices
            batch_top_k = [(terms[i], similarities[i].item()) for i in batch_top_k_indices]

            # Merge with global top-k
            top_k_results.extend(batch_top_k)

            offset += self.batch_size
            del embeddings
            torch.cuda.empty_cache()

            if any(sim >= similarity_threshold for _, sim in top_k_results):
                break

        top_k_results = sorted(top_k_results, key=lambda x: x[1], reverse=True)[:k]
        return [term for term, _ in top_k_results]

    def concept_grounding_from_embeddings(self, embeddings: torch.Tensor, num_terms_to_retrieve: int = 1,
                                          similarity_threshold: float = 0.9999) -> Dict:

        """
        Retrieve the top-k most similar terms using GPU for fast computation.
        This method processes embeddings in batches to avoid memory overflow.
        """
        top_k_results = []
        offset = 0

        while offset <= 5 * self.batch_size:
            query = f"SELECT term, embedding FROM embeddings LIMIT {self.batch_size} OFFSET {offset}"
            results = self.cursor.execute(query).fetchall()

            # Parallel deserialization
            # def load_embedding(row):
            #     return pickle.loads(row[1])
            #
            # with ThreadPoolExecutor(max_workers=10) as executor:
            #     embeddings_list = list(executor.map(load_embedding, results))
            #
            # embeddings_np = np.array(embeddings_list, dtype=np.float32)
            # existing_embeddings = torch.from_numpy(embeddings_np).to(self.device)
            #terms = [row[0] for row in results]

            if not results:
                break

            terms = [row[0] for row in results]
            existing_embeddings = torch.stack(
                [torch.tensor(pickle.loads(row[1]), dtype=torch.float32) for row in results]
            ).to(self.device)

            # Compute cosine similarities in the batch
            similarities = cosine_similarity(embeddings, existing_embeddings, dim=1).squeeze(0)

            # Get top-k within this batch
            batch_top_k_indices = torch.topk(similarities, min(num_terms_to_retrieve, len(terms))).indices
            batch_top_k = [(terms[i], similarities[i].item()) for i in batch_top_k_indices]

            # Merge with global top-k
            top_k_results.extend(batch_top_k)

            offset += self.batch_size
            del existing_embeddings
            torch.cuda.empty_cache()

            if any(sim >= similarity_threshold for _, sim in top_k_results):
                break
        top_k_results = sorted(top_k_results, key=lambda x: x[1], reverse=True)[:num_terms_to_retrieve]

        words = [term for term, _ in top_k_results]
        return {
            "terms": words
        }

    def concept_grounding_from_words(self, words: List[str], num_terms_to_retrieve: int = 1,
                                     similarity_threshold: float = 0.9999) -> Dict:
        embeddings = [self.retrieve_embedding(term) for term in words]
        embeddings = [embedding for embedding in embeddings if embedding is not None]

        if embeddings:
            average_embedding = torch.mean(torch.stack(embeddings), dim=0)
            result = self.retrieve_top_k_by_embeddings(embedding=average_embedding,
                                                       k=num_terms_to_retrieve,
                                                       similarity_threshold=similarity_threshold)

            return {
                "terms": result
            }
        else:
            print("No valid embeddings found for the provided terms.")
            return {
                "terms": ["None"]
            }

    def retrieve_top_k_by_string(self, term: str, k: int) -> List[str]:
        embedding = self.retrieve_embedding(term)
        if embedding is None:
            return []

        offset = 0
        top_k_results = []
        fetch_time = []
        stack_time = []
        sim_time = []
        rest_time = []
        while offset <= 5 * self.batch_size:
            start_time = time.time()
            query = f"SELECT term, embedding FROM embeddings LIMIT {self.batch_size} OFFSET {offset}"
            results = self.cursor.execute(query).fetchall()
            fetch_time.append(time.time() - start_time)

            if not results:
                break

            start_time = time.time()
            terms = [row[0] for row in results]
            embeddings = torch.stack([torch.tensor(pickle.loads(row[1]), dtype=torch.float32) for row in results]).to(
                self.device)
            stack_time.append(time.time() - start_time)

            start_time = time.time()
            similarities = cosine_similarity(embedding.unsqueeze(0), embeddings).squeeze(0)
            batch_top_k_indices = torch.topk(similarities, min(k, len(terms))).indices
            batch_top_k = [(terms[i], similarities[i].item()) for i in batch_top_k_indices]
            sim_time.append(time.time() - start_time)

            start_time = time.time()
            top_k_results.extend(batch_top_k)
            # top_k_results = sorted(top_k_results, key=lambda x: x[1], reverse=True)[:k]

            offset += self.batch_size
            del embeddings
            torch.cuda.empty_cache()
            rest_time.append(time.time() - start_time)

        # logging.info(f"Fetch_time: {np.mean(fetch_time):.4f} ({np.std(fetch_time):.4f})")
        # logging.info(f"Stack_time: {np.mean(stack_time):.4f} ({np.std(stack_time):.4f})")
        # logging.info(f"Similarity_time: {np.mean(sim_time):.4f} ({np.std(sim_time):.4f})")
        # logging.info(f"Rest_time: {np.mean(rest_time):.4f} ({np.std(rest_time):.4f})")

        top_k_results = sorted(top_k_results, key=lambda x: x[1], reverse=True)[:k]
        return [term for term, _ in top_k_results]

    # def retrieve_top_k_by_string2(self, term: str, k: int) -> List[str]:
    #     embedding = self.retrieve_embedding(term)
    #     if embedding is None:
    #         return []
    #
    #     top_k_results = []
    #     last_row_id = 0
    #
    #     fetch_time = []
    #     stack_time = []
    #     sim_time = []
    #     rest_time = []
    #
    #     # ThreadPool for parallel batch loading
    #     executor = ThreadPoolExecutor(max_workers=2)
    #
    #     # Helper function to fetch and process embeddings asynchronously
    #     def _fetch_and_process_batch(row_id: int) -> Tuple[List[str], torch.Tensor]:
    #         # Fetch data from SQLite
    #         query = f"SELECT rowid, term, embedding FROM embeddings WHERE rowid > ? LIMIT {self.batch_size}"
    #         results = self.cursor.execute(query, (row_id,)).fetchall()
    #         if not results:
    #             return [], None
    #
    #         terms = [row[1] for row in results]
    #         embeddings = torch.stack(
    #             [torch.tensor(pickle.loads(row[2]), dtype=torch.float32) for row in results]
    #         ).to(self.device, non_blocking=True)
    #
    #         return terms, embeddings
    #
    #     # Start pre-loading the first batch
    #     future: Future = executor.submit(_fetch_and_process_batch, last_row_id)
    #
    #     while True:
    #         start_time = time.time()
    #         # Wait for the future to complete and get results
    #         terms, embeddings = future.result()
    #         fetch_time.append(time.time() - start_time)
    #
    #         if not terms or embeddings is None:
    #             break
    #
    #         last_row_id += len(terms)
    #
    #         # Start loading the next batch asynchronously
    #         future = executor.submit(_fetch_and_process_batch, last_row_id)
    #
    #         # Similarity computation
    #         start_time = time.time()
    #         similarities = cosine_similarity(embedding.unsqueeze(0), embeddings).squeeze(0)
    #         batch_top_k_indices = torch.topk(similarities, min(k, len(terms))).indices
    #         batch_top_k = [(terms[i], similarities[i].item()) for i in batch_top_k_indices]
    #         sim_time.append(time.time() - start_time)
    #
    #         # Maintain global top-k
    #         start_time = time.time()
    #         top_k_results.extend(batch_top_k)
    #         top_k_results = sorted(top_k_results, key=lambda x: x[1], reverse=True)[:k]
    #
    #         del embeddings
    #         torch.cuda.empty_cache()
    #         rest_time.append(time.time() - start_time)
    #
    #     executor.shutdown(wait=True)
    #
    #     # Log performance metrics
    #     logging.info(f"Fetch_time: {np.mean(fetch_time):.4f}s (±{np.std(fetch_time):.4f}s)")
    #     logging.info(f"Stack_time: {np.mean(stack_time):.4f}s (±{np.std(stack_time):.4f}s)")
    #     logging.info(f"Similarity_time: {np.mean(sim_time):.4f}s (±{np.std(sim_time):.4f}s)")
    #     logging.info(f"Rest_time: {np.mean(rest_time):.4f}s (±{np.std(rest_time):.4f}s)")
    #
    #     return [term for term, _ in top_k_results]


def main_english():
    # Define paths for the model and the database
    language = "english"
    DATASET = "IMDB"

    logger.info(f"==================== START OF NEW EMBEDDINGS ORACLE: {DATASET} ====================")

    model_path = 'data/external/embeddings/enwiki_20180420_100d.bin'
    db_path = f"data/oracle/embeddings_{DATASET}.db"

    # Initialize the EmbeddingOracle
    logger.info("Initializing EmbeddingOracle...")
    start_time = time.time()
    embedding_oracle = EmbeddingOracle(model_path=model_path, db_path=db_path)
    logger.info(f"EmbeddingOracle initialized in {time.time() - start_time:.2f}s")

    # Test adding new terms
    test_terms = ["corte di cassazione"]
    for term in test_terms:
        logger.info(f"Adding term: {term}")
        embedding_oracle.add_term(term)

    # Test bulk addition of terms
    terms_to_add = [
        "machine learning", "deep learning", "artificial intelligence",
        "natural language processing", "computer vision"
    ]
    logger.info(f"Bulk adding {len(terms_to_add)} terms...")
    start_time = time.time()
    embedding_oracle.add_terms_bulk(terms_to_add, batch_size=5000)
    logger.info(f"Bulk addition completed in {time.time() - start_time:.2f}s")

    # Test retrieving embedding for a term
    test_term = "artificial intelligence"
    logger.info(f"Retrieving embedding for term: {test_term}")
    embedding = embedding_oracle.retrieve_embedding(test_term)
    if embedding is not None:
        logger.info(f"Embedding for '{test_term}': {embedding[:5]}... (truncated)")

    # Test retrieving top-k similar terms by string
    test_query_term = "apple"
    k = 10
    logger.info(f"Retrieving top-{k} similar terms to '{test_query_term}' by string...")
    start_time = time.time()
    top_k_by_string = embedding_oracle.retrieve_top_k_by_string(test_query_term, k=k)
    logger.info(f"Top {k} terms similar to '{test_query_term}': {top_k_by_string}")
    logger.info(f"Retrieved in {time.time() - start_time:.2f}s")

    # Test retrieving top-k similar terms by embedding
    if embedding is not None:
        logger.info(f"Retrieving top-{k} similar terms by embedding...")
        start_time = time.time()
        top_k_by_embeddings = embedding_oracle.retrieve_top_k_by_embeddings(embedding, k=k)
        logger.info(f"Top {k} terms similar to the embedding of '{test_term}': {top_k_by_embeddings}")
        logger.info(f"Retrieved in {time.time() - start_time:.2f}s")

    # Test retrieving the most similar term to a list of terms
    terms_list = ["apple", "banana", "fruit"]
    logger.info(f"Retrieving most similar terms to the list: {terms_list}")
    start_time = time.time()
    most_similar_to_list = embedding_oracle.retrieve_most_similar_to_list(terms_list, k=k)
    logger.info(f"Most similar terms to the list {terms_list}: {most_similar_to_list}")
    logger.info(f"Retrieved in {time.time() - start_time:.2f}s")

    # Performance and Memory Check
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        logger.info(f"GPU Memory Usage: {gpu_memory:.2f} MB")

    logger.info(f"==================== END OF EMBEDDINGS ORACLE TEST: {DATASET} ====================")


def main_italian():
    # Define paths for the model and the database
    language = "italian"
    DATASET = "Imprisonment-IT"

    logger.info(f"==================== START OF NEW EMBEDDINGS ORACLE: {DATASET} ====================")

    model_path = 'data/external/embeddings/itwiki_20180420_100d.bin'
    db_path = f"data/oracle/embeddings_{DATASET}.db"

    # Inizializza l'EmbeddingOracle
    logger.info("Inizializzazione di EmbeddingOracle...")
    start_time = time.time()
    embedding_oracle = EmbeddingOracle(model_path=model_path, db_path=db_path)
    logger.info(f"EmbeddingOracle inizializzato in {time.time() - start_time:.2f}s")

    # Test aggiunta di nuovi termini
    test_terms = ["corte di cassazione", "diritto amministrativo", "giurisprudenza italiana"]
    for term in test_terms:
        logger.info(f"Aggiunta del termine: {term}")
        embedding_oracle.add_term(term)

    # Test aggiunta in blocco di termini
    terms_to_add = [
        "apprendimento automatico", "rete neurale", "intelligenza artificiale",
        "elaborazione del linguaggio naturale", "visione artificiale"
    ]
    logger.info(f"Aggiunta in blocco di {len(terms_to_add)} termini...")
    start_time = time.time()
    embedding_oracle.add_terms_bulk(terms_to_add, batch_size=5000)
    logger.info(f"Aggiunta in blocco completata in {time.time() - start_time:.2f}s")

    # Test recupero dell'embedding per un termine
    test_term = "intelligenza artificiale"
    logger.info(f"Recupero dell'embedding per il termine: {test_term}")
    embedding = embedding_oracle.retrieve_embedding(test_term)
    if embedding is not None:
        logger.info(f"Embedding per '{test_term}': {embedding[:5]}... (troncato)")

    # Test recupero dei primi-k termini simili per stringa
    test_query_term = "mela"
    k = 10
    logger.info(f"Recupero dei primi {k} termini simili a '{test_query_term}' per stringa...")
    start_time = time.time()
    top_k_by_string = embedding_oracle.retrieve_top_k_by_string(test_query_term, k=k)
    logger.info(f"Primi {k} termini simili a '{test_query_term}': {top_k_by_string}")
    logger.info(f"Recuperato in {time.time() - start_time:.2f}s")

    # Test recupero dei primi-k termini simili per embedding
    if embedding is not None:
        logger.info(f"Recupero dei primi {k} termini simili per embedding...")
        start_time = time.time()
        top_k_by_embeddings = embedding_oracle.retrieve_top_k_by_embeddings(embedding, k=k)
        logger.info(f"Primi {k} termini simili all'embedding di '{test_term}': {top_k_by_embeddings}")
        logger.info(f"Recuperato in {time.time() - start_time:.2f}s")

    # Test recupero del termine più simile a una lista di termini
    terms_list = ["mela", "banana", "frutta"]
    logger.info(f"Recupero dei termini più simili alla lista: {terms_list}")
    start_time = time.time()
    most_similar_to_list = embedding_oracle.retrieve_most_similar_to_list(terms_list, k=k)
    logger.info(f"Termini più simili alla lista {terms_list}: {most_similar_to_list}")
    logger.info(f"Recuperato in {time.time() - start_time:.2f}s")

    # Controllo delle prestazioni e della memoria
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        logger.info(f"Utilizzo della memoria GPU: {gpu_memory:.2f} MB")

    logger.info(f"==================== FINE DEL TEST EMBEDDINGS ORACLE: {DATASET} ====================")


if __name__ == '__main__':
    # main_english()
    main_italian()
