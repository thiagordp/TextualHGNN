import logging

from src.models.graph_explainability.embeddings_oracle import EmbeddingOracle

# Define paths for the model and the database
language = "portuguese_voto"
DATASET = "STF_HC_Voto_Relatorio"
log_file = f"logs/embeddings_oracle_{DATASET}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler(log_file)  # File output
    ]
)

def main():


    logging.info(f"==================== START OF NEW EMBEDDINGS ORACLE: {DATASET} ====================")

    model_path = 'data/external/embeddings/glove_legal_100.bin'
    db_path = f"data/oracle/embeddings_{DATASET}.db"

    # Initialize the EmbeddingOracle
    embedding_oracle = EmbeddingOracle(model_path=model_path, db_path=db_path)

    # Test adding a new term
    test_term = "machine learning"
    embedding_oracle.add_term(test_term)

    test_term = "deep learning"
    embedding_oracle.add_term(test_term)

    test_term = "artificial intelligence"
    embedding_oracle.add_term(test_term)

    terms_to_add = ["machine learning", "deep learning", "artificial intelligence", "natural language processing",
                    "computer vision"]
    embedding_oracle.add_terms_bulk(terms_to_add, batch_size=5000)

    # Test retrieving embedding for a term
    embedding = embedding_oracle.retrieve_embedding(test_term)
    if embedding is not None:
        logging.info(f"Embedding for '{test_term}': {embedding[:5]}... (truncated)")

    # Test retrieving top-k similar terms by string
    top_k_by_string = embedding_oracle.retrieve_top_k_by_string("apple", k=10)
    logging.info(f"Top 5 terms similar to 'apple': {top_k_by_string}")

    # Test retrieving top-k similar terms by embedding
    if embedding is not None:
        top_k_by_embeddings = embedding_oracle.retrieve_top_k_by_embeddings(embedding, k=10)
        logging.info(f"Top 5 terms similar to the embedding of '{test_term}': {top_k_by_embeddings}")

    # Test retrieving the most similar term to a list of terms
    terms_list = ["apple", "banana", "fruit"]
    most_similar_to_list = embedding_oracle.retrieve_most_similar_to_list(terms_list, k=10)
    logging.info(f"Most similar terms to the list {terms_list}: {most_similar_to_list}")


if __name__ == '__main__':
    # Example usage
    # Run the main function
    main()
