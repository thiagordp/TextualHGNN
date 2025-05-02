from gensim.models import KeyedVectors

EMBEDDINGS_PATH_TXT = "data/external/embeddings/enwiki_20180420_100d.txt"
EMBEDDINGS_PATH_BIN = EMBEDDINGS_PATH_TXT.replace(".txt", ".bin")

print("Converting embeddings to binary (Word2Vec format)...")
print(f"Text file path: {EMBEDDINGS_PATH_TXT}")
print(f"Binary file path: {EMBEDDINGS_PATH_BIN}")

# Load the model
model = KeyedVectors.load_word2vec_format(EMBEDDINGS_PATH_TXT, binary=False)

# Save in Word2Vec binary format (usable with load_word2vec_format(binary=True))
model.save_word2vec_format(EMBEDDINGS_PATH_BIN, binary=True)

print("Finished.")
