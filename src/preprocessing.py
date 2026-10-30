
import os
import numpy as np
import pandas as pd


# Project directories
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def load_data():

    books_path = os.path.join(
        MODEL_DIR, "merged_books.pkl"
    )

    embeddings_path = os.path.join(
        MODEL_DIR, "book_embeddings.npy"
    )

    # Load dataset
    books = pd.read_pickle(books_path)

    # Load embeddings
    embeddings = np.load(embeddings_path)

    # Validate dataset and embeddings
    if len(books) != len(embeddings):
        raise ValueError(
            "Dataset and embeddings have different row counts!"
        )

    print("Dataset loaded successfully!")
    print("Books shape:", books.shape)
    print("Embeddings shape:", embeddings.shape)

    return books, embeddings
