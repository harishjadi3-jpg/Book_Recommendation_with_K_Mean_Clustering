
from src.preprocessing import load_data
from src.clustering import train_kmeans, save_model
from src.evaluation import (
    evaluate_model,
    display_cluster_distribution
)


def main():

    books, embeddings = load_data()

    kmeans, labels = train_kmeans(
        embeddings,
        n_clusters=9
    )

    evaluate_model(embeddings, labels)

    clustered_books = display_cluster_distribution(
        books,
        labels
    )

    save_model(kmeans, labels)

    clustered_books.to_pickle(
        "models/clustered_books_vscode.pkl"
    )

    print("\nComplete clustering process finished!")


if __name__ == "__main__":
    main()
