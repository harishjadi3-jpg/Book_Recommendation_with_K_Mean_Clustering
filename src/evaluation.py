
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)
import pandas as pd


def evaluate_model(embeddings, labels):

    print("\nMODEL EVALUATION")
    print("-" * 40)

    # Silhouette Score
    silhouette = silhouette_score(
        embeddings,
        labels,
        metric="cosine",
        sample_size=2000,
        random_state=42
    )

    # Davies-Bouldin Index
    db_score = davies_bouldin_score(
        embeddings,
        labels
    )

    # Calinski-Harabasz Index
    ch_score = calinski_harabasz_score(
        embeddings,
        labels
    )

    print("Silhouette Score:", silhouette)
    print("Davies-Bouldin Index:", db_score)
    print("Calinski-Harabasz Index:", ch_score)

    return silhouette, db_score, ch_score




def display_cluster_distribution(books, labels):

    results = books.copy()
    results["Cluster"] = labels

    distribution = (
        results["Cluster"]
        .value_counts()
        .sort_index()
    )

    print("\nCLUSTER DISTRIBUTION")
    print("-" * 40)

    for cluster, count in distribution.items():

        percentage = (count / len(results)) * 100

        print(
            f"Cluster {cluster}: {count} books "
            f"({percentage:.2f}%)"
        )

    return results

