
import os
import joblib
import numpy as np

from sklearn.cluster import KMeans


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")


def train_kmeans(embeddings, n_clusters=9):

    print("\nTraining K-Means Clustering...")

    kmeans = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=300,
        random_state=42
    )

    labels = kmeans.fit_predict(embeddings)

    print("K-Means training completed!")
    print("Number of clusters:", len(np.unique(labels)))
    print("Inertia:", kmeans.inertia_)
    print("Iterations:", kmeans.n_iter_)

    return kmeans, labels


def save_model(kmeans, labels):

    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(
        kmeans,
        os.path.join(MODEL_DIR, "kmeans_model_vscode.pkl")
    )

    np.save(
        os.path.join(MODEL_DIR, "cluster_labels_vscode.npy"),
        labels
    )

    print("\nModel and cluster labels saved successfully!")
