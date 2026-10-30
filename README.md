# 📚 Book Recommendation System Using Unsupervised Learning

A Book Recommendation System built using **Unsupervised Machine Learning** and **Natural Language Processing (NLP)**. The project uses Sentence Transformers to understand the semantic meaning of book descriptions and K-Means clustering to group books with similar content.

The objective is to organize books based on their descriptions and help users discover books with similar themes and topics.

## 🎯 Project Objectives

- Understand the semantic meaning of book descriptions.
- Convert textual descriptions into numerical embeddings.
- Group similar books using K-Means clustering.
- Evaluate clustering performance using multiple metrics.
- Explore how unsupervised learning can support book discovery and recommendations.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data loading and preprocessing |
| NumPy | Numerical operations |
| Sentence Transformers | Semantic text embeddings |
| Scikit-learn | K-Means clustering and evaluation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Joblib | Model serialization |

## 📂 Dataset

The project uses book information and descriptions from the Book-Crossing dataset and a supplementary book-description dataset.

The datasets contain information such as:

- ISBN
- Book title
- Book author
- Publication year
- Publisher
- Book image URLs
- Book description

### Data preprocessing

The following steps were performed:

1. Loaded the book information and description datasets.
2. Merged the datasets using ISBN.
3. Checked for missing values and duplicate records.
4. Analyzed description lengths.
5. Retained all books, including books with shorter descriptions.

**Final dataset size:** 7,009 books.

## 🧠 Implementation Workflow

### 1. Data Preprocessing

Loaded the book metadata and description files using Pandas and merged them using ISBN.

The resulting dataset was inspected for missing values, duplicate records, and description quality.

### 2. Semantic Embedding Generation

Used the pre-trained Sentence Transformer model:

`all-MiniLM-L6-v2`

This model converts each book description into a numerical vector that captures its semantic meaning.

**Embedding dimensions:**

- Number of books: 7,009
- Embedding dimensions: 384
- Embedding matrix shape: `(7009, 384)`

The generated embeddings were saved as a NumPy file for reuse.

### 3. K-Means Clustering

Applied the K-Means clustering algorithm to group books based on their semantic embeddings.

Final model configuration:

```python
KMeans(
    n_clusters=9,
    init="k-means++",
    n_init=10,
    max_iter=300,
    random_state=42
)
```

**Why K-Means with K=9?**

Different values of K were evaluated using the Elbow Method and Silhouette Score.

Although K=3 achieved a slightly higher Silhouette Score, K=9 was selected to provide more granular groups and make it easier to explore different book categories.

The selection was based on practical interpretability and cluster granularity rather than the Silhouette Score alone.

### 4. Model Evaluation

The final clustering model was evaluated using three metrics.

| Evaluation Metric | Result |
|---|---:|
| Silhouette Score (Cosine) | 0.05808 |
| Davies-Bouldin Index | 4.39108 |
| Calinski-Harabasz Index | 112.93894 |
| Inertia | 5496.82373 |
| Number of Iterations | 45 |

**Metric interpretation:**

- **Silhouette Score:** Measures how similar a book is to its own cluster compared with other clusters. The low score indicates that semantic clusters overlap considerably.
- **Davies-Bouldin Index:** Measures similarity between clusters. Lower values are generally preferred.
- **Calinski-Harabasz Index:** Measures the ratio of between-cluster dispersion to within-cluster dispersion. Higher values are generally preferred.
- **Inertia:** Measures the within-cluster sum of squared distances to cluster centroids. It is useful for comparing K-Means configurations on the same feature representation.

## 📊 Cluster Distribution

The final K-Means model generated 9 clusters containing all 7,009 books.

| Cluster | Number of Books | Percentage |
|---:|---:|---:|
| 0 | 234 | 3.34% |
| 1 | 1,338 | 19.09% |
| 2 | 621 | 8.86% |
| 3 | 790 | 11.27% |
| 4 | 889 | 12.68% |
| 5 | 1,183 | 16.88% |
| 6 | 871 | 12.43% |
| 7 | 407 | 5.81% |
| 8 | 676 | 9.64% |
| **Total** | **7,009** | **100%** |

### Observed Cluster Themes

Based on manually inspecting sample books from each cluster, the following broad themes were observed:

| Cluster | Observed Theme |
|---:|---|
| 0 | International and translated literature |
| 1 | Mixed fiction, biography and memoir |
| 2 | Spirituality and personal development |
| 3 | Mystery, crime and thriller |
| 4 | Fiction, fantasy and science fiction |
| 5 | Literature, history and science |
| 6 | Education, reference and travel |
| 7 | Romance and romantic suspense |
| 8 | Children's books and adventure |

*Note: These are approximate themes inferred from sample books, not predefined or ground-truth categories. Some clusters contain mixed topics.*

## 📁 Project Structure

```text
Book-Recommendation-System/
│
├── data/
│   ├── BX-Books.csv
│   └── Books_Descriptions.csv
│
├── models/
│   ├── merged_books.pkl
│   ├── book_embeddings.npy
│   ├── kmeans_model.pkl
│   ├── cluster_labels.npy
│   └── clustered_books.pkl
│
├── notebooks/
│   └── Final_KMeans_Clustering.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── clustering.py
│   └── evaluation.py
│
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Book-Recommendation-System.git
cd Book-Recommendation-System
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
```

### 3. Activate the virtual environment

For Linux/macOS:

```bash
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Prepare the required files

The runtime pipeline expects these two files inside the `models/` directory:

- `merged_books.pkl`
- `book_embeddings.npy`

They are generated during the preprocessing and embedding-generation workflow in the Jupyter notebook.

The dataset and model artifacts are excluded from GitHub through `.gitignore`. Generate them locally using the notebook before running the application.

### 6. Run the project

```bash
python main.py
```

The application loads the dataset and embeddings, trains the K-Means model, evaluates its performance, displays cluster distribution, and saves the resulting model and cluster labels.

## 💾 Saved Model Artifacts

The project saves the following files:

| File | Description |
|---|---|
| `book_embeddings.npy` | Semantic embeddings of book descriptions |
| `merged_books.pkl` | Preprocessed book dataset |
| `kmeans_model_vscode.pkl` | Trained K-Means model from the VS Code pipeline |
| `cluster_labels_vscode.npy` | Cluster assignment for each book |
| `clustered_books_vscode.pkl` | Book dataset with cluster labels |

## 🔍 Key Learnings

- Understanding unsupervised learning and clustering.
- Generating semantic text embeddings using Sentence Transformers.
- Applying K-Means clustering to high-dimensional text representations.
- Selecting the number of clusters using the Elbow Method and Silhouette Score.
- Evaluating clustering quality using multiple metrics.
- Organizing a machine learning project into reusable Python modules.
- Saving and loading trained machine learning models.

## 🚀 Future Improvements

- Develop a user-friendly web interface.
- Build a book recommendation function using cosine similarity.
- Allow users to search for books by title or description.
- Display similar books with their authors and cover images.
- Improve recommendations by combining semantic similarity with book metadata.

## 👨‍💻 Author

**Your Name**

GitHub: [Your GitHub Profile](https://github.com/YOUR_USERNAME)

---

⭐ If you find this project interesting, consider giving it a star!