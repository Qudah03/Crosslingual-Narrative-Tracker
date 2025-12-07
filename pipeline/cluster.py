import pandas as pd # pyright: ignore[reportMissingModuleSource]
import numpy as np # type: ignore
import hdbscan # pyright: ignore[reportMissingImports]
import os
import umap # pyright: ignore[reportMissingImports]


EMBEDDINGS_FILE = "data/processed/embeddings.parquet"
OUTPUT_FILE = "data/processed/clustered_articles.parquet"

# Load embeddings from Parquet
df = pd.read_parquet(EMBEDDINGS_FILE)
df["embedding"] = df["embedding"].apply(np.array)  # convert lists back to numpy arrays
X = np.stack(df["embedding"].values)               # shape: (num_articles, embedding_dim)
print(f"Loaded {len(df)} articles with embeddings.")

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    n_components=5,    # reduces to 5 dimensions
    metric="cosine",
    random_state=42
)
X_reduced = reducer.fit_transform(X)
print("Dimensionality reduction completed.")

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=5,  # minimum articles per cluster
    metric="euclidean",
    cluster_selection_method="eom"
)
labels = clusterer.fit_predict(X_reduced)
df["cluster"] = labels

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_parquet(OUTPUT_FILE, index=False)
print(f"Saved clustered articles to {OUTPUT_FILE}")
print(f"Identified {len(set(labels)) - (1 if -1 in labels else 0)} clusters.")
print("Cluster size distribution:")
print(df["cluster"].value_counts())

# to run this script, use the command:
# python pipeline/cluster.py