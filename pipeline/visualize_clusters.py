import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import numpy as np
from matplotlib.cm import get_cmap
from arabic_reshaper import reshape
from bidi.algorithm import get_display

# Load clustered articles
df = pd.read_parquet("data/processed/clustered_articles.parquet")

# Convert embedding strings back to lists (if needed)
if isinstance(df.loc[0, "embedding"], str):
    df["embedding"] = df["embedding"].apply(ast.literal_eval)

# Prepare 2D coordinates for plotting
if "umap_x" not in df.columns or "umap_y" not in df.columns:
    import umap
    embeddings = list(df["embedding"])
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric="cosine", random_state=42)
    embedding_2d = reducer.fit_transform(embeddings)
    df["umap_x"] = embedding_2d[:, 0]
    df["umap_y"] = embedding_2d[:, 1]

# Determine unique clusters
clusters = sorted(df["cluster"].unique())
num_clusters = len([c for c in clusters if c != -1])
print(f"Detected {num_clusters} clusters (excluding outliers).")

# Assign a color palette dynamically
palette = sns.color_palette("tab20", n_colors=max(num_clusters, 1))
cluster_colors = {c: palette[i % len(palette)] for i, c in enumerate([c for c in clusters if c != -1])}
cluster_colors[-1] = (0.8, 0.8, 0.8)  # gray for outliers

# Plot clusters
plt.figure(figsize=(12, 8))
for c in clusters:
    cluster_df = df[df["cluster"] == c]
    plt.scatter(
        cluster_df["umap_x"],
        cluster_df["umap_y"],
        s=80,
        c=[cluster_colors[c]] * len(cluster_df),
        label=f"Cluster {c}" if c != -1 else "Outliers",
        alpha=0.8,
        edgecolors="k"
    )
plt.title("Article Clusters")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.legend()
plt.show()

# Optional: Plot by language if 'language' column exists
if "language" in df.columns:
    plt.figure(figsize=(12, 8))
    languages = df["language"].unique()
    lang_palette = sns.color_palette("Set2", n_colors=len(languages))
    for i, lang in enumerate(languages):
        lang_df = df[df["language"] == lang]
        plt.scatter(
            lang_df["umap_x"],
            lang_df["umap_y"],
            s=80,
            c=[lang_palette[i]] * len(lang_df),
            label=lang,
            alpha=0.8,
            edgecolors="k"
        )
    plt.title("Articles by Language")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.legend()
    plt.show()

# Print top articles per cluster with proper Arabic shaping
for cluster_id in clusters:
    cluster_articles = df[df["cluster"] == cluster_id]
    print(f"\nCluster {cluster_id} ({len(cluster_articles)} articles)")
    for title in cluster_articles["title"].head(5):
        # Fix Arabic text display if it contains Arabic letters
        if any("\u0600" <= c <= "\u06FF" for c in title):
            title = get_display(reshape(title))
        print(" -", title) # Use a font that supports Arabic

# to run: python pipeline/visualize_clusters.py