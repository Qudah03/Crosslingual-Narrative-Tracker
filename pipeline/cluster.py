import pandas as pd # pyright: ignore[reportMissingModuleSource]
import numpy as np # type: ignore
import hdbscan # pyright: ignore[reportMissingImports]
import os
import sys
import umap # pyright: ignore[reportMissingImports]
import logging

EMBEDDINGS_FILE = "data/processed/embeddings.parquet"
OUTPUT_FILE = "data/processed/clustered_articles.parquet"

# Configure logging BEFORE any processing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    try:
        # Validate embeddings file exists
        if not os.path.exists(EMBEDDINGS_FILE):
            logger.error(f"Embeddings file not found: {EMBEDDINGS_FILE}")
            logger.info("Run 'python pipeline/embed.py' first to generate embeddings.")
            return False
        
        # Load embeddings from Parquet
        logger.info(f"Loading embeddings from {EMBEDDINGS_FILE}...")
        df = pd.read_parquet(EMBEDDINGS_FILE)
        logger.info(f"Loaded {len(df)} articles with embeddings.")
        
        if df.empty:
            logger.error("No articles found in embeddings file.")
            return False
        
        # Convert embeddings to numpy arrays
        df["embedding"] = df["embedding"].apply(np.array)
        X = np.stack(df["embedding"].values)
        logger.info(f"Embedding shape: {X.shape}")
        
        # Dimensionality reduction
        logger.info("Starting dimensionality reduction with UMAP...")
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            metric="cosine",
            random_state=42
        )
        X_reduced = reducer.fit_transform(X)

        df['x_coord'] = X_reduced[:, 0]
        df['y_coord'] = X_reduced[:, 1]
        
        logger.info("Dimensionality reduction completed.")
        
        # Clustering
        logger.info("Starting clustering with HDBSCAN...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=5,
            metric="euclidean",
            cluster_selection_method="eom"
        )
        labels = clusterer.fit_predict(X_reduced)
        df["cluster"] = labels
        
        # Save results
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        df.to_parquet(OUTPUT_FILE, index=False)
        logger.info(f"Saved clustered articles to {OUTPUT_FILE}")
        
        # Log statistics
        num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info(f"Identified {num_clusters} clusters.")
        logger.info("Cluster size distribution:")
        logger.info(df["cluster"].value_counts().to_string())
        
        return True
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return False
    except Exception as e:
        logger.error(f"Error in clustering pipeline: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

# to run this script, use the command:
# python pipeline/cluster.py