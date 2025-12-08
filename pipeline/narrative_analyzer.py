"""
Narrative Analysis Module
Analyzes articles to:
1. Extract narrative themes (bias, perspective, topic)
2. Group articles by topic similarity
3. Detect narrative patterns across languages
"""

import pandas as pd
import numpy as np
import logging
import sys
import os

# --- FIX: Add project root to path so we can import 'pipeline' ---
# This allows the script to see the 'pipeline' folder even when running inside it
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# ------------------------------------------------------------------

from typing import Dict, List
from sentence_transformers import SentenceTransformer
from pipeline.llm_summarizer import LocalSummarizer

# --------------------------
# Configure logging
# --------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --------------------------
# Source bias map
# --------------------------
# Use this dict to classify sources by bias/reliability
source_map = {
    "BBC": "neutral",
    "Reuters": "neutral",
    "AP": "neutral",
    "Al Jazeera": "neutral",
    "DW": "neutral",
    "Le Monde": "neutral",
    "NYT": "neutral",
    "CNN": "neutral",
    "The Guardian": "neutral",
    "Fox News": "biased_right",
    "Breitbart": "biased_right",
    "RT": "biased_left"
    # You can expand this dynamically or from a JSON/CSV
}

# --------------------------
# Narrative keywords
# --------------------------
narrative_keywords = {
    "conflict": ["war", "conflict", "attack", "violence", "military"],
    "politics": ["election", "government", "parliament", "minister", "policy"],
    "economy": ["economy", "trade", "market", "finance", "business"],
    "environment": ["climate", "environment", "pollution", "sustainability"],
    "health": ["health", "disease", "pandemic", "vaccine", "medical"],
    "technology": ["technology", "digital", "ai", "software", "innovation"]
}

class NarrativeAnalyzer:
    """Analyzes narrative themes and topics in articles."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-large"):
        """Load multilingual sentence transformer for embeddings."""
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            
            # --- FIX: Define source_map inside the class instance ---
            self.source_map = {
                "BBC": "neutral",
                "Reuters": "neutral",
                "AP": "neutral",
                "Al Jazeera": "neutral",
                "DW": "neutral",
                "Le Monde": "neutral",
                "NYT": "neutral",
                "CNN": "neutral",
                "The Guardian": "neutral",
                "Fox News": "biased_right",
                "Breitbart": "biased_right",
                "RT": "biased_left"
            }
            # ---------------------------------------------------------
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    # --------------------------
    # Extract narrative features
    # --------------------------
    def extract_narrative_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds features to the DataFrame:
        - article_length: proxy for depth
        - source_bias: maps source to bias category
        - language: keeps track of article language
        - narrative_X: binary flags if keywords appear in text
        """
        try:
            logger.info("Extracting narrative features...")

            # Article length = title + summary lengths
            df["article_length"] = (
                df["title"].fillna("").str.len() +
                df.get("summary", pd.Series("", index=df.index)).fillna("").str.len()
            )

            # Normalize sources and map bias
            df["source_clean"] = df["source"].fillna("unknown").str.strip()
            df["source_bias"] = df["source_clean"].map(self.source_map).fillna("unknown")

            # Language
            df["language"] = df.get("language", "unknown")

            # Pre-clean text once (lowercase)
            text = df["text"].fillna("").astype(str).str.lower()

            # Narrative keyword detection
            for narrative, keywords in narrative_keywords.items():
                pattern = "|".join([k.lower() for k in keywords])
                df[f"narrative_{narrative}"] = text.str.contains(pattern, na=False, regex=True).astype(int)

            logger.info("Narrative features extracted successfully")
            return df

        except Exception as e:
            logger.error(f"Error extracting narrative features: {e}")
            raise

    def generate_smart_labels(self, df):
        """
        Generates LLM summaries for each cluster.
        """
        from pipeline.llm_summarizer import LocalSummarizer
        
        logger.info("Initializing Local LLM (Phi-3) on RTX 4050...")
        llm = LocalSummarizer()
        
        # Create a map: {cluster_id: "Summary text"}
        cluster_summaries = {}
        unique_clusters = df['cluster'].unique()
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:
                cluster_summaries[cluster_id] = "Noise / Outliers"
                continue
                
            # Get headlines for this cluster
            titles = df[df['cluster'] == cluster_id]['title'].tolist()
            
            logger.info(f"Summarizing Cluster {cluster_id} ({len(titles)} articles)...")
            summary = llm.summarize_cluster(titles)
            cluster_summaries[cluster_id] = summary
            
        # Map back to dataframe
        df['llm_label'] = df['cluster'].map(cluster_summaries)
        return df
    
    # --------------------------
    # Group articles by topic similarity
    # --------------------------
    def group_by_topic_similarity(self, df: pd.DataFrame, embeddings: np.ndarray, threshold: float = 0.75) -> (pd.DataFrame, List[Dict]):
        """
        Clusters articles by cosine similarity of embeddings.
        Returns:
        - df: DataFrame with 'cluster' column
        - topic_groups: list of clusters and their info
        """
        try:
            logger.info(f"Grouping articles by topic similarity (threshold={threshold})...")

            # Cosine similarity between embeddings
            norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            similarities = np.dot(norm_embeddings, norm_embeddings.T)

            topic_groups = []
            used = set()

            for i in range(len(df)):
                if i in used:
                    continue
                similar_indices = np.where(similarities[i] > threshold)[0]
                topic_group = {
                    "cluster": len(topic_groups),
                    "member_count": len(similar_indices),
                    "members": similar_indices.tolist(),
                    "main_title": df.iloc[i]["title"],
                    "avg_similarity": similarities[i][similar_indices].mean()
                }
                topic_groups.append(topic_group)
                used.update(similar_indices)

            # Assign cluster IDs
            df["cluster"] = -1
            for topic in topic_groups:
                for idx in topic["members"]:
                    df.loc[idx, "cluster"] = topic["cluster"]

            logger.info(f"Found {len(topic_groups)} distinct topics")
            return df, topic_groups

        except Exception as e:
            logger.error(f"Error grouping by topic: {e}", exc_info=True)
            raise

    # --------------------------
    # Detect narrative patterns
    # --------------------------
    def detect_narrative_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Returns patterns including:
        - top narratives across corpus
        - language coverage
        - source distribution
        - narratives per topic cluster
        """
        try:
            logger.info("Detecting narrative patterns...")

            patterns = {
                "top_narratives": {},
                "language_coverage": {},
                "source_distribution": {},
                "topic_narratives": {}
            }

            narrative_cols = [c for c in df.columns if c.startswith("narrative_")]

            # 1. Top narratives
            for col in narrative_cols:
                patterns["top_narratives"][col.replace("narrative_", "")] = int(df[col].sum())

            # 2. Language coverage
            if "language" in df.columns:
                patterns["language_coverage"] = df["language"].value_counts().to_dict()

            # 3. Source distribution
            if "source_clean" in df.columns:
                patterns["source_distribution"] = df["source_clean"].value_counts().to_dict()

            # 4. Narratives per cluster
            if "cluster" in df.columns:
                for cluster in df["cluster"].unique():
                    topic_df = df[df["cluster"] == cluster]
                    narratives = {col.replace("narrative_", ""): int(topic_df[col].sum()) for col in narrative_cols}
                    patterns["topic_narratives"][f"topic_{int(cluster)}"] = narratives

            logger.info("Narrative patterns detected")
            return patterns

        except Exception as e:
            logger.error(f"Error detecting patterns: {e}", exc_info=True)
            raise

    # --------------------------
    # Generate human-readable summary
    # --------------------------
    def generate_topic_summary(self, df: pd.DataFrame, topic_groups: List[Dict]) -> str:
        """
        Create a textual report of top topics, including:
        - number of articles
        - main title
        - avg similarity
        - sources and languages
        - detected narratives
        """
        try:
            summary = "\n" + "="*60 + "\n"
            summary += "NARRATIVE ANALYSIS REPORT\n"
            summary += "="*60 + "\n\n"

            for topic in topic_groups[:10]:  # Top 10 topics
                summary += f"Topic {topic['cluster']}: {topic['member_count']} articles\n"
                summary += f"  Main title: {topic['main_title'][:80]}\n"
                summary += f"  Avg similarity: {topic['avg_similarity']:.2f}\n"

                topic_articles = df[df["cluster"] == topic["cluster"]]
                sources = topic_articles.get("source_clean", pd.Series("unknown")).unique()
                languages = topic_articles.get("language", pd.Series("unknown")).unique()
                summary += f"  Sources: {', '.join(map(str, sources[:3]))}\n"
                summary += f"  Languages: {', '.join(map(str, languages[:3]))}\n"

                narrative_cols = [c for c in topic_articles.columns if c.startswith("narrative_")]
                narratives = [c.replace("narrative_", "") for c in narrative_cols if topic_articles[c].sum() > 0]
                if narratives:
                    summary += f"  Narratives: {', '.join(narratives)}\n"

                summary += "\n"

            summary += "="*60 + "\n"
            return summary

        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return ""

# --------------------------
# Main pipeline execution
# --------------------------
def main():
    """Run narrative analysis on clustered articles."""
    try:
        input_file = "data/processed/clustered_articles.parquet"
        output_file = "data/processed/narrative_analysis.parquet"

        logger.info(f"Loading articles from {input_file}...")
        df = pd.read_parquet(input_file)
        if df.empty:
            logger.error("No articles found!")
            return False

        analyzer = NarrativeAnalyzer()
        df = analyzer.extract_narrative_features(df)

        # Ensure embeddings column exists and is an array
        embeddings = np.array(df["embedding"].tolist())
        df, topic_groups = analyzer.group_by_topic_similarity(df, embeddings, threshold=0.75)

        # --- Run LLM Summarization ---
        df = analyzer.generate_smart_labels(df)
        # ----------------------------------

        patterns = analyzer.detect_narrative_patterns(df)
        logger.info(f"Top narratives: {patterns['top_narratives']}")
        logger.info(f"Language coverage: {patterns['language_coverage']}")
        logger.info(f"Source distribution: {patterns['source_distribution']}")

        summary = analyzer.generate_topic_summary(df, topic_groups)
        logger.info(summary)

        df.to_parquet(output_file, index=False)
        logger.info(f"Saved narrative analysis to {output_file}")
        return True

    except Exception as e:
        logger.error(f"Error in narrative analysis: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
