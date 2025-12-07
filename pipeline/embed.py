# pipeline/embed.py
from sentence_transformers import SentenceTransformer # pyright: ignore[reportMissingImports]
import pandas as pd # pyright: ignore[reportMissingModuleSource]
import json
import logging
import os
import sys
import glob

MODEL_NAME = "intfloat/multilingual-e5-large"

# Configure logging BEFORE any functions
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_articles(paths):
    """Load articles from JSON files."""
    articles = []
    if not paths:
        logger.warning("No JSON files found in data/raw/")
        return pd.DataFrame()
    
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    logger.warning(f"File {path} is empty, skipping.")
                    continue
                if content.startswith("["):  # full JSON array
                    arr = json.loads(content)
                    articles.extend(arr)
                else:  # JSON lines
                    for line in content.splitlines():
                        if line.strip():
                            articles.append(json.loads(line))
            logger.info(f"Loaded {len(articles)} articles from {path}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {path}: {e}")
        except IOError as e:
            logger.error(f"Failed to read {path}: {e}")
    
    return pd.DataFrame(articles)

def build_text(row):
    title = row.get("title", "")
    summary = row.get("summary", "")
    return (title + " — " + summary).strip()

def main(output_file="data/processed/embeddings.parquet"):
    try:
        # Automatically find all JSON files in data/raw
        input_files = glob.glob("data/raw/*.json")
        logger.info(f"Found {len(input_files)} JSON files: {input_files}")

        logger.info("Loading articles...")
        df = load_articles(input_files)
        if df.empty:
            logger.error("No articles found! Check data/raw/ directory.")
            return False

        df["text"] = df.apply(build_text, axis=1)
        logger.info(f"Encoding {len(df)} articles with {MODEL_NAME}...")
        model = SentenceTransformer(MODEL_NAME)
        embeddings = model.encode(df["text"].tolist(), normalize_embeddings=True, show_progress_bar=True)

        df["embedding"] = [x.tolist() for x in embeddings]
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        df.to_parquet(output_file, index=False)
        logger.info(f"Saved embeddings to {output_file}")
        return True
    
    except Exception as e:
        logger.error(f"Error in embed pipeline: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

# to run this script, use the command:
# python pipeline/embed.py
