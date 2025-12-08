from sentence_transformers import SentenceTransformer
import pandas as pd
import json
import logging
import os
import sys
import glob
import numpy as np

MODEL_NAME = "intfloat/multilingual-e5-large"
PROCESSED_FILE = "data/processed/embeddings.parquet"

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_text(row):
    return (row.get("title", "") + " — " + row.get("summary", "")).strip()

def main():
    try:
        # 1. Load NEW Raw Data
        input_files = glob.glob("data/raw/*.json")
        new_df = pd.DataFrame()
        for path in input_files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        new_df = pd.concat([new_df, pd.DataFrame(data)])
            except Exception as e:
                logger.error(f"Error reading {path}: {e}")

        if new_df.empty:
            logger.warning("No new raw data found.")
            return True

        # Ensure 'link' column exists for deduplication
        if 'link' not in new_df.columns:
            new_df['link'] = new_df['title'] # Fallback if no link

        # 2. Load OLD Processed Data (History)
        if os.path.exists(PROCESSED_FILE):
            logger.info(f"Loading existing history from {PROCESSED_FILE}...")
            history_df = pd.read_parquet(PROCESSED_FILE)
            
            # Find strictly NEW articles (compare links)
            existing_links = set(history_df['link'].unique())
            new_df = new_df[~new_df['link'].isin(existing_links)]
            
            logger.info(f"Found {len(new_df)} new articles to process.")
        else:
            logger.info("No history found. Starting fresh.")
            history_df = pd.DataFrame()

        # 3. Embed ONLY the New Stuff (Saves compute time)
        if not new_df.empty:
            new_df["text"] = new_df.apply(build_text, axis=1)
            
            logger.info(f"Encoding {len(new_df)} new articles...")
            model = SentenceTransformer(MODEL_NAME)
            embeddings = model.encode(new_df["text"].tolist(), normalize_embeddings=True, show_progress_bar=True)
            
            new_df["embedding"] = [x.tolist() for x in embeddings]
            
            # 4. Merge and Save
            combined_df = pd.concat([history_df, new_df], ignore_index=True)
            
            # Clean up duplicates just in case
            combined_df = combined_df.drop_duplicates(subset=['link'], keep='last')
            
            os.makedirs(os.path.dirname(PROCESSED_FILE), exist_ok=True)
            combined_df.to_parquet(PROCESSED_FILE, index=False)
            logger.info(f"✅ Saved updated dataset with {len(combined_df)} total articles.")
        else:
            logger.info("No new unique articles to add.")

        return True
    
    except Exception as e:
        logger.error(f"Error in embed pipeline: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    sys.exit(0 if main() else 1)