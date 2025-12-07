# pipeline/embed.py
from sentence_transformers import SentenceTransformer
import pandas as pd
import json
import os
import glob

MODEL_NAME = "intfloat/multilingual-e5-large"

def load_articles(paths):
    articles = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                continue
            if content.startswith("["):  # full JSON array
                arr = json.loads(content)
                articles.extend(arr)
            else:  # JSON lines
                for line in content.splitlines():
                    if line.strip():
                        articles.append(json.loads(line))
    return pd.DataFrame(articles)

def build_text(row):
    title = row.get("title", "")
    summary = row.get("summary", "")
    return (title + " — " + summary).strip()

def main(output_file="data/processed/embeddings.parquet"):
    # Automatically find all JSON files in data/raw
    input_files = glob.glob("data/raw/*.json")
    print(f"Found {len(input_files)} JSON files: {input_files}")

    print("Loading articles...")
    df = load_articles(input_files)
    if df.empty:
        print("No articles found!")
        return

    df["text"] = df.apply(build_text, axis=1)
    print(f"Encoding {len(df)} articles with {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(df["text"].tolist(), normalize_embeddings=True, show_progress_bar=True)

    df["embedding"] = list(map(lambda x: x.tolist(), embeddings))
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_parquet(output_file, index=False)
    print(f"Saved embeddings to {output_file}")

if __name__ == "__main__":
    main()
