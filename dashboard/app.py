import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# CONFIG
# ----------------------------
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
DATA_PATH = "data/processed/clustered_articles.parquet"

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_data(path):
    df = pd.read_parquet(path)
    # Ensure embeddings are numpy arrays
    df["embedding"] = df["embedding"].apply(lambda x: np.array(x))
    return df

df = load_data(DATA_PATH)

st.title("Cross-lingual Narrative Tracker")
st.markdown("Visualize article clusters and explore narratives across multiple sources.")

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model(model_name):
    return SentenceTransformer(model_name)

model = load_model(EMBEDDING_MODEL)

# ----------------------------
# 2D UMAP projection
# ----------------------------
if "umap_x" not in df.columns or "umap_y" not in df.columns:
    from umap import UMAP
    embeddings = np.stack(df["embedding"].to_numpy())
    umap = UMAP(n_components=2, random_state=42)
    umap_proj = umap.fit_transform(embeddings)
    df["umap_x"], df["umap_y"] = umap_proj[:,0], umap_proj[:,1]

# ----------------------------
# SCATTER PLOT: Clusters & Topics
# ----------------------------
fig = px.scatter(
    df,
    x="umap_x",
    y="umap_y",
    color="cluster",  # <-- use the actual column from your parquet
    hover_data=["title", "source"] + [col for col in df.columns if col.startswith("narrative_")],
    title="Article Clusters by Topic",
    color_discrete_sequence=px.colors.qualitative.Safe  # discrete colors for clusters
)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# KEYWORD SEARCH
# ----------------------------
st.sidebar.header("Keyword Search")
query = st.sidebar.text_input("Enter keyword or phrase:")

if query:
    query_emb = model.encode([query])
    all_embeddings = np.stack(df["embedding"].to_numpy())
    sims = cosine_similarity(all_embeddings, query_emb).flatten()
    df["similarity"] = sims
    top_matches = df.sort_values("similarity", ascending=False).head(10)
    
    st.sidebar.subheader(f"Top articles for '{query}':")
    for i, row in top_matches.iterrows():
        st.sidebar.markdown(f"**{row['title']}**")
        st.sidebar.markdown(f"Source: {row['source']}, Topic: {row['cluster']}, Similarity: {row['similarity']:.2f}")
        narratives = [col.replace("narrative_", "") for col in df.columns if col.startswith("narrative_") and row[col]==1]
        if narratives:
            st.sidebar.markdown(f"Narratives: {', '.join(narratives)}")
        st.sidebar.markdown("---")

# ----------------------------
# Optional: Topic similarity heatmap
# ----------------------------
st.header("Topic Similarity (Average Embeddings)")

topic_embeddings = df.groupby("cluster")["embedding"].apply(lambda x: np.mean(np.stack(x), axis=0))
clusters = topic_embeddings.index.tolist()
topic_matrix = np.stack(topic_embeddings.to_numpy())
sim_matrix = cosine_similarity(topic_matrix)

import plotly.figure_factory as ff
fig_sim = ff.create_annotated_heatmap(
    z=sim_matrix,
    x=[f"Topic {i}" for i in clusters],
    y=[f"Topic {i}" for i in clusters],
    colorscale="Viridis"
)
st.plotly_chart(fig_sim, use_container_width=True)
st.markdown("This heatmap shows the cosine similarity between average embeddings of different topics.")