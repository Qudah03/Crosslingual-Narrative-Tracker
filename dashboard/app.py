import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Page Config
st.set_page_config(page_title="Narrative Tracker", layout="wide")

DATA_PATH = "data/processed/narrative_analysis.parquet"

def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_parquet(DATA_PATH)

def main():
    st.title("Cross-Lingual Narrative Tracker")
    st.markdown("Analyzing how political narratives shift across languages and sources.")

    df = load_data()

    if df is None:
        st.error("Data not found. Run the pipeline first!")
        return

    # Sidebar Filters
    st.sidebar.header("Filters")
    selected_lang = st.sidebar.multiselect("Language", df['language'].unique(), default=df['language'].unique())
    selected_source = st.sidebar.multiselect("Source", df['source_clean'].unique(), default=df['source_clean'].unique())

    # Filter Data
    filtered_df = df[df['language'].isin(selected_lang) & df['source_clean'].isin(selected_source)]

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Articles", len(filtered_df))
    col2.metric("Sources", filtered_df['source_clean'].nunique())
    col3.metric("Topics Detected", filtered_df['cluster'].nunique())

    # Main Visual: Cluster Map
    st.subheader("Topic Cluster Map")
    if 'x_coord' in filtered_df.columns:
        fig = px.scatter(
            filtered_df,
            x='x_coord', y='y_coord',
            color='cluster',
            hover_data=['title', 'source', 'language'],
            template='plotly_dark',
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Coordinates not found. Update cluster.py to save UMAP embeddings.")

    # Data Table
    st.subheader("Deep Dive")
    st.dataframe(filtered_df[['title', 'source', 'language', 'cluster', 'narrative_conflict', 'narrative_politics']])

if __name__ == "__main__":
    main()