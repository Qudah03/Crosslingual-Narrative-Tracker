import streamlit as st
import pandas as pd
import plotly.express as px
import os
from collections import Counter
import string

# -------------------------------------------------------------------------
# 1. CONFIGURATION
# -------------------------------------------------------------------------
st.set_page_config(
    page_title="Narrative Tracker", 
    layout="wide",
    page_icon="🌍"
)

DATA_PATH = "data/processed/narrative_analysis.parquet"

# Map categories to Emojis for visual cues
CATEGORY_ICONS = {
    "conflict": "⚔️ Conflict",
    "politics": "🏛️ Politics",
    "economy": "💰 Economy",
    "environment": "🌱 Environment",
    "health": "🏥 Health",
    "technology": "💻 Technology"
}

# -------------------------------------------------------------------------
# 2. DATA LOADING (Cached for Speed)
# -------------------------------------------------------------------------
@st.cache_data
def load_data():
    """Loads the processed data from Parquet. Cached to prevent reloading."""
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_parquet(DATA_PATH)

def get_smart_label_fallback(df, cluster_id):
    """
    Fallback logic: Creates a label like 'Conflict: War, Sudan, Army'
    if the AI summary is missing.
    """
    if cluster_id == -1: return "⚠️ Noise / Outliers"
    
    cluster_df = df[df['cluster'] == cluster_id]
    
    # Find Dominant Narrative (e.g., Conflict)
    narrative_cols = [c for c in df.columns if c.startswith('narrative_')]
    dominant_category = "General"
    
    if narrative_cols:
        counts = {col: cluster_df[col].sum() for col in narrative_cols}
        if counts:
            best_col = max(counts, key=counts.get)
            if counts[best_col] > 0:
                cat_name = best_col.replace("narrative_", "")
                dominant_category = CATEGORY_ICONS.get(cat_name, cat_name.title())

    # Find Top Keywords
    titles = cluster_df['title'].astype(str).str.cat(sep=' ')
    titles = titles.lower().translate(str.maketrans('', '', string.punctuation))
    stop_words = set(['the', 'and', 'for', 'that', 'with', 'from', 'this', 'after', 'says', 'will', 'have', 'been'])
    words = [w for w in titles.split() if len(w) > 3 and w not in stop_words]
    keywords = ", ".join([w[0].title() for w in Counter(words).most_common(3)]) if words else "Misc"

    return f"{dominant_category}: {keywords}"

# -------------------------------------------------------------------------
# 3. MAIN APPLICATION
# -------------------------------------------------------------------------
def main():
    st.title("🌍 Cross-Lingual Narrative Tracker")
    st.markdown("Analyzing how political narratives shift across languages (Powered by Phi-3 AI).")

    df = load_data()

    if df is None:
        st.error(f"Data not found at {DATA_PATH}. Run the pipeline first!")
        return

    # --- SIDEBAR: FILTER LOGIC ---
    if 'cluster' in df.columns:
        unique_clusters = sorted(df['cluster'].unique())
        
        # Determine Label Source (AI vs Fallback)
        if 'llm_label' in df.columns:
            st.sidebar.success("✨ AI Summaries Active")
            label_map = df[['cluster', 'llm_label']].drop_duplicates().set_index('cluster')['llm_label'].to_dict()
        else:
            st.sidebar.warning("⚠️ Using Keyword Fallback")
            label_map = {c: get_smart_label_fallback(df, c) for c in unique_clusters}
        
        st.sidebar.header("Filters")
        
        # Filter Dropdown
        selected_cluster_ids = st.sidebar.multiselect(
            "Select Narratives", 
            options=unique_clusters,
            format_func=lambda x: f"{x}: {label_map.get(x, 'Unknown')[:50]}...", 
            default=unique_clusters[:3] if len(unique_clusters) > 0 else None
        )
        
        # Apply Filter
        if selected_cluster_ids:
            filtered_df = df[df['cluster'].isin(selected_cluster_ids)].copy()
        else:
            filtered_df = df.copy()
    else:
        st.warning("No clusters found. Showing all data.")
        filtered_df = df.copy()
        label_map = {}

    # --- TOP METRICS ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Articles", len(filtered_df))
    col2.metric("Sources", filtered_df['source_clean'].nunique())
    col3.metric("Topics Selected", filtered_df['cluster'].nunique() if 'cluster' in filtered_df.columns else 0)

    st.markdown("---")

    # --- TABS LAYOUT (Cleaner Look) ---
    tab1, tab2, tab3 = st.tabs(["🗺️ Narrative Map", "📈 Trends Over Time", "📋 Deep Dive"])

    # TAB 1: CLUSTER MAP
    with tab1:
        if 'x_coord' in filtered_df.columns:
            # Prepare Legend Label (Truncated)
            filtered_df['Narrative'] = filtered_df['cluster'].map(label_map)
            filtered_df['Legend_Label'] = filtered_df['Narrative'].apply(lambda x: x[:60] + "..." if isinstance(x, str) and len(x) > 60 else x)
            
            fig = px.scatter(
                filtered_df,
                x='x_coord', y='y_coord',
                color='Legend_Label', 
                hover_data={'Narrative': True, 'title': True, 'source': True, 'Legend_Label': False},
                template='plotly_dark',
                height=650,
                title="Semantic Cluster Map"
            )
            # Fix Squeezing: Move Legend to Bottom
            fig.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
                margin=dict(b=100)
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Coordinates not found.")

    # TAB 2: TEMPORAL ANALYSIS
    with tab2:
        if 'scraped_at' in filtered_df.columns:
            # Aggregation
            timeline_df = filtered_df.groupby(['scraped_at', 'cluster']).size().reset_index(name='count')
            
            # Map Labels
            if label_map:
                timeline_df['Narrative'] = timeline_df['cluster'].map(label_map)
                timeline_df['Legend_Label'] = timeline_df['Narrative'].apply(lambda x: x[:60] + "..." if isinstance(x, str) and len(x) > 60 else x)
            else:
                timeline_df['Legend_Label'] = timeline_df['cluster'].astype(str)
            
            fig_time = px.line(
                timeline_df, 
                x='scraped_at', y='count', 
                color='Legend_Label',
                title='Narrative Volume Over Time', 
                template='plotly_dark', 
                markers=True,
                height=500
            )
            # Fix Squeezing: Move Legend to Bottom here too
            fig_time.update_layout(
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                margin=dict(b=100)
            )
            st.plotly_chart(fig_time, use_container_width=True)
        else:
            st.info("No timeline data available yet. Run the scraper on multiple days to see trends.")

    # TAB 3: RAW DATA
    with tab3:
        display_cols = ['title', 'source', 'language', 'scraped_at']
        # Show AI Label if available
        if 'llm_label' in df.columns:
            display_cols.insert(0, 'llm_label')
        elif 'cluster' in filtered_df.columns:
            display_cols.insert(0, 'cluster')
        
        st.dataframe(
            filtered_df[display_cols], 
            use_container_width=True,
            hide_index=True
        )

if __name__ == "__main__":
    main()