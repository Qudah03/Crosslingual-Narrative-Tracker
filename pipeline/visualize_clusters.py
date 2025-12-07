import pandas as pd
import plotly.express as px
import logging
import sys
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

INPUT_FILE = "data/processed/narrative_analysis.parquet"
OUTPUT_HTML = "data/processed/cluster_map.html"

def main():
    try:
        if not os.path.exists(INPUT_FILE):
            logger.error(f"Input file not found: {INPUT_FILE}")
            return False

        logger.info(f"Loading data from {INPUT_FILE}...")
        df = pd.read_parquet(INPUT_FILE)

        if 'x_coord' not in df.columns:
            logger.error("Dataframe missing 'x_coord' and 'y_coord'. Did you update cluster.py?")
            return False

        # Clean hover text
        df['hover_title'] = df['title'].str.wrap(50).apply(lambda x: x.replace('\n', '<br>'))

        logger.info("Generating interactive scatter plot...")
        fig = px.scatter(
            df,
            x='x_coord',
            y='y_coord',
            color='cluster',
            hover_name='source_clean',
            hover_data={'hover_title': True, 'language': True, 'cluster': True, 'x_coord': False, 'y_coord': False},
            title='Cross-Lingual Narrative Clusters',
            template='plotly_dark',
            color_continuous_scale=px.colors.qualitative.Bold
        )

        fig.update_layout(legend_title_text='Topic Cluster')
        
        # Save to HTML
        fig.write_html(OUTPUT_HTML)
        logger.info(f"Visualization saved to {OUTPUT_HTML}")
        return True

    except Exception as e:
        logger.error(f"Visualization failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)