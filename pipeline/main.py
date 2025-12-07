"""
Master Pipeline: Automated ML Workflow
Orchestrates: Embed → Cluster → Analyze Narratives → Visualize
"""

import os
import subprocess
import sys
import logging
from datetime import datetime

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

def run_step(step_num: int, step_name: str, script_path: str) -> bool:
    """Run a pipeline step and handle errors."""
    try:
        logger.info(f"{'='*60}")
        logger.info(f"Step {step_num}: {step_name}")
        logger.info(f"{'='*60}")
        
        result = subprocess.run(
            [sys.executable, script_path],
            check=True,
            capture_output=False
        )
        logger.info(f"[OK] Step {step_num} completed successfully\n")
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ Step {step_num} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        logger.error(f"✗ Step {step_num} error: {e}")
        return False


def main():
    """Execute complete ML pipeline."""
    
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    logger.info("\n" + "="*60)
    logger.info("CROSSLINGUAL NARRATIVE TRACKER - ML PIPELINE")
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*60 + "\n")
    
    steps = [
        (1, "Generate Embeddings", "pipeline/embed.py"),
        (2, "Cluster Articles", "pipeline/cluster.py"),
        (3, "Analyze Narratives", "pipeline/narrative_analyzer.py"),
        (4, "Visualize Results", "pipeline/visualize_clusters.py"),
    ]
    
    completed = 0
    failed_step = None
    
    for step_num, step_name, script_path in steps:
        # Check if script exists
        if not os.path.exists(script_path):
            logger.warning(f"Step {step_num} skipped: {script_path} not found")
            continue
        
        success = run_step(step_num, step_name, script_path)
        
        if success:
            completed += 1
        else:
            failed_step = step_name
            logger.error(f"Pipeline halted at Step {step_num}")
            break
    
    # Final summary
    logger.info("="*60)
    logger.info("PIPELINE SUMMARY")
    logger.info("="*60)
    logger.info(f"Completed: {completed}/{len(steps)} steps")
    
    if failed_step:
        logger.error(f"Failed at: {failed_step}")
        logger.info("Check logs/pipeline.log for details")
        return False
    else:
        logger.info("✓ All steps completed successfully!")
        logger.info(f"Ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("\nOutput files:")
        logger.info("  - data/processed/embeddings.parquet")
        logger.info("  - data/processed/clustered_articles.parquet")
        logger.info("  - data/processed/narrative_analysis.parquet")
        logger.info("\nNext: Run 'streamlit run dashboard/app.py' to view results")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)