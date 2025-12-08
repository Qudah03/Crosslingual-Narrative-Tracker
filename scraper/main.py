import os
import sys
import json
from scrapy.crawler import CrawlerProcess
from datetime import datetime 

# 1. Setup Paths
# Add the current directory to path so we can import the spiders
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import your spiders
# Ensure these match your actual class names and file locations
from news_spiders.bbc_spider import BBCSpider
from news_spiders.aljazeera_ar_spider import AlJazeeraArabicSpider
from news_spiders.dw_spider import DWSpider
from news_spiders.le_monde_spider import LeMondeSpider

def run_all_spiders():
    # We use a temporary "JSON Lines" file.
    # This allows multiple spiders to write to one file without corrupting it.
    temp_output = "data/raw/temp_headlines.jsonl"
    final_output = "data/raw/all_headlines.json"
    
    # Clean up old temp file if it exists
    if os.path.exists(temp_output):
        os.remove(temp_output)

    # 2. Configure Scrapy
    # We set the FEED here globally. All spiders will dump into this ONE file.
    process = CrawlerProcess(settings={
        "FEEDS": {
            temp_output: {
                "format": "jsonlines",  # Crucial: Use jsonlines, not json
                "encoding": "utf8",
                "overwrite": True
            }
        },
        "LOG_LEVEL": "INFO",
        "USER_AGENT": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    })

    # 3. Queue the Spiders
    print("🕷️  Queuing spiders...")
    process.crawl(BBCSpider)
    process.crawl(AlJazeeraArabicSpider)
    process.crawl(DWSpider)
    process.crawl(LeMondeSpider) # You can comment this out if you want to skip it

    # 4. Start Scraping (This blocks until finished)
    print("⏳ Running spiders... (This may take a moment)")
    process.start()

   # 5. Convert JSONL -> JSON
    print("🔄 Formatting data...")
    all_articles = []
    
    # Get today's date string (e.g., "2023-10-27")
    today_str = datetime.now().strftime("%Y-%m-%d")

    if os.path.exists(temp_output):
        with open(temp_output, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        article = json.loads(line)
                        # --- NEW: Add Timestamp ---
                        article['scraped_at'] = today_str
                        # --------------------------
                        all_articles.append(article)
                    except json.JSONDecodeError:
                        pass
        
        # Cleanup
        os.remove(temp_output)

    # Save the final clean file
    os.makedirs(os.path.dirname(final_output), exist_ok=True)
    with open(final_output, 'w', encoding='utf-8') as f:
        json.dump(all_articles, f, indent=4, ensure_ascii=False)

    print(f"✅ Success! Saved {len(all_articles)} articles to {final_output}")

if __name__ == "__main__":
    run_all_spiders()