# ML Model: Narrative Understanding & Topic Grouping

## How It Works

Your ML pipeline now understands narratives and groups articles by topic across multiple languages.

### 1. **Embedding Layer** (`embed.py`)
- Converts article text to 1024-dimensional vectors using a multilingual transformer
- Same embedding for all languages → cross-lingual understanding
- Model: `intfloat/multilingual-e5-large` (supports 100+ languages)

### 2. **Clustering Layer** (`cluster.py`)
- Groups similar articles together using HDBSCAN
- Uses UMAP for dimensionality reduction
- Creates distinct clusters even across languages

### 3. **Narrative Analysis Layer** (`narrative_analyzer.py`) ⭐ NEW
This is the key component that understands narratives:

#### A. **Narrative Feature Extraction**
Detects narrative themes in articles:
- **Conflict**: war, violence, attack, military
- **Politics**: elections, government, parliament
- **Economy**: trade, markets, finance
- **Environment**: climate, pollution, sustainability
- **Health**: disease, pandemic, vaccines
- **Technology**: AI, digital, innovation

#### B. **Topic Grouping by Similarity**
- Groups articles by semantic similarity (not just keywords)
- Threshold: 0.75 (adjustable)
- Same article topic can appear in multiple languages

#### C. **Pattern Detection**
Identifies:
- Which narratives are most common
- How topics vary by language/source
- Narrative patterns within each topic

### 4. **Visualization Layer** (`visualize_clusters.py`)
- 2D scatter plots of article clusters
- Color-coded by topic and language
- Shows narrative distribution

---

## Running the Full Pipeline

```powershell
# Activate venv
.\.venv311\Scripts\Activate.ps1

# Run complete pipeline (all 4 steps)
python pipeline/main.py
```

This will:
1. ✓ Generate embeddings from articles
2. ✓ Cluster articles into topics
3. ✓ Analyze narratives in each topic
4. ✓ Create visualizations
5. ✓ Save results to `data/processed/narrative_analysis.parquet`

---

## Understanding the Output

### Narrative Analysis Results

The pipeline outputs:
- **embeddings.parquet**: Raw embeddings for each article
- **clustered_articles.parquet**: Articles with cluster IDs
- **narrative_analysis.parquet**: Full narrative analysis with:
  - `cluster`: Which topic group the article belongs to
  - `narrative_*`: Boolean flags for detected narratives
  - `article_length`: Depth of coverage
  - `source_bias`: Source credibility level
  - `language`: Article language

### Example Output in logs

```
Top narratives: {
    'conflict': 45,
    'politics': 32,
    'economy': 28,
    'environment': 15
}

Language coverage: {
    'en': 89,
    'fr': 42,
    'ar': 38,
    'de': 35
}

Topic 0: 12 articles
  Main title: "Global Trade War Escalates"
  Narratives: conflict, economy
  Sources: BBC, Reuters, Le Monde
  Languages: en, fr
```

---

## How to Customize Narratives

Edit `pipeline/narrative_analyzer.py` line 59-69:

```python
narrative_keywords = {
    "conflict": ["war", "conflict", "attack", "violence"],
    "politics": ["election", "government", "parliament"],
    "YOUR_TOPIC": ["keyword1", "keyword2", "keyword3"],  # Add custom
}
```

---

## How Articles Are Grouped by Topic

1. **Embedding-based**: Articles with embeddings >0.75 cosine similarity form a topic
2. **Cross-language**: Same story in Arabic, English, French = 1 topic
3. **Narrative-aware**: Topics labeled by detected narratives

Example: BBC "War in Gaza" + Al Jazeera "الصراع في غزة" = Same topic

---

## Advanced: Fine-tune the Model

To improve topic grouping, adjust in `cluster.py`:

```python
# Lower = fewer, larger clusters
min_cluster_size=3   # (default: 5)

# Higher = stricter similarity threshold
threshold=0.80       # (default: 0.75 in narrator_analyzer.py)
```

---

## Automation Options

### Daily Automatic Execution (Windows Task Scheduler)

```powershell
# Create scheduled task
$action = New-ScheduledTaskAction -Execute "python" -Argument "pipeline/main.py" -WorkingDirectory "C:\Users\LOQ\Desktop\SideProject\Crosslingual-Narrative-Tracker"
$trigger = New-ScheduledTaskTrigger -Daily -At 10:00AM
Register-ScheduledTask -TaskName "NewsNarrativeTracker" -Action $action -Trigger $trigger
```

### View Dashboard

```powershell
streamlit run dashboard/app.py
```

Then open `http://localhost:8501`

---

## Troubleshooting

**Error: "No articles found"**
- Check `data/raw/` has JSON files
- Run scrapers first to populate data

**Error: "embeddings.parquet not found"**
- Run `python pipeline/embed.py` first

**Error: "Model download failed"**
- Check internet connection
- Model is ~1.3GB, first run takes time

**Low topic quality**
- Increase `threshold` in narrative_analyzer.py (0.8, 0.85)
- Add more narrative keywords
- Check article quality in data/raw/

---

## Next Steps

1. ✓ Collect data in `data/raw/`
2. ✓ Run `python pipeline/main.py`
3. ✓ View results in logs/pipeline.log
4. ✓ Launch dashboard with streamlit
5. ✓ Schedule for daily automation

For feedback on narratives, check `logs/pipeline.log` after each run!
