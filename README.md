# 🎬 Movie Recommendation System

A powerful AI-powered movie recommendation system featuring multiple search modes, natural language SQL queries, semantic search with RAG (Retrieval-Augmented Generation), and traditional content-based filtering.

## ✨ Features

### 🔍 Four Search Modes

1. **Browse Movies** - Browse and explore movies with detailed information from TMDB
2. **SQL Search** - Natural language queries powered by LangChain ReAct agent
3. **Recommendations** - Content-based filtering using TF-IDF and cosine similarity
4. **Smart Search (RAG)** - Semantic search using sentence transformers and ChromaDB

### 🤖 AI-Powered Features

- **Natural Language SQL** - Ask questions in plain English, get SQL results
- **Semantic Search** - Find movies by describing themes, plots, or moods
- **Content-Based Filtering** - Get recommendations based on movie similarity
- **ReAct Agent** - Uses Qwen2.5-72B-Instruct for intelligent query reasoning

### 📊 Database Architecture

- **Two-table design** for optimal performance:
  - `processed_movies` - Optimized for ML recommendations (id, title, tags)
  - `movies_metadata` - Rich metadata for SQL queries (11 columns including ratings, genres, runtime)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip
- Virtual environment (recommended)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/MovieRecommendationPlusSearch.git
cd MovieRecommendationPlusSearch
```

2. **Create virtual environment**

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up environment variables**

```bash
# Copy example env file
copy .env.example .env  # Windows
# cp .env.example .env  # Linux/Mac

# Edit .env and add your API keys
notepad .env  # Windows
# nano .env  # Linux/Mac
```

Add these keys to `.env`:

```
TMDB_API_KEY=your_tmdb_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
```

**Get API Keys:**

- **TMDB**: https://www.themoviedb.org/settings/api
- **HuggingFace**: https://huggingface.co/settings/tokens

5. **Run ETL Pipeline** (creates database tables)

```bash
python main.py --mode etl
```

Expected output:

```
Starting ETL pipeline...
Extracted 4803 movies and 4803 credits
Processed 4803 movies for recommendations table
Processed 4803 movies for metadata table
✓ ETL pipeline completed!
```

6. **Train recommendation model** (optional)

```bash
python main.py --mode train --model content
```

7. **Launch the app**

```bash
streamlit run app_enhanced.py
```

The app will open at `http://localhost:8501`

## 📖 Usage Guide

### 1. Browse Movies Tab

- Select any movie from the dropdown
- View poster, ratings, genres, runtime, and overview
- Get instant recommendations for similar movies

### 2. SQL Search Tab

Ask natural language questions like:

- "Show me the top 10 highest rated movies"
- "What are the 5 longest movies?"
- "Find Action movies with more than 1000 votes"
- "Show me movies from 2015"

The ReAct agent will:

1. Understand your question
2. Generate appropriate SQL query
3. Execute and return results

### 3. Recommendations Tab

- Select a movie
- Choose number of recommendations (5-20)
- Get content-based similar movies using ML

### 4. Smart Search (RAG) Tab

**First time setup:**

1. Click "🔨 Build Vector Index" (one-time, ~2-3 minutes)
2. Wait for completion

**Then search with descriptions:**

- "Movies about space exploration and aliens"
- "Romantic comedies set in Paris"
- "Films about artificial intelligence"
- "Time travel adventures"

The RAG system finds semantically similar movies using embeddings.

## 🏗️ Architecture

```
MovieRecommendationPlusSearch/
├── app_enhanced.py           # Main Streamlit web UI
├── main.py                   # CLI entry point (ETL/Training)
├── config.py                 # Configuration management
│
├── data/
│   ├── movies.db            # SQLite database (generated)
│   └── raw/                 # Raw CSV files
│
├── data_scripts/
│   ├── data_loader.py       # CSV loading
│   ├── data_processor.py    # Data transformation
│   └── database_manager.py  # SQLite operations
│
├── langchain_utils/
│   ├── llm_setup.py         # HuggingFace LLM setup
│   └── sql_agent.py         # ReAct SQL Agent
│
├── models/
│   ├── content_based.py     # Content-based recommender
│   ├── matrix_factorization.py
│   └── neural_cf.py
│
├── pipeline/
│   ├── etl_pipeline.py      # Extract-Transform-Load
│   └── training_pipeline.py # Model training
│
├── rag/
│   └── vector_store.py      # ChromaDB vector store
│
└── utils/
    ├── tmdb_api.py          # TMDB API client
    └── logger.py            # Logging utilities
```

### Database Schema

#### `movies_metadata` (SQL queries)

- `id`, `title`, `genres`, `overview`
- `vote_average`, `vote_count`, `release_year`
- `runtime`, `popularity`, `original_language`

#### `processed_movies` (ML recommendations)

- `id`, `title`, `tags` (combined features)

See [TWO_TABLE_ARCHITECTURE.md](TWO_TABLE_ARCHITECTURE.md) for details.

## 🔧 Configuration

Edit `config/config.yaml` to customize:

- Data paths
- Model parameters
- Database settings
- Training hyperparameters

## 📚 Documentation

- [RAG_LANGCHAIN_GUIDE.md](RAG_LANGCHAIN_GUIDE.md) - RAG and LangChain setup
- [TWO_TABLE_ARCHITECTURE.md](TWO_TABLE_ARCHITECTURE.md) - Database design
- [HOW_TO_TRAIN.md](HOW_TO_TRAIN.md) - Model training guide
- [DATABASE_SCHEMA_FIX.md](DATABASE_SCHEMA_FIX.md) - Schema enhancement details

## 🧪 Available Commands

```bash
# Run ETL pipeline
python main.py --mode etl

# Train content-based model
python main.py --mode train --model content

# Train all models
python main.py --mode all

# Check database schema
python check_db_schema.py

# Launch web app
streamlit run app_enhanced.py
```

## 🛠️ Troubleshooting

### SQL Agent Not Working

- Verify HuggingFace API key in `.env`
- Restart the app after adding keys
- Check terminal for error messages

### RAG Search Not Available

- Click "Build Vector Index" button in Smart Search tab
- Wait for indexing to complete (~2-3 minutes)
- Index persists for future sessions

### No Recommendations

- Train the content model first:
  ```bash
  python main.py --mode train --model content
  ```

### Database Errors

- Re-run ETL pipeline:
  ```bash
  python main.py --mode etl
  ```

## 📦 Dependencies

- **Web Framework**: Streamlit
- **ML/AI**: scikit-learn, sentence-transformers, langchain
- **Database**: SQLite, ChromaDB
- **APIs**: TMDB, HuggingFace Inference
- **Deep Learning**: PyTorch (for NCF model)

See `requirements.txt` for complete list.

## 🙏 Acknowledgments

- **TMDB** - Movie data and images
- **HuggingFace** - LLM inference API
- **LangChain** - ReAct agent framework
- **ChromaDB** - Vector database for RAG
- **Streamlit** - Web UI framework
