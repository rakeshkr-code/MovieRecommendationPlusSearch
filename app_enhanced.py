# app_enhanced.py
"""
Enhanced Streamlit Web UI with RAG and LangChain Integration
Includes 4 search modes: Browse, SQL Agent, Recommendations, and RAG Search
"""

import streamlit as st
import pandas as pd
import os
from pathlib import Path
from dotenv import load_dotenv
from typing import List, Tuple, Optional

from config import Config
from data_scripts import DatabaseManager
from models import ContentBasedRecommender, MFRecommender, NCFRecommender
from utils import get_tmdb_client, TMDBClient
from rag import MovieVectorStore
from langchain_utils import MovieSQLAgent

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #E50914;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 1.5rem;
    }
    .movie-title {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .rating {
        font-size: 1.5rem;
        color: #FFA500;
    }
    .genre-tag {
        display: inline-block;
        background-color: #E50914;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        margin: 0.2rem;
        font-size: 0.9rem;
    }
    .stButton>button {
        background-color: #E50914;
        color: white;
        font-size: 1.1rem;
        padding: 0.6rem 2rem;
        border-radius: 5px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #B20710;
    }
    .st-tabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .st-tabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0px 24px;
        background-color: #f0f0f0;
        border-radius: 5px 5px 0px 0px;
    }
    .st-tabs [aria-selected="true"] {
        background-color: #E50914;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_config():
    """Load configuration"""
    return Config()


@st.cache_resource
def load_movies_data():
    """Load movie data from database"""
    config = load_config()
    db_manager = DatabaseManager(config.data.db_path)
    
    try:
        with db_manager as db:
            df = db.load_dataframe(config.data.processed_table)
        return df
    except Exception as e:
        st.error(f"Failed to load movie data: {e}")
        st.info("Please run the ETL pipeline first: `python main.py --mode etl`")
        return None


@st.cache_resource
def load_model(model_type: str):
    """Load trained model"""
    model_paths = {
        'content': './artifacts/content_based_model.pkl',
        'mf': './artifacts/mf_model.pkl',
        'ncf': './artifacts/ncf_model.pkl'
    }
    
    model_path = Path(model_paths[model_type])
    
    if not model_path.exists():
        return None
    
    try:
        if model_type == 'content':
            return ContentBasedRecommender.load_model(str(model_path))
        elif model_type == 'mf':
            return MFRecommender.load_model(str(model_path))
        elif model_type == 'ncf':
            return NCFRecommender.load_model(str(model_path))
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


@st.cache_resource
def get_tmdb_api_client():
    """Get TMDB API client"""
    api_key = os.getenv('TMDB_API_KEY')
    return get_tmdb_client(api_key)


@st.cache_resource
def get_vector_store():
    """Get or create vector store"""
    try:
        vector_store = MovieVectorStore()
        return vector_store
    except Exception as e:
        st.error(f"Failed to initialize vector store: {e}")
        return None


@st.cache_resource
def get_sql_agent():
    """Get SQL Agent"""
    try:
        config = load_config()
        hf_api_key = os.getenv('HUGGINGFACE_API_KEY')
        sql_agent = MovieSQLAgent(config.data.db_path, hf_api_key)
        return sql_agent
    except Exception as e:
        st.error(f"Failed to initialize SQL Agent: {e}")
        return None


def display_movie_card(movie_id: int, movie_title: str, tmdb_client: TMDBClient, show_full_details: bool = True):
    """Display movie card with poster and details"""
    details = tmdb_client.get_movie_with_details(movie_id)
    
    if show_full_details:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            if details.get('poster_url'):
                st.image(details['poster_url'], use_container_width=True)
            else:
                st.image("https://via.placeholder.com/500x750?text=No+Poster", use_container_width=True)
        
        with col2:
            release_year = details['release_date'][:4] if details.get('release_date') and len(details['release_date']) >= 4 else 'Unknown'
            st.markdown(f"<div class='movie-title'>{details['title']} ({release_year})</div>", unsafe_allow_html=True)
            
            vote_avg = details.get('vote_average', 0)
            vote_count = details.get('vote_count', 0)
            st.markdown(f"<div class='rating'>⭐ {vote_avg:.1f}/10 ({vote_count:,} votes)</div>", unsafe_allow_html=True)
            
            st.markdown("---")
            
            if details.get('genres'):
                st.markdown("**Genres:**")
                genres_html = " ".join([f"<span class='genre-tag'>{genre}</span>" for genre in details['genres']])
                st.markdown(genres_html, unsafe_allow_html=True)
            
            if details.get('runtime'):
                st.markdown(f"**Runtime:** {details['runtime']} minutes")
            
            if details.get('tagline'):
                st.markdown(f"*\"{details['tagline']}\"*")
            
            st.markdown("**Overview:**")
            st.write(details.get('overview', 'No overview available.'))
    
    return details


def display_movie_grid(movies: List[Tuple[int, float]], movies_df: pd.DataFrame, tmdb_client: TMDBClient, score_label: str = "Similarity"):
    """Display movies in a grid layout"""
    cols_per_row = 5
    num_movies = len(movies)
    
    for i in range(0, num_movies, cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j in range(cols_per_row):
            idx = i + j
            if idx >= num_movies:
                break
            
            movie_id, score = movies[idx]
            movie_row = movies_df[movies_df['id'] == movie_id]
            
            if movie_row.empty:
                continue
            
            details = tmdb_client.get_movie_with_details(movie_id)
            
            with cols[j]:
                if details.get('poster_url'):
                    st.image(details['poster_url'], use_container_width=True)
                else:
                    st.image("https://via.placeholder.com/300x450?text=No+Poster", use_container_width=True)
                
                st.markdown(f"**{details['title']}**")
                st.markdown(f"{score_label}: {score:.3f}")
                
                vote_avg = details.get('vote_average', 0)
                if vote_avg > 0:
                    st.markdown(f"⭐ {vote_avg:.1f}/10")


def tab_browse_movies():
    """Tab 1: Browse movies by dropdown"""
    st.markdown("### 🔍 Browse Movies")
    st.caption("Select a movie from the dropdown to view details and get recommendations")
    
    movies_df = load_movies_data()
    tmdb_client = get_tmdb_api_client()
    
    if movies_df is None:
        return
    
    movie_titles = movies_df['title'].tolist()
    movie_ids_map = dict(zip(movies_df['title'], movies_df['id']))
    movie_idx_map = dict(zip(movies_df['title'], movies_df.index))
    
    selected_movie = st.selectbox(
        "Search for a movie:",
        options=movie_titles,
        index=None,
        placeholder="Type to search..."
    )
    
    if selected_movie:
        st.session_state['browse_selected_movie'] = selected_movie
        st.session_state['browse_selected_id'] = movie_ids_map[selected_movie]
        st.session_state['browse_selected_idx'] = movie_idx_map[selected_movie]
        
        st.markdown("---")
        display_movie_card(movie_ids_map[selected_movie], selected_movie, tmdb_client)
        
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            top_n = st.slider("Number of recommendations", 5, 20, 10, key="browse_top_n")
            get_recs = st.button("🎯 Get Similar Movies", use_container_width=True, type="primary", key="browse_get_recs")
        
        if get_recs:
            with st.spinner("Loading model..."):
                model = load_model('content')
            
            if model:
                with st.spinner("Generating recommendations..."):
                    recs = model.recommend(movie_idx_map[selected_movie], top_n=top_n)
                    st.session_state['browse_recs'] = recs
        
        if 'browse_recs' in st.session_state and st.session_state.get('browse_selected_movie') == selected_movie:
            st.markdown("---")
            st.markdown("### 🎯 Recommended Movies")
            display_movie_grid(st.session_state['browse_recs'], movies_df, tmdb_client, "Similarity")


def tab_sql_search():
    """Tab 2: SQL Agent search"""
    st.markdown("### 💬 Natural Language SQL Search")
    st.caption("Ask questions about movies in natural language - AI will generate and execute SQL queries")
    
    sql_agent = get_sql_agent()
    movies_df = load_movies_data()
    tmdb_client = get_tmdb_api_client()
    
    if not sql_agent or not sql_agent.is_ready:
        st.warning("⚠️ SQL Agent requires HuggingFace API key")
        st.info("Add your HuggingFace API key to `.env` file:")
        st.code("HUGGINGFACE_API_KEY=your_key_here", language="bash")
        
        with st.expander("📖 How to get HuggingFace API key"):
            st.markdown("""
            1. Go to https://huggingface.co/settings/tokens
            2. Click "New token"
            3. Copy the token
            4. Add to`.env` file
            5. Restart the app
            """)
        return
    
    # Example queries
    st.markdown("**Example queries:**")
    examples = sql_agent.get_sample_queries()
    
    cols = st.columns(2)
    for i, example in enumerate(examples[:4]):
        with cols[i % 2]:
            if st.button(f"💡 {example}", key=f"example_{i}"):
                st.session_state['sql_query_input'] = example
    
    st.markdown("---")
    
    # Query input
    query = st.text_area(
        "Enter your question:",
        value=st.session_state.get('sql_query_input', ''),
        placeholder="e.g., Show me top 10 action movies from 2015",
        height=100
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        execute_btn = st.button("🔍 Search", use_container_width=True, type="primary", key="sql_search_btn")
    
    if execute_btn and query:
        with st.spinner("Generating SQL and executing query..."):
            result = sql_agent.query(query)
        
        if result['success']:
            st.success("✅ Query executed successfully!")
            
            if result['sql_query']:
                with st.expander("📝 Generated SQL Query"):
                    st.code(result['sql_query'], language="sql")
            
            st.markdown("### Results")
            st.write(result['result'])
        else:
            st.error(f"❌ Error: {result['error']}")


def tab_recommendations():
    """Tab 3: Content-based recommendations"""
    st.markdown("### 🎯 Get Movie Recommendations")
    st.caption("Select a movie and get similar recommendations using content-based filtering")
    
    movies_df = load_movies_data()
    tmdb_client = get_tmdb_api_client()
    
    if movies_df is None:
        return
    
    movie_titles = movies_df['title'].tolist()
    movie_ids_map = dict(zip(movies_df['title'], movies_df['id']))
    movie_idx_map = dict(zip(movies_df['title'], movies_df.index))
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        selected_movie = st.selectbox(
            "Select a movie:",
            options=movie_titles,
            index=None,
            placeholder="Type to search...",
            key="rec_movie_select"
        )
    
    with col2:
        top_n = st.number_input("Top-N", min_value=5, max_value=20, value=10, key="rec_top_n")
    
    if selected_movie:    
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            get_recs = st.button("🎯 Get Recommendations", use_container_width=True, type="primary", key="rec_get_recs")
        
        if get_recs:
            with st.spinner("Generating recommendations..."):
                model = load_model('content')
                if model:
                    recs = model.recommend(movie_idx_map[selected_movie], top_n=top_n)
                    st.session_state['tab3_recs'] = recs
                    st.session_state['tab3_movie'] = selected_movie
        
        if 'tab3_recs' in st.session_state and st.session_state.get('tab3_movie') == selected_movie:
            st.markdown("---")
            st.markdown(f"### Movies similar to **{selected_movie}**")
            display_movie_grid(st.session_state['tab3_recs'], movies_df, tmdb_client, "Similarity")


def tab_rag_search():
    """Tab 4: RAG semantic search"""
    st.markdown("### 🧠 Smart Semantic Search (RAG)")
    st.caption("Search for movies using natural language descriptions - AI finds semantically similar movies")
    
    vector_store = get_vector_store()
    movies_df = load_movies_data()
    tmdb_client = get_tmdb_api_client()
    
    if movies_df is None:
        return
    
    # Check if index exists
    if not vector_store or not vector_store.is_ready():
        st.warning("⚠️ Vector store not initialized")
        st.info("Click the button below to create the vector index (one-time setup, takes ~2-3 minutes)")
        
        if st.button("🔨 Build Vector Index", type="primary", key="build_vector_index"):
            with st.spinner("Creating embeddings and building index... This may take a few minutes."):
                # vector_store.create_index(movies_df, text_column='tags')
                vector_store.create_index(movies_df, text_column='description')
                st.success("✅ Vector index created (with description column) successfully!")
                st.rerun()
        return
    
    # Display stats
    stats = vector_store.get_collection_stats()
    text_col_label = stats.get('text_column', 'unknown')
    st.info(f"📊 Vector store ready: **{stats['count']}** movies indexed on column **`{text_col_label}`**")
    
    # ── Rebuild section ──────────────────────────────────────────────────────
    with st.expander("⚙️ Rebuild Vector Index with a Different Column"):
        st.markdown("Select a text column from the dataset to embed and rebuild the index.")
        
        # Offer only object (string) columns that are non-empty
        candidate_cols = [
            c for c in movies_df.columns
            if movies_df[c].dtype == object and movies_df[c].notna().any()
        ]
        rebuild_col = st.selectbox(
            "Text column to embed:",
            options=candidate_cols,
            index=candidate_cols.index(text_col_label) if text_col_label in candidate_cols else 0,
            key="rebuild_col_select"
        )
        
        st.warning(
            f"⚠️ This will **DELETE** the current index (built on `{text_col_label}`) "
            f"and rebuild it using `{rebuild_col}`. Takes ~2-3 minutes."
        )
        
        if st.button("🔨 Rebuild Index", key="rebuild_vector_btn", type="primary"):
            with st.spinner(f"Rebuilding vector index using column '{rebuild_col}'…"):
                vector_store.create_index(movies_df, text_column=rebuild_col, force_recreate=True)
            st.success(f"✅ Index rebuilt successfully using column: `{rebuild_col}`")
            # Clear cached resource so the next load picks up the fresh index
            get_vector_store.clear()
            st.rerun()
    # ─────────────────────────────────────────────────────────────────────────
    
    # Example queries
    st.markdown("**Example searches:**")
    example_queries = [
        "Movies about space exploration and aliens",
        "Romantic comedies set in Paris",
        "Action movies with car chases",
        "Films about artificial intelligence",
        "Time travel adventures",
        "Superhero team-up movies"
    ]
    
    cols = st.columns(3)
    for i, example in enumerate(example_queries[:6]):
        with cols[i % 3]:
            if st.button(f"💡 {example}", key=f"rag_example_{i}"):
                st.session_state['rag_query_input'] = example
    
    st.markdown("---")
    
    # Query input
    query = st.text_input(
        "Describe the type of movies you're looking for:",
        value=st.session_state.get('rag_query_input', ''),
        placeholder="e.g., Movies about overcoming adversity and personal growth"
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        top_k = st.number_input("Results", min_value=5, max_value=20, value=10, key="rag_top_k")
    
    with col2:
        search_btn = st.button("🔍 Search", use_container_width=True, type="primary", key="rag_search_btn")
    
    if search_btn and query:
        with st.spinner("Searching..."):
            results = vector_store.search(query, top_k=top_k)
        
        if results:
            st.markdown("### 🎬 Results")
            st.caption(f"Found {len(results)} semantically similar movies")
            
            # Convert results to format for display_movie_grid
            movie_scores = [(r['movie_id'], r['similarity']) for r in results]
            display_movie_grid(movie_scores, movies_df, tmdb_client, "Relevance")
        else:
            st.warning("No results found. Try a different search query.")


def main():
    """Main app with tabs"""
    
    # Header
    st.markdown("<div class='main-header'>🎬 Movie Recommendation System</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Discover movies with AI-powered search and recommendations</div>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ About")
        st.markdown("""
        This app provides 4 ways to discover movies:
        
        1. **Browse** - Select from dropdown
        2. **SQL Search** - Natural language queries
        3. **Recommendations** - Content-based filtering
        4. **Smart Search** - Semantic/RAG search
        """)
        
        st.markdown("---")
        st.subheader("📊 System Status")
        
        # Check components
        movies_df = load_movies_data()
        if movies_df is not None:
            st.success(f"✅ Database: {len(movies_df)} movies")
        else:
            st.error("❌ Database not found")
        
        model = load_model('content')
        if model:
            st.success("✅ Content model loaded")
        else:
            st.warning("⚠️ Content model not trained")
        
        vector_store = get_vector_store()
        if vector_store and vector_store.is_ready():
            st.success("✅ RAG vector store ready")
        else:
            st.warning("⚠️ RAG not initialized")
        
        sql_agent = get_sql_agent()
        if sql_agent and sql_agent.is_ready:
            st.success("✅ SQL Agent ready")
        else:
            st.warning("⚠️ SQL Agent needs API key")
        
        st.markdown("---")
        st.caption("Powered by TMDB, LangChain & ChromaDB")
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔍 Browse Movies",
        "💬 SQL Search",
        "🎯 Recommendations",
        "🧠 Smart Search (RAG)"
    ])
    
    with tab1:
        tab_browse_movies()
    
    with tab2:
        tab_sql_search()
    
    with tab3:
        tab_recommendations()
    
    with tab4:
        tab_rag_search()


if __name__ == "__main__":
    main()
