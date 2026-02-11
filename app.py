# app.py
"""
Streamlit Web UI for Movie Recommendation System
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
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
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
    .rec-card {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    .rec-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
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


def display_movie_details(movie_id: int, movie_title: str, tmdb_client: TMDBClient):
    """Display detailed movie information"""
    
    col1, col2 = st.columns([1, 2])
    
    # Get TMDB details
    details = tmdb_client.get_movie_with_details(movie_id)
    
    with col1:
        # Display poster
        if details.get('poster_url'):
            st.image(details['poster_url'], use_container_width=True)
        else:
            st.image("https://via.placeholder.com/500x750?text=No+Poster", use_container_width=True)
    
    with col2:
        # Movie title and year
        release_year = details['release_date'][:4] if details.get('release_date') and len(details['release_date']) >= 4 else 'Unknown'
        st.markdown(f"<div class='movie-title'>{details['title']} ({release_year})</div>", unsafe_allow_html=True)
        
        # Rating
        vote_avg = details.get('vote_average', 0)
        vote_count = details.get('vote_count', 0)
        st.markdown(f"<div class='rating'>⭐ {vote_avg:.1f}/10 ({vote_count:,} votes)</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Genres
        if details.get('genres'):
            st.markdown("**Genres:**")
            genres_html = " ".join([f"<span class='genre-tag'>{genre}</span>" for genre in details['genres']])
            st.markdown(genres_html, unsafe_allow_html=True)
        
        # Runtime
        if details.get('runtime'):
            st.markdown(f"**Runtime:** {details['runtime']} minutes")
        
        # Tagline
        if details.get('tagline'):
            st.markdown(f"*\"{details['tagline']}\"*")
        
        # Overview
        st.markdown("**Overview:**")
        st.write(details.get('overview', 'No overview available.'))


def display_recommendations(recommendations: List[Tuple[int, float]], movies_df: pd.DataFrame, tmdb_client: TMDBClient):
    """Display recommended movies in a grid"""
    
    st.markdown("---")
    st.markdown("### 🎯 Recommended Movies")
    
    # Create grid layout
    cols_per_row = 5
    num_recs = len(recommendations)
    
    for i in range(0, num_recs, cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j in range(cols_per_row):
            idx = i + j
            if idx >= num_recs:
                break
            
            movie_id, score = recommendations[idx]
            
            # Get movie info from dataframe
            movie_row = movies_df[movies_df['id'] == movie_id]
            
            if movie_row.empty:
                continue
            
            movie_title = movie_row.iloc[0]['title']
            
            # Get TMDB details
            details = tmdb_client.get_movie_with_details(movie_id)
            
            with cols[j]:
                # Poster
                if details.get('poster_url'):
                    st.image(details['poster_url'], use_container_width=True)
                else:
                    st.image("https://via.placeholder.com/300x450?text=No+Poster", use_container_width=True)
                
                # Title
                st.markdown(f"**{details['title']}**")
                
                # Score/Rating
                if 'mf' in st.session_state.get('model_type', 'content') or 'ncf' in st.session_state.get('model_type', 'content'):
                    st.markdown(f"Score: {score:.3f}")
                else:
                    st.markdown(f"Similarity: {score:.3f}")
                
                # TMDB Rating
                vote_avg = details.get('vote_average', 0)
                if vote_avg > 0:
                    st.markdown(f"⭐ {vote_avg:.1f}/10")


def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown("<div class='main-header'>🎬 Movie Recommendation System</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Discover your next favorite movie</div>", unsafe_allow_html=True)
    
    # Load data
    movies_df = load_movies_data()
    
    if movies_df is None:
        st.stop()
    
    # Get TMDB client
    tmdb_client = get_tmdb_api_client()
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Model selection
        st.subheader("Recommendation Model")
        model_type = st.selectbox(
            "Select Model",
            options=['content', 'mf', 'ncf'],
            format_func=lambda x: {
                'content': 'Content-Based Filtering',
                'mf': 'Matrix Factorization',
                'ncf': 'Neural Collaborative Filtering'
            }[x],
            index=0
        )
        st.session_state['model_type'] = model_type
        
        # Top-N slider
        top_n = st.slider("Number of Recommendations", min_value=5, max_value=20, value=10, step=1)
        
        st.markdown("---")
        
        # Instructions
        st.subheader("📖 How to Use")
        st.markdown("""
        1. Select a movie from the dropdown
        2. View movie details and poster
        3. Click "Get Recommendations"
        4. Browse similar movies
        """)
        
        st.markdown("---")
        st.caption("Powered by TMDB API")
    
    # Movie selection
    st.markdown("### 🔍 Select a Movie")
    
    # Create searchable dropdown
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
        st.session_state['selected_movie'] = selected_movie
        st.session_state['selected_movie_id'] = movie_ids_map[selected_movie]
        st.session_state['selected_movie_idx'] = movie_idx_map[selected_movie]
        
        st.markdown("---")
        
        # Display movie details
        display_movie_details(
            st.session_state['selected_movie_id'],
            selected_movie,
            tmdb_client
        )
        
        st.markdown("---")
        
        # Recommendation button
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            get_recs_button = st.button("🎯 Get Recommendations", use_container_width=True, type="primary")
        
        if get_recs_button:
            # Check model type compatibility
            if model_type in ['mf', 'ncf']:
                st.warning(f"⚠️ {model_type.upper()} model is designed for user-based recommendations. Falling back to Content-Based model for movie-to-movie similarity.")
                model_type = 'content'
                st.session_state['model_type'] = 'content'
            
            # Load model
            with st.spinner(f"Loading {model_type} model..."):
                model = load_model(model_type)
            
            if model is None:
                st.error(f"❌ {model_type} model not found. Please train the model first.")
                st.code(f"python main.py --mode train --model {model_type}", language="bash")
                st.stop()
            
            # Get recommendations
            with st.spinner("Generating recommendations..."):
                try:
                    recommendations = model.recommend(
                        st.session_state['selected_movie_idx'],
                        top_n=top_n
                    )
                    
                    st.session_state['recommendations'] = recommendations
                    
                except Exception as e:
                    st.error(f"Error generating recommendations: {e}")
                    st.stop()
        
        # Display recommendations if available
        if 'recommendations' in st.session_state and st.session_state.get('selected_movie') == selected_movie:
            display_recommendations(
                st.session_state['recommendations'],
                movies_df,
                tmdb_client
            )
    
    else:
        # Welcome message
        st.info("👆 Select a movie from the dropdown above to get started!")
        
        # Display some popular movies as suggestions
        st.markdown("### 🌟 Popular Movies")
        st.caption("Try searching for these movies:")
        
        popular_movies = [
            "Avatar",
            "The Dark Knight",
            "Inception",
            "Interstellar",
            "The Avengers",
            "Pulp Fiction",
            "The Godfather",
            "Forrest Gump"
        ]
        
        cols = st.columns(4)
        for i, movie in enumerate(popular_movies):
            with cols[i % 4]:
                st.markdown(f"• {movie}")


if __name__ == "__main__":
    main()
