# data/data_processor.py
import pandas as pd
import ast
import numpy as np
from typing import List, Dict, Any
from nltk.stem.porter import PorterStemmer

class DataProcessor:
    """Processes raw movie data into clean features"""
    
    def __init__(self):
        self.stemmer = PorterStemmer()
        
    def parse_json_column(self, text: str) -> List[str]:
        """Parse JSON-like string columns"""
        try:
            data = ast.literal_eval(text)
            return [item['name'] for item in data]
        except (ValueError, SyntaxError, TypeError):
            return []
    
    def get_top_n_items(self, text: str, n: int = 3) -> List[str]:
        """Extract top N items from JSON column"""
        try:
            data = ast.literal_eval(text)
            return [item['name'] for item in data[:n]]
        except (ValueError, SyntaxError, TypeError):
            return []
    
    def get_director(self, text: str) -> List[str]:
        """Extract director name from crew column"""
        try:
            data = ast.literal_eval(text)
            for person in data:
                if person.get('job') == 'Director':
                    return [person['name']]
        except (ValueError, SyntaxError, TypeError):
            pass
        return []
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if pd.isna(text):
            return ""
        return text.lower().replace(" ", "")
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        """Apply stemming to tokens"""
        return [self.stemmer.stem(token) for token in tokens]
    
    def process_dataframe(self, movies_df: pd.DataFrame, credits_df: pd.DataFrame) -> pd.DataFrame:
        """Main processing pipeline for processed_movies table (unchanged for content-based recommendations)"""
        # Merge dataframes
        df = movies_df.merge(credits_df, left_on='id', right_on='movie_id', how='inner')

        df = df.rename(columns={'title_x': 'title'})
        
        # Select relevant columns
        df = df[['id', 'title', 'genres', 'keywords', 'overview', 'cast', 'crew']]
        
        # Handle missing values
        df.dropna(inplace=True)
        
        # Parse JSON columns
        df['genres'] = df['genres'].apply(self.parse_json_column)
        df['keywords'] = df['keywords'].apply(self.parse_json_column)
        df['cast'] = df['cast'].apply(lambda x: self.get_top_n_items(x, n=3))
        df['crew'] = df['crew'].apply(self.get_director)
        
        # Process overview
        df['overview'] = df['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])
        
        # Clean all list columns
        for col in ['genres', 'keywords', 'cast', 'crew']:
            df[col] = df[col].apply(lambda x: [self.clean_text(i) for i in x])
        
        # Create tags column (combining all features)
        df['tags'] = df['overview'] + df['genres'] + df['keywords'] + df['cast'] + df['crew']
        df['tags'] = df['tags'].apply(lambda x: " ".join(x))
        
        # Keep only id, title, and tags for final output
        processed_df = df[['id', 'title', 'tags']].copy()
        processed_df['tags'] = processed_df['tags'].apply(lambda x: x.lower())
        
        return processed_df
    
    def process_metadata_dataframe(self, movies_df: pd.DataFrame, credits_df: pd.DataFrame) -> pd.DataFrame:
        """Process dataframe for movies_metadata table (for SQL queries with full metadata)"""
        # Merge dataframes
        df = movies_df.merge(credits_df, left_on='id', right_on='movie_id', how='inner')
        df = df.rename(columns={'title_x': 'title'})
        
        # Select columns with metadata
        columns_to_keep = ['id', 'title', 'genres', 'overview', 'vote_average', 'vote_count', 
                          'release_date', 'runtime', 'popularity', 'original_language']
        columns_to_keep = [col for col in columns_to_keep if col in df.columns]
        df = df[columns_to_keep]
        
        # Handle missing values
        if 'vote_average' in df.columns:
            df['vote_average'] = df['vote_average'].fillna(0.0)
        if 'vote_count' in df.columns:
            df['vote_count'] = df['vote_count'].fillna(0)
        if 'runtime' in df.columns:
            df['runtime'] = df['runtime'].fillna(0)
        if 'popularity' in df.columns:
            df['popularity'] = df['popularity'].fillna(0.0)
        if 'release_date' in df.columns:
            df['release_date'] = df['release_date'].fillna('Unknown')
            # Extract year
            df['release_year'] = df['release_date'].apply(
                lambda x: int(x[:4]) if isinstance(x, str) and len(x) >= 4 and x[:4].isdigit() else None
            )
        
        # Parse genres to readable text
        if 'genres' in df.columns:
            df['genres'] = df['genres'].fillna('[]')
            df['genres_list'] = df['genres'].apply(self.parse_json_column)
            df['genres'] = df['genres_list'].apply(lambda x: ', '.join(x) if x else '')
            df = df.drop('genres_list', axis=1)
        
        # Clean overview
        if 'overview' in df.columns:
            df['overview'] = df['overview'].fillna('')
        
        # Drop rows with missing critical data
        df.dropna(subset=['id', 'title'], inplace=True)
        
        return df
