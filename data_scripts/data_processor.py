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
        df = df[['id', 'title', 'tagline', 'overview', 'genres', 'keywords', 'cast', 'crew']]
        
        # ## Handle missing values
        # df.dropna(inplace=True)
        df[['overview', 'tagline']] = df[['overview', 'tagline']].fillna('not available for this movie').astype(str)
        
        # Parse JSON columns
        df['genres'] = df['genres'].apply(self.parse_json_column)
        df['genres'] = df['genres'].apply(lambda x: ', '.join(x) if isinstance(x, list) else 'Genres not available')
        df['keywords'] = df['keywords'].apply(self.parse_json_column)
        df['keywords'] = df['keywords'].apply(lambda x: ', '.join(x) if isinstance(x, list) else 'Keywords not available')
        df['cast'] = df['cast'].apply(lambda x: self.get_top_n_items(x, n=3))
        df['cast'] = df['cast'].apply(lambda x: ', '.join(x) if isinstance(x, list) else 'Cast not available')
        df['crew'] = df['crew'].apply(self.get_director)
        df['crew'] = df['crew'].apply(lambda x: ', '.join(x) if isinstance(x, list) else 'Director name not available')
        
        # # Process overview
        # df['overview'] = df['overview'].apply(lambda x: x.split() if isinstance(x, str) else [])
        # Not doing any preprocessing on overview to keep it just normal text for better recommendations

        # # Clean all list columns
        # for col in ['genres', 'keywords', 'cast', 'crew']:
        #     df[col] = df[col].apply(lambda x: [self.clean_text(i) for i in x])
        
        # Create tags column (combining all features)
        df['tags'] = df['overview'] + df['genres'] + df['keywords'] + df['cast'] + df['crew']
        # df['tags'] = df['tags'].apply(lambda x: " ".join(x))

        # Description column
        df['description'] = (
            "Movie Name/Title: " + df['title'] + ".\n"
            + "Director of the Movie: " + df['crew'] + ".\n"
            + "Main tagline of the Movie: " + df['tagline'] + ".\n"
            + "Overview of the Movie: " + df['overview'] + ".\n"
            + "Genres of the Movie: " + df['genres'] + ".\n"
            + "Keywords related to the Movie: " + df['keywords'] + ".\n"
            + "Main 3 Actors/cast of the Movie: " + df['cast'] + "."
        )
        
        # # Keep only id, title, and tags for final output
        # processed_df = df[['id', 'title', 'tags']].copy()
        # processed_df['tags'] = processed_df['tags'].apply(lambda x: x.lower())

        processed_df = df[['id', 'title', 'tags', 'description']].copy()
        processed_df['tags'] = processed_df['tags'].apply(lambda x: x.lower())
        processed_df['description'] = processed_df['description'].apply(lambda x: x.lower())
        
        return processed_df
    
    def process_metadata_dataframe(self, movies_df: pd.DataFrame, credits_df: pd.DataFrame) -> pd.DataFrame:
        """Process dataframe for movies_metadata table (for SQL queries with full metadata)"""
        # Merge dataframes
        df = movies_df.merge(credits_df, left_on='id', right_on='movie_id', how='inner')
        df = df.rename(columns={'title_x': 'title'})
        
        # Select columns with metadata
        columns_to_keep = ['id', 'title', 'genres', 'overview', 'vote_average', 'vote_count', 
        # columns_to_keep = ['id', 'title', 'vote_average', 'vote_count', 
                          'release_date', 'runtime', 'popularity', 'original_language',
                          'budget', 'revenue', 'status']
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
                # lambda x: int(x[:4]) if isinstance(x, str) and len(x) >= 4 and x[:4].isdigit() else None
                lambda x: int(x[-4:]) if isinstance(x, str) and len(x) >= 10 and x[-4:].isdigit() else "NA" # None
            )
        
        # Parse genres to readable text
        if 'genres' in df.columns:
            df['genres'] = df['genres'].fillna('[]')
            df['genres_list'] = df['genres'].apply(self.parse_json_column)
            df['genres'] = df['genres_list'].apply(lambda x: ', '.join(x) if x else '')
            df = df.drop('genres_list', axis=1)
        # Clean overview
        if 'overview' in df.columns:
            df['overview'] = df['overview'].fillna('not available for this movie').astype(str)
        
        # Drop rows with missing critical data
        df.dropna(subset=['id', 'title'], inplace=True)
        
        return df

if __name__ == "__main__":
    # Example usage
    movies_df = pd.read_csv('data/raw/tmdb_5000_movies.csv')
    credits_df = pd.read_csv('data/raw/tmdb_5000_credits.csv')
    
    processor = DataProcessor()
    processed_movies = processor.process_dataframe(movies_df, credits_df)    
    print(processed_movies.head())
    print(processed_movies['description'].iloc[0])  # Print description of the first movie for verification
    # ## put the description of the first movie in a text file for better readability
    # with open('artifacts/sample_movie_description.txt', 'w') as f:
    #     f.write("Description of the first movie in processed_movies:\n\n")
    #     f.write(processed_movies['description'].iloc[0])
    #     # Tags
    #     f.write("\n\nTags of the first movie in processed_movies:\n\n")
    #     f.write(processed_movies['tags'].iloc[0])
    # Process metadata dataframe
    processed_metadata = processor.process_metadata_dataframe(movies_df, credits_df)
    print(processed_metadata.head())