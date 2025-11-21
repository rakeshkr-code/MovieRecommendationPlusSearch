# data/data_loader.py
import pandas as pd
from typing import Tuple
from pathlib import Path

class DataLoader:
    """Handles loading raw data from CSV files"""
    
    def __init__(self, movies_path: str, credits_path: str):
        self.movies_path = Path(movies_path)
        self.credits_path = Path(credits_path)
        
    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load movies and credits CSV files"""
        if not self.movies_path.exists():
            raise FileNotFoundError(f"Movies file not found: {self.movies_path}")
        if not self.credits_path.exists():
            raise FileNotFoundError(f"Credits file not found: {self.credits_path}")
            
        movies_df = pd.read_csv(self.movies_path)
        credits_df = pd.read_csv(self.credits_path)
        
        return movies_df, credits_df
    
    def validate_data(self, movies_df: pd.DataFrame, credits_df: pd.DataFrame) -> bool:
        """Validate loaded data"""
        required_movie_cols = ['id', 'title', 'genres', 'keywords', 'overview']
        required_credit_cols = ['movie_id', 'cast', 'crew']
        
        missing_movie = set(required_movie_cols) - set(movies_df.columns)
        missing_credit = set(required_credit_cols) - set(credits_df.columns)
        
        if missing_movie:
            raise ValueError(f"Missing columns in movies: {missing_movie}")
        if missing_credit:
            raise ValueError(f"Missing columns in credits: {missing_credit}")
            
        return True
