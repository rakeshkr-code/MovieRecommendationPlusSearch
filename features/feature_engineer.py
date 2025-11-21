# features/feature_engineer.py
import pandas as pd
import numpy as np
from typing import Dict, Any

class FeatureEngineer:
    """Engineer additional features for models"""
    
    def __init__(self):
        self.feature_stats: Dict[str, Any] = {}
    
    def create_movie_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional movie features"""
        features_df = df.copy()
        
        # Text length features
        if 'tags' in features_df.columns:
            features_df['tag_length'] = features_df['tags'].str.len()
            features_df['tag_word_count'] = features_df['tags'].str.split().str.len()
        
        # Store statistics
        self.feature_stats['tag_length_mean'] = features_df['tag_length'].mean()
        self.feature_stats['tag_word_count_mean'] = features_df['tag_word_count'].mean()
        
        return features_df
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize feature matrix"""
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std[std == 0] = 1  # Avoid division by zero
        
        normalized = (features - mean) / std
        return normalized
