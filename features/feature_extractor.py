# features/feature_extractor.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from typing import Tuple

class FeatureExtractor:
    """Extract features for different model types"""
    
    def __init__(self, max_features: int = 5000):
        self.max_features = max_features
        self.count_vectorizer = None
        self.tfidf_vectorizer = None
        
    def extract_count_features(self, texts: pd.Series) -> np.ndarray:
        """Extract count-based features (for content-based filtering)"""
        self.count_vectorizer = CountVectorizer(
            max_features=self.max_features,
            stop_words='english'
        )
        features = self.count_vectorizer.fit_transform(texts).toarray()
        return features
    
    def extract_tfidf_features(self, texts: pd.Series) -> np.ndarray:
        """Extract TF-IDF features"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words='english'
        )
        features = self.tfidf_vectorizer.fit_transform(texts).toarray()
        return features
    
    def prepare_collaborative_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for collaborative filtering models
        For TMDB dataset without explicit ratings, we create implicit feedback
        """
        # Create synthetic user interactions based on popularity and vote_average
        # In production, you'd use actual user-item interactions
        
        # For demonstration: create implicit ratings from vote_average
        movie_ids = df['id'].values
        
        # Generate synthetic user-item pairs
        # This is a placeholder - in production use real interaction data
        num_users = 1000  # Synthetic user count
        interactions = []
        
        for movie_idx, movie_id in enumerate(movie_ids):
            # Simulate 5-20 users interacting with each movie
            num_interactions = np.random.randint(5, 20)
            user_ids = np.random.choice(num_users, num_interactions, replace=False)
            
            for user_id in user_ids:
                # Implicit feedback (1 for interaction, 0 for no interaction)
                interactions.append([user_id, movie_idx, 1])
        
        interactions = np.array(interactions)
        
        return interactions[:, 0], interactions[:, 1], interactions[:, 2]
