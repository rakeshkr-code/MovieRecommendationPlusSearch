# models/content_based.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
from .base_model import BaseRecommender

class ContentBasedRecommender(BaseRecommender):
    """Content-based recommender using cosine similarity"""
    
    def __init__(self):
        super().__init__("ContentBased")
        self.similarity_matrix = None
        self.movie_ids = None
    
    def train(self, features: np.ndarray, movie_ids: np.ndarray):
        """
        Train by computing similarity matrix
        
        Args:
            features: Feature matrix (n_movies, n_features)
            movie_ids: Array of movie IDs
        """
        self.movie_ids = movie_ids
        self.similarity_matrix = cosine_similarity(features)
        self.is_trained = True
    
    def predict(self, movie_idx: int) -> np.ndarray:
        """Get similarity scores for a movie"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        return self.similarity_matrix[movie_idx]
    
    def recommend(self, movie_idx: int, top_n: int = 10) -> List[Tuple[int, float]]:
        """
        Recommend similar movies
        
        Args:
            movie_idx: Index of the movie
            top_n: Number of recommendations
            
        Returns:
            List of (movie_id, similarity_score) tuples
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        similarity_scores = self.similarity_matrix[movie_idx]
        
        # Get top N similar movies (excluding the movie itself)
        similar_indices = similarity_scores.argsort()[::-1][1:top_n+1]
        
        recommendations = [
            (self.movie_ids[idx], similarity_scores[idx])
            for idx in similar_indices
        ]
        
        return recommendations
