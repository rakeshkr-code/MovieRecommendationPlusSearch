# models/base_model.py
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Optional
import pickle

class BaseRecommender(ABC):
    """Abstract base class for all recommender models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.is_trained = False
    
    @abstractmethod
    def train(self, *args, **kwargs):
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, *args, **kwargs) -> np.ndarray:
        """Make predictions"""
        pass
    
    @abstractmethod
    def recommend(self, item_id: int, top_n: int = 10) -> List[Tuple[int, float]]:
        """Recommend top N items"""
        pass
    
    def save_model(self, path: str):
        """Save model to disk"""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load_model(path: str) -> 'BaseRecommender':
        """Load model from disk"""
        with open(path, 'rb') as f:
            return pickle.load(f)
