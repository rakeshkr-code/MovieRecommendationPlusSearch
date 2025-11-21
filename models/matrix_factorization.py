# models/matrix_factorization.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional
from .base_model import BaseRecommender

class MatrixFactorization(nn.Module):
    """Matrix Factorization model using PyTorch"""
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 50):
        super(MatrixFactorization, self).__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        
        # Initialize embeddings
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        user_b = self.user_bias(user_ids).squeeze()
        item_b = self.item_bias(item_ids).squeeze()
        
        # Dot product + biases
        prediction = (user_emb * item_emb).sum(dim=1) + user_b + item_b
        
        return prediction

class MFRecommender(BaseRecommender):
    """Matrix Factorization Recommender wrapper"""
    
    def __init__(self, num_users: int, num_items: int, embedding_dim: int = 50):
        super().__init__("MatrixFactorization")
        self.model = MatrixFactorization(num_users, num_items, embedding_dim)
        self.num_users = num_users
        self.num_items = num_items
        self.movie_ids = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train(self, user_ids: np.ndarray, item_ids: np.ndarray, ratings: np.ndarray,
              epochs: int = 20, batch_size: int = 256, learning_rate: float = 0.001,
              movie_id_mapping: Optional[np.ndarray] = None):
        """
        Train the matrix factorization model
        
        Args:
            user_ids: Array of user IDs
            item_ids: Array of item IDs (indices)
            ratings: Array of ratings/interactions
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            movie_id_mapping: Mapping from indices to movie IDs
        """
        self.movie_ids = movie_id_mapping
        
        # Convert to tensors
        user_tensor = torch.LongTensor(user_ids).to(self.device)
        item_tensor = torch.LongTensor(item_ids).to(self.device)
        rating_tensor = torch.FloatTensor(ratings).to(self.device)
        
        # Setup training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        dataset = torch.utils.data.TensorDataset(user_tensor, item_tensor, rating_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_users, batch_items, batch_ratings in dataloader:
                optimizer.zero_grad()
                
                predictions = self.model(batch_users, batch_items)
                loss = criterion(predictions, batch_ratings)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True
    
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict rating for user-item pair"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id]).to(self.device)
            item_tensor = torch.LongTensor([item_id]).to(self.device)
            prediction = self.model(user_tensor, item_tensor)
            
        return prediction.item()
    
    def recommend(self, user_id: int, top_n: int = 10) -> List[Tuple[int, float]]:
        """
        Recommend items for a user
        
        Args:
            user_id: User ID
            top_n: Number of recommendations
            
        Returns:
            List of (movie_id, predicted_score) tuples
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        with torch.no_grad():
            user_tensor = torch.LongTensor([user_id] * self.num_items).to(self.device)
            item_tensor = torch.LongTensor(range(self.num_items)).to(self.device)
            predictions = self.model(user_tensor, item_tensor).cpu().numpy()
        
        # Get top N items
        top_indices = predictions.argsort()[::-1][:top_n]
        
        if self.movie_ids is not None:
            recommendations = [
                (self.movie_ids[idx], predictions[idx])
                for idx in top_indices
            ]
        else:
            recommendations = [
                (idx, predictions[idx])
                for idx in top_indices
            ]
        
        return recommendations
