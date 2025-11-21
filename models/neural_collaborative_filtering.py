# models/neural_collaborative_filtering.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional
from .base_model import BaseRecommender

class NeuralCF(nn.Module):
    """Neural Collaborative Filtering model"""
    
    def __init__(self, num_users: int, num_items: int, 
                 embedding_dim: int = 64, mlp_layers: List[int] = [128, 64, 32, 16],
                 dropout: float = 0.2):
        super(NeuralCF, self).__init__()
        
        # GMF (Generalized Matrix Factorization) component
        self.gmf_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.gmf_item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP component
        self.mlp_user_embedding = nn.Embedding(num_users, embedding_dim)
        self.mlp_item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # MLP layers
        mlp_modules = []
        input_size = embedding_dim * 2
        
        for layer_size in mlp_layers:
            mlp_modules.append(nn.Linear(input_size, layer_size))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(dropout))
            input_size = layer_size
        
        self.mlp = nn.Sequential(*mlp_modules)
        
        # Final prediction layer
        self.output_layer = nn.Linear(embedding_dim + mlp_layers[-1], 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        nn.init.normal_(self.gmf_user_embedding.weight, std=0.01)
        nn.init.normal_(self.gmf_item_embedding.weight, std=0.01)
        nn.init.normal_(self.mlp_user_embedding.weight, std=0.01)
        nn.init.normal_(self.mlp_item_embedding.weight, std=0.01)
    
    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # GMF part
        gmf_user_emb = self.gmf_user_embedding(user_ids)
        gmf_item_emb = self.gmf_item_embedding(item_ids)
        gmf_output = gmf_user_emb * gmf_item_emb  # Element-wise product
        
        # MLP part
        mlp_user_emb = self.mlp_user_embedding(user_ids)
        mlp_item_emb = self.mlp_item_embedding(item_ids)
        mlp_input = torch.cat([mlp_user_emb, mlp_item_emb], dim=1)
        mlp_output = self.mlp(mlp_input)
        
        # Concatenate GMF and MLP outputs
        concat_output = torch.cat([gmf_output, mlp_output], dim=1)
        
        # Final prediction
        prediction = self.output_layer(concat_output).squeeze()
        
        return torch.sigmoid(prediction)

class NCFRecommender(BaseRecommender):
    """Neural Collaborative Filtering Recommender wrapper"""
    
    def __init__(self, num_users: int, num_items: int, 
                 embedding_dim: int = 64, mlp_layers: List[int] = [128, 64, 32, 16],
                 dropout: float = 0.2):
        super().__init__("NeuralCollaborativeFiltering")
        self.model = NeuralCF(num_users, num_items, embedding_dim, mlp_layers, dropout)
        self.num_users = num_users
        self.num_items = num_items
        self.movie_ids = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    def train(self, user_ids: np.ndarray, item_ids: np.ndarray, labels: np.ndarray,
              epochs: int = 20, batch_size: int = 256, learning_rate: float = 0.001,
              movie_id_mapping: Optional[np.ndarray] = None):
        """
        Train the NCF model
        
        Args:
            user_ids: Array of user IDs
            item_ids: Array of item IDs (indices)
            labels: Array of binary labels (1 for interaction, 0 for no interaction)
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            movie_id_mapping: Mapping from indices to movie IDs
        """
        self.movie_ids = movie_id_mapping
        
        # Convert to tensors
        user_tensor = torch.LongTensor(user_ids).to(self.device)
        item_tensor = torch.LongTensor(item_ids).to(self.device)
        label_tensor = torch.FloatTensor(labels).to(self.device)
        
        # Setup training
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        dataset = torch.utils.data.TensorDataset(user_tensor, item_tensor, label_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_users, batch_items, batch_labels in dataloader:
                optimizer.zero_grad()
                
                predictions = self.model(batch_users, batch_items)
                loss = criterion(predictions, batch_labels)
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
        self.is_trained = True
    
    def predict(self, user_id: int, item_id: int) -> float:
        """Predict interaction probability for user-item pair"""
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
