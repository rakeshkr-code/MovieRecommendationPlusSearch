# config/config.py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class DataConfig:
    """Data configuration settings"""
    raw_movies_path: str = "../archive/tmdb_5000_movies.csv"
    raw_credits_path: str = "../archive/tmdb_5000_credits.csv"
    db_path: str = "./data/movies.db"
    processed_table: str = "processed_movies"
    raw_movies_table: str = "raw_movies"
    raw_credits_table: str = "raw_credits"
    
@dataclass
class ModelConfig:
    """Model configuration settings"""
    # Matrix Factorization
    mf_embedding_dim: int = 50
    mf_learning_rate: float = 0.001
    mf_epochs: int = 20
    mf_batch_size: int = 256
    
    # Neural Collaborative Filtering
    ncf_embedding_dim: int = 64
    ncf_mlp_layers: list = None
    ncf_learning_rate: float = 0.001
    ncf_epochs: int = 20
    ncf_batch_size: int = 256
    ncf_dropout: float = 0.2
    
    # Content-Based (your current approach)
    cb_max_features: int = 5000
    cb_top_n: int = 10
    
    def __post_init__(self):
        if self.ncf_mlp_layers is None:
            self.ncf_mlp_layers = [128, 64, 32, 16]

@dataclass
class MLFlowConfig:
    """MLflow tracking configuration"""
    tracking_uri: str = "http://localhost:5000"  # Adjust for your GCP setup
    experiment_name: str = "movie_recommender"
    artifact_location: Optional[str] = "./mlruns"

@dataclass
class Config:
    """Main configuration class"""
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    mlflow: MLFlowConfig = MLFlowConfig()
    random_seed: int = 42
