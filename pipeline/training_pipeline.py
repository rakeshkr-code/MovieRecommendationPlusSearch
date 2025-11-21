# pipeline/training_pipeline.py
import pandas as pd
import numpy as np
from typing import Dict, Any
from ..data_scripts import DatabaseManager
from ..features import FeatureExtractor, FeatureEngineer
from ..models import ContentBasedRecommender, MFRecommender, NCFRecommender
from ..utils import Logger, MLFlowTracker

class TrainingPipeline:
    """Training pipeline for recommender models"""
    
    def __init__(self, config):
        self.config = config
        self.logger = Logger.get_logger(__name__)
        self.db_manager = DatabaseManager(config.data.db_path)
        self.feature_extractor = FeatureExtractor(config.model.cb_max_features)
        self.feature_engineer = FeatureEngineer()
        self.mlflow_tracker = MLFlowTracker(config.mlflow)
    
    def load_data(self) -> pd.DataFrame:
        """Load processed data from database"""
        self.logger.info("Loading data from database...")
        with self.db_manager as db:
            df = db.load_dataframe(self.config.data.processed_table)
        self.logger.info(f"Loaded {len(df)} movies")
        return df
    
    def train_content_based(self, df: pd.DataFrame) -> ContentBasedRecommender:
        """Train content-based recommender"""
        self.logger.info("Training Content-Based Recommender...")
        
        with self.mlflow_tracker.start_run("content_based"):
            # Extract features
            features = self.feature_extractor.extract_count_features(df['tags'])
            
            # Initialize and train model
            model = ContentBasedRecommender()
            model.train(features, df['id'].values)
            
            # Log to MLflow
            self.mlflow_tracker.log_params({
                'model_type': 'content_based',
                'max_features': self.config.model.cb_max_features,
                'num_movies': len(df)
            })
            
            # Save model
            model_path = "./artifacts/content_based_model.pkl"
            model.save_model(model_path)
            self.mlflow_tracker.log_artifact(model_path)
            
            self.logger.info("Content-Based model trained successfully")
            return model
    
    def train_matrix_factorization(self, df: pd.DataFrame) -> MFRecommender:
        """Train matrix factorization model"""
        self.logger.info("Training Matrix Factorization Recommender...")
        
        with self.mlflow_tracker.start_run("matrix_factorization"):
            # Prepare collaborative data
            user_ids, item_ids, ratings = self.feature_extractor.prepare_collaborative_data(df)
            
            # Initialize model
            num_users = int(user_ids.max()) + 1
            num_items = len(df)
            
            model = MFRecommender(
                num_users=num_users,
                num_items=num_items,
                embedding_dim=self.config.model.mf_embedding_dim
            )
            
            # Train
            model.train(
                user_ids=user_ids,
                item_ids=item_ids,
                ratings=ratings,
                epochs=self.config.model.mf_epochs,
                batch_size=self.config.model.mf_batch_size,
                learning_rate=self.config.model.mf_learning_rate,
                movie_id_mapping=df['id'].values
            )
            
            # Log to MLflow
            self.mlflow_tracker.log_params({
                'model_type': 'matrix_factorization',
                'embedding_dim': self.config.model.mf_embedding_dim,
                'epochs': self.config.model.mf_epochs,
                'batch_size': self.config.model.mf_batch_size,
                'learning_rate': self.config.model.mf_learning_rate,
                'num_users': num_users,
                'num_items': num_items
            })
            
            # Save model
            model_path = "./artifacts/mf_model.pkl"
            model.save_model(model_path)
            self.mlflow_tracker.log_artifact(model_path)
            
            self.logger.info("Matrix Factorization model trained successfully")
            return model
    
    def train_neural_cf(self, df: pd.DataFrame) -> NCFRecommender:
        """Train neural collaborative filtering model"""
        self.logger.info("Training Neural Collaborative Filtering Recommender...")
        
        with self.mlflow_tracker.start_run("neural_cf"):
            # Prepare collaborative data
            user_ids, item_ids, labels = self.feature_extractor.prepare_collaborative_data(df)
            
            # Initialize model
            num_users = int(user_ids.max()) + 1
            num_items = len(df)
            
            model = NCFRecommender(
                num_users=num_users,
                num_items=num_items,
                embedding_dim=self.config.model.ncf_embedding_dim,
                mlp_layers=self.config.model.ncf_mlp_layers,
                dropout=self.config.model.ncf_dropout
            )
            
            # Train
            model.train(
                user_ids=user_ids,
                item_ids=item_ids,
                labels=labels,
                epochs=self.config.model.ncf_epochs,
                batch_size=self.config.model.ncf_batch_size,
                learning_rate=self.config.model.ncf_learning_rate,
                movie_id_mapping=df['id'].values
            )
            
            # Log to MLflow
            self.mlflow_tracker.log_params({
                'model_type': 'neural_cf',
                'embedding_dim': self.config.model.ncf_embedding_dim,
                'mlp_layers': str(self.config.model.ncf_mlp_layers),
                'dropout': self.config.model.ncf_dropout,
                'epochs': self.config.model.ncf_epochs,
                'batch_size': self.config.model.ncf_batch_size,
                'learning_rate': self.config.model.ncf_learning_rate,
                'num_users': num_users,
                'num_items': num_items
            })
            
            # Save model
            model_path = "./artifacts/ncf_model.pkl"
            model.save_model(model_path)
            self.mlflow_tracker.log_artifact(model_path)
            
            self.logger.info("Neural CF model trained successfully")
            return model
    
    def run(self, model_type: str = 'all') -> Dict[str, Any]:
        """
        Run training pipeline
        
        Args:
            model_type: 'content', 'mf', 'ncf', or 'all'
        """
        self.logger.info(f"Starting training pipeline for model type: {model_type}")
        
        # Load data
        df = self.load_data()
        
        models = {}
        
        if model_type in ['content', 'all']:
            models['content_based'] = self.train_content_based(df)
        
        if model_type in ['mf', 'all']:
            models['matrix_factorization'] = self.train_matrix_factorization(df)
        
        if model_type in ['ncf', 'all']:
            models['neural_cf'] = self.train_neural_cf(df)
        
        self.logger.info("Training pipeline completed successfully")
        return models
