# utils/mlflow_tracker.py
import mlflow
from typing import Dict, Any, Optional
from contextlib import contextmanager

class MLFlowTracker:
    """MLflow experiment tracking utility"""
    
    def __init__(self, config):
        self.config = config
        mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_experiment(config.experiment_name)
    
    @contextmanager
    def start_run(self, run_name: str):
        """Context manager for MLflow runs"""
        with mlflow.start_run(run_name=run_name):
            yield
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters"""
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_artifact(self, local_path: str):
        """Log artifact"""
        mlflow.log_artifact(local_path)
    
    def log_model(self, model, artifact_path: str):
        """Log model"""
        mlflow.pytorch.log_model(model, artifact_path)
