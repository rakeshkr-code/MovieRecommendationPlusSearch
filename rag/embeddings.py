# rag/embeddings.py
"""
Embedding generation for movie descriptions using sentence-transformers
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Optional
import logging


class EmbeddingGenerator:
    """Generate embeddings for text using sentence-transformers"""
    
    DEFAULT_MODEL = 'sentence-transformers/all-MiniLM-L6-v2'
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize embedding generator
        
        Args:
            model_name: Sentence transformer model name
                       Default: all-MiniLM-L6-v2 (lightweight, fast)
                       Alternative: all-mpnet-base-v2 (better quality)
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Loading embedding model: {self.model_name}")
        try:
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self.logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def encode(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            numpy array of embeddings (n_texts, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=True  # L2 normalization for better similarity
            )
            return embeddings
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {e}")
            raise
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text
            
        Returns:
            numpy array of embedding (embedding_dim,)
        """
        return self.encode([text], show_progress=False)[0]
    
    def get_dimension(self) -> int:
        """Get embedding dimension"""
        return self.embedding_dim
