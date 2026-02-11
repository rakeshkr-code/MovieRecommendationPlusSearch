# rag/vector_store.py
"""
Vector store for semantic movie search using ChromaDB
"""

import chromadb
from chromadb.config import Settings
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path

from .embeddings import EmbeddingGenerator


class MovieVectorStore:
    """Vector store for semantic search of movies"""
    
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "movies"):
        """
        Initialize vector store
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.logger = logging.getLogger(__name__)
        
        # Create persist directory
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Initialize embedding generator
        self.embedder = EmbeddingGenerator()
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(collection_name)
            self.logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = None
            self.logger.info(f"Collection {collection_name} not found. Will create on index creation.")
    
    def create_index(self, df: pd.DataFrame, text_column: str = 'tags', force_recreate: bool = False):
        """
        Create vector index from dataframe
        
        Args:
            df: DataFrame with movie data
            text_column: Column containing text to embed
            force_recreate: If True, delete existing collection and recreate
        """
        if force_recreate and self.collection is not None:
            self.logger.info("Deleting existing collection...")
            self.client.delete_collection(self.collection_name)
            self.collection = None
        
        if self.collection is not None and not force_recreate:
            count = self.collection.count()
            if count > 0:
                self.logger.info(f"Collection already exists with {count} documents. Use force_recreate=True to rebuild.")
                return
        
        # Create collection
        self.logger.info(f"Creating new collection: {self.collection_name}")
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        
        # Prepare data
        texts = df[text_column].fillna("").tolist()
        ids = [str(movie_id) for movie_id in df['id'].tolist()]
        
        # Create metadata
        metadatas = []
        for _, row in df.iterrows():
            metadata = {
                'movie_id': int(row['id']),
                'title': str(row['title'])
            }
            metadatas.append(metadata)
        
        # Generate embeddings
        self.logger.info(f"Generating embeddings for {len(texts)} movies...")
        embeddings = self.embedder.encode(texts, batch_size=32, show_progress=True)
        
        # Add to collection in batches
        batch_size = 500
        total = len(texts)
        
        for i in range(0, total, batch_size):
            end_idx = min(i + batch_size, total)
            
            self.collection.add(
                ids=ids[i:end_idx],
                embeddings=embeddings[i:end_idx].tolist(),
                documents=texts[i:end_idx],
                metadatas=metadatas[i:end_idx]
            )
            
            self.logger.info(f"Indexed {end_idx}/{total} documents")
        
        self.logger.info(f"Indexing complete! Total documents: {self.collection.count()}")
    
    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Semantic search for movies
        
        Args:
            query: Natural language query
            top_k: Number of results to return
            
        Returns:
            List of dictionaries with movie_id, title, distance/similarity
        """
        if self.collection is None:
            raise ValueError("Vector store not initialized. Call create_index() first.")
        
        # Generate query embedding
        query_embedding = self.embedder.encode_single(query)
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            include=['metadatas', 'documents', 'distances']
        )
        
        # Format results
        formatted_results = []
        
        if results['ids'] and len(results['ids'][0]) > 0:
            for i in range(len(results['ids'][0])):
                result = {
                    'movie_id': results['metadatas'][0][i]['movie_id'],
                    'title': results['metadatas'][0][i]['title'],
                    'distance': results['distances'][0][i],
                    'similarity': 1 - results['distances'][0][i],  # Convert distance to similarity
                    'text': results['documents'][0][i][:200] + "..."  # Preview text
                }
                formatted_results.append(result)
        
        return formatted_results
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        if self.collection is None:
            return {'status': 'not_initialized', 'count': 0}
        
        return {
            'status': 'ready',
            'count': self.collection.count(),
            'name': self.collection_name,
            'embedding_dim': self.embedder.get_dimension()
        }
    
    def is_ready(self) -> bool:
        """Check if vector store is ready for search"""
        return self.collection is not None and self.collection.count() > 0
