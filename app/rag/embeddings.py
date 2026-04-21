"""
embeddings.py
Part of the app/rag module.
MiniLM-based embeddings for semantic search and retrieval.
"""

import numpy as np
from typing import List, Dict, Optional, Union
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class MiniLMEncoder:
    """
    MiniLM-based encoder for generating text embeddings.
    Lightweight and efficient for real-time trading applications.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize MiniLM encoder
        
        Args:
            model_name: Sentence-transformers model name
        """
        self.model_name = model_name
        self.model = None
        self.embedding_dim = 384  # MiniLM-L6 output dimension
        self.is_initialized = False
        
    def initialize(self):
        """Lazy initialization of the model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.is_initialized = True
            logger.info(f"MiniLM encoder initialized: {self.model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed. Using fallback embeddings.")
            self.is_initialized = False
    
    def encode(self, texts: Union[str, List[str]], 
               normalize: bool = True) -> np.ndarray:
        """
        Encode text(s) to embeddings
        
        Args:
            texts: Single text or list of texts
            normalize: Whether to normalize embeddings
        
        Returns:
            Embedding array of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        if not self.is_initialized:
            self.initialize()
        
        if self.model is None:
            # Fallback: random embeddings (for development only)
            logger.warning("Using fallback random embeddings")
            embeddings = np.random.randn(len(texts), self.embedding_dim)
        else:
            embeddings = self.model.encode(texts, normalize_embeddings=normalize)
        
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a search query"""
        return self.encode(query)[0]
    
    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings"""
        if embedding1.ndim == 1:
            embedding1 = embedding1.reshape(1, -1)
        if embedding2.ndim == 1:
            embedding2 = embedding2.reshape(1, -1)
        
        # Normalize if needed
        embedding1 = embedding1 / (np.linalg.norm(embedding1, axis=1, keepdims=True) + 1e-8)
        embedding2 = embedding2 / (np.linalg.norm(embedding2, axis=1, keepdims=True) + 1e-8)
        
        similarity = np.dot(embedding1, embedding2.T)
        return float(similarity[0, 0])


class EmbeddingGenerator:
    """
    Generates and manages embeddings for documents and queries.
    Supports batch processing and caching.
    """
    
    def __init__(self, cache_size: int = 10000):
        """
        Initialize embedding generator
        
        Args:
            cache_size: Maximum size of embedding cache
        """
        self.encoder = MiniLMEncoder()
        self.cache: Dict[str, np.ndarray] = {}
        self.cache_size = cache_size
        
    def get_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Get embedding for text (with caching)
        
        Args:
            text: Input text
            use_cache: Whether to use cache
        
        Returns:
            Embedding vector
        """
        if use_cache and text in self.cache:
            return self.cache[text].copy()
        
        embedding = self.encoder.encode(text)[0]
        
        if use_cache:
            self._add_to_cache(text, embedding)
        
        return embedding
    
    def get_embeddings_batch(self, texts: List[str], 
                             use_cache: bool = True) -> List[np.ndarray]:
        """
        Get embeddings for multiple texts
        
        Args:
            texts: List of input texts
            use_cache: Whether to use cache
        
        Returns:
            List of embedding vectors
        """
        uncached_texts = []
        uncached_indices = []
        results = [None] * len(texts)
        
        if use_cache:
            for i, text in enumerate(texts):
                if text in self.cache:
                    results[i] = self.cache[text].copy()
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        if uncached_texts:
            new_embeddings = self.encoder.encode(uncached_texts)
            
            for idx, embedding in zip(uncached_indices, new_embeddings):
                results[idx] = embedding
                if use_cache:
                    self._add_to_cache(texts[idx], embedding)
        
        return results
    
    def _add_to_cache(self, text: str, embedding: np.ndarray):
        """Add embedding to cache with LRU eviction"""
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[text] = embedding
    
    def clear_cache(self):
        """Clear embedding cache"""
        self.cache.clear()
        logger.info("Embedding cache cleared")
    
    def compute_similarity_matrix(self, texts: List[str]) -> np.ndarray:
        """
        Compute similarity matrix for a list of texts
        
        Args:
            texts: List of texts
        
        Returns:
            Similarity matrix of shape (n, n)
        """
        embeddings = self.get_embeddings_batch(texts)
        embeddings_matrix = np.vstack(embeddings)
        
        # Normalize
        norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        embeddings_matrix = embeddings_matrix / (norms + 1e-8)
        
        # Compute similarity matrix
        similarity = np.dot(embeddings_matrix, embeddings_matrix.T)
        
        return similarity
    
    def find_most_similar(self, query: str, candidates: List[str], 
                          top_k: int = 5) -> List[tuple]:
        """
        Find most similar texts to a query
        
        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of results to return
        
        Returns:
            List of (text, similarity_score) tuples
        """
        query_embedding = self.get_embedding(query)
        candidate_embeddings = self.get_embeddings_batch(candidates)
        
        similarities = []
        for text, emb in zip(candidates, candidate_embeddings):
            sim = self.encoder.similarity(query_embedding, emb)
            similarities.append((text, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]