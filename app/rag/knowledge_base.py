"""
knowledge_base.py
Part of the app/rag module.
Vector database management for trading knowledge.
"""

import json
import pickle
import numpy as np
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import logging
import os

from .embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Document structure for knowledge base"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        if self.embedding is not None:
            data['embedding'] = self.embedding.tolist()
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Document':
        """Create document from dictionary"""
        if 'embedding' in data and data['embedding']:
            data['embedding'] = np.array(data['embedding'])
        if 'timestamp' in data and isinstance(data['timestamp'], str):
            data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class VectorStore:
    """
    Vector database for semantic search and retrieval
    """
    
    def __init__(self, dimension: int = 384):
        """
        Initialize vector store
        
        Args:
            dimension: Embedding dimension
        """
        self.dimension = dimension
        self.documents: List[Document] = []
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.id_to_index: Dict[str, int] = {}
        
    def add_documents(self, documents: List[Document]):
        """
        Add documents to vector store
        
        Args:
            documents: List of documents to add
        """
        for doc in documents:
            if doc.id not in self.id_to_index:
                self.documents.append(doc)
                self.id_to_index[doc.id] = len(self.documents) - 1
        
        self._rebuild_index()
        logger.info(f"Added {len(documents)} documents. Total: {len(self.documents)}")
    
    def _rebuild_index(self):
        """Rebuild embeddings matrix for efficient search"""
        if not self.documents:
            self.embeddings_matrix = None
            return
        
        embeddings = []
        for doc in self.documents:
            if doc.embedding is not None:
                embeddings.append(doc.embedding)
            else:
                # Placeholder for documents without embeddings
                embeddings.append(np.zeros(self.dimension))
        
        self.embeddings_matrix = np.vstack(embeddings)
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True)
        self.embeddings_matrix = self.embeddings_matrix / (norms + 1e-8)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 10) -> List[Tuple[Document, float]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
        
        Returns:
            List of (document, similarity_score) tuples
        """
        if self.embeddings_matrix is None or len(self.documents) == 0:
            return []
        
        # Normalize query
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Compute similarities
        similarities = np.dot(self.embeddings_matrix, query_norm)
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.3:  # Similarity threshold
                results.append((self.documents[idx], float(similarities[idx])))
        
        return results
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID"""
        if doc_id in self.id_to_index:
            return self.documents[self.id_to_index[doc_id]]
        return None
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document by ID"""
        if doc_id in self.id_to_index:
            idx = self.id_to_index[doc_id]
            del self.documents[idx]
            del self.id_to_index[doc_id]
            # Rebuild index after deletion
            self._rebuild_index()
            return True
        return False
    
    def save(self, filepath: str):
        """Save vector store to disk"""
        data = {
            'dimension': self.dimension,
            'documents': [doc.to_dict() for doc in self.documents]
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Vector store saved to {filepath}")
    
    def load(self, filepath: str):
        """Load vector store from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.dimension = data['dimension']
        self.documents = [Document.from_dict(d) for d in data['documents']]
        self._rebuild_index()
        
        logger.info(f"Vector store loaded from {filepath}")


class KnowledgeBase:
    """
    Complete knowledge base management system
    """
    
    def __init__(self, storage_path: str = "data/knowledge/"):
        """
        Initialize knowledge base
        
        Args:
            storage_path: Path for persistent storage
        """
        self.storage_path = storage_path
        self.embedder = EmbeddingGenerator()
        self.vector_store = VectorStore()
        self.categories: Dict[str, List[str]] = {}
        
        # Ensure storage directory exists
        os.makedirs(storage_path, exist_ok=True)
        
    def add_document(self, content: str, metadata: Dict = None) -> Document:
        """
        Add a document to the knowledge base
        
        Args:
            content: Document content
            metadata: Document metadata
        
        Returns:
            Created document
        """
        doc_id = f"doc_{datetime.now().timestamp()}_{hash(content) % 10000}"
        
        document = Document(
            id=doc_id,
            content=content,
            metadata=metadata or {},
            embedding=self.embedder.get_embedding(content)
        )
        
        self.vector_store.add_documents([document])
        
        # Index by category
        category = metadata.get('category', 'general') if metadata else 'general'
        if category not in self.categories:
            self.categories[category] = []
        self.categories[category].append(doc_id)
        
        logger.info(f"Added document: {doc_id} (category: {category})")
        
        return document
    
    def add_documents_batch(self, documents: List[Dict]) -> List[Document]:
        """
        Add multiple documents
        
        Args:
            documents: List of {'content': str, 'metadata': dict}
        
        Returns:
            List of created documents
        """
        created = []
        for doc_data in documents:
            doc = self.add_document(doc_data['content'], doc_data.get('metadata'))
            created.append(doc)
        return created
    
    def search(self, query: str, category: str = None, top_k: int = 10) -> List[Dict]:
        """
        Search knowledge base
        
        Args:
            query: Search query
            category: Filter by category
            top_k: Number of results
        
        Returns:
            List of search results with content and metadata
        """
        query_embedding = self.embedder.get_embedding(query)
        results = self.vector_store.search(query_embedding, top_k=top_k * 2)
        
        # Apply category filter
        if category:
            results = [(doc, score) for doc, score in results 
                      if doc.metadata.get('category') == category]
        
        # Format results
        formatted_results = []
        for doc, score in results[:top_k]:
            formatted_results.append({
                'id': doc.id,
                'content': doc.content,
                'metadata': doc.metadata,
                'relevance_score': score,
                'timestamp': doc.timestamp.isoformat()
            })
        
        return formatted_results
    
    def get_context(self, query: str, top_k: int = 5) -> str:
        """
        Get context string for RAG prompt
        
        Args:
            query: Search query
            top_k: Number of documents to include
        
        Returns:
            Concatenated context string
        """
        results = self.search(query, top_k=top_k)
        
        if not results:
            return "No relevant context available."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[{i}] {result['content']}")
        
        return "\n\n".join(context_parts)
    
    def build_trading_knowledge_base(self):
        """Build initial trading knowledge base"""
        logger.info("Building trading knowledge base...")
        
        trading_knowledge = [
            {
                'content': """Technical Analysis: RSI above 70 indicates overbought conditions, 
                below 30 indicates oversold. MACD crossover signals trend changes. 
                Bollinger Bands show volatility expansion and contraction.""",
                'metadata': {'category': 'technical', 'source': 'docs'}
            },
            {
                'content': """Risk Management: Never risk more than 2% of capital on a single trade. 
                Use stop-loss orders at 1.5x ATR. Position sizing should be based on Kelly Criterion 
                or fixed fractional method.""",
                'metadata': {'category': 'risk', 'source': 'docs'}
            },
            {
                'content': """Market Regimes: Trending markets favor momentum strategies. 
                Ranging markets favor mean reversion. High volatility requires smaller positions. 
                Low volatility may indicate impending breakout.""",
                'metadata': {'category': 'strategy', 'source': 'docs'}
            },
            {
                'content': """Sentiment Indicators: High funding rates indicate over-leverage. 
                Low exchange reserves suggest accumulation. Social volume spikes often precede 
                volatility. Put/call ratio extremes signal reversals.""",
                'metadata': {'category': 'sentiment', 'source': 'docs'}
            },
            {
                'content': """Entry Signals: Look for confluence of multiple indicators. 
                Higher timeframe trend alignment increases probability. Volume confirmation 
                validates breakouts. Divergences signal potential reversals.""",
                'metadata': {'category': 'entry', 'source': 'docs'}
            },
            {
                'content': """Exit Strategies: Scale out at predetermined targets. 
                Trailing stops protect profits. Time-based exits for mean reversion. 
                Volatility-based stops adapt to market conditions.""",
                'metadata': {'category': 'exit', 'source': 'docs'}
            }
        ]
        
        self.add_documents_batch(trading_knowledge)
        logger.info(f"Knowledge base built with {len(trading_knowledge)} documents")
    
    def save(self):
        """Save knowledge base to disk"""
        self.vector_store.save(f"{self.storage_path}/vector_store.pkl")
        
        # Save categories
        with open(f"{self.storage_path}/categories.json", 'w') as f:
            json.dump(self.categories, f)
        
        logger.info("Knowledge base saved")
    
    def load(self):
        """Load knowledge base from disk"""
        vector_path = f"{self.storage_path}/vector_store.pkl"
        if os.path.exists(vector_path):
            self.vector_store.load(vector_path)
        
        categories_path = f"{self.storage_path}/categories.json"
        if os.path.exists(categories_path):
            with open(categories_path, 'r') as f:
                self.categories = json.load(f)
        
        logger.info("Knowledge base loaded")