# app/rag/knowledge_loader.py
# Knowledge base loader for RAG system

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class KnowledgeBaseLoader:
    """Load and manage knowledge base for RAG system"""
    
    def __init__(self, knowledge_path: str = "data/knowledge/"):
        """
        Initialize knowledge base loader
        
        Args:
            knowledge_path: Path to knowledge base directory
        """
        self.knowledge_path = Path(knowledge_path)
        self.documents = []
        self.embeddings = None
        self.index = None
        self.model = None
        
    def load(self) -> bool:
        """
        Load knowledge base from disk
        
        Returns:
            Success status
        """
        try:
            # Load documents
            docs_file = self.knowledge_path / "documents.json"
            if not docs_file.exists():
                logger.error(f"Documents file not found: {docs_file}")
                return False
            
            with open(docs_file, 'r') as f:
                data = json.load(f)
            self.documents = data['documents']
            
            # Load embeddings
            embeddings_file = self.knowledge_path / "embeddings.npy"
            if embeddings_file.exists():
                self.embeddings = np.load(embeddings_file)
                logger.info(f"Loaded embeddings: {self.embeddings.shape}")
            
            # Load FAISS index
            index_file = self.knowledge_path / "vectors.faiss"
            if index_file.exists():
                import faiss
                self.index = faiss.read_index(str(index_file))
                logger.info(f"Loaded FAISS index: {self.index.ntotal} vectors")
            
            logger.info(f"Loaded {len(self.documents)} documents from knowledge base")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load knowledge base: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search knowledge base for relevant documents
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of relevant documents with scores
        """
        if self.index is None:
            logger.warning("FAISS index not loaded")
            return []
        
        try:
            from sentence_transformers import SentenceTransformer
            
            # Lazy load model
            if self.model is None:
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Generate query embedding
            query_embedding = self.model.encode([query], normalize_embeddings=True)
            
            # Search
            scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
            
            results = []
            for idx, score in zip(indices[0], scores[0]):
                if idx >= 0 and idx < len(self.documents):
                    results.append({
                        'document': self.documents[idx],
                        'score': float(score)
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_context(self, query: str, top_k: int = 3) -> str:
        """
        Get context string for RAG prompt
        
        Args:
            query: Search query
            top_k: Number of documents to include
        
        Returns:
            Context string
        """
        results = self.search(query, top_k)
        
        if not results:
            return "No relevant context available."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            doc = result['document']
            context_parts.append(f"[Source: {doc['metadata']['category']}]\n{doc['content']}")
        
        return "\n\n---\n\n".join(context_parts)
    
    def get_document_count(self) -> int:
        """Get total number of documents"""
        return len(self.documents)
    
    def get_categories(self) -> List[str]:
        """Get unique categories in knowledge base"""
        categories = set()
        for doc in self.documents:
            categories.add(doc['metadata'].get('category', 'general'))
        return list(categories)


# Singleton instance
_knowledge_base = None


def get_knowledge_base() -> KnowledgeBaseLoader:
    """Get or create knowledge base singleton"""
    global _knowledge_base
    if _knowledge_base is None:
        _knowledge_base = KnowledgeBaseLoader()
        _knowledge_base.load()
    return _knowledge_base