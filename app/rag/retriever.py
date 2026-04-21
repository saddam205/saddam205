"""
retriever.py
Part of the app/rag module.
Document retrieval with hybrid search (semantic + keyword).
"""

import re
from typing import List, Dict, Optional, Tuple
from collections import Counter
import logging

from .knowledge_base import KnowledgeBase, Document
from .embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


class DocumentRetriever:
    """
    Hybrid document retriever combining semantic and keyword search
    """
    
    def __init__(self, knowledge_base: KnowledgeBase):
        """
        Initialize retriever
        
        Args:
            knowledge_base: Knowledge base instance
        """
        self.kb = knowledge_base
        self.embedder = EmbeddingGenerator()
        
    def keyword_search(self, query: str, documents: List[Document], 
                       top_k: int = 10) -> List[Tuple[Document, float]]:
        """
        Perform keyword-based search using TF-IDF style scoring
        
        Args:
            query: Search query
            documents: List of documents to search
            top_k: Number of results
        
        Returns:
            List of (document, score) tuples
        """
        # Extract keywords from query
        keywords = re.findall(r'\b\w+\b', query.lower())
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                    'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        keywords = [k for k in keywords if k not in stopwords and len(k) > 2]
        
        if not keywords:
            return []
        
        # Score documents based on keyword frequency
        scores = []
        for doc in documents:
            doc_text = doc.content.lower()
            score = sum(doc_text.count(kw) for kw in keywords)
            # Boost by keyword uniqueness
            unique_keywords = len(set(keywords))
            score = score * (1 + 0.1 * unique_keywords)
            scores.append(score)
        
        # Normalize and sort
        max_score = max(scores) if scores else 1
        normalized_scores = [s / max_score for s in scores]
        
        scored_docs = list(zip(documents, normalized_scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return scored_docs[:top_k]
    
    def hybrid_search(self, query: str, category: str = None, 
                      top_k: int = 10, semantic_weight: float = 0.7) -> List[Dict]:
        """
        Hybrid search combining semantic and keyword methods
        
        Args:
            query: Search query
            category: Category filter
            top_k: Number of results
            semantic_weight: Weight for semantic search (0-1)
        
        Returns:
            List of search results
        """
        # Get all documents (or filtered by category)
        if category:
            doc_ids = self.kb.categories.get(category, [])
            documents = [self.kb.vector_store.get_document(doc_id) 
                        for doc_id in doc_ids if self.kb.vector_store.get_document(doc_id)]
        else:
            documents = self.kb.vector_store.documents
        
        if not documents:
            return []
        
        # Semantic search
        query_embedding = self.embedder.get_embedding(query)
        semantic_results = self.kb.vector_store.search(query_embedding, top_k=top_k * 2)
        semantic_scores = {doc.id: score for doc, score in semantic_results}
        
        # Keyword search
        keyword_results = self.keyword_search(query, documents, top_k=top_k * 2)
        keyword_scores = {doc.id: score for doc, score in keyword_results}
        
        # Combine scores
        all_doc_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
        
        combined_results = []
        for doc_id in all_doc_ids:
            doc = self.kb.vector_store.get_document(doc_id)
            if not doc:
                continue
            
            sem_score = semantic_scores.get(doc_id, 0)
            kw_score = keyword_scores.get(doc_id, 0)
            
            # Weighted combination
            combined_score = (semantic_weight * sem_score + 
                            (1 - semantic_weight) * kw_score)
            
            combined_results.append((doc, combined_score))
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        # Format results
        formatted_results = []
        for doc, score in combined_results[:top_k]:
            formatted_results.append({
                'id': doc.id,
                'content': doc.content,
                'metadata': doc.metadata,
                'relevance_score': score,
                'semantic_score': semantic_scores.get(doc.id, 0),
                'keyword_score': keyword_scores.get(doc.id, 0),
                'timestamp': doc.timestamp.isoformat()
            })
        
        return formatted_results
    
    def retrieve_context(self, query: str, top_k: int = 5) -> str:
        """
        Retrieve context string for RAG
        
        Args:
            query: Search query
            top_k: Number of documents
        
        Returns:
            Context string
        """
        results = self.hybrid_search(query, top_k=top_k)
        
        if not results:
            return "No relevant context available."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(f"[Document {i}] {result['content']}")
        
        return "\n\n".join(context_parts)


class HybridRetriever:
    """
    Advanced hybrid retriever with reranking and filtering
    """
    
    def __init__(self, knowledge_base: KnowledgeBase):
        """
        Initialize hybrid retriever
        
        Args:
            knowledge_base: Knowledge base instance
        """
        self.kb = knowledge_base
        self.embedder = EmbeddingGenerator()
        self.reranker = None  # Can add cross-encoder reranker
        
    def retrieve(self, query: str, filters: Dict = None, 
                 top_k: int = 10) -> List[Dict]:
        """
        Retrieve documents with advanced filtering
        
        Args:
            query: Search query
            filters: Metadata filters (e.g., {'category': 'technical'})
            top_k: Number of results
        
        Returns:
            List of retrieved documents
        """
        # Get initial candidates
        query_embedding = self.embedder.get_embedding(query)
        candidates = self.kb.vector_store.search(query_embedding, top_k=top_k * 3)
        
        # Apply filters
        if filters:
            filtered_candidates = []
            for doc, score in candidates:
                match = all(doc.metadata.get(k) == v for k, v in filters.items())
                if match:
                    filtered_candidates.append((doc, score))
            candidates = filtered_candidates
        
        # Rerank (simplified - using query-document similarity)
        reranked = []
        for doc, score in candidates:
            # Boost score based on metadata
            boost = 1.0
            if doc.metadata.get('source') == 'verified':
                boost *= 1.2
            if doc.metadata.get('recency', 0) > 0.9:
                boost *= 1.1
            
            final_score = score * boost
            reranked.append((doc, final_score))
        
        # Sort by final score
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        # Format results
        results = []
        for doc, score in reranked[:top_k]:
            results.append({
                'id': doc.id,
                'content': doc.content,
                'metadata': doc.metadata,
                'relevance_score': score,
                'timestamp': doc.timestamp.isoformat()
            })
        
        return results
    
    def retrieve_by_metadata(self, metadata_filter: Dict, top_k: int = 50) -> List[Dict]:
        """
        Retrieve documents by metadata
        
        Args:
            metadata_filter: Metadata filter conditions
            top_k: Maximum number of results
        
        Returns:
            List of matching documents
        """
        results = []
        for doc in self.kb.vector_store.documents:
            match = True
            for key, value in metadata_filter.items():
                if doc.metadata.get(key) != value:
                    match = False
                    break
            
            if match:
                results.append({
                    'id': doc.id,
                    'content': doc.content,
                    'metadata': doc.metadata,
                    'timestamp': doc.timestamp.isoformat()
                })
                
                if len(results) >= top_k:
                    break
        
        return results